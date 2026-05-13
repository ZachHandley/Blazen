//! `ToolHandler` foreign-callback trampoline + `blazen_agent_new` constructor.
//! Bridges a C vtable of function pointers into a Rust `Arc<dyn ToolHandler>`
//! so the agent loop can invoke foreign-language tool implementations.
//!
//! Phase R5 Agent B.
//!
//! ## Why a vtable, not a single callback
//!
//! Foreign hosts (Ruby's `ffi` gem, Dart's `NativeCallable`, Crystal's `->`
//! procs, Lua's `ffi.cast`, PHP's FFI) each have their own way of
//! materialising a function pointer that can be invoked from Rust. To keep
//! the C side trivially describable in any of those, the cabi surface accepts
//! a flat `#[repr(C)]` vtable carrying `(user_data, drop_user_data, execute)`
//! function pointers. The foreign side owns the lifecycle of `user_data`
//! (its boxed Ruby block, its Dart isolate token, etc.) and provides the
//! `drop_user_data` thunk we call when the wrapper is dropped.
//!
//! ## Thread-safety
//!
//! `BlazenToolHandlerVTable` is `Send + Sync`: the foreign side is
//! responsible for making `user_data` and the `execute` thunk safe to call
//! from arbitrary tokio worker threads. Ruby's `ffi` gem reacquires the GVL
//! around the callback; Dart's `NativeCallable` shuttles work back to the
//! isolate; Crystal and Lua run their callbacks on the calling thread. None
//! of those mechanisms are visible from Rust, so we trust the foreign side
//! to uphold the `Send + Sync` contract on its end.
//!
//! ## Call-path
//!
//! The agent loop invokes `ToolHandler::execute` from inside a tokio task.
//! We don't know whether the foreign `execute` thunk is willing to block
//! (Ruby's GVL acquisition can park), so the trampoline always dispatches
//! the call through `tokio::task::spawn_blocking`. That keeps a single slow
//! tool handler from starving other tasks on the runtime's worker threads.

use std::ffi::{CString, c_char, c_void};
use std::sync::Arc;

use async_trait::async_trait;
use blazen_uniffi::agent::{Agent as InnerAgent, ToolHandler};
use blazen_uniffi::errors::{BlazenError as InnerError, BlazenResult};

use crate::agent::BlazenAgent;
use crate::error::BlazenError;
use crate::llm::BlazenCompletionModel;
use crate::llm_records::BlazenTool;
use crate::string::cstr_to_opt_string;

// ---------------------------------------------------------------------------
// Error-out helpers (module-private, mirror the agent.rs / workflow.rs shape)
// ---------------------------------------------------------------------------

/// Writes `e` to the out-param if non-null and returns `-1`.
///
/// # Safety
///
/// `out_err` must be null OR a valid destination for a single
/// `*mut BlazenError` write.
unsafe fn write_error(out_err: *mut *mut BlazenError, e: InnerError) -> i32 {
    if !out_err.is_null() {
        // SAFETY: caller-supplied out-param; per the function-level contract
        // it's either null (handled above) or a valid destination for a
        // single pointer-sized write.
        unsafe {
            *out_err = BlazenError::from(e).into_ptr();
        }
    }
    -1
}

/// Writes a synthesised `Internal` error to the out-param and returns `-1`.
///
/// # Safety
///
/// Same contract as [`write_error`].
unsafe fn write_internal_error(out_err: *mut *mut BlazenError, msg: &str) -> i32 {
    // SAFETY: forwarded to `write_error`; caller upholds the same contract.
    unsafe {
        write_error(
            out_err,
            InnerError::Internal {
                message: msg.into(),
            },
        )
    }
}

// ---------------------------------------------------------------------------
// VTable
// ---------------------------------------------------------------------------

/// Function-pointer vtable a foreign caller fills out to implement
/// [`ToolHandler`] across the C ABI.
///
/// `user_data` is opaque to Rust — it's whatever the foreign side wants to
/// associate with the handler (a boxed Ruby block, a Dart isolate token, a
/// Crystal proc, etc.). It is passed back unchanged to every `execute`
/// invocation and to `drop_user_data` when the wrapper is dropped.
///
/// ## `execute` contract
///
/// - `tool_name` and `arguments_json` are caller-owned C strings that remain
///   valid for the duration of the call. The callback MUST NOT free them.
/// - On success, the callback writes a heap-allocated NUL-terminated UTF-8
///   string into `*out_result_json`. The string must have been allocated in
///   a way that is compatible with [`crate::string::blazen_string_free`]
///   (i.e. `CString::into_raw` on the Rust side, or anything the foreign
///   side hands back through `blazen_string_alloc` if one exists). The
///   cabi surface frees the string with `blazen_string_free`.
/// - On failure, the callback writes a caller-owned `*mut BlazenError` into
///   `*out_err`. The cabi surface frees it with `blazen_error_free`.
/// - The return value is `0` on success and `-1` on failure. Any other
///   non-zero value is treated as a failure but logged as a protocol
///   violation in the synthesised `Internal` error message.
///
/// ## `drop_user_data` contract
///
/// Called exactly once when the [`CToolHandler`] wrapper is dropped — i.e.
/// when the [`InnerAgent`] referencing this handler is dropped and the last
/// reference goes away. The foreign side releases whatever resources back
/// `user_data` here.
#[repr(C)]
pub struct BlazenToolHandlerVTable {
    /// Opaque foreign-side pointer. Passed back to `execute` and
    /// `drop_user_data` unchanged.
    pub user_data: *mut c_void,
    /// Invoked when the wrapper is dropped. The foreign side releases
    /// `user_data` here.
    pub drop_user_data: extern "C" fn(user_data: *mut c_void),
    /// Executes the named tool. See the struct-level docs for the contract.
    pub execute: extern "C" fn(
        user_data: *mut c_void,
        tool_name: *const c_char,
        arguments_json: *const c_char,
        out_result_json: *mut *mut c_char,
        out_err: *mut *mut BlazenError,
    ) -> i32,
}

// SAFETY: the foreign side guarantees its `user_data` + `execute` thunk are
// safe to use from arbitrary tokio worker threads. Ruby's `ffi` gem
// reacquires the GVL around the callback; Dart's `NativeCallable` ships work
// back to its owning isolate; Crystal / Lua / PHP callbacks run on the
// calling thread. None of these guarantees are visible from Rust — the
// vtable is a pure foreign-language contract, and the cabi surface trusts
// the foreign side to uphold it on its end.
unsafe impl Send for BlazenToolHandlerVTable {}
// SAFETY: see the `Send` impl. Calling `execute` concurrently from multiple
// threads is part of the same foreign-side contract.
unsafe impl Sync for BlazenToolHandlerVTable {}

// ---------------------------------------------------------------------------
// Wrapper struct
// ---------------------------------------------------------------------------

/// Rust-side wrapper that owns a [`BlazenToolHandlerVTable`] and implements
/// [`ToolHandler`] by dispatching through the vtable's function pointers.
///
/// One instance is constructed per [`blazen_agent_new`] call. The wrapper
/// is held inside an `Arc<dyn ToolHandler>` that the inner agent owns; once
/// the agent is dropped, the wrapper drops, and the foreign-side
/// `user_data` is released via `drop_user_data`.
pub(crate) struct CToolHandler {
    vtable: BlazenToolHandlerVTable,
}

impl Drop for CToolHandler {
    fn drop(&mut self) {
        // SAFETY: by the vtable contract, `drop_user_data` is the foreign
        // side's release thunk for `user_data` and is safe to call exactly
        // once when the wrapper is destroyed. We haven't called it before
        // (no other path in this module invokes it after a `CToolHandler`
        // is constructed).
        (self.vtable.drop_user_data)(self.vtable.user_data);
    }
}

#[async_trait]
impl ToolHandler for CToolHandler {
    // `InnerError` is large (it carries every variant's payload inline), but
    // it's the shared error type across `blazen_uniffi` and we don't get to
    // choose its representation here.
    #[allow(clippy::result_large_err)]
    async fn execute(&self, tool_name: String, arguments_json: String) -> BlazenResult<String> {
        // Capture pointer + fn-pointer as primitives so the spawn_blocking
        // closure doesn't need to borrow `self` (which is `&CToolHandler`,
        // not `'static`).
        let user_data_addr = self.vtable.user_data as usize;
        let execute_fn = self.vtable.execute;

        let join = tokio::task::spawn_blocking(move || -> Result<String, InnerError> {
            // Build C strings from the inputs. `CString::new` fails on
            // interior NUL bytes — surface that as an Internal error rather
            // than a panic.
            let tool_name_c = CString::new(tool_name).map_err(|_| InnerError::Internal {
                message: "tool_name contains interior NUL byte".into(),
            })?;
            let args_c = CString::new(arguments_json).map_err(|_| InnerError::Internal {
                message: "arguments_json contains interior NUL byte".into(),
            })?;
            let user_data = user_data_addr as *mut c_void;

            let mut out_json: *mut c_char = std::ptr::null_mut();
            let mut out_err: *mut BlazenError = std::ptr::null_mut();

            let status = execute_fn(
                user_data,
                tool_name_c.as_ptr(),
                args_c.as_ptr(),
                &raw mut out_json,
                &raw mut out_err,
            );

            if status == 0 {
                if out_json.is_null() {
                    return Err(InnerError::Internal {
                        message: "tool handler returned success but null result".into(),
                    });
                }
                // Reclaim the foreign-side result string. By the vtable
                // contract this is a `CString::into_raw`-compatible
                // allocation — taking it back here matches how the cabi
                // would otherwise free it via `blazen_string_free`.
                //
                // SAFETY: per the `execute` contract, `out_json` is a heap
                // NUL-terminated UTF-8 buffer allocated such that
                // `CString::from_raw` is sound. The foreign side has
                // transferred ownership to us.
                let owned = unsafe { CString::from_raw(out_json) };
                Ok(owned.to_string_lossy().into_owned())
            } else {
                if out_err.is_null() {
                    return Err(InnerError::Internal {
                        message: format!(
                            "tool handler returned non-zero status ({status}) without setting out_err"
                        ),
                    });
                }
                // SAFETY: per the `execute` contract on failure, `out_err`
                // is a caller-owned `*mut BlazenError` produced by the cabi
                // surface (or a foreign-side equivalent that goes through
                // `BlazenError::into_ptr`). Reclaim ownership so the box
                // drops at end of scope; we only need the inner error.
                let be = unsafe { Box::from_raw(out_err) };
                Err(be.inner)
            }
        })
        .await;

        match join {
            Ok(Ok(s)) => Ok(s),
            Ok(Err(e)) => Err(e),
            Err(join_err) => Err(InnerError::Internal {
                message: format!("tool handler task panicked: {join_err}"),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// C entry point: blazen_agent_new
// ---------------------------------------------------------------------------

/// Constructs a [`BlazenAgent`] from a completion model, optional system
/// prompt, tool list, tool-handler vtable, and max iterations.
///
/// On success returns `0` and writes a fresh `*mut BlazenAgent` into
/// `*out_agent`. On failure returns `-1` and writes a fresh `*mut BlazenError`
/// into `*out_err`. Both out-params may be null to discard the corresponding
/// side of the result.
///
/// ## Ownership
///
/// - `model` is BORROWED — the underlying `Arc<CompletionModel>` is cloned
///   into the agent. The caller retains its handle and is still responsible
///   for freeing it.
/// - `system_prompt` is BORROWED for the duration of this call (it is copied
///   into the agent before the call returns). A null pointer means "no
///   system prompt".
/// - `tools` is BORROWED at the array level, but each `*mut BlazenTool`
///   element is CONSUMED — the inner `Tool` record is moved into the agent's
///   tool vec. Callers must NOT free the individual tool handles afterwards
///   (the array itself remains caller-owned).
/// - `tool_handler` is CONSUMED — ownership of `user_data` transfers to the
///   constructed [`CToolHandler`], which releases it via `drop_user_data`
///   when the wrapper drops. Early-return error paths still invoke
///   `drop_user_data` so the foreign side doesn't leak.
/// - `out_agent` receives a caller-owned `*mut BlazenAgent`. Free with
///   [`crate::agent::blazen_agent_free`].
///
/// # Safety
///
/// `model` must be null OR a live `BlazenCompletionModel` produced by the
/// cabi surface. `system_prompt` must be null OR a NUL-terminated UTF-8
/// buffer valid for the duration of this call. When `tools_count > 0`,
/// `tools` must point to an array of exactly `tools_count` valid
/// `*mut BlazenTool` entries, each produced by the cabi surface; ownership
/// of each element transfers to this function (the array itself stays
/// caller-owned). `tool_handler.user_data` and the `execute` /
/// `drop_user_data` thunks must satisfy the contracts documented on
/// [`BlazenToolHandlerVTable`]. `out_agent` and `out_err` must each be null
/// OR a writable slot for a single `*mut` write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_agent_new(
    model: *const BlazenCompletionModel,
    system_prompt: *const c_char,
    tools: *const *mut BlazenTool,
    tools_count: usize,
    tool_handler: BlazenToolHandlerVTable,
    max_iterations: u32,
    out_agent: *mut *mut BlazenAgent,
    out_err: *mut *mut BlazenError,
) -> i32 {
    // ---- Validate model + reclaim tools (consuming them on the happy path,
    // dropping them on the error path) -----------------------------------

    if model.is_null() {
        // No tools were reclaimed yet (we error out before the loop). The
        // tool_handler hasn't been wrapped, so we still need to drop its
        // user_data to honour the consume-on-call contract.
        (tool_handler.drop_user_data)(tool_handler.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "blazen_agent_new: null model") };
    }
    if tools_count > 0 && tools.is_null() {
        (tool_handler.drop_user_data)(tool_handler.user_data);
        // SAFETY: `out_err` upholds the function-level contract.
        return unsafe { write_internal_error(out_err, "blazen_agent_new: null tools array") };
    }

    // Reclaim each tool pointer. Collect them in a Vec<Box<BlazenTool>>
    // first so a later validation failure can drop them via Vec's RAII
    // instead of leaking. We refuse null elements eagerly: the foreign side
    // is supposed to fill the array with live handles.
    let mut tool_boxes: Vec<Box<BlazenTool>> = Vec::with_capacity(tools_count);
    for i in 0..tools_count {
        // SAFETY: caller guarantees `tools` indexes `tools_count` valid
        // pointer-sized slots when `tools_count > 0`.
        let ptr = unsafe { *tools.add(i) };
        if ptr.is_null() {
            // `tool_boxes` is dropped at end of scope — each reclaimed
            // `BlazenTool` is freed cleanly.
            (tool_handler.drop_user_data)(tool_handler.user_data);
            // SAFETY: `out_err` upholds the function-level contract.
            return unsafe {
                write_internal_error(out_err, "blazen_agent_new: null tool pointer in array")
            };
        }
        // SAFETY: per the function-level contract each element is a live
        // `BlazenTool` produced by the cabi surface, and ownership transfers
        // to this function.
        let boxed = unsafe { Box::from_raw(ptr) };
        tool_boxes.push(boxed);
    }

    // ---- Borrow / copy remaining inputs --------------------------------

    // SAFETY: caller has guaranteed `model` is a live `BlazenCompletionModel`.
    let model_handle = unsafe { &*model };
    let model_arc = Arc::clone(&model_handle.0);

    // SAFETY: caller upholds the NUL-terminated UTF-8 contract on
    // `system_prompt` (null is valid and becomes `None`).
    let system_prompt_opt = unsafe { cstr_to_opt_string(system_prompt) };

    // Move each inner Tool record out of its Box.
    let inner_tools: Vec<_> = tool_boxes.into_iter().map(|b| b.0).collect();

    // ---- Build the agent ------------------------------------------------

    let handler: Arc<dyn ToolHandler> = Arc::new(CToolHandler {
        vtable: tool_handler,
    });
    // `tool_handler` is now owned by the `CToolHandler` inside `handler` —
    // any subsequent early-return MUST NOT call `drop_user_data` again
    // (Drop will fire when the Arc is dropped).

    let inner_agent: Arc<InnerAgent> = InnerAgent::new(
        model_arc,
        system_prompt_opt,
        inner_tools,
        handler,
        max_iterations,
    );

    let cabi_agent = BlazenAgent::from(inner_agent);
    let raw = cabi_agent.into_ptr();

    if out_agent.is_null() {
        // No out_agent slot — caller has thrown away the handle. Free it
        // immediately rather than leak. (The contract documents this as a
        // discard path on success.)
        // SAFETY: `raw` was just produced by `Box::into_raw` in
        // `into_ptr`, so reconstructing the `Box` is sound.
        drop(unsafe { Box::from_raw(raw) });
    } else {
        // SAFETY: caller-supplied out-param; per the function-level contract
        // it is either null (handled above) or a valid destination for a
        // single pointer-sized write.
        unsafe {
            *out_agent = raw;
        }
    }
    0
}
