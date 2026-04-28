//! Typed `wasm-bindgen` wrappers for [`blazen_core`] types that are not
//! exposed elsewhere in the SDK.
//!
//! These wrappers fill the gaps in the public WASM surface: the workflow
//! engine produces and accepts these types, but bindings consumers were
//! previously forced to round-trip them through `serde-wasm-bindgen` and
//! reach for plain JS objects. With these wrappers, JS callers can hold
//! typed handles, perform binding-free serialisation (`MessagePack` and
//! JSON), and inspect snapshot / registry / state-value internals directly.

pub mod session_ref;
pub mod snapshot;
pub mod step;
pub mod value;
pub mod workflow_result;

pub use session_ref::{
    WasmRefLifetime, WasmRegistryKey, WasmRemoteRefDescriptor, WasmSessionRefRegistry,
};
pub use snapshot::WasmWorkflowSnapshot;
pub use step::{WasmStepOutput, WasmStepOutputEvent, WasmStepOutputKind, WasmStepRegistration};
pub use value::{WasmBytesWrapper, WasmStateValue};
pub use workflow_result::WasmWorkflowResult;

// ---------------------------------------------------------------------------
// Block-on-local helper.
// ---------------------------------------------------------------------------

/// Poll a future to completion on the current thread, panicking if it
/// returns `Pending`.
///
/// Mirrors the block-on-local helper in [`crate::context`]; duplicated here
/// (rather than re-exported) so the `core_types` module has no dependency
/// on the rest of the SDK and can be inspected in isolation. Same safety
/// argument: WASM is single-threaded, so any `Arc<RwLock<_>>` future
/// resolves on the first poll.
///
/// # Panics
///
/// Panics if the future yields `Pending`.
pub(crate) fn block_on_local<F: std::future::Future>(fut: F) -> F::Output {
    use std::pin::Pin;
    use std::task::{Context as TaskContext, Poll};
    let mut fut = Box::pin(fut);
    let waker = noop_waker();
    let mut cx = TaskContext::from_waker(&waker);
    match Pin::as_mut(&mut fut).poll(&mut cx) {
        Poll::Ready(v) => v,
        Poll::Pending => panic!(
            "core_types sync method polled future to Pending — only no-I/O \
             operations are allowed through the sync API"
        ),
    }
}

/// Construct a no-op `Waker`.
fn noop_waker() -> std::task::Waker {
    use std::task::{RawWaker, RawWakerVTable, Waker};

    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        |_| RawWaker::new(std::ptr::null(), &VTABLE),
        |_| {},
        |_| {},
        |_| {},
    );

    // SAFETY: the vtable functions are all no-ops and never dereference
    // the data pointer. A null pointer is therefore safe.
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}
