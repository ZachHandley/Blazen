//! Async-result handle for the cabi surface.
//!
//! Every async cabi entry point exposes two C functions:
//!
//! - `blazen_X_method_blocking(...)` — synchronous, drives the future on the
//!   cabi tokio runtime via `runtime().block_on(...)`.
//! - `blazen_X_method(...)` — returns a `*mut BlazenFuture` immediately, spawns
//!   the underlying async task onto the runtime, and signals completion via a
//!   one-byte write to an internal pipe whose **read** end is observable
//!   through [`blazen_future_fd`]. Ruby calls `IO.for_fd(fd).wait_readable`,
//!   which yields to `Fiber.scheduler` if one is installed — that's the path
//!   that makes the `async` gem cooperate with Blazen calls.
//!
//! Phase R1 lays the type-erased infrastructure down. Phase R3+ adds typed
//! `blazen_future_take_<result>` extern functions per result type — each of
//! those wraps [`BlazenFuture::take_typed`] with the concrete `T` it expects
//! (because C can't see Rust generics, we monomorphise on the FFI boundary).
//!
//! ## Cross-platform pipes
//!
//! [`std::io::pipe`] (stable since Rust 1.87; workspace MSRV is 1.91) returns
//! a `(PipeReader, PipeWriter)` pair that works on unix and windows. On unix
//! the reader's `AsRawFd::as_raw_fd()` is the integer file descriptor for
//! `poll` / `epoll` / `kqueue`. On windows the reader's
//! `AsRawHandle::as_raw_handle()` is a HANDLE; we cast it to `i64` so the C
//! signature stays uniform. Windows FFI hosts that can't wait on raw HANDLEs
//! fall back to [`blazen_future_wait`].

// Spawn + take_typed are crate-private foundations used by Phase R3+ typed
// wrappers. The public extern functions are linker-preserved regardless.
#![allow(dead_code)]

use std::any::Any;
use std::future::Future;
use std::io::Read;

use parking_lot::Mutex;

use blazen_uniffi::errors::BlazenError as InnerError;

/// Cross-platform pipe pair plus the cached raw fd / handle for the reader.
///
/// The reader is kept inside a mutex so [`blazen_future_wait`] can pull it out,
/// drain the completion byte, and put it back. Storing the raw fd separately
/// means [`blazen_future_fd`] doesn't have to lock the mutex (and doesn't have
/// to keep a borrow alive across the lock guard — locking through a mutex to
/// pull the raw fd and then immediately releasing the guard would create a
/// dangling fd if the consumer also tried to `wait` concurrently).
///
/// The cached `raw_fd` is only valid as long as `reader` is `Some(_)` — i.e.
/// for the entire lifetime of the [`BlazenFuture`]. We never call
/// `libc::close` on the cached value; the `PipeReader`'s `Drop` handles
/// closing when the future is freed.
struct Pipe {
    reader: Mutex<Option<std::io::PipeReader>>,
    writer: Mutex<Option<std::io::PipeWriter>>,
    /// Cached raw fd (unix) or raw HANDLE-as-i64 (windows). See module docs.
    raw_fd: i64,
}

/// Internal completion state for a [`BlazenFuture`].
enum FutureState {
    /// Spawned task is still running.
    Pending,
    /// Spawned task finished. Box holds the typed result (downcastable via
    /// [`BlazenFuture::take_typed`]) or the error path.
    Ready(Result<Box<dyn Any + Send>, InnerError>),
    /// Result has already been consumed by [`BlazenFuture::take_typed`].
    /// A second `take_*` call returns an `Internal` error.
    Taken,
}

/// Opaque async-result handle. cbindgen renders the C side as a
/// forward-declared `typedef struct BlazenFuture BlazenFuture;` — FFI hosts
/// never inspect the layout directly.
///
/// Deliberately not `#[repr(C)]`.
pub struct BlazenFuture {
    state: Mutex<FutureState>,
    pipe: Pipe,
}

impl BlazenFuture {
    /// Spawns an async task onto the cabi tokio runtime and hands back an
    /// opaque future handle. On completion the task stores the typed result
    /// into the handle's state mutex, writes one byte to the internal pipe,
    /// and drops the writer (closing it). Consumers observe completion by
    /// either polling [`blazen_future_poll`], blocking via
    /// [`blazen_future_wait`], or waiting on the fd returned by
    /// [`blazen_future_fd`].
    ///
    /// The returned pointer is owned by the C caller. They must release it
    /// with [`blazen_future_free`] after they have observed completion AND
    /// taken the typed result (or chosen to discard it).
    pub(crate) fn spawn<T, F>(fut: F) -> *mut BlazenFuture
    where
        T: Send + 'static,
        F: Future<Output = Result<T, InnerError>> + Send + 'static,
    {
        let (reader, writer) = std::io::pipe().expect("std::io::pipe failed");

        // Capture the raw fd / HANDLE before the reader is moved into the
        // mutex so `blazen_future_fd` can return it without locking. The fd
        // stays valid as long as the `PipeReader` inside the mutex is alive
        // (i.e. for the lifetime of this `BlazenFuture`).
        let raw_fd: i64 = {
            #[cfg(unix)]
            {
                use std::os::fd::AsRawFd;
                i64::from(reader.as_raw_fd())
            }
            #[cfg(windows)]
            {
                use std::os::windows::io::AsRawHandle;
                reader.as_raw_handle() as i64
            }
            #[cfg(not(any(unix, windows)))]
            {
                // Should be unreachable on supported platforms, but keep the
                // build going with a clearly-invalid sentinel rather than a
                // compile error.
                -1
            }
        };

        let bf = Box::new(BlazenFuture {
            state: Mutex::new(FutureState::Pending),
            pipe: Pipe {
                reader: Mutex::new(Some(reader)),
                writer: Mutex::new(Some(writer)),
                raw_fd,
            },
        });
        let ptr = Box::into_raw(bf);

        // Pass the pointer through `usize` so the spawned task doesn't capture
        // a raw pointer (which would tank `Send` inference). The consumer is
        // contractually required not to free the handle until they've observed
        // completion, so dereferencing here is sound.
        let raw = ptr as usize;
        crate::runtime::runtime().spawn(async move {
            let result = fut.await;
            let boxed: Result<Box<dyn Any + Send>, InnerError> = match result {
                Ok(v) => Ok(Box::new(v) as Box<dyn Any + Send>),
                Err(e) => Err(e),
            };

            // SAFETY: the C-side contract on the cabi async surface mandates
            // that `blazen_future_free` is only called AFTER the consumer has
            // observed completion (via poll, wait, or the fd becoming
            // readable). Until then `raw` points at a live `BlazenFuture`.
            // This task is the sole producer reference; there's no aliased
            // `&mut`.
            let bf = unsafe { &*(raw as *const BlazenFuture) };
            {
                let mut state = bf.state.lock();
                *state = FutureState::Ready(boxed);
            }
            // Signal completion: write one byte, then drop the writer to send
            // EOF. The byte is what makes the reader-fd `readable` for
            // poll/select/epoll/IO.wait_readable.
            if let Some(mut w) = bf.pipe.writer.lock().take() {
                use std::io::Write;
                // The write can only fail if the reader has been closed
                // (which would only happen if the consumer freed the handle
                // early — a contract violation). Ignore the result either
                // way; the state is already set.
                let _ = w.write_all(&[1u8]);
                // `w` drops here, closing the write end.
            }
        });

        ptr
    }

    /// Pops the typed result out of the future. Called from monomorphised
    /// `blazen_future_take_<T>` extern functions in Phase R3+.
    ///
    /// Returns `Err(Internal)` if:
    /// - the future is still pending
    /// - the result has already been taken
    /// - the stored type doesn't match `T` (indicates a cabi wiring bug — a
    ///   typed `take_*` was called against the wrong future kind)
    ///
    /// On the success path the boxed result is moved out and unboxed; on the
    /// error path the inner `BlazenError` is moved out.
    ///
    /// # Safety
    ///
    /// `ptr` must be a non-null pointer produced by [`BlazenFuture::spawn`],
    /// not yet freed, and not concurrently freed from another thread.
    // `InnerError` is large (~168 bytes), but every cabi consumer ultimately
    // funnels it into a `Box<BlazenError>` via `BlazenError::into_ptr` before
    // crossing the FFI boundary, so the temporary `Result<T, InnerError>`
    // doesn't actually appear on a hot path. Allow the lint here rather than
    // forcing a `Box<InnerError>` indirection.
    #[allow(clippy::result_large_err)]
    pub(crate) unsafe fn take_typed<T: Send + 'static>(
        ptr: *mut BlazenFuture,
    ) -> Result<T, InnerError> {
        if ptr.is_null() {
            return Err(InnerError::Internal {
                message: "blazen_future_take_*: null future pointer".into(),
            });
        }
        // SAFETY: caller has guaranteed `ptr` is a live `BlazenFuture`.
        let bf = unsafe { &*ptr };
        let mut state = bf.state.lock();
        let current = std::mem::replace(&mut *state, FutureState::Taken);
        match current {
            FutureState::Pending => {
                // Restore the state — nothing was taken.
                *state = FutureState::Pending;
                Err(InnerError::Internal {
                    message: "blazen_future_take_*: future not ready".into(),
                })
            }
            FutureState::Taken => Err(InnerError::Internal {
                message: "blazen_future_take_*: result already consumed".into(),
            }),
            FutureState::Ready(Err(e)) => Err(e),
            FutureState::Ready(Ok(boxed)) => match boxed.downcast::<T>() {
                Ok(t) => Ok(*t),
                Err(_) => Err(InnerError::Internal {
                    message: "blazen_future_take_*: type mismatch".into(),
                }),
            },
        }
    }
}

/// Returns a read-only file descriptor that becomes readable once the future
/// completes. On unix this is the raw pipe fd produced by `pipe(2)`. On
/// windows it is the raw HANDLE for the pipe's read end, cast to `i64`;
/// windows FFI hosts that can't wait on raw HANDLEs should fall back to
/// [`blazen_future_wait`] instead.
///
/// Use the returned fd with `poll(2)` / `select(2)` / `IO.wait_readable`
/// (Ruby) / `epoll`. Do NOT read from the fd directly — call the appropriate
/// `blazen_future_take_*` after the fd indicates readiness. The fd is owned
/// by the future; closing it manually (via `close(2)` / `CloseHandle`) is
/// undefined behavior.
///
/// Returns `-1` if `fut` is null or the host platform doesn't expose the pipe
/// as either a unix fd or a windows HANDLE.
///
/// # Safety
///
/// `fut` must be null OR a valid pointer to a `BlazenFuture` produced by the
/// cabi surface (and not yet freed).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_fd(fut: *const BlazenFuture) -> i64 {
    if fut.is_null() {
        return -1;
    }
    // SAFETY: caller has guaranteed `fut` is a live `BlazenFuture` pointer.
    let bf = unsafe { &*fut };
    bf.pipe.raw_fd
}

/// Non-blocking readiness check. Returns:
/// - `1` if the future has completed and the result is available to take
///   (whether the result is `Ok` or `Err`)
/// - `0` if the future is still pending
/// - `-1` if `fut` is null
///
/// Does not consume the result — safe to call repeatedly. After this returns
/// `1`, call the matching typed `blazen_future_take_*` to pop the result.
///
/// # Safety
///
/// `fut` must be null OR a valid pointer to a `BlazenFuture` produced by the
/// cabi surface (and not yet freed).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_poll(fut: *const BlazenFuture) -> i32 {
    if fut.is_null() {
        return -1;
    }
    // SAFETY: caller has guaranteed `fut` is a live `BlazenFuture` pointer.
    let bf = unsafe { &*fut };
    let state = bf.state.lock();
    match &*state {
        FutureState::Pending => 0,
        FutureState::Ready(_) | FutureState::Taken => 1,
    }
}

/// Blocks the calling thread until the future completes. Returns `0` on
/// success, `-1` if `fut` is null. The typed result remains available for a
/// subsequent `blazen_future_take_*`.
///
/// Internally drains the one completion byte from the pipe's read end. If
/// the byte has already been drained by a previous `blazen_future_wait`, the
/// second call sees EOF and still returns `0` (the future is definitionally
/// complete at that point — the writer has already been dropped).
///
/// Roughly equivalent to `IO.for_fd(blazen_future_fd(fut)).read(1)` from
/// Ruby, but without round-tripping through the Ruby IO layer.
///
/// # Safety
///
/// `fut` must be null OR a valid pointer to a `BlazenFuture` produced by the
/// cabi surface (and not yet freed).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_wait(fut: *mut BlazenFuture) -> i32 {
    if fut.is_null() {
        return -1;
    }
    // SAFETY: caller has guaranteed `fut` is a live `BlazenFuture` pointer.
    let bf = unsafe { &*fut };

    // Fast path: if the state is already `Ready` or `Taken`, the spawned task
    // has finished and the pipe byte is either still in the kernel buffer
    // (harmless) or already drained. Either way we can short-circuit without
    // touching the reader.
    {
        let state = bf.state.lock();
        if !matches!(&*state, FutureState::Pending) {
            return 0;
        }
    }

    // Slow path: drain one byte from the pipe. Pull the reader out of the
    // mutex temporarily so the read isn't holding the mutex across the
    // blocking syscall — important if another caller polls concurrently.
    let Some(mut reader) = bf.pipe.reader.lock().take() else {
        // Another thread is concurrently waiting on the same future. The
        // double-wait case isn't part of the documented contract (FFI hosts
        // should serialise their own access to a single future), but be
        // defensive: spin until the state flips out of `Pending`, then
        // return.
        loop {
            if !matches!(&*bf.state.lock(), FutureState::Pending) {
                return 0;
            }
            std::thread::yield_now();
        }
    };

    let mut byte = [0u8; 1];
    // A successful read of 1 byte means the completion signal arrived.
    // `read_exact` may return `UnexpectedEof` if the writer was dropped
    // before writing — that shouldn't happen in normal flow (we always
    // write before dropping the writer), but treat it as completion anyway:
    // the only way the writer can be dropped without a write is if the
    // spawned task itself panicked, in which case the state mutex was
    // poisoned and parking_lot will have surfaced that via its own panic.
    let _ = reader.read_exact(&mut byte);

    // Put the reader back so its `Drop` runs at `blazen_future_free` time
    // (which closes the fd). This also makes a second `wait` call see EOF
    // immediately instead of erroring on a missing reader.
    *bf.pipe.reader.lock() = Some(reader);

    0
}

/// Frees the future handle. If the typed result was never consumed by a
/// `blazen_future_take_*`, the boxed value (or the unread `BlazenError`) is
/// dropped here. No-op on a null pointer.
///
/// Closing the internal pipe fds happens automatically when the contained
/// `PipeReader` and `PipeWriter` drop — callers must NOT separately close
/// the fd returned by [`blazen_future_fd`].
///
/// # Safety
///
/// `fut` must be null OR a pointer previously produced by the cabi async
/// surface (and not yet freed). Calling this twice on the same non-null
/// pointer is a double-free; calling it while the spawned task hasn't yet
/// signalled completion is undefined behavior on the C side (the spawned
/// task is still holding a non-aliased reference to the handle's state).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blazen_future_free(fut: *mut BlazenFuture) {
    if fut.is_null() {
        return;
    }
    // SAFETY: per the contract above, `fut` was produced by `Box::into_raw`
    // over a `BlazenFuture`, so reconstructing the `Box` here is sound and
    // `drop` releases the original allocation (including the pipe halves and
    // any unread typed result).
    drop(unsafe { Box::from_raw(fut) });
}
