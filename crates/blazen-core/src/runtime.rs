//! Cross-target async runtime abstractions.
//!
//! Native delegates to tokio. wasm32 uses single-threaded
//! [`wasm_bindgen_futures::spawn_local`] together with
//! [`futures_util::stream::FuturesUnordered`]. The public API names
//! (`spawn`, `JoinHandle`, `JoinSet`, `JoinError`, `Instant`) match tokio
//! exactly so callers can swap `tokio::{spawn, task::*}` for
//! `crate::runtime::*` without further changes.
//!
//! Only the small subset of the tokio API that `blazen-core` actually uses
//! is mirrored:
//!
//! - [`spawn`] returning a [`JoinHandle<T>`] that is `Future<Output = Result<T, JoinError>>`.
//! - [`JoinSet<T>`] with `new`, `spawn`, `join_next`, `is_empty`, `len`.
//! - [`Instant`] re-exported from `std::time` on native, `web_time` on wasm32.

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code, unused_imports)]
mod imp {
    pub use std::time::Instant;
    pub use tokio::task::{JoinError, JoinHandle, JoinSet};
    pub use tokio::time::sleep;
    pub use tokio::time::timeout;

    /// Native [`tokio::spawn`].
    pub fn spawn<F>(fut: F) -> JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        tokio::spawn(fut)
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
#[allow(dead_code, unused_imports)]
mod imp {
    pub use web_time::Instant;

    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, OnceLock};
    use std::task::{Context, Poll};
    use std::time::Duration;

    use futures_util::future::BoxFuture;
    use futures_util::stream::{FuturesUnordered, StreamExt};
    use tokio::sync::{Notify, oneshot};

    /// Boxed future the host spawner is asked to drive to completion.
    pub type WasiBoxFuture = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

    /// Process-wide spawner registered once by the embedding host (e.g.
    /// `blazen-node`). The host typically schedules each future on the JS
    /// microtask queue via a `ThreadsafeFunction` + `Promise.resolve()` so
    /// futures advance without `std::thread::spawn`'d block_on threads —
    /// which Cloudflare workerd's single-isolate WASI runtime forbids.
    pub type Spawner = Arc<dyn Fn(WasiBoxFuture) + Send + Sync>;

    static SPAWNER: OnceLock<Spawner> = OnceLock::new();

    /// Register the process-wide spawner. Called once at napi `module_init`
    /// by `blazen-node`. First writer wins; subsequent calls return `Err`
    /// carrying the rejected spawner.
    ///
    /// # Errors
    ///
    /// Returns the rejected spawner if a spawner is already registered.
    pub fn register_spawner(f: Spawner) -> Result<(), Spawner> {
        SPAWNER.set(f)
    }

    /// Boxed future returned by the host sleeper, resolved when the requested
    /// duration has elapsed.
    pub type SleepFut = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

    /// Process-wide sleeper registered once by the embedding host. The host
    /// typically schedules the resolution via JS `setTimeout` (napi env) so
    /// timers fire without tokio's time wheel — which Cloudflare workerd's
    /// single-isolate WASI runtime cannot drive.
    pub type SleeperFn = Arc<dyn Fn(Duration) -> SleepFut + Send + Sync>;

    static SLEEPER: OnceLock<SleeperFn> = OnceLock::new();

    /// Register a process-wide sleeper. Called once at napi `module_exports`
    /// by `blazen-node` (uses JS `setTimeout` via napi env to schedule the
    /// resolution). First writer wins.
    ///
    /// # Errors
    ///
    /// Returns the rejected sleeper if one is already registered.
    pub fn register_sleeper(f: SleeperFn) -> Result<(), SleeperFn> {
        SLEEPER.set(f)
    }

    /// Wasi sleep — delegates to the registered sleeper (typically JS
    /// `setTimeout`). Falls back to `tokio::time::sleep` if no sleeper is
    /// registered (compat for native unit tests that exercise the wasi cfg
    /// path in CI).
    pub async fn sleep(dur: Duration) {
        if let Some(s) = SLEEPER.get() {
            s(dur).await;
        } else {
            tokio::time::sleep(dur).await;
        }
    }

    /// Wasi mirror of `tokio::time::error::Elapsed`.
    #[derive(Debug)]
    pub struct Elapsed;

    impl core::fmt::Display for Elapsed {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str("deadline has elapsed")
        }
    }

    impl std::error::Error for Elapsed {}

    /// Wasi timeout — races the future against `sleep(dur)`. Returns
    /// `Ok(F::Output)` if the future completes first, `Err(Elapsed)` if the
    /// sleep fires first. Mirrors the browser-wasm32 polyfill below.
    pub async fn timeout<F>(dur: Duration, fut: F) -> Result<F::Output, Elapsed>
    where
        F: Future,
    {
        tokio::select! {
            v = fut => Ok(v),
            () = sleep(dur) => Err(Elapsed),
        }
    }

    /// Dispatch a boxed future onto the registered spawner, falling back to
    /// `futures_executor::block_on` when no host spawner has been
    /// registered. The fallback drives the future inline — that surfaces
    /// host misconfiguration immediately rather than silently hanging.
    fn dispatch(work: WasiBoxFuture) {
        if let Some(spawner) = SPAWNER.get() {
            spawner(work);
        } else {
            futures_executor::block_on(work);
        }
    }

    /// Wasi mirror of [`tokio::task::JoinError`].
    ///
    /// On wasi (single-isolate workerd) there are no panics-across-threads
    /// or runtime cancellation semantics; the only way a task ends with an
    /// error is via [`JoinHandle::abort`].
    #[derive(Debug, thiserror::Error)]
    #[error("task was aborted")]
    pub struct JoinError;

    /// Wasi mirror of [`tokio::task::JoinHandle<T>`].
    ///
    /// Backed by the registered `SPAWNER` (or the inline `block_on`
    /// fallback) and a [`tokio::sync::oneshot`] channel for the output,
    /// plus an abort signal.
    ///
    /// The abort signal uses [`tokio::sync::Notify`] (rather than a
    /// `oneshot::Sender<()>`) so **dropping the `JoinHandle` does not
    /// abort the spawned task** — matching native
    /// `tokio::task::JoinHandle`'s detached-on-drop behaviour. With
    /// `oneshot`, the `Sender`'s `Drop` impl wakes its `Receiver` with
    /// an error, which a `tokio::select! { _ = abort_rx => ... }`
    /// branch matches as "abort"; the result is that
    /// `runtime::spawn(fut)` would silently kill `fut` the instant the
    /// caller dropped its `JoinHandle` (e.g.
    /// `runtime::spawn(execute_pipeline(...))` with no `let` binding).
    /// `Notify::notified()` only resolves on an explicit `notify_one`,
    /// not on `Notify` drop, so detached spawn is safe.
    pub struct JoinHandle<T> {
        rx: oneshot::Receiver<Result<T, JoinError>>,
        abort: Arc<Notify>,
    }

    impl<T> JoinHandle<T> {
        /// Signal the spawned future to stop. Best-effort: if the future has
        /// already produced a value the abort is a no-op.
        pub fn abort(&self) {
            self.abort.notify_one();
        }
    }

    impl<T> Future for JoinHandle<T> {
        type Output = Result<T, JoinError>;

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            match Pin::new(&mut self.rx).poll(cx) {
                Poll::Ready(Ok(res)) => Poll::Ready(res),
                Poll::Ready(Err(_)) => Poll::Ready(Err(JoinError)),
                Poll::Pending => Poll::Pending,
            }
        }
    }

    /// Wasi mirror of [`tokio::spawn`].
    ///
    /// Keeps the native `Send + 'static` bound for API symmetry — the
    /// workflow internals are already `Send`-clean since they have to
    /// compile against the native build.
    pub fn spawn<F>(fut: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let (out_tx, out_rx) = oneshot::channel();
        let abort = Arc::new(Notify::new());
        let abort_for_task = Arc::clone(&abort);
        let work: WasiBoxFuture = Box::pin(async move {
            tokio::select! {
                v = fut => {
                    let _ = out_tx.send(Ok(v));
                }
                () = abort_for_task.notified() => {
                    let _ = out_tx.send(Err(JoinError));
                }
            }
        });
        dispatch(work);
        JoinHandle { rx: out_rx, abort }
    }

    /// Wasi mirror of [`tokio::task::JoinSet<T>`].
    ///
    /// Backed by [`FuturesUnordered`]. Only the methods used by
    /// `blazen-core::event_loop` are provided: `new`, `spawn`, `join_next`,
    /// `is_empty`, `len`, `abort_all`.
    pub struct JoinSet<T> {
        inner: FuturesUnordered<BoxFuture<'static, Result<T, JoinError>>>,
    }

    impl<T: Send + 'static> JoinSet<T> {
        #[must_use]
        pub fn new() -> Self {
            Self {
                inner: FuturesUnordered::new(),
            }
        }

        /// Spawn a future into the set.
        ///
        /// Drives the future immediately via the registered `SPAWNER` (or
        /// the inline `block_on` fallback) so callers don't need to poll
        /// `join_next` for the future to make progress. The
        /// `FuturesUnordered` then carries only the completion signal so
        /// `join_next` still resolves in spawn order.
        pub fn spawn<F>(&mut self, fut: F)
        where
            F: Future<Output = T> + Send + 'static,
        {
            let (tx, rx) = oneshot::channel::<T>();
            let work: WasiBoxFuture = Box::pin(async move {
                let v = fut.await;
                let _ = tx.send(v);
            });
            dispatch(work);
            self.inner.push(Box::pin(async move {
                match rx.await {
                    Ok(v) => Ok(v),
                    Err(_) => Err(JoinError),
                }
            }));
        }

        /// Await the next completed task. Returns `None` when the set is
        /// empty.
        pub async fn join_next(&mut self) -> Option<Result<T, JoinError>> {
            self.inner.next().await
        }

        #[must_use]
        pub fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }

        #[must_use]
        pub fn len(&self) -> usize {
            self.inner.len()
        }

        /// Abort every spawned task. Mirrors
        /// [`tokio::task::JoinSet::abort_all`]. The underlying tasks were
        /// already dispatched via the SPAWNER and cannot be cancelled
        /// cooperatively after spawn — the set is simply drained so
        /// subsequent `join_next` calls return `None`.
        pub fn abort_all(&mut self) {
            self.inner = FuturesUnordered::new();
        }
    }

    impl<T: Send + 'static> Default for JoinSet<T> {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
#[allow(dead_code, unused_imports)]
mod imp {
    pub use web_time::Instant;

    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;
    use std::task::{Context, Poll};

    use futures_util::future::LocalBoxFuture;
    use futures_util::stream::{FuturesUnordered, StreamExt};
    use tokio::sync::oneshot;

    /// Wasm32 mirror of [`tokio::task::JoinError`].
    ///
    /// On wasm32 there are no panics-across-threads or runtime cancellation
    /// semantics; the only way a task ends with an error is via
    /// [`JoinHandle::abort`].
    #[derive(Debug, thiserror::Error)]
    #[error("task was aborted")]
    pub struct JoinError;

    /// Wasm32 mirror of [`tokio::task::JoinHandle<T>`].
    ///
    /// Backed by [`wasm_bindgen_futures::spawn_local`] and a
    /// [`tokio::sync::oneshot`] channel for the output, plus an abort flag.
    /// The abort sender is held behind a `Mutex` so `abort` can take
    /// `&self`, matching tokio's signature.
    pub struct JoinHandle<T> {
        rx: oneshot::Receiver<Result<T, JoinError>>,
        abort_tx: Mutex<Option<oneshot::Sender<()>>>,
    }

    impl<T> JoinHandle<T> {
        /// Signal the spawned future to stop. Best-effort: if the future has
        /// already produced a value the abort is a no-op.
        pub fn abort(&self) {
            if let Some(tx) = self.abort_tx.lock().ok().and_then(|mut g| g.take()) {
                let _ = tx.send(());
            }
        }
    }

    impl<T> Future for JoinHandle<T> {
        type Output = Result<T, JoinError>;

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            match Pin::new(&mut self.rx).poll(cx) {
                Poll::Ready(Ok(res)) => Poll::Ready(res),
                Poll::Ready(Err(_)) => Poll::Ready(Err(JoinError)),
                Poll::Pending => Poll::Pending,
            }
        }
    }

    /// Wasm32 mirror of [`tokio::spawn`].
    ///
    /// Note the relaxed `Send` bound: wasm32 is single-threaded, so spawned
    /// futures need not be `Send`.
    pub fn spawn<F>(fut: F) -> JoinHandle<F::Output>
    where
        F: Future + 'static,
        F::Output: 'static,
    {
        let (out_tx, out_rx) = oneshot::channel();
        let (abort_tx, abort_rx) = oneshot::channel();
        wasm_bindgen_futures::spawn_local(async move {
            tokio::select! {
                v = fut => {
                    let _ = out_tx.send(Ok(v));
                }
                _ = abort_rx => {
                    let _ = out_tx.send(Err(JoinError));
                }
            }
        });
        JoinHandle {
            rx: out_rx,
            abort_tx: Mutex::new(Some(abort_tx)),
        }
    }

    /// Wasm32 mirror of [`tokio::task::JoinSet<T>`].
    ///
    /// Backed by [`FuturesUnordered`]. Only the methods used by
    /// `blazen-core::event_loop` are provided: `new`, `spawn`, `join_next`,
    /// `is_empty`, `len`.
    pub struct JoinSet<T> {
        inner: FuturesUnordered<LocalBoxFuture<'static, Result<T, JoinError>>>,
    }

    impl<T: 'static> JoinSet<T> {
        #[must_use]
        pub fn new() -> Self {
            Self {
                inner: FuturesUnordered::new(),
            }
        }

        /// Spawn a future into the set. The future must be `'static` but is
        /// not required to be `Send` on wasm32.
        ///
        /// Unlike `FuturesUnordered::push`, this drives the future immediately
        /// via [`wasm_bindgen_futures::spawn_local`] so callers don't need to
        /// poll `join_next` for the future to make progress. The
        /// `FuturesUnordered` then carries only the completion signal so
        /// `join_next` still resolves in spawn order.
        pub fn spawn<F>(&mut self, fut: F)
        where
            F: Future<Output = T> + 'static,
        {
            let (tx, rx) = oneshot::channel::<T>();
            wasm_bindgen_futures::spawn_local(async move {
                let v = fut.await;
                let _ = tx.send(v);
            });
            self.inner.push(Box::pin(async move {
                match rx.await {
                    Ok(v) => Ok(v),
                    Err(_) => Err(JoinError),
                }
            }));
        }

        /// Await the next completed task. Returns `None` when the set is
        /// empty.
        pub async fn join_next(&mut self) -> Option<Result<T, JoinError>> {
            self.inner.next().await
        }

        #[must_use]
        pub fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }

        #[must_use]
        pub fn len(&self) -> usize {
            self.inner.len()
        }

        /// Abort every spawned task. Mirrors
        /// [`tokio::task::JoinSet::abort_all`]. On `wasm32-unknown-unknown`
        /// the underlying tasks were dispatched via `spawn_local` and
        /// cannot be cancelled cooperatively after spawn — the set is
        /// simply drained so subsequent `join_next` calls return `None`.
        /// Called by sub-workflow fan-out cleanup paths in
        /// `blazen-core::event_loop`.
        pub fn abort_all(&mut self) {
            self.inner = FuturesUnordered::new();
        }
    }

    impl<T: 'static> Default for JoinSet<T> {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Wasm32 mirror of [`tokio::time::sleep`].
    ///
    /// Delegates to JS `setTimeout` via `wasm-bindgen-futures`. Works in
    /// browsers, Workers, Node.js, and Deno — anywhere a `setTimeout`
    /// global exists.
    pub async fn sleep(dur: std::time::Duration) {
        use js_sys::Promise;
        use wasm_bindgen::JsCast;
        use wasm_bindgen::prelude::*;
        use wasm_bindgen_futures::JsFuture;

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let ms = dur.as_millis() as i32;
        let promise = Promise::new(&mut |resolve, _reject| {
            let global = js_sys::global();
            // Try the standard browser/Workers `setTimeout` global.
            // Workers, browsers, Node.js, Deno all expose this.
            let set_timeout = js_sys::Reflect::get(&global, &JsValue::from_str("setTimeout"))
                .expect("setTimeout missing from global scope");
            let set_timeout: js_sys::Function =
                set_timeout.dyn_into().expect("setTimeout not a function");
            let _ = set_timeout.call2(&JsValue::NULL, &resolve, &JsValue::from(ms));
        });
        let _ = JsFuture::from(promise).await;
    }

    /// Wasm32 mirror of [`tokio::time::timeout`].
    ///
    /// Races the supplied future against a [`sleep`] of `dur` using
    /// [`tokio::select!`] (the `macros` feature is available on wasm32; the
    /// `time` feature is not, which is why this polyfill exists). Returns
    /// `Ok(F::Output)` if the future completes first, `Err(Elapsed)` if the
    /// sleep fires first.
    ///
    /// The error type is a unit-like marker so callers using `is_err()` /
    /// `Ok(_)` / `Err(_)` keep working unchanged.
    #[derive(Debug)]
    pub struct Elapsed;

    impl core::fmt::Display for Elapsed {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str("deadline has elapsed")
        }
    }

    impl std::error::Error for Elapsed {}

    pub async fn timeout<F>(dur: std::time::Duration, fut: F) -> Result<F::Output, Elapsed>
    where
        F: Future,
    {
        tokio::select! {
            v = fut => Ok(v),
            () = sleep(dur) => Err(Elapsed),
        }
    }
}

#[allow(unused_imports)]
pub use imp::*;
