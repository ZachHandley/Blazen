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

    /// Native [`tokio::spawn`].
    pub fn spawn<F>(fut: F) -> JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        tokio::spawn(fut)
    }
}

#[cfg(target_arch = "wasm32")]
#[allow(dead_code, unused_imports)]
mod imp {
    pub use web_time::Instant;

    use std::future::Future;
    use std::pin::Pin;
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
    pub struct JoinHandle<T> {
        rx: oneshot::Receiver<Result<T, JoinError>>,
        abort_tx: Option<oneshot::Sender<()>>,
    }

    impl<T> JoinHandle<T> {
        /// Signal the spawned future to stop. Best-effort: if the future has
        /// already produced a value the abort is a no-op.
        pub fn abort(&mut self) {
            if let Some(tx) = self.abort_tx.take() {
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
            abort_tx: Some(abort_tx),
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
}

#[allow(unused_imports)]
pub use imp::*;
