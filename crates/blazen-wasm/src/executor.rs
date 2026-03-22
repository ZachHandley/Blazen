//! Minimal async executor for WASIp2.
//!
//! All I/O in [`WasiHttpClient`](crate::http_wasi::WasiHttpClient) uses
//! `pollable.block()` internally, so futures returned by provider methods
//! complete on the first poll without needing a real async runtime.

/// Execute an async future synchronously.
///
/// This works because `WasiHttpClient::send()` performs all I/O via WASI
/// preview2 polling (`subscribe().block()`), making the future resolve
/// immediately on the first poll.
///
/// # Panics
///
/// Panics if the future returns `Pending`, which indicates a bug -- all
/// I/O in the WASM component should be synchronous under the hood.
pub fn wasi_block_on<F: std::future::Future>(fut: F) -> F::Output {
    let mut fut = std::pin::pin!(fut);
    let waker = futures_util::task::noop_waker();
    let mut cx = std::task::Context::from_waker(&waker);
    match fut.as_mut().poll(&mut cx) {
        std::task::Poll::Ready(val) => val,
        std::task::Poll::Pending => panic!(
            "wasi_block_on: future returned Pending -- all I/O should be synchronous via pollable.block()"
        ),
    }
}
