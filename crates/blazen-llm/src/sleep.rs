//! Cross-platform async sleep.
//!
//! Uses `tokio::time::sleep` on native targets, `setTimeout` via JS interop
//! on `wasm32-unknown-unknown`, and is a no-op on `WASIp2` (no timer support).

use std::time::Duration;

/// Sleep for the given duration, yielding to the async runtime.
///
/// - **Native**: delegates to [`tokio::time::sleep`].
/// - **Browser/Node WASM**: creates a JS `Promise` backed by `setTimeout`.
/// - **`WASIp2`**: no-op (no timer support).
pub(crate) async fn sleep(duration: Duration) {
    cfg_sleep(duration).await;
}

#[cfg(not(any(target_os = "wasi", target_arch = "wasm32")))]
async fn cfg_sleep(duration: Duration) {
    tokio::time::sleep(duration).await;
}

#[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
async fn cfg_sleep(duration: Duration) {
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use wasm_bindgen::JsCast;

    // `JsFuture` is `!Send`, but on wasm32 everything is single-threaded.
    // Wrap it so callers with `+ Send` trait bounds compile.
    struct SendJsFuture(wasm_bindgen_futures::JsFuture);

    // SAFETY: wasm32 is single-threaded — there is no other thread to send to.
    unsafe impl Send for SendJsFuture {}

    impl Future for SendJsFuture {
        type Output = Result<wasm_bindgen::JsValue, wasm_bindgen::JsValue>;

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            // SAFETY: we never move the inner JsFuture out of the Pin.
            unsafe { self.map_unchecked_mut(|s| &mut s.0) }.poll(cx)
        }
    }

    let ms = i32::try_from(duration.as_millis()).unwrap_or(i32::MAX);
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        // Use globalThis.setTimeout — works in browsers, workers, and Node.
        let global = js_sys::global();
        let _ = js_sys::Reflect::get(&global, &wasm_bindgen::JsValue::from_str("setTimeout"))
            .expect("setTimeout not found on globalThis")
            .unchecked_into::<js_sys::Function>()
            .call2(
                &wasm_bindgen::JsValue::NULL,
                &resolve,
                &wasm_bindgen::JsValue::from(ms),
            );
    });
    let _ = SendJsFuture(wasm_bindgen_futures::JsFuture::from(promise)).await;
}

#[cfg(target_os = "wasi")]
async fn cfg_sleep(duration: Duration) {
    let _ = duration;
}
