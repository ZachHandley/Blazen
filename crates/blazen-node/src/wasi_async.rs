//! Wasi-only async dispatcher for [`blazen_core::runtime::spawn`].
//!
//! Drives Rust futures via the JS microtask queue (`Promise.resolve().then`)
//! because tokio's `std::thread::spawn`-based runtime driver doesn't work on
//! workerd's single-isolate WASI model. Mirrors what
//! [`wasm_bindgen_futures::spawn_local`] does on browser `wasm32-unknown`,
//! but uses napi-rs's [`ThreadsafeFunction`] primitive for JS interop instead
//! of `wasm-bindgen`.
//!
//! Wired up at module load via [`install`] in `lib.rs`. After install, every
//! `runtime::spawn` from `blazen-core` (and the migrated 6C call sites in
//! `blazen-llm`, `blazen-pipeline`, etc.) ends up on this queue and is polled
//! cooperatively from JS microtasks. The worker code on the JS side wires up
//! `globalThis.__blazenDrainAsyncQueue` to the napi-exported [`drain`] so the
//! `Promise.resolve().then(...)` callback can call back into Rust.
//!
//! The whole module is single-isolate: `RefCell` over the queue is fine, no
//! `Mutex` is needed. Wakers are `Arc<TaskWaker>` so they remain `Send + Sync`
//! to satisfy the `Waker` contract, but they don't actually carry per-task
//! state — they just ask the dispatcher to schedule a drain.

#![cfg(all(target_arch = "wasm32", target_os = "wasi"))]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;

/// Boxed future the registered spawner is asked to drive.
///
/// Mirrors `blazen_core::runtime::WasiBoxFuture` so the signature passed to
/// [`blazen_core::runtime::register_spawner`] type-checks.
type WasiBoxFuture = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

/// Boxed future the registered sleeper produces. Mirrors
/// `blazen_core::runtime::SleepFut` so the closure handed to
/// [`blazen_core::runtime::register_sleeper`] type-checks.
type WasiSleepFut = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

thread_local! {
    /// Pending tasks awaiting their next poll. Drained on every microtask
    /// fire; tasks that return `Poll::Pending` are pushed back on.
    static QUEUE: RefCell<VecDeque<Task>> = const { RefCell::new(VecDeque::new()) };

    /// Idempotency flag for [`schedule_drain`] — flips back to `false` on the
    /// next [`drain`] entry so a wake during a drain reschedules another one.
    static SCHEDULED: RefCell<bool> = const { RefCell::new(false) };
}

/// JS-side scheduler whose body is `() => Promise.resolve().then(() =>
/// globalThis.__blazenDrainAsyncQueue())`. Calling `.call(())` queues a
/// microtask that ultimately re-enters Rust via [`drain`].
///
/// Set once during [`install`]; first writer wins.
static SCHEDULER_TSFN: OnceLock<ThreadsafeFunction<(), Unknown<'static>, (), Status, false>> =
    OnceLock::new();

/// JS-side sleeper whose body is `(ms) => new Promise(r =>
/// globalThis.setTimeout(r, ms))`. Calling `.call_async(ms)` schedules a
/// `setTimeout` and returns a Rust-awaitable handle to the underlying
/// Promise that resolves when the timer fires.
///
/// Set once during [`install`]; first writer wins. The `weak::<true>()`
/// flavour matches every other `Promise`-returning TSfn in this crate
/// (`manager.rs`, `pipeline/builder.rs`, `content/store.rs`) so the napi
/// runtime doesn't keep the host alive solely on this handle.
///
/// Wrapped in `Arc` so the `sleep_impl` closure can hand a fresh handle
/// to each spawned future without taking the TSfn out of the `OnceLock`
/// (`ThreadsafeFunction` itself is not `Clone` in napi-rs 3).
static SLEEPER_TSFN: OnceLock<Arc<ThreadsafeFunction<u32, Promise<()>, u32, Status, false, true>>> =
    OnceLock::new();

/// One unit of pending work tracked by the dispatcher.
struct Task {
    fut: WasiBoxFuture,
    waker: Waker,
}

/// Stateless waker — every wake just re-schedules a queue drain. We don't
/// need to track which task woke because a drain re-polls every queued task,
/// which is correct (if not maximally efficient) and matches what
/// `wasm_bindgen_futures` does on browser wasm32.
struct TaskWaker;

impl Wake for TaskWaker {
    fn wake(self: Arc<Self>) {
        schedule_drain();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        schedule_drain();
    }
}

/// Ask the JS host to call us back on the next microtask, if not already
/// scheduled. Idempotent within a single drain cycle so a flurry of wakes
/// only produces one re-entry.
fn schedule_drain() {
    let already = SCHEDULED.with(|s| {
        let was = *s.borrow();
        *s.borrow_mut() = true;
        was
    });
    if already {
        return;
    }
    if let Some(tsfn) = SCHEDULER_TSFN.get() {
        let _ = tsfn.call((), ThreadsafeFunctionCallMode::NonBlocking);
    }
    // If the TSfn isn't installed yet, this wake is dropped. `install`
    // must run before any `spawn` in the module's lifecycle — if you hit
    // this branch you have a module-init ordering bug, not a runtime bug.
}

/// Pump the queue once. Exported to JS as `__blazenDrainAsyncQueue` so the
/// `Promise.resolve().then(...)` body can call back in.
///
/// Drains the current queue into a local buffer, polls each task once, and
/// re-queues anything still pending. If the queue is non-empty after the
/// pass (either from re-queues or fresh spawns triggered by polled tasks),
/// schedules another drain.
#[napi(js_name = "__blazenDrainAsyncQueue")]
pub fn drain() {
    SCHEDULED.with(|s| *s.borrow_mut() = false);

    let mut tasks: VecDeque<Task> = QUEUE.with(|q| std::mem::take(&mut *q.borrow_mut()));
    let mut still_pending: VecDeque<Task> = VecDeque::with_capacity(tasks.len());

    while let Some(mut task) = tasks.pop_front() {
        let mut cx = Context::from_waker(&task.waker);
        match task.fut.as_mut().poll(&mut cx) {
            Poll::Ready(()) => {}
            Poll::Pending => still_pending.push_back(task),
        }
    }

    if !still_pending.is_empty() {
        QUEUE.with(|q| {
            let mut q = q.borrow_mut();
            // Splice any tasks spawned during this drain (sitting in `q`
            // because we already took the original queue contents) AFTER
            // the re-queued pendings — newer spawns trail older ones in
            // overall FIFO order. After this `q == still_pending` with
            // freshly-spawned tasks appended to its tail.
            still_pending.append(&mut q);
            *q = still_pending;
        });
    }

    let pending = QUEUE.with(|q| !q.borrow().is_empty());
    if pending {
        schedule_drain();
    }
}

/// Spawner registered into [`blazen_core::runtime::register_spawner`].
///
/// Pushes the future onto the thread-local queue and asks JS to schedule a
/// drain. The future starts making progress on the next microtask.
fn spawn(fut: WasiBoxFuture) {
    let waker = Waker::from(Arc::new(TaskWaker));
    QUEUE.with(|q| q.borrow_mut().push_back(Task { fut, waker }));
    schedule_drain();
}

/// Sleeper registered into [`blazen_core::runtime::register_sleeper`].
///
/// Builds a future that, when polled, asks the JS side (via
/// [`SLEEPER_TSFN`]) to call `setTimeout` and hand back the matching
/// `Promise<void>`. Awaiting that Promise resolves when the timer fires —
/// at which point the Rust future returned here completes.
///
/// Saturates the duration at [`u32::MAX`] milliseconds (~49 days), which
/// is more than enough for any real-world `runtime::sleep` / `runtime::timeout`
/// caller in this workspace and matches the `as i32` saturation used by
/// the browser-wasm32 `sleep` polyfill in `blazen_core::runtime`.
///
/// If [`SLEEPER_TSFN`] is unset (i.e. [`install`] has not run yet) we
/// resolve immediately rather than hang. That mirrors the
/// `register_spawner`-not-yet-registered branch in `schedule_drain` and
/// surfaces module-init ordering bugs as zero-delay sleeps rather than
/// deadlocks.
fn sleep_impl(dur: Duration) -> WasiSleepFut {
    #[allow(clippy::cast_possible_truncation)]
    let ms: u32 = u32::try_from(dur.as_millis()).unwrap_or(u32::MAX);
    let tsfn = SLEEPER_TSFN.get().map(Arc::clone);
    Box::pin(async move {
        let Some(tsfn) = tsfn else {
            return;
        };
        // Phase 1: schedule the JS callback and capture the returned
        // Promise. A dispatch error means the napi host went away — drop
        // the sleep silently rather than hang.
        let Ok(promise) = tsfn.call_async(ms).await else {
            return;
        };
        // Phase 2: drive the JS Promise to completion. setTimeout's
        // resolver takes no argument, so the Promise yields `()`. Errors
        // are likewise dropped: a rejection from the JS side is also a
        // host malfunction we can't recover from here.
        let _ = promise.await;
    })
}

/// Wire up the dispatcher. Builds the JS scheduler + sleeper closures via
/// [`Env::run_script`], wraps each in a [`ThreadsafeFunction`], stores them
/// in [`SCHEDULER_TSFN`] / [`SLEEPER_TSFN`], and registers [`spawn`] +
/// [`sleep_impl`] with `blazen-core`.
///
/// Called once from the napi `module_exports` hook in `lib.rs`. Subsequent
/// calls are no-ops on both sides (the `OnceLock::set` returns `Err` and
/// `register_spawner` / `register_sleeper` do the same).
///
/// # Errors
///
/// Returns the underlying napi error if `run_script` fails or any TSfn build
/// fails. Both indicate a host-environment bug (bad eval support, threadsafe
/// function unsupported), not a recoverable condition.
pub fn install(env: &Env) -> napi::Result<()> {
    // IIFE returns a fresh closure each time `install` is called, but we
    // only ever hit this path once — the `OnceLock::set` below guards
    // re-registration. The closure body runs on the main JS thread when the
    // microtask fires.
    let scheduler: Function<'_, (), Unknown<'_>> = env.run_script(
        "(() => () => { Promise.resolve().then(() => globalThis.__blazenDrainAsyncQueue()); })()",
    )?;

    let tsfn: ThreadsafeFunction<(), Unknown<'static>, (), Status, false> = scheduler
        .build_threadsafe_function::<()>()
        .callee_handled::<false>()
        .build()?;

    // First writer wins; if some other code already registered a scheduler
    // (impossible in practice — only `install` writes here) we drop ours
    // silently rather than panic, matching `register_spawner` semantics.
    let _ = SCHEDULER_TSFN.set(tsfn);

    // Hand the spawner to blazen-core. After this point every
    // `runtime::spawn` in the workspace dispatches through `spawn` above.
    let _ = blazen_core::runtime::register_spawner(Arc::new(spawn));

    // Sleeper: a JS function `(ms) => new Promise(r => setTimeout(r, ms))`.
    // workerd exposes `globalThis.setTimeout` natively, so this works on
    // Cloudflare Workers' WASI runtime (where tokio's time wheel doesn't).
    // The IIFE shape mirrors the scheduler IIFE above for symmetry.
    let sleeper: Function<'_, u32, Promise<()>> =
        env.run_script("(() => (ms) => new Promise(r => globalThis.setTimeout(r, ms)))()")?;

    let sleeper_tsfn: ThreadsafeFunction<u32, Promise<()>, u32, Status, false, true> = sleeper
        .build_threadsafe_function::<u32>()
        .callee_handled::<false>()
        .weak::<true>()
        .build()?;

    let _ = SLEEPER_TSFN.set(Arc::new(sleeper_tsfn));

    // Register the sleeper. After this point every `runtime::sleep` /
    // `runtime::timeout` from `blazen-core` resolves via JS `setTimeout`
    // instead of falling back to `tokio::time::sleep` (which panics on
    // workerd's WASI).
    let _ = blazen_core::runtime::register_sleeper(Arc::new(sleep_impl));

    // Override napi-rs's wasm async executor to use our microtask scheduler
    // instead of `std::thread::spawn(|| block_on(...))` (which traps on
    // workerd because it doesn't implement the WASI threads ABI). The
    // closure type matches `napi::bindgen_prelude::AsyncExecutorFn` and our
    // `spawn` takes `WasiBoxFuture = Pin<Box<dyn Future<Output = ()> + Send
    // + 'static>>`, which is exactly what napi-rs hands us.
    let _ = napi::bindgen_prelude::set_async_executor(Arc::new(|fut| spawn(fut)));

    Ok(())
}
