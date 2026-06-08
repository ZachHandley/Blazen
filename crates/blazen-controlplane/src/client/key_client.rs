//! Worker-side control-plane key client.
//!
//! [`ControlPlaneKeyClient`] is the worker's link in the
//! [`blazen_llm::KeyResolver`] cascade that fetches per-place provider
//! keys from the control plane over the EXISTING authenticated
//! `WorkerSession` bidi stream. It is installed AFTER the worker-local
//! key resolver, so the order is worker-local → control-plane → env
//! terminal.
//!
//! ## Sync resolve, async fetch
//!
//! [`blazen_llm::KeyResolver::resolve`] is synchronous, but fetching a
//! key requires a round-trip over the session. The client bridges the
//! two with a cache:
//!
//! - [`ControlPlaneKeyClient::resolve`] reads ONLY the cache (honoring
//!   TTL expiry). It never blocks on the network.
//! - [`ControlPlaneKeyClient::pre_warm`] is the async fetch path: it
//!   sends [`KeyRequest`](crate::protocol::KeyRequest)s and awaits the
//!   matching [`KeyResponse`](crate::protocol::KeyResponse)s to populate
//!   the cache. The worker calls it on assignment receipt (after
//!   [`blazen_llm::set_current_place`]).
//! - [`ControlPlaneKeyClient::on_key_response`] is called by the inbound
//!   pump when a [`ServerToWorker::KeyResponse`](crate::protocol::ServerToWorker::KeyResponse)
//!   frame arrives; it fulfils the pending correlation oneshot and caches
//!   the result.
//!
//! The key value is NEVER logged — the wire [`KeyResponse`] redacts it in
//! `Debug`, and this module never formats the key directly.

use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::error::ControlPlaneError;
use crate::protocol::{ENVELOPE_VERSION, KeyRequest, KeyResponse, WorkerToServer};

/// A cached provider key plus the metadata needed to honor TTL expiry and
/// version supersession.
#[derive(Clone)]
struct CacheEntry {
    /// The secret key material. Never logged — `CacheEntry` deliberately
    /// has no `Debug` impl, and the resolver returns owned `String`s the
    /// caller is responsible for not logging.
    value: String,
    /// When this entry was populated (used with `ttl` for expiry).
    fetched_at: Instant,
    /// Optional TTL; `None` = cache for the session lifetime.
    ttl: Option<Duration>,
    /// Server-assigned monotonic version. A `KeyResponse` with a strictly
    /// higher version supersedes the cached entry.
    version: u64,
}

impl CacheEntry {
    /// Whether this entry has aged past its TTL. Entries with no TTL never
    /// expire on their own.
    fn is_expired(&self) -> bool {
        match self.ttl {
            Some(ttl) => self.fetched_at.elapsed() >= ttl,
            None => false,
        }
    }
}

/// Worker-side [`blazen_llm::KeyResolver`] that fetches and caches
/// per-place provider keys from the control plane over the authenticated
/// session.
///
/// Cheaply cloneable — all interior state is behind `Arc`/`DashMap`, so a
/// clone shares the same cache and pending-correlation maps. The worker
/// installs one instance into the global resolver chain ONCE (the chain is
/// process-global), keeps a clone to drive [`Self::pre_warm`] /
/// [`Self::on_key_response`], and rebinds the per-session outbound sender
/// on every (re)connect via [`Self::rebind_outbound`].
pub struct ControlPlaneKeyClient {
    inner: Arc<Inner>,
}

/// One outstanding [`KeyRequest`] awaiting its response: the provider it
/// asked for (so [`ControlPlaneKeyClient::on_key_response`] can cache the
/// result without a provider field on the wire response) and the oneshot
/// [`ControlPlaneKeyClient::pre_warm`] is blocked on.
struct Pending {
    provider: String,
    waiter: oneshot::Sender<()>,
}

struct Inner {
    /// Outbound sender into the worker's CURRENT session. Sending a
    /// [`WorkerToServer::KeyRequest`] here threads the frame onto the gRPC
    /// stream towards the control plane. Held behind an `RwLock<Option<_>>`
    /// so the worker can rebind it on each reconnect (the resolver itself
    /// is installed into the process-global chain once and outlives any
    /// single session). `None` before the first handshake.
    outbound: RwLock<Option<mpsc::Sender<WorkerToServer>>>,
    /// Cache: `provider` → entry. Read by the sync `resolve`, written by
    /// `on_key_response`.
    cache: DashMap<String, CacheEntry>,
    /// Pending correlation map: `request_id` → the provider + the
    /// notify-oneshot that [`Self::pre_warm`] is awaiting. Consumed by
    /// [`Self::on_key_response`], which caches by the stored provider then
    /// signals the waiter.
    pending: DashMap<Uuid, Pending>,
}

impl ControlPlaneKeyClient {
    /// Build a client bound to the worker session's current outbound
    /// sender.
    #[must_use]
    pub fn new(outbound: mpsc::Sender<WorkerToServer>) -> Self {
        Self {
            inner: Arc::new(Inner {
                outbound: RwLock::new(Some(outbound)),
                cache: DashMap::new(),
                pending: DashMap::new(),
            }),
        }
    }

    /// Build a client with no bound session yet. The worker constructs it
    /// at startup (before the first handshake) and calls
    /// [`Self::rebind_outbound`] once each session is established. Until
    /// then, [`Self::resolve`] still works (cache reads) but
    /// [`Self::pre_warm`] returns a transport error.
    #[must_use]
    pub fn disconnected() -> Self {
        Self {
            inner: Arc::new(Inner {
                outbound: RwLock::new(None),
                cache: DashMap::new(),
                pending: DashMap::new(),
            }),
        }
    }

    /// Rebind the outbound sender for a new session. Called by the worker
    /// after each (re)connect so the (process-global, install-once)
    /// resolver keeps sending [`KeyRequest`]s onto the live stream.
    ///
    /// # Panics
    ///
    /// Panics if the internal outbound lock has been poisoned by a prior
    /// panic while it was held.
    pub fn rebind_outbound(&self, outbound: mpsc::Sender<WorkerToServer>) {
        *self
            .inner
            .outbound
            .write()
            .expect("key client outbound lock poisoned") = Some(outbound);
    }

    /// Snapshot the current outbound sender, if any.
    fn outbound(&self) -> Option<mpsc::Sender<WorkerToServer>> {
        self.inner
            .outbound
            .read()
            .expect("key client outbound lock poisoned")
            .clone()
    }

    /// Pre-warm the cache for `providers` in tenant `place` by sending a
    /// [`KeyRequest`] for each and awaiting the matching [`KeyResponse`].
    ///
    /// Called by the worker on assignment receipt (after
    /// [`blazen_llm::set_current_place`]), so the synchronous `resolve`
    /// hit later finds a populated cache. The `place` is advisory here —
    /// the control plane scopes the lookup to the worker's
    /// server-authenticated place regardless — but is accepted so callers
    /// can document intent and so a future store could honor it.
    ///
    /// Each request waits up to `timeout` for its response; a timed-out or
    /// dropped response leaves that provider un-cached (the worker falls
    /// through to env on the next `resolve`). Errors are per-provider and
    /// non-fatal — a failure to warm one provider does not abort the rest.
    ///
    /// # Errors
    ///
    /// Returns [`ControlPlaneError::Transport`] only if the outbound
    /// channel is closed (the session disconnected) when trying to send a
    /// request. Per-provider timeouts are NOT errors — they simply leave
    /// the cache unpopulated for that provider.
    pub async fn pre_warm(
        &self,
        _place: &str,
        providers: &[String],
        timeout: Duration,
    ) -> Result<(), ControlPlaneError> {
        let Some(outbound) = self.outbound() else {
            return Err(ControlPlaneError::Transport(
                "key client has no bound session outbound (not connected)".into(),
            ));
        };
        for provider in providers {
            // Skip providers already cached and fresh — avoid redundant
            // round-trips on repeat assignments.
            if self
                .inner
                .cache
                .get(provider)
                .is_some_and(|e| !e.is_expired())
            {
                continue;
            }

            let request_id = Uuid::new_v4();
            let (tx, rx) = oneshot::channel();
            self.inner.pending.insert(
                request_id,
                Pending {
                    provider: provider.clone(),
                    waiter: tx,
                },
            );

            let frame = WorkerToServer::KeyRequest(KeyRequest {
                envelope_version: ENVELOPE_VERSION,
                request_id,
                provider: provider.clone(),
            });

            if outbound.send(frame).await.is_err() {
                // Session gone — drop the pending entry and surface the
                // transport failure so the caller can react.
                self.inner.pending.remove(&request_id);
                return Err(ControlPlaneError::Transport(
                    "worker outbound closed before KeyRequest".into(),
                ));
            }

            // Await the inbound pump's `on_key_response` (which caches the
            // key and signals this oneshot), bounded by `timeout`. On
            // success the cache is already populated; on a timeout / dropped
            // sender, clean up the pending entry and leave the provider
            // un-cached (the worker falls through to env on the next
            // resolve).
            if !matches!(tokio::time::timeout(timeout, rx).await, Ok(Ok(()))) {
                self.inner.pending.remove(&request_id);
                tracing::debug!(
                    provider = %provider,
                    "pre_warm: no key response (timeout or dropped); leaving uncached"
                );
            }
        }
        Ok(())
    }

    /// Route an inbound [`KeyResponse`] from the session pump: cache the
    /// key under the provider recorded in the pending correlation map
    /// (when present, honoring version supersession), then signal the
    /// [`Self::pre_warm`] waiter.
    ///
    /// Safe to call for an unknown `request_id` (a late / duplicate
    /// response with no pending entry is dropped — the worker has no
    /// provider context for it). The key value is never logged.
    pub fn on_key_response(&self, resp: &KeyResponse) {
        let Some((_, pending)) = self.inner.pending.remove(&resp.request_id) else {
            tracing::debug!(
                request_id = %resp.request_id,
                "on_key_response with no pending request (late/duplicate); dropping"
            );
            return;
        };

        // Cache positive responses by the requested provider. `key: None`
        // means the server brokers no key — leave the cache untouched so
        // `resolve` falls through to env.
        self.cache_for_provider(&pending.provider, resp);

        // Signal the waiter; the receiver may have already dropped (timed
        // out) — ignore.
        let _ = pending.waiter.send(());
    }

    /// Insert (or supersede) a cached entry for `provider` from a resolved
    /// [`KeyResponse`]. A response with a strictly lower `version` than the
    /// cached entry is ignored (stale, out-of-order delivery).
    fn cache_for_provider(&self, provider: &str, resp: &KeyResponse) {
        let Some(value) = &resp.key else {
            return;
        };
        let ttl = resp.ttl_secs.map(Duration::from_secs);
        let entry = CacheEntry {
            value: value.clone(),
            fetched_at: Instant::now(),
            ttl,
            version: resp.version,
        };
        match self.inner.cache.entry(provider.to_string()) {
            Entry::Occupied(mut occ) => {
                // Only overwrite with an equal-or-newer version, so an
                // out-of-order older response cannot clobber a fresh key.
                if resp.version >= occ.get().version {
                    occ.insert(entry);
                }
            }
            Entry::Vacant(vac) => {
                vac.insert(entry);
            }
        }
    }

    /// Cached, non-expired key for `provider`, or `None`. Pure cache read
    /// — never touches the network. Shared by [`Self::resolve`] and tests.
    fn cached(&self, provider: &str) -> Option<String> {
        self.inner.cache.get(provider).and_then(|e| {
            if e.is_expired() {
                None
            } else {
                Some(e.value.clone())
            }
        })
    }
}

impl Clone for ControlPlaneKeyClient {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl blazen_llm::KeyResolver for ControlPlaneKeyClient {
    /// Synchronous cache read. Returns a cached, non-expired key for
    /// `provider`, or `None` to defer to the next resolver (env terminal).
    /// `place` is ignored here — the control plane already scoped the
    /// cached key to the worker's authenticated place when it served it.
    fn resolve(&self, provider: &str, _place: Option<&str>) -> Option<String> {
        self.cached(provider)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use blazen_llm::KeyResolver as _;

    fn response(
        request_id: Uuid,
        key: Option<&str>,
        ttl_secs: Option<u64>,
        version: u64,
    ) -> KeyResponse {
        KeyResponse {
            envelope_version: ENVELOPE_VERSION,
            request_id,
            key: key.map(ToString::to_string),
            ttl_secs,
            version,
        }
    }

    #[test]
    fn resolve_misses_empty_cache() {
        let (tx, _rx) = mpsc::channel(8);
        let client = ControlPlaneKeyClient::new(tx);
        assert!(client.resolve("openai", None).is_none());
    }

    #[test]
    fn cache_for_provider_then_resolve_hits() {
        let (tx, _rx) = mpsc::channel(8);
        let client = ControlPlaneKeyClient::new(tx);
        let resp = response(Uuid::new_v4(), Some("sk-cached"), None, 1);
        client.cache_for_provider("openai", &resp);
        assert_eq!(client.resolve("openai", None).as_deref(), Some("sk-cached"));
    }

    #[test]
    fn none_key_response_does_not_cache() {
        let (tx, _rx) = mpsc::channel(8);
        let client = ControlPlaneKeyClient::new(tx);
        let resp = response(Uuid::new_v4(), None, None, 1);
        client.cache_for_provider("openai", &resp);
        assert!(client.resolve("openai", None).is_none());
    }

    #[test]
    fn expired_entry_resolves_none() {
        let (tx, _rx) = mpsc::channel(8);
        let client = ControlPlaneKeyClient::new(tx);
        // Insert an already-expired entry (ttl 0).
        let resp = response(Uuid::new_v4(), Some("sk-stale"), Some(0), 1);
        client.cache_for_provider("openai", &resp);
        // ttl=0 means elapsed (>=0) is immediately expired.
        assert!(client.resolve("openai", None).is_none());
    }

    #[test]
    fn higher_version_supersedes_lower() {
        let (tx, _rx) = mpsc::channel(8);
        let client = ControlPlaneKeyClient::new(tx);
        client.cache_for_provider("fal", &response(Uuid::new_v4(), Some("v1"), None, 1));
        client.cache_for_provider("fal", &response(Uuid::new_v4(), Some("v2"), None, 2));
        assert_eq!(client.resolve("fal", None).as_deref(), Some("v2"));
    }

    #[test]
    fn lower_version_does_not_clobber_newer() {
        let (tx, _rx) = mpsc::channel(8);
        let client = ControlPlaneKeyClient::new(tx);
        client.cache_for_provider("fal", &response(Uuid::new_v4(), Some("v5"), None, 5));
        // An out-of-order older response (version 3) must not overwrite v5.
        client.cache_for_provider("fal", &response(Uuid::new_v4(), Some("v3"), None, 3));
        assert_eq!(client.resolve("fal", None).as_deref(), Some("v5"));
    }

    #[test]
    fn on_key_response_caches_by_pending_provider_and_signals() {
        let (tx, _rx) = mpsc::channel(8);
        let client = ControlPlaneKeyClient::new(tx);
        let request_id = Uuid::new_v4();
        let (otx, mut orx) = oneshot::channel();
        client.inner.pending.insert(
            request_id,
            Pending {
                provider: "openai".to_string(),
                waiter: otx,
            },
        );

        let resp = response(request_id, Some("sk-pending"), None, 1);
        client.on_key_response(&resp);

        // The waiter was signalled.
        assert!(orx.try_recv().is_ok(), "pre_warm waiter must be signalled");
        // The pending entry was consumed.
        assert!(!client.inner.pending.contains_key(&request_id));
        // And the key was cached under the requested provider.
        assert_eq!(
            client.resolve("openai", None).as_deref(),
            Some("sk-pending")
        );
    }

    #[test]
    fn on_key_response_unknown_request_id_is_dropped() {
        let (tx, _rx) = mpsc::channel(8);
        let client = ControlPlaneKeyClient::new(tx);
        // No pending entry for this id — must not panic, must not cache.
        let resp = response(Uuid::new_v4(), Some("sk-orphan"), None, 1);
        client.on_key_response(&resp);
        assert!(client.resolve("openai", None).is_none());
    }

    #[tokio::test]
    async fn pre_warm_populates_cache_via_correlated_response() {
        // Wire an outbound channel and a fake "server" that echoes a key
        // response for each request_id it sees. The server returns after
        // handling exactly `expected` requests so the task terminates even
        // though it holds a client clone (which keeps the sender alive).
        let (tx, mut rx) = mpsc::channel::<WorkerToServer>(8);
        let client = ControlPlaneKeyClient::new(tx);
        let client_for_server = client.clone();

        let expected = 2usize;
        let server = tokio::spawn(async move {
            let mut handled = 0usize;
            while handled < expected {
                let Some(frame) = rx.recv().await else { break };
                if let WorkerToServer::KeyRequest(req) = frame {
                    let resp = KeyResponse {
                        envelope_version: ENVELOPE_VERSION,
                        request_id: req.request_id,
                        key: Some(format!("sk-{}", req.provider)),
                        ttl_secs: None,
                        version: 1,
                    };
                    // Mirror the worker's inbound pump: route the response
                    // into `on_key_response`, which caches by the pending
                    // provider and signals the pre_warm waiter.
                    client_for_server.on_key_response(&resp);
                    handled += 1;
                }
            }
        });

        client
            .pre_warm(
                "acme",
                &["openai".to_string(), "fal".to_string()],
                Duration::from_secs(5),
            )
            .await
            .expect("pre_warm");

        assert_eq!(client.resolve("openai", None).as_deref(), Some("sk-openai"));
        assert_eq!(client.resolve("fal", None).as_deref(), Some("sk-fal"));

        // The server task has handled both requests and returned.
        server.await.expect("server task joins");
    }

    #[tokio::test]
    async fn pre_warm_errors_when_outbound_closed() {
        let (tx, rx) = mpsc::channel::<WorkerToServer>(8);
        drop(rx); // close the channel — sends will fail
        let client = ControlPlaneKeyClient::new(tx);
        let err = client
            .pre_warm("acme", &["openai".to_string()], Duration::from_millis(50))
            .await
            .expect_err("closed outbound must error");
        assert!(matches!(err, ControlPlaneError::Transport(_)));
    }

    #[tokio::test]
    async fn pre_warm_timeout_leaves_provider_uncached() {
        // No server responds; pre_warm should time out and leave the cache
        // empty without erroring.
        let (tx, _rx) = mpsc::channel::<WorkerToServer>(8);
        let client = ControlPlaneKeyClient::new(tx);
        client
            .pre_warm("acme", &["ghost".to_string()], Duration::from_millis(50))
            .await
            .expect("pre_warm tolerates a timeout");
        assert!(client.resolve("ghost", None).is_none());
    }
}
