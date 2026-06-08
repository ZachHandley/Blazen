//! Per-place provider-key store for the control plane.
//!
//! The control plane stores per-`(place, provider)` API keys and serves
//! them to a worker over the existing authenticated `WorkerSession` bidi
//! stream (see [`super::session`] and
//! [`crate::client::key_client::ControlPlaneKeyClient`]). Keys NEVER
//! touch an [`Assignment`](crate::protocol::Assignment) and are NEVER
//! logged — every type that carries a key value redacts it in its
//! [`std::fmt::Debug`] impl.
//!
//! [`KeyStore`] is the swappable seam (mirroring
//! [`AssignmentStore`](super::store::AssignmentStore)). The default
//! [`EnvFileKeyStore`] resolves keys from environment variables and,
//! optionally, files under a configured directory — both are namespaced
//! by place so one tenant can never read another's key.

use std::collections::BTreeMap;
use std::path::PathBuf;

use async_trait::async_trait;

use crate::error::ControlPlaneError;

/// Environment-variable prefix for per-place provider keys read by
/// [`EnvFileKeyStore`]. The full variable name is
/// `BLAZEN_PLACE_KEY__<PLACE>__<PROVIDER>` with `<PLACE>` and
/// `<PROVIDER>` uppercased.
const ENV_KEY_PREFIX: &str = "BLAZEN_PLACE_KEY__";

/// Environment variable naming the on-disk directory root scanned by
/// [`EnvFileKeyStore`]. Keys live at `$BLAZEN_PLACE_KEYS_DIR/<place>/<provider>`.
const ENV_KEYS_DIR: &str = "BLAZEN_PLACE_KEYS_DIR";

/// A resolved provider key plus its cache-control metadata.
///
/// The `value` field is a secret. Its [`std::fmt::Debug`] impl REDACTS
/// it — printing `SharedKey { value: "<REDACTED>", .. }` — so a key can
/// never leak through a `tracing` field, `dbg!`, panic message, or test
/// assertion that formats the struct.
#[derive(Clone, PartialEq, Eq)]
pub struct SharedKey {
    /// The secret key material. NEVER logged — see the manual `Debug`
    /// impl below, which redacts this field.
    pub value: String,
    /// Optional time-to-live in seconds. `Some(ttl)` lets the worker-side
    /// cache expire the key and re-fetch; `None` means cache for the
    /// session's lifetime.
    pub ttl_secs: Option<u64>,
    /// Monotonic version counter. Lets the worker-side cache notice a
    /// rotated key (a higher `version` for the same provider supersedes
    /// the cached one).
    pub version: u64,
}

impl std::fmt::Debug for SharedKey {
    /// Redacts [`SharedKey::value`] so the secret never reaches a log
    /// line, panic message, or test assertion that formats the struct.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedKey")
            .field("value", &"<REDACTED>")
            .field("ttl_secs", &self.ttl_secs)
            .field("version", &self.version)
            .finish()
    }
}

/// Per-place provider-key persistence seam.
///
/// Mirrors [`AssignmentStore`](super::store::AssignmentStore): the
/// session handler reads through this trait so the backing store can be
/// swapped (env/file by default, or a tenant-aware secret manager in a
/// hosted deployment) without touching the wire protocol.
///
/// Implementations are `Send + Sync + std::fmt::Debug` so the server can
/// hold an `Arc<dyn KeyStore>`. The `Debug` bound is satisfied without
/// leaking secrets — implementations carry only configuration, never the
/// keys themselves, in their own `Debug` output.
#[async_trait]
pub trait KeyStore: Send + Sync + std::fmt::Debug {
    /// Resolve the provider key for `place`. Returns `None` when the
    /// store has no key for that `(place, provider)` pair — the worker
    /// then falls through to the next link in its resolver chain (env).
    ///
    /// The `place` argument is the worker session's server-authenticated
    /// place (see [`super::session`]); it is NEVER a place named by the
    /// worker, so one tenant can never read another's key.
    ///
    /// # Errors
    /// Returns a [`ControlPlaneError`] when the underlying store cannot
    /// be reached or returns a malformed payload. A missing key is
    /// `Ok(None)`, not an error.
    async fn get_key(
        &self,
        place: &str,
        provider: &str,
    ) -> Result<Option<SharedKey>, ControlPlaneError>;
}

/// Default [`KeyStore`]: resolves keys from environment variables and,
/// optionally, files under a configured directory.
///
/// Resolution order for a `(place, provider)` lookup:
///
/// 1. **File** (wins, if present): `$BLAZEN_PLACE_KEYS_DIR/<place>/<provider>`
///    — the file's trimmed contents are the key. Only consulted when
///    `BLAZEN_PLACE_KEYS_DIR` is set.
/// 2. **Environment**: `BLAZEN_PLACE_KEY__<PLACE>__<PROVIDER>` with both
///    `<PLACE>` and `<PROVIDER>` uppercased.
///
/// Both sources are place-scoped, so place A's key is never visible to a
/// lookup for place B. Keys resolved here carry `ttl_secs: None` (cache
/// for the session) and `version: 0` (the env/file store has no rotation
/// counter — a redeploy with a new value is the rotation mechanism).
#[derive(Debug, Clone, Default)]
pub struct EnvFileKeyStore {
    /// On-disk directory root, from `BLAZEN_PLACE_KEYS_DIR`. `None`
    /// disables the file source (env-only). Holds only a path — never a
    /// key — so the derived `Debug` leaks nothing.
    keys_dir: Option<PathBuf>,
    /// In-memory override for the env-variable lookup, keyed by full
    /// variable name. `None` (the production default) reads the live
    /// process environment via `std::env::var`; `Some(map)` reads the map.
    /// Tests inject a map so they exercise the place-scoping / precedence
    /// logic WITHOUT mutating (unsafe in edition 2024) the process-global
    /// environment, and without cross-test env races. Values here are
    /// configuration test fixtures, never production secrets.
    env_override: Option<BTreeMap<String, String>>,
}

impl EnvFileKeyStore {
    /// Construct an env-only store (no file source) reading the live
    /// process environment.
    #[must_use]
    pub fn new() -> Self {
        Self {
            keys_dir: None,
            env_override: None,
        }
    }

    /// Construct a store from the ambient environment: reads
    /// `BLAZEN_PLACE_KEYS_DIR` to decide whether the file source is
    /// active. The env-variable source is always active.
    #[must_use]
    pub fn from_env() -> Self {
        let keys_dir = std::env::var_os(ENV_KEYS_DIR)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);
        Self {
            keys_dir,
            env_override: None,
        }
    }

    /// Look up `var`, returning `Some(value)` only for a present, non-empty
    /// value. Reads the in-memory override when set, else the live process
    /// environment.
    fn env_lookup(&self, var: &str) -> Option<String> {
        match &self.env_override {
            Some(map) => map.get(var).filter(|v| !v.is_empty()).cloned(),
            None => std::env::var(var).ok().filter(|v| !v.is_empty()),
        }
    }

    /// Build the env-variable name for a `(place, provider)` pair:
    /// `BLAZEN_PLACE_KEY__<PLACE>__<PROVIDER>`, both segments uppercased.
    fn env_var_name(place: &str, provider: &str) -> String {
        format!(
            "{ENV_KEY_PREFIX}{}__{}",
            place.to_uppercase(),
            provider.to_uppercase()
        )
    }
}

#[async_trait]
impl KeyStore for EnvFileKeyStore {
    async fn get_key(
        &self,
        place: &str,
        provider: &str,
    ) -> Result<Option<SharedKey>, ControlPlaneError> {
        // 1. File source (wins). Only when a keys dir is configured.
        if let Some(dir) = &self.keys_dir {
            let path = dir.join(place).join(provider);
            match tokio::fs::read_to_string(&path).await {
                Ok(contents) => {
                    let value = contents.trim().to_string();
                    if !value.is_empty() {
                        return Ok(Some(SharedKey {
                            value,
                            ttl_secs: None,
                            version: 0,
                        }));
                    }
                    // Empty file → fall through to env.
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    // No file for this place/provider → fall through to env.
                }
                Err(e) => {
                    return Err(ControlPlaneError::Transport(format!(
                        "read place key file: {e}"
                    )));
                }
            }
        }

        // 2. Environment source.
        let var = Self::env_var_name(place, provider);
        match self.env_lookup(&var) {
            Some(value) => Ok(Some(SharedKey {
                value,
                ttl_secs: None,
                version: 0,
            })),
            // Unset, empty, or non-unicode → no key for this pair.
            None => Ok(None),
        }
    }
}

/// Construct a default env/file store wrapped in `Arc<dyn KeyStore>` for
/// callers that want the trait object directly (e.g. the server
/// constructor).
#[must_use]
pub fn default_key_store() -> std::sync::Arc<dyn KeyStore> {
    std::sync::Arc::new(EnvFileKeyStore::from_env())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an env-only store whose env source is an in-memory map. Keeps
    /// the tests off the (edition-2024-`unsafe`) `std::env::set_var` and
    /// free of cross-test env races.
    fn store_with_env(pairs: &[(&str, &str)]) -> EnvFileKeyStore {
        let map = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        EnvFileKeyStore {
            keys_dir: None,
            env_override: Some(map),
        }
    }

    #[tokio::test]
    async fn env_lookup_resolves_known_place_provider() {
        let store = store_with_env(&[("BLAZEN_PLACE_KEY__ACME__OPENAI", "sk-acme-secret")]);
        let key = store
            .get_key("acme", "openai")
            .await
            .expect("get_key")
            .expect("Some(key)");
        assert_eq!(key.value, "sk-acme-secret");
        assert_eq!(key.version, 0);
        assert!(key.ttl_secs.is_none());
    }

    #[tokio::test]
    async fn missing_key_is_none() {
        let store = store_with_env(&[]);
        let missing = store.get_key("nobody", "ghost").await.expect("get_key");
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn place_scoping_isolates_tenants() {
        // Only place A has a FAL key.
        let store = store_with_env(&[("BLAZEN_PLACE_KEY__PLACEA__FAL", "key-for-a")]);

        let in_a = store
            .get_key("placea", "fal")
            .await
            .expect("get_key a")
            .expect("Some for place a");
        assert_eq!(in_a.value, "key-for-a");

        // Place B has no FAL key — must NOT see place A's key.
        let in_b = store.get_key("placeb", "fal").await.expect("get_key b");
        assert!(
            in_b.is_none(),
            "place B must not resolve place A's key (tenant isolation)"
        );
    }

    #[tokio::test]
    async fn file_source_wins_over_env() {
        let dir = tempfile::tempdir().expect("tempdir");
        let place_dir = dir.path().join("acme");
        tokio::fs::create_dir_all(&place_dir)
            .await
            .expect("mkdir place");
        tokio::fs::write(place_dir.join("openai"), "file-key\n")
            .await
            .expect("write key file");

        // Env has a DIFFERENT value for the same place/provider; the file
        // must win.
        let map: BTreeMap<String, String> = [(
            "BLAZEN_PLACE_KEY__ACME__OPENAI".to_string(),
            "env-key".to_string(),
        )]
        .into_iter()
        .collect();
        let store = EnvFileKeyStore {
            keys_dir: Some(dir.path().to_path_buf()),
            env_override: Some(map),
        };
        let key = store
            .get_key("acme", "openai")
            .await
            .expect("get_key")
            .expect("Some(key)");
        // File wins, and trailing whitespace is trimmed.
        assert_eq!(key.value, "file-key");
    }

    #[tokio::test]
    async fn empty_env_value_falls_through_to_none() {
        // A present-but-empty variable must be treated as absent.
        let store = store_with_env(&[("BLAZEN_PLACE_KEY__ACME__OPENAI", "")]);
        let missing = store.get_key("acme", "openai").await.expect("get_key");
        assert!(missing.is_none());
    }

    #[test]
    fn shared_key_debug_redacts_value() {
        let key = SharedKey {
            value: "sk-super-secret-do-not-leak".to_string(),
            ttl_secs: Some(300),
            version: 7,
        };
        let rendered = format!("{key:?}");
        assert!(
            !rendered.contains("sk-super-secret-do-not-leak"),
            "Debug must not contain the secret value, got: {rendered}"
        );
        assert!(
            rendered.contains("REDACT"),
            "Debug must signal redaction, got: {rendered}"
        );
        // Non-secret metadata is still visible for triage.
        assert!(rendered.contains("300"));
        assert!(rendered.contains('7'));
    }

    #[test]
    fn env_var_name_uppercases_both_segments() {
        assert_eq!(
            EnvFileKeyStore::env_var_name("acme", "openai"),
            "BLAZEN_PLACE_KEY__ACME__OPENAI"
        );
    }
}
