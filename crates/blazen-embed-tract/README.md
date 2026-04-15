# blazen-embed-tract

Pure-Rust ONNX inference backend for Blazen embeddings, using [`tract-onnx`](https://crates.io/crates/tract-onnx) instead of the C++ ONNX Runtime.

> **You almost never import this crate directly.** Depend on `blazen-embed` (the facade) instead — it re-exports `EmbedModel` / `EmbedOptions` / `EmbedModelName` and picks the right underlying implementation automatically per target. On `musl`/`wasm` targets the facade resolves to this crate; on glibc/macOS/Windows it resolves to `blazen-embed-fastembed`.

## Why this crate exists

`blazen-embed-fastembed` (the default embedding backend on glibc/macOS/Windows) pulls `fastembed` → `ort` → Microsoft's prebuilt ONNX Runtime binaries. ONNX Runtime is not published by Microsoft for several target triples, including:

- `x86_64-unknown-linux-musl` and `aarch64-unknown-linux-musl` (Alpine Linux, Docker-on-Graviton)
- `wasm32-*` (browser / server WASM)

This crate provides a pure-Rust alternative so Blazen ships embedding support on those platforms. Performance on CPU-only workloads is typically **2–4× slower than ORT**, with no GPU support (tract is CPU-only). On supported platforms, the facade keeps you on `blazen-embed-fastembed` for peak throughput.

## Design

Mirrors the public API of `blazen-embed-fastembed`: same `from_options` / `embed` / `model_id` / `dims` methods, same model names, same response shape. Both crates expose identical type names (`EmbedModel`, `EmbedOptions`, `EmbedModelName`) so the facade can substitute one for the other without any downstream code changes.

Components:

| Concern | Crate |
|---|---|
| Model download (HuggingFace) | `blazen-model-cache` (reused) |
| Tokenization | `tokenizers` (already in workspace) |
| ONNX inference | `tract-onnx` (new) |
| Tensor math (pooling, normalization) | `ndarray` (new) |
| Async wrapper | `tokio::task::spawn_blocking` |

## Usage

Use the facade, not this crate directly:

```rust
use blazen_embed::{EmbedModel, EmbedOptions, EmbedModelName};

let options = EmbedOptions {
    model: EmbedModelName::BgeSmallEnV15,
    cache_dir: None,
    batch_size: Some(32),
};
let model = EmbedModel::from_options(options).await?;
let response = model.embed(vec!["hello world".into()]).await?;
assert_eq!(response.embeddings.len(), 1);
```

When built for `x86_64-unknown-linux-musl`, `aarch64-unknown-linux-musl`, or `wasm32-*`, the `EmbedModel` you get back is from this crate. When built for glibc/macOS/Windows, it comes from `blazen-embed-fastembed`. The calling code is identical.

## Backend selection

There is **no feature flag** for choosing between tract and fastembed. `blazen-embed`'s `Cargo.toml` uses target-cfg dependencies to dispatch automatically:

```toml
# blazen-embed/Cargo.toml (illustrative)
[target.'cfg(any(target_env = "musl", target_family = "wasm"))'.dependencies]
blazen-embed-tract = { path = "../blazen-embed-tract" }

[target.'cfg(not(any(target_env = "musl", target_family = "wasm")))'.dependencies]
blazen-embed-fastembed = { path = "../blazen-embed-fastembed" }
```

Wheels built for musl targets automatically use tract; wheels for glibc/mac/windows use fastembed. Consumers never choose — they always depend on `blazen-embed` and get the right backend for the target triple they compiled against.

## Related

- `crates/blazen-embed/` — the facade you should depend on
- `crates/blazen-embed-fastembed/` — the glibc/macOS/Windows backend the facade selects by default
- `crates/blazen-embed-candle/` — candle-based alternative (BERT-family only; good reference for pure-Rust tokenize + tensor-math patterns)
- `crates/blazen-model-cache/` — shared HuggingFace download + cache layer
