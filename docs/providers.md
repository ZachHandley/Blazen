# Blazen LLM Providers

> All providers compile unconditionally. The only opt-in features are `reqwest` (native HTTP client) and `tiktoken` (exact BPE token counting).

## Architecture

Every provider implements the `CompletionModel` trait (`complete()` + `stream()`) and optionally `ModelRegistry` (`list_models()` + `get_model()`) and `ProviderInfo` (`provider_name()`, `base_url()`, `capabilities()`).

Providers fall into two categories:

- **Native providers** — implement their own API format (OpenAI, Anthropic, Gemini, Azure, fal.ai)
- **OpenAI-compatible providers** — wrap `OpenAiCompatProvider` internally, which handles the OpenAI wire format

### HTTP Client

All providers accept `Arc<dyn HttpClient>`. Three implementations exist:

| Implementation | Target | Crate |
|---|---|---|
| `ReqwestHttpClient` | Native (Linux/macOS/Windows) | `blazen-llm` (feature `reqwest`) |
| `FetchHttpClient` | Browser WASM (`wasm32-unknown-unknown`) | `blazen-llm` (auto) |
| `WasiHttpClient` | WASIp2 (`wasm32-wasip2`) | `blazen-wasm` |

Constructors:
- `Provider::new(api_key)` — uses platform default HTTP client (not available on WASIp2)
- `Provider::new_with_client(api_key, client)` — always available, accepts any `Arc<dyn HttpClient>`

---

## Native Providers

### OpenAI

```rust
use blazen_llm::providers::openai::OpenAiProvider;

let model = OpenAiProvider::new("sk-...")
    .with_model("gpt-4.1");
```

| | |
|---|---|
| **Module** | `blazen_llm::providers::openai` |
| **Base URL** | `https://api.openai.com/v1` |
| **Auth** | Bearer token |
| **Models API** | `GET /v1/models` |
| **Streaming** | SSE |
| **Tool calling** | Yes |
| **Structured output** | Yes (`response_format`) |
| **Vision** | Yes |
| **Embeddings** | Yes (`OpenAiEmbeddingModel`) |

### Anthropic

```rust
use blazen_llm::providers::anthropic::AnthropicProvider;

let model = AnthropicProvider::new("sk-ant-...")
    .with_model("claude-sonnet-4-20250514");
```

| | |
|---|---|
| **Module** | `blazen_llm::providers::anthropic` |
| **Base URL** | `https://api.anthropic.com/v1` |
| **Auth** | `x-api-key` header + `anthropic-version` header |
| **Models API** | Static registry (no public `/models` endpoint) |
| **Streaming** | Typed SSE events (`message_start`, `content_block_delta`, etc.) |
| **Tool calling** | Yes |
| **Vision** | Yes |

### Gemini

```rust
use blazen_llm::providers::gemini::GeminiProvider;

let model = GeminiProvider::new("AIza...")
    .with_model("gemini-2.5-flash");
```

| | |
|---|---|
| **Module** | `blazen_llm::providers::gemini` |
| **Base URL** | `https://generativelanguage.googleapis.com/v1beta` |
| **Auth** | API key as query parameter |
| **Models API** | `GET /v1beta/models` |
| **Streaming** | SSE |
| **Tool calling** | Yes (via `functionCall`/`functionResponse`) |
| **Structured output** | Yes |
| **Vision** | Yes |

### Azure OpenAI

```rust
use blazen_llm::providers::azure::AzureOpenAiProvider;

let model = AzureOpenAiProvider::new("key", "my-resource", "gpt-4o-deployment");
```

| | |
|---|---|
| **Module** | `blazen_llm::providers::azure` |
| **Base URL** | `https://{resource}.openai.azure.com/openai/deployments/{deployment}` |
| **Auth** | `api-key` header |
| **Streaming** | SSE (OpenAI format) |
| **Tool calling** | Yes |
| **Structured output** | Yes |
| **Vision** | Yes |

### fal.ai

```rust
use blazen_llm::providers::fal::{FalProvider, FalLlmEndpoint};

// Default: OpenAI-compatible chat at
// `openrouter/router/openai/v1/chat/completions`.
let model = FalProvider::new("fal-...")
    .with_llm_model("openai/gpt-4o");
```

`FalProvider` is a native provider with its own endpoint matrix. The
LLM surface is controlled by [`FalLlmEndpoint`](#fal-llm-endpoints) and
an independently-chosen `llm_model`. Setting a model does **not** change
the URL path — endpoint and model are fully orthogonal.

| | |
|---|---|
| **Module** | `blazen_llm::providers::fal` |
| **Default LLM endpoint** | `FalLlmEndpoint::OpenAiChat` (`openrouter/router/openai/v1/chat/completions`) |
| **Default LLM execution** | Sync (`https://fal.run/...`) for OpenAI-compat endpoints; queue (`https://queue.fal.run/...`) for prompt-string endpoints |
| **Auth** | `Authorization: Key <key>` |
| **Streaming** | Real SSE for OpenAI-compat endpoints; cumulative-output SSE deltas for prompt-string endpoints |
| **Compute traits** | `ImageGeneration`, `VideoGeneration`, `AudioGeneration`, `Transcription`, `ThreeDGeneration`, `EmbeddingModel` |

#### fal LLM endpoints

`FalLlmEndpoint` selects the URL path and wire format. Model selection
is a **separate** field (`llm_model`) and is passed through in the body
of the request.

| Variant | URL path | Body format |
|---|---|---|
| `OpenAiChat` (default) | `openrouter/router/openai/v1/chat/completions` | OpenAI messages |
| `OpenAiResponses` | `openrouter/router/openai/v1/responses` | OpenAI Responses |
| `OpenAiEmbeddings` | `openrouter/router/openai/v1/embeddings` | OpenAI embeddings |
| `OpenRouter { enterprise: false }` | `openrouter/router` | Prompt string |
| `OpenRouter { enterprise: true }` | `openrouter/router/enterprise` | Prompt string |
| `AnyLlm { enterprise: false }` | `fal-ai/any-llm` | Prompt string |
| `AnyLlm { enterprise: true }` | `fal-ai/any-llm/enterprise` | Prompt string |
| `Vision { family, enterprise }` | `{openrouter\|fal-ai/any-llm}/.../vision[/enterprise]` | Prompt string + images |
| `Audio { family, enterprise }` | `{openrouter\|fal-ai/any-llm}/.../audio[/enterprise]` | Prompt string + audio |
| `Video { family, enterprise }` | `{openrouter\|fal-ai/any-llm}/.../video[/enterprise]` | Prompt string + video |
| `Custom { path, body_format }` | caller-supplied path | caller-selected format |

See [fal.ai API Reference](./fal-ai-api-reference.md) for the full
endpoint matrix, the 4800-char prompt-format budget, and auto-routing
rules.

#### Enterprise mode

```rust
use blazen_llm::providers::fal::{FalProvider, FalLlmEndpoint};

// Start on OpenRouter, promote to the enterprise variant.
let model = FalProvider::new("fal-...")
    .with_llm_endpoint(FalLlmEndpoint::OpenRouter { enterprise: false })
    .with_enterprise();
```

`with_enterprise()` promotes the current endpoint to its enterprise
variant (`openrouter/router/enterprise`, `fal-ai/any-llm/enterprise`,
`.../vision/enterprise`, etc.). The OpenAI-compat endpoints
(`OpenAiChat`, `OpenAiResponses`) have no enterprise variant, so
`with_enterprise()` on those promotes to
`AnyLlm { enterprise: true }` (the body format switches from OpenAI
messages to prompt-string — a log warning is emitted).

#### Auto-routing vision / audio / video

When the configured endpoint is `OpenRouter` or `AnyLlm` and a request
contains image, audio, or video content parts, `FalProvider`
transparently switches to the matching `Vision` / `Audio` / `Video`
sub-endpoint for the duration of that call. This is on by default;
disable with `.with_auto_route_modality(false)`.

```rust
use blazen_llm::{CompletionModel, CompletionRequest, ContentPart};
use blazen_llm::providers::fal::{FalProvider, FalLlmEndpoint};

let model = FalProvider::new("fal-...")
    .with_llm_endpoint(FalLlmEndpoint::OpenRouter { enterprise: false });

// A request with an image content part routes to
// `openrouter/router/vision` automatically.
```

#### Embeddings

```rust
use blazen_llm::providers::fal::FalProvider;
use blazen_llm::EmbeddingModel;

let provider = FalProvider::new("fal-...");
let em = provider.embedding_model();
let resp = em.embed(&["hello".into(), "world".into()]).await?;
```

`FalEmbeddingModel` implements the `EmbeddingModel` trait and posts to
`openrouter/router/openai/v1/embeddings` (the `OpenAiEmbeddings`
endpoint).

#### Extra compute methods

In addition to the standard image/video/audio traits, `FalProvider`
exposes:

- `generate_3d(...)` — 3D asset generation
- `remove_background(...)` — background removal
- `upscale_image_aura(...)` — Aura-SR upscaler
- `upscale_image_clarity(...)` — Clarity upscaler
- `upscale_image_creative(...)` — Creative upscaler

---

## OpenAI-Compatible Providers

All wrap `OpenAiCompatProvider` internally. Same constructor pattern, same trait implementations.

### Groq

```rust
use blazen_llm::providers::groq::GroqProvider;
let model = GroqProvider::new("gsk-...").with_model("llama-3.3-70b-versatile");
```

| | |
|---|---|
| **Module** | `blazen_llm::providers::groq` |
| **Base URL** | `https://api.groq.com/openai/v1` |
| **Default model** | `llama-3.3-70b-versatile` |
| **Models API** | Yes (`GET /openai/v1/models`) |
| **Key env var** | `GROQ_API_KEY` |

### OpenRouter

```rust
use blazen_llm::providers::openrouter::OpenRouterProvider;
let model = OpenRouterProvider::new("sk-or-...").with_model("anthropic/claude-sonnet-4");
```

| | |
|---|---|
| **Module** | `blazen_llm::providers::openrouter` |
| **Base URL** | `https://openrouter.ai/api/v1` |
| **Default model** | `openai/gpt-4.1` |
| **Models API** | Yes (400+ models with pricing) |
| **Key env var** | `OPENROUTER_API_KEY` |

### Together AI

```rust
use blazen_llm::providers::together::TogetherProvider;
let model = TogetherProvider::new("...").with_model("meta-llama/Llama-3.3-70B-Instruct-Turbo");
```

| | |
|---|---|
| **Module** | `blazen_llm::providers::together` |
| **Base URL** | `https://api.together.xyz/v1` |
| **Default model** | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| **Models API** | Yes (with pricing) |
| **Embeddings** | Yes |
| **Key env var** | `TOGETHER_API_KEY` |

### Mistral AI

```rust
use blazen_llm::providers::mistral::MistralProvider;
let model = MistralProvider::new("...").with_model("mistral-large-latest");
```

| | |
|---|---|
| **Module** | `blazen_llm::providers::mistral` |
| **Base URL** | `https://api.mistral.ai/v1` |
| **Default model** | `mistral-large-latest` |
| **Models API** | Yes |
| **Embeddings** | Yes |
| **Key env var** | `MISTRAL_API_KEY` |

### DeepSeek

```rust
use blazen_llm::providers::deepseek::DeepSeekProvider;
let model = DeepSeekProvider::new("...").with_model("deepseek-chat");
```

| | |
|---|---|
| **Module** | `blazen_llm::providers::deepseek` |
| **Base URL** | `https://api.deepseek.com` |
| **Default model** | `deepseek-chat` |
| **Models API** | No |
| **Key env var** | `DEEPSEEK_API_KEY` |

### Fireworks AI

```rust
use blazen_llm::providers::fireworks::FireworksProvider;
let model = FireworksProvider::new("...").with_model("accounts/fireworks/models/llama-v3p3-70b-instruct");
```

| | |
|---|---|
| **Module** | `blazen_llm::providers::fireworks` |
| **Base URL** | `https://api.fireworks.ai/inference/v1` |
| **Default model** | `accounts/fireworks/models/llama-v3p3-70b-instruct` |
| **Models API** | Yes |
| **Embeddings** | Yes |
| **Key env var** | `FIREWORKS_API_KEY` |

### Perplexity

```rust
use blazen_llm::providers::perplexity::PerplexityProvider;
let model = PerplexityProvider::new("...").with_model("sonar-pro");
```

| | |
|---|---|
| **Module** | `blazen_llm::providers::perplexity` |
| **Base URL** | `https://api.perplexity.ai` |
| **Default model** | `sonar-pro` |
| **Models API** | No |
| **Key env var** | `PERPLEXITY_API_KEY` |

### xAI (Grok)

```rust
use blazen_llm::providers::xai::XaiProvider;
let model = XaiProvider::new("...").with_model("grok-3");
```

| | |
|---|---|
| **Module** | `blazen_llm::providers::xai` |
| **Base URL** | `https://api.x.ai/v1` |
| **Default model** | `grok-3` |
| **Models API** | Yes |
| **Embeddings** | Yes |
| **Key env var** | `XAI_API_KEY` |

### Cohere

```rust
use blazen_llm::providers::cohere::CohereProvider;
let model = CohereProvider::new("...").with_model("command-a-08-2025");
```

| | |
|---|---|
| **Module** | `blazen_llm::providers::cohere` |
| **Base URL** | `https://api.cohere.ai/compatibility/v1` |
| **Default model** | `command-a-08-2025` |
| **Models API** | No |
| **Embeddings** | Yes |
| **Key env var** | `COHERE_API_KEY` |

### AWS Bedrock

```rust
use blazen_llm::providers::bedrock::BedrockProvider;
let model = BedrockProvider::new("key", "us-east-1")
    .with_model("anthropic.claude-sonnet-4-20250514-v1:0");
```

| | |
|---|---|
| **Module** | `blazen_llm::providers::bedrock` |
| **Base URL** | `https://bedrock-mantle.{region}.api.aws/v1` |
| **Default model** | `anthropic.claude-sonnet-4-20250514-v1:0` |
| **Models API** | Yes |
| **Key env var** | `BEDROCK_API_KEY` + `BEDROCK_REGION` |

---

## Custom OpenAI-Compatible Endpoints

For any OpenAI-compatible API not listed above, use `OpenAiCompatProvider` directly:

```rust
use blazen_llm::providers::openai_compat::{OpenAiCompatProvider, OpenAiCompatConfig, AuthMethod};

let provider = OpenAiCompatProvider::new_with_client(
    OpenAiCompatConfig {
        provider_name: "my-provider".into(),
        base_url: "https://my-llm.example.com/v1".into(),
        api_key: "sk-...".into(),
        default_model: "my-model".into(),
        auth_method: AuthMethod::Bearer,
        extra_headers: vec![],
        query_params: vec![],
        supports_model_listing: false,
    },
    http_client,
);
```

In the WASM component, custom providers can be registered at runtime via `POST /v1/providers/register`.

---

## Text-to-Speech providers

Blazen exposes text-to-speech through the `OpenAiProvider` compute class (separate from `CompletionModel.openai()`, which is the chat completions entry point). Because `OpenAiProvider` speaks the stock `/v1/audio/speech` wire format, the **same class** works against real OpenAI and against any OpenAI-compatible TTS service — including a local [zvoice](https://github.com/zachhandley/zvoice) instance wrapping the [VoxCPM2](https://github.com/OpenBMB/VoxCPM2) model.

Setting `api_key` to an empty string makes the provider omit the `Authorization` header entirely, which is required for most local services.

### Rust

```rust
use blazen_llm::providers::openai::OpenAiProvider;
use blazen_llm::compute::{AudioGeneration, SpeechRequest};

// Real OpenAI.
let openai = OpenAiProvider::new("sk-...");
let audio = openai
    .text_to_speech(SpeechRequest::new("Hello, world!").with_voice("alloy"))
    .await?;

// Local zvoice wrapping VoxCPM2 — empty key omits the Authorization header.
let zvoice = OpenAiProvider::new("").with_base_url("http://beastpc.lan:8900/v1");
let audio = zvoice
    .text_to_speech(SpeechRequest::new("Hello, world!"))
    .await?;
```

### Python

```python
from blazen import OpenAiProvider, ProviderOptions, SpeechRequest

# Real OpenAI
openai = OpenAiProvider(options=ProviderOptions(api_key="sk-..."))
audio = await openai.text_to_speech(SpeechRequest(text="Hello, world!", voice="alloy"))

# Local zvoice wrapping VoxCPM2 — empty api_key omits Authorization header
zvoice = OpenAiProvider(options=ProviderOptions(
    api_key="",
    base_url="http://beastpc.lan:8900/v1",
))
audio = await zvoice.text_to_speech(SpeechRequest(text="Hello, world!"))
```

### Node.js / TypeScript

```typescript
import { OpenAiProvider } from "blazen";

const openai = OpenAiProvider.create({ apiKey: "sk-..." });
const audio = await openai.textToSpeech({ text: "Hello, world!", voice: "alloy" });

const zvoice = OpenAiProvider.create({ apiKey: "", baseUrl: "http://beastpc.lan:8900/v1" });
const localAudio = await zvoice.textToSpeech({ text: "Hello, world!" });
```

The same base-URL override pattern works with `OpenAiCompatProvider` for any other OpenAI-compatible TTS service (Groq, Together AI, etc.) that ships a `/v1/audio/speech` endpoint.

---

## Local models

Blazen's local in-process LLM backends (`mistral.rs`, `llama.cpp`, `candle`) load model weights into memory or VRAM on the current process. Unlike remote providers, these models are a finite resource — you only want one (or a few) loaded at once, and you want explicit control over when the GPU memory is freed.

Every `CompletionModel` exposes four lifecycle methods:

| Method | Returns | Behaviour |
|---|---|---|
| `load()` | `None` / `void` | For local providers, synchronously downloads (if needed) and loads the model so the next inference call pays no startup cost. For remote providers, raises `NotImplementedError` / throws. Idempotent. |
| `unload()` | `None` / `void` | For local providers, drops the model and frees its GPU/CPU memory so the process can load a different model. For remote providers, raises `NotImplementedError` / throws. Idempotent. |
| `is_loaded()` / `isLoaded()` | `bool` | Whether the model is currently resident in memory. Always `false` for remote providers (they have no local weights). |
| `vram_bytes()` / `vramBytes()` | `int?` / `number?` | Approximate VRAM footprint in bytes, if the underlying implementation reports one. `None` / `null` for remote providers and for local providers that don't yet expose memory usage. |

Inference methods (`complete`, `stream`) still auto-load the model on first call if you never called `load()` explicitly — the explicit calls are only there when you need deterministic timing or want to unload between runs.

Today the Python and Node factories only expose `CompletionModel.mistralrs()`; `llama.cpp` and `candle` are Rust-only until their host-language factories land, but all three already implement the load/unload plumbing in Rust.

### Python

```python
from blazen import CompletionModel, MistralRsOptions, ChatMessage

model = CompletionModel.mistralrs(options=MistralRsOptions(
    model_id="mistralai/Mistral-7B-Instruct-v0.3",
))

# Explicit load (otherwise happens lazily on first `complete` call).
await model.load()
assert await model.is_loaded()

# Inference
response = await model.complete([ChatMessage.user("Hello")])

# Free the GPU memory when idle.
await model.unload()
assert not await model.is_loaded()

# Remote providers raise NotImplementedError on load.
openai = CompletionModel.openai()
try:
    await openai.load()
except NotImplementedError:
    print("openai is a remote provider, nothing to load")
```

### Node.js / TypeScript

```typescript
import { CompletionModel } from "blazen";

const model = CompletionModel.mistralrs({
    modelId: "mistralai/Mistral-7B-Instruct-v0.3",
});

await model.load();
console.log(await model.isLoaded()); // true

// Inference ...

await model.unload();
console.log(await model.isLoaded()); // false

// Remote providers throw on load.
const openai = CompletionModel.openai({});
try {
    await openai.load();
} catch (e) {
    console.log("openai is remote:", (e as Error).message);
}
```

Notes:

- `load` / `unload` are idempotent — calling them repeatedly is a no-op.
- `is_loaded` / `isLoaded` always returns `false` for remote providers.
- `vram_bytes` / `vramBytes` returns `None` / `null` for providers that don't report memory usage. Today that is every provider — the plumbing is wired through but no backend exposes a number yet.
- If you never call `load()` explicitly, inference methods still auto-load on first call. The explicit lifecycle is only there so you can front-load startup cost and unload between runs.

---

## SDK Usage

### Python

```python
from blazen import CompletionModel, ProviderOptions

model = CompletionModel.groq(options=ProviderOptions(api_key="gsk-..."))
# Or: .openai(), .anthropic(), .gemini(), .azure(), .fal(),
#     .openrouter(), .together(), .mistral(), .deepseek(),
#     .fireworks(), .perplexity(), .xai(), .cohere(), .bedrock()
# All provider methods take `options=ProviderOptions(api_key="...")`.

response = await model.complete(
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    tools=[{"name": "get_weather", "description": "...", "parameters": {...}}],
)
```

For `fal.ai`, pass a `FalOptions` instead of loose kwargs:

```python
from blazen import CompletionModel, FalOptions, FalLlmEndpoint, FalProvider

# LLM completion model via CompletionModel.fal():
model = CompletionModel.fal(
    options=FalOptions(
        api_key="fal-...",
        model="openai/gpt-4o",
        endpoint=FalLlmEndpoint.OPENAI_CHAT,  # default
        enterprise=False,
        auto_route_modality=True,
    ),
)

# Or build a full FalProvider for compute + embeddings:
fal = FalProvider(
    options=FalOptions(
        api_key="fal-...",
        enterprise=True,
    ),
)
embeddings = fal.embedding_model()
resp = await embeddings.embed(["hello", "world"])

# Extra compute methods
await fal.generate_3d(request)
await fal.remove_background(image_url="https://...")
await fal.upscale_image_aura(request)
await fal.upscale_image_clarity(request)
await fal.upscale_image_creative(request)
```

### Node.js / TypeScript

```typescript
import { CompletionModel } from 'blazen';

const model = CompletionModel.groq({ apiKey: 'gsk-...' });

const response = await model.completeWithOptions(
    [{ role: 'user', content: 'Hello' }],
    { temperature: 0.7, maxTokens: 1000, topP: 0.9 }
);
```

### WASM (WASIp2)

The WASM component exposes an OpenAI-compatible HTTP API. Route to any provider via the model string:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "groq/llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "Hello"}]}'
```

Provider is resolved from the `provider/model` prefix. If no slash, defaults to `openai`.

---

## Writing a custom provider

Not every service Blazen users want to reach ships an OpenAI-compatible API, and not every capability maps onto one of the built-in providers above. For those cases Blazen exposes a **custom provider** escape hatch in all three host languages: you write a normal class in Rust, Python, or TypeScript whose methods match Blazen's capability trait method names, hand it to Blazen, and get back an object that behaves exactly like a first-class provider.

The workflow engine treats the wrapped object as implementing every capability trait whose methods the host object actually defines; missing methods raise `UnsupportedError` at the call site.

### Rust

Downstream Rust crates can implement the capability traits directly on any type of their choosing — this is the baseline and has always been supported. The reference example is embedded in [`crates/blazen-llm/src/compute/traits.rs`](../crates/blazen-llm/src/compute/traits.rs) lines 1–33, which walks through a minimal `ComputeProvider` + `ImageGeneration` impl.

```rust
use async_trait::async_trait;
use blazen_llm::compute::{ComputeProvider, ImageGeneration, ImageRequest, ImageResult};
use blazen_llm::BlazenError;

struct MyImageProvider { /* ... */ }

#[async_trait]
impl ImageGeneration for MyImageProvider {
    async fn generate_image(&self, request: ImageRequest) -> Result<ImageResult, BlazenError> {
        // Your implementation.
        todo!()
    }
    // upscale_image(...) ...
}
```

The Rust path does not need a wrapper type — you just `impl` the traits you care about, and the rest of Blazen picks up the implementation through its usual trait machinery.

### Python (`blazen.CustomProvider`)

On the Python side, write a normal class with `async def` methods whose names match Blazen's capability trait method names (snake_case: `text_to_speech`, `clone_voice`, `generate_image`, ...), then wrap an instance in `CustomProvider(instance, provider_id="...")`.

Requests arrive as plain Python dicts shaped like the corresponding `SpeechRequest` / `ImageRequest` / ... classes — you do not need to construct typed request objects yourself. Responses must be dicts whose keys match the underlying Rust result struct fields exactly (`AudioResult`, `VoiceHandle`, `ImageResult`, ...), because Blazen deserialises them through serde.

```python
import base64
from blazen import CustomProvider, SpeechRequest

class MyElevenLabsProvider:
    def __init__(self, api_key: str):
        from elevenlabs.client import AsyncElevenLabs
        self._client = AsyncElevenLabs(api_key=api_key)

    async def text_to_speech(self, request):
        # `request` is a dict shaped like SpeechRequest (serde JSON).
        audio_bytes = await self._client.text_to_speech.convert(
            voice_id=request["voice"],
            text=request["text"],
            model_id="eleven_multilingual_v2",
        )
        # Return a dict shaped like AudioResult.
        return {
            "audio": [{
                "media": {
                    "base64": base64.b64encode(audio_bytes).decode(),
                    "media_type": "mpeg",
                    "metadata": {},
                },
            }],
            "timing": {"total_ms": 0, "queue_ms": None, "execution_ms": None},
            "metadata": {},
        }

    async def clone_voice(self, request):
        voice = await self._client.voices.add(
            name=request["name"],
            files=[...],  # fetch request["reference_urls"][0] to bytes
        )
        return {
            "id": voice.voice_id,
            "name": voice.name,
            "provider": "elevenlabs",
            "metadata": {},
        }

# Wrap the instance as a Blazen provider.
my_provider = CustomProvider(
    MyElevenLabsProvider(api_key="..."),
    provider_id="elevenlabs",
)

# Now use it exactly like a built-in provider.
audio = await my_provider.text_to_speech(SpeechRequest(
    text="Hello from ElevenLabs!",
    voice="rachel",
))
```

Notes:

- Method names match the Rust trait method names (`text_to_speech`, **not** `textToSpeech`).
- Requests arrive as Python dicts. You do not need to write `SpeechRequest(...)` constructors inside your host class.
- Responses must be dicts that match the corresponding result struct layout (`AudioResult`, `VoiceHandle`, etc.). Dict keys must match the Rust struct field names exactly — the Rust side deserialises them via serde.
- Only methods you actually implement count. If your class doesn't define a `generate_music` method, calling `my_provider.generate_music(...)` raises `UnsupportedError`.
- Full list of supported capability method names (snake_case): `text_to_speech`, `generate_music`, `generate_sfx`, `clone_voice`, `list_voices`, `delete_voice`, `generate_image`, `upscale_image`, `text_to_video`, `image_to_video`, `transcribe`, `generate_3d`, `remove_background`, `submit`, `status`, `result`, `cancel`.

### Node.js / TypeScript (`CustomProvider`)

Same idea on the JavaScript side, but method names use camelCase (`textToSpeech`, `cloneVoice`, `generateImage`, ...) to match JavaScript conventions. The Rust shim calls `Function.prototype.bind(hostObject)` under the hood so `this` refers to the original instance when your methods are invoked — you can safely write class methods that read `this.client` or `this.apiKey`.

```typescript
import { CustomProvider } from "blazen";
import { ElevenLabsClient } from "@elevenlabs/elevenlabs-js";

class MyElevenLabsProvider {
    private client: ElevenLabsClient;

    constructor(apiKey: string) {
        this.client = new ElevenLabsClient({ apiKey });
    }

    async textToSpeech(request: any) {
        const audio = await this.client.textToSpeech.convert({
            voiceId: request.voice,
            text: request.text,
            modelId: "eleven_multilingual_v2",
        });
        return {
            audio: [{
                media: {
                    base64: Buffer.from(audio).toString("base64"),
                    mediaType: "mpeg",
                    metadata: {},
                },
            }],
            timing: { totalMs: 0, queueMs: null, executionMs: null },
            metadata: {},
        };
    }

    async cloneVoice(request: any) {
        const voice = await this.client.voices.add({
            name: request.name,
            files: [/* fetch from request.referenceUrls[0] */],
        });
        return {
            id: voice.voiceId,
            name: voice.name,
            provider: "elevenlabs",
            metadata: {},
        };
    }
}

const provider = new CustomProvider(
    new MyElevenLabsProvider("..."),
    { providerId: "elevenlabs" },
);

const audio = await provider.textToSpeech({
    text: "Hello!",
    voice: "rachel",
});
```

Notes:

- Method names on the JS side are camelCase (`textToSpeech`, `cloneVoice`, `generateImage`, ...).
- The Rust shim uses `Function.prototype.bind(hostObject)` internally so `this` resolves to the right instance when your methods are invoked.
- Missing methods throw `UnsupportedError` at the call site.
- Full list of supported capability method names (camelCase): `textToSpeech`, `generateMusic`, `generateSfx`, `cloneVoice`, `listVoices`, `deleteVoice`, `generateImage`, `upscaleImage`, `textToVideo`, `imageToVideo`, `transcribe`, `generate3d`, `removeBackground`, `submit`, `status`, `result`, `cancel`.
