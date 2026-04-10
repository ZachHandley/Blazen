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
