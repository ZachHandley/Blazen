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
use blazen_llm::providers::fal::FalProvider;

let model = FalProvider::new("fal-...")
    .with_llm_model("anthropic/claude-sonnet-4.5");
```

| | |
|---|---|
| **Module** | `blazen_llm::providers::fal` |
| **Base URL** | `https://queue.fal.run` |
| **Auth** | `Authorization: Key <key>` |
| **Execution** | Queue-based (submit → poll → result) |
| **Compute traits** | `ImageGeneration`, `VideoGeneration`, `AudioGeneration`, `Transcription` |

See [fal.ai API Reference](./fal-ai-api-reference.md) for detailed endpoint documentation.

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
from blazen import CompletionModel

model = CompletionModel.groq("gsk-...")
# Or: .openai(), .anthropic(), .gemini(), .azure(), .fal(),
#     .openrouter(), .together(), .mistral(), .deepseek(),
#     .fireworks(), .perplexity(), .xai(), .cohere(), .bedrock()

response = await model.complete(
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    tools=[{"name": "get_weather", "description": "...", "parameters": {...}}],
)
```

### Node.js / TypeScript

```typescript
import { CompletionModel } from 'blazen';

const model = CompletionModel.groq('gsk-...');

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
