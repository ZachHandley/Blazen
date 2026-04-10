# Local Chat Browser Example

On-device LLM chat using the Blazen WASM SDK with [WebLLM](https://webllm.mlc.ai/) for local inference. No API key required after the model is downloaded.

## What it demonstrates

- Creating a WebLLM engine for a small on-device model
- Wrapping it as a Blazen `CompletionModel` with `CompletionModel.fromJsHandler()`
- Multi-turn chat using `model.complete(messages)`
- A fallback-to-API pattern using `CompletionModel.withFallback()` (commented out, ready to enable)

## Prerequisites

```bash
npm install @blazen/sdk @mlc-ai/web-llm
```

## Browser requirements

WebLLM requires **WebGPU** support:

- Chrome / Edge 113+
- Safari 26+ (Sep 2025, including iOS / iPadOS)
- Firefox 141+ (Windows) / 145+ (macOS ARM)

On iPhones, only iPhone 15 Pro and newer have enough RAM for even 1B models.

## Running

```bash
npx serve .
```

Then open `http://localhost:3000`.

### First load

The first visit downloads model weights (~600 MB for the 1B model). They are cached by WebLLM in IndexedDB, so subsequent visits start much faster.

### Model compilation

WebLLM compiles the model for your specific GPU on first use. This takes 30-60 seconds. After compilation, the shader cache persists and future loads are near-instant.

## How it works

1. **`init()`** loads the Blazen WASM binary.
2. **`webllm.CreateMLCEngine()`** downloads, caches, and compiles the model for WebGPU.
3. **`CompletionModel.fromJsHandler()`** wraps the WebLLM engine as a Blazen `CompletionModel`.
4. **`model.complete(messages)`** sends the conversation to the local model and returns the response.

## Choosing a model

Stick to 1B-3B parameter models for a usable experience:

| Model | Size | Cold start | Tokens/sec |
|---|---|---|---|
| `Llama-3.2-1B-Instruct-q4f16_1-MLC` | ~600 MB | ~30s | 40-60 |
| `Llama-3.2-3B-Instruct-q4f16_1-MLC` | ~1.8 GB | ~60s | 15-30 |

Models at 7B+ parameters require 4+ GB of GPU memory and cause 1-3 minute cold starts. They are not recommended for browser use.

## API fallback pattern

The commented-out section at the bottom of the error handler demonstrates how to set up a hosted API fallback. This is the recommended production pattern -- try local inference first, fall back to a cloud API if WebGPU is unavailable:

```js
const localModel = CompletionModel.fromJsHandler('local', handler);
const apiModel = CompletionModel.openrouter();
const model = CompletionModel.withFallback([localModel, apiModel]);
```
