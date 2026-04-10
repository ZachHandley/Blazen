# Local RAG Browser Example

In-browser Retrieval-Augmented Generation (RAG) using the Blazen WASM SDK with [transformers.js](https://huggingface.co/docs/transformers.js) for local embeddings. No API key required -- the embedding model runs entirely in the browser.

## What it demonstrates

- Loading the `Xenova/all-MiniLM-L6-v2` embedding model via `transformers.js`
- Wrapping it as a Blazen `EmbeddingModel` with `EmbeddingModel.fromJsHandler()`
- Creating a `Memory` store and inserting documents
- Performing semantic search with `memory.search()`

## Prerequisites

```bash
npm install @blazen/sdk @huggingface/transformers
```

## Running

The simplest approach is a local static file server:

```bash
npx serve .
```

Then open `http://localhost:3000` (or whichever port is shown).

### With a bundler

If you use Vite, webpack, or another bundler, copy this `index.html` into your project and the `@blazen/sdk` and `@huggingface/transformers` imports will resolve from `node_modules` automatically.

### With import maps (no build step)

Uncomment the `<script type="importmap">` block in `index.html` and adjust the CDN URLs to the versions you need. This works in all modern browsers without any bundler.

## How it works

1. **`init()`** loads the Blazen WASM binary.
2. **`pipeline('feature-extraction', ...)`** from transformers.js downloads and caches the embedding model (~23 MB, one-time).
3. **`EmbeddingModel.fromJsHandler()`** wraps the pipeline as a Blazen `EmbeddingModel` so it can be used with `Memory`.
4. **`new Memory(embedder)`** creates an in-memory vector store backed by the local embedding model.
5. **`memory.search(query, limit)`** embeds the query locally and performs similarity search.

## Performance notes

- First load downloads the model weights (~23 MB). Subsequent loads use the browser cache.
- Embedding latency is approximately 170 ms per query on WASM (CPU), or ~35 ms with WebGPU if the browser supports it.
- The `Memory` store is in-memory and is lost on page reload.
