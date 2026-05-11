# blazen-wasm

WASIp2 WASM component for deploying Blazen as an edge HTTP handler via ZLayer (or any WASI-compatible host).

## What it does

`blazen-wasm` implements `wasi:http/incoming-handler` to serve an OpenAI-compatible API backed by Blazen's LLM providers. Outbound LLM API calls go through `wasi:http/outgoing-handler`.

```text
[Client] --(HTTP)--> [wasmtime host / ZLayer]
                          |
                   wasi:http/incoming-handler
                          |
                     [blazen-wasm]
                          |
                   wasi:http/outgoing-handler
                          |
                  [LLM Provider APIs]
```

### Endpoints

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| GET | `/health` | Working | Health check |
| POST | `/v1/chat/completions` | Working | Chat completion (dispatches to real providers) |
| GET | `/v1/providers` | Working | List available providers (built-in + custom) |
| POST | `/v1/providers/register` | Working | Register a custom OpenAI-compatible provider |
| POST | `/v1/images/generations` | Not yet implemented (501) | Image generation |
| POST | `/v1/audio/speech` | Not yet implemented (501) | Text-to-speech |
| POST | `/v1/agent/run` | Not yet implemented (501) | Agent execution |

## Build

Requires [cargo-component](https://github.com/bytecodealliance/cargo-component):

```bash
cargo component build --target wasm32-wasip2 --release
```

The output `.wasm` file lands in `target/wasm32-wasip2/release/`.

## Deploy to ZLayer

```bash
zlayer deploy target/wasm32-wasip2/release/blazen_wasm.wasm
```

## API key strategies

- Pass provider API keys via request headers (`X-API-Key`, `Authorization: Bearer ...`)
- Or configure them as environment variables on the host / ZLayer deployment

## Docs

Full documentation at [blazen.dev/docs/getting-started/introduction](https://blazen.dev/docs/getting-started/introduction).

## License

MPL-2.0
