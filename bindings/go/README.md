# Blazen for Go

Idiomatic Go bindings for [Blazen](https://github.com/ZachHandley/Blazen), the event-driven AI workflow engine written in Rust. The binding wraps the same `blazen-uniffi` core that powers the Swift and Kotlin SDKs, so workflow steps, LLM calls, streaming, agents, and telemetry all behave identically across languages.

## Status

This is a v0 binding. Linux `amd64` and `arm64` ship today as prebuilt static archives under `internal/clib/<GOOS>_<GOARCH>/libblazen_uniffi.a`; Windows and macOS land via Forgejo CI release builds. The Ruby binding is a separate work-in-progress crate under `crates/blazen-uniffi/src/bin/` and is unrelated to this Go module.

## Install

Go 1.22+ with cgo enabled (a host `gcc` or `clang`):

```bash
go get github.com/zachhandley/Blazen/bindings/go@latest
```

The bundled cgo link directive already passes `-lstdc++` against the prebuilt static archive, so no extra `CGO_LDFLAGS` plumbing is needed on the caller side.

## Hello workflow

A one-step workflow that consumes a `StartEvent` and emits a `StopEvent`. Note the `blazen::` namespace prefix on event types -- that's the FFI wire format the engine routes on. `StartEvent` payloads arrive wrapped in `{"data": <user-input>}`, and `StopEvent` payloads are emitted as `{"result": <output>}`.

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "time"

    blazen "github.com/zachhandley/Blazen/bindings/go"
)

type greetHandler struct{}

func (greetHandler) Invoke(_ context.Context, e blazen.Event) (blazen.StepOutput, error) {
    var envelope struct {
        Data struct{ Name string `json:"name"` } `json:"data"`
    }
    if err := json.Unmarshal([]byte(e.DataJSON), &envelope); err != nil {
        return nil, err
    }
    out, _ := json.Marshal(map[string]any{
        "result": map[string]string{"greeting": "Hello, " + envelope.Data.Name + "!"},
    })
    return blazen.NewStepOutputSingle(blazen.Event{
        EventType: "blazen::StopEvent", DataJSON: string(out),
    }), nil
}

func main() {
    blazen.Init()
    defer blazen.Shutdown()

    builder, err := blazen.NewWorkflowBuilder("greeter").Step(
        "greet",
        []string{"blazen::StartEvent"}, []string{"blazen::StopEvent"},
        greetHandler{},
    )
    if err != nil { panic(err) }
    wf, err := builder.Build()
    if err != nil { panic(err) }
    defer wf.Close()

    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    result, err := wf.Run(ctx, map[string]string{"name": "Zach"})
    if err != nil { panic(err) }
    fmt.Printf("%s -> %s\n", result.Event.EventType, result.Event.DataJSON)
}
```

A complete runnable version -- with full error classification across the typed-error tree -- lives in [`examples/hello_workflow/main.go`](examples/hello_workflow/main.go).

## Error handling

Every error returned from the binding implements `blazen.Error`. Use `errors.As` to recover the concrete variant. Most call sites only need to dispatch on a handful:

```go
result, err := wf.Run(ctx, input)
if err != nil {
    var rl *blazen.RateLimitError
    if errors.As(err, &rl) {
        // Provider Retry-After is in milliseconds; 0 means no hint.
        time.Sleep(time.Duration(rl.RetryAfterMs) * time.Millisecond)
        return retry()
    }
    var ve *blazen.ValidationError
    if errors.As(err, &ve) {
        return fmt.Errorf("bad input: %s", ve.Message)
    }
    return err
}
```

See `errors.go` for the full set: `AuthError`, `RateLimitError`, `TimeoutError`, `ValidationError`, `ContentPolicyError`, `UnsupportedError`, `ComputeError`, `MediaError`, `ProviderError`, `WorkflowError`, `ToolError`, `PeerError`, `PersistError`, `PromptError`, `MemoryError`, `CacheError`, `CancelledError`, `InternalError`.

## Native library distribution

The prebuilt static archives under `internal/clib/<GOOS>_<GOARCH>/` are committed to the repository, so end users do not need to build anything beyond running `go get`. If you are hacking on the binding itself, rebuild the archives locally with:

```bash
./scripts/build-uniffi-lib.sh
```

from the repository root.

## Where to go from here

- **LLM completions** -- see [`llm.go`](llm.go) and the [Go LLM guide](https://blazen.dev/docs/guides/go/llm).
- **Streaming** -- see [`streaming.go`](streaming.go) and the [Go Streaming guide](https://blazen.dev/docs/guides/go/streaming).
- **Agent loop** -- see [`agent.go`](agent.go) and the [Go Agent guide](https://blazen.dev/docs/guides/go/agent).
- **Full quickstart** -- [Go Quickstart on blazen.dev](https://blazen.dev/docs/guides/go/quickstart).

The hosted guides go live alongside the Phase 8 docs release; until then the surface is fully documented inline via Go doc comments (`go doc github.com/zachhandley/Blazen/bindings/go`).

## License

[MPL-2.0](../../LICENSE). The same terms as the rest of the Blazen workspace.
