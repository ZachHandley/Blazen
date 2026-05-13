# Blazen for Ruby

Idiomatic Ruby bindings for [Blazen](https://github.com/ZachHandley/Blazen), the event-driven AI workflow engine written in Rust. The same Rust core that powers the Go, Swift, Kotlin, Python, Node, and WASM SDKs — exposed through a hand-written C ABI (`libblazen_cabi`) and wrapped with the `ffi` gem, so workflow steps, LLM calls, streaming, agents, and persistence all run in-process against actual Rust.

## Status

Working as of 2026-05-12. Native code is loaded via `cbindgen`-generated C headers plus the Ruby [`ffi`](https://github.com/ffi/ffi) gem (no UniFFI Ruby template required). Async-aware: every `_blocking` method has an awaitable counterpart that yields via `Fiber.scheduler`, so the binding composes cleanly with the [`async`](https://github.com/socketry/async) gem.

## Installation

Maven Central / RubyGems publishing is pending. For now, build from source:

```bash
cd bindings/ruby
bundle install
rake build              # produces pkg/blazen-<version>.gem
gem install pkg/blazen-*.gem
```

System requirements:

- Ruby 3.0+ (tested against 3.4)
- Linux, macOS, or Windows (the `ffi` gem handles platform dispatch)
- The bundled `ext/blazen/libblazen_cabi.{so,dylib,dll}` shared library, refreshed via `./scripts/build-uniffi-lib.sh` from the workspace root

## Quickstart

A single-step workflow with a Ruby-defined `StepHandler` block. The `StartEvent` payload arrives wrapped as `{"data": <user-input>}`, and `StopEvent` payloads are emitted as `{"result": <output>}` — the same wire format every other Blazen binding uses.

```ruby
require "blazen"

Blazen.init

workflow = Blazen.workflow("echo") do |b|
  b.step("echo",
         accepts: ["blazen::StartEvent"],
         emits:   ["blazen::StopEvent"]) do |evt|
    payload = evt.data["data"] || {}
    Blazen::Workflow::StepOutput.single(
      Blazen::Workflow::Event.create(
        event_type: "blazen::StopEvent",
        data: { result: payload },
      ),
    )
  end
end

result = workflow.run_blocking({ hello: "world" })
puts result.event_data.inspect   # => {"result" => {"hello" => "world"}}
```

`Blazen.init` warms the shared Tokio runtime and installs a default tracing subscriber. It is idempotent — call it once at process start.

## LLM completion

Every provider lives behind a keyword-arg factory on `Blazen::Providers`. Models implement both `#complete_blocking` (thread-blocking) and `#complete` (future-returning, composes with `Fiber.scheduler`).

```ruby
model = Blazen::Providers.openai(api_key: ENV.fetch("OPENAI_API_KEY"))
req = Blazen::Llm.completion_request(
  messages: [Blazen::Llm.user("Hello!")],
  max_tokens: 64,
)
response = model.complete_blocking(req)
puts response.content
```

## Async with the `async` gem

The non-`_blocking` variants of `Workflow#run`, `Pipeline#run`, `CompletionModel#complete`, and friends yield via `Fiber.scheduler`. Under [`async`](https://github.com/socketry/async), three concurrent workflow runs overlap without spawning extra threads:

```ruby
require "async"
require "blazen"

Blazen.init

Async do
  tasks = (1..3).map { |i| Async { workflow.run({ task: i }) } }
  results = tasks.map(&:wait)
  results.each { |r| pp r.event_data }
end
```

## Streaming completions

`Blazen::Streaming.complete` accepts either explicit `on_chunk:`/`on_done:`/`on_error:` kwargs or a single block that receives `(kind, *args)`:

```ruby
Blazen::Streaming.complete(model, req) do |kind, *args|
  case kind
  when :chunk then print args.first.content_delta
  when :done  then puts                                # blank line on completion
  when :error then warn "stream error: #{args.first.message}"
  end
end
```

`complete_async` returns when the streaming future resolves; the sink callbacks fire on Tokio worker threads while the calling fiber yields.

## Agents with tool calls

`Blazen::Agents.new` builds a tool-calling loop backed by a Ruby `tool_handler` proc. The handler receives `(tool_name, args_hash)` and returns either a `String` (forwarded verbatim) or any JSON-serialisable value (encoded via `JSON.dump`).

```ruby
tools = [
  Blazen::Llm.tool(
    name: "get_weather",
    description: "Get current weather for a city",
    parameters: { type: "object", properties: { city: { type: "string" } } },
  ),
]

agent = Blazen::Agents.new(
  model: model,
  tools: tools,
  tool_handler: ->(name, args) {
    case name
    when "get_weather" then "It's sunny in #{args['city']}, 72°F."
    else raise "unknown tool: #{name}"
    end
  },
  max_iterations: 5,
)

result = agent.run_blocking("What is the weather in San Francisco?")
puts result.final_message
puts "iterations=#{result.iterations} tool_calls=#{result.tool_call_count}"
```

## Error handling

Every native error surfaces as an idiomatic Ruby exception under [`Blazen::Error`](lib/blazen/errors.rb):

```ruby
begin
  workflow.run_blocking(input)
rescue Blazen::RateLimitError => e
  sleep((e.retry_after_ms || 1_000) / 1_000.0)
  retry
rescue Blazen::AuthError
  rotate_api_key
rescue Blazen::ProviderError => e
  log.warn("provider #{e.provider} failed: #{e.kind} (status=#{e.status})")
rescue Blazen::Error => e
  # Catch-all for the rest: WorkflowError, TimeoutError, ValidationError,
  # ContentPolicyError, UnsupportedError, ComputeError, MediaError,
  # ToolError, PeerError, PersistError, PromptError, MemoryError,
  # CacheError, CancelledError, InternalError.
  raise e
end
```

`RateLimitError`, `TimeoutError`, `ProviderError`, `PeerError`, `PromptError`, `MemoryError`, and `CacheError` carry structured fields (`retry_after_ms`, `elapsed_ms`, `kind`, `provider`, `status`, …) so callers branch on type instead of parsing messages.

## Building from source

```bash
# Refresh the native shared library (linux_amd64 by default — pass the target
# triple for cross-arch builds).
./scripts/build-uniffi-lib.sh

# Rebuild the gem (the rake task re-copies the native lib into ext/blazen/
# before packaging).
cd bindings/ruby
rake build
```

## Where to go from here

- [Ruby quickstart](https://blazen.dev/docs/guides/ruby/quickstart)
- [Ruby LLM guide](https://blazen.dev/docs/guides/ruby/llm)
- [Ruby streaming guide](https://blazen.dev/docs/guides/ruby/streaming)
- [Ruby agent guide](https://blazen.dev/docs/guides/ruby/agent)

The surface is fully YARD-documented inline; once the gem is published to RubyGems, `gem rdoc blazen` (or [rubydoc.info](https://rubydoc.info/)) will host the API reference.

## License

[MPL-2.0](LICENSE) — same terms as the rest of the Blazen workspace.
