# frozen_string_literal: true

require "spec_helper"
require "json"
require "tempfile"
require "securerandom"

# Optional: the +async+ gem is a dev dependency used only by the
# Fiber.scheduler interop spec. If it isn't installed (e.g. on a slim CI
# image), the corresponding example is marked +pending+ instead of erroring
# the whole suite. Install with: `gem install async` (or `bundle install`).
begin
  require "async"
  ASYNC_AVAILABLE = true
rescue LoadError
  ASYNC_AVAILABLE = false
end

# --- Test helpers --------------------------------------------------------

# Skip the current example unless +name+ is set in the environment. Used to
# gate live-network specs (Providers / Agents / Streaming) so the suite
# stays green on machines without API credentials.
def skip_unless_env(name)
  skip "set #{name} to run this spec" if ENV[name].nil? || ENV[name].empty?
end

# Convenience: build an echo workflow whose single step wraps the
# +StartEvent+ payload into a +StopEvent+. Used by several specs.
def build_echo_workflow(name = "echo")
  Blazen.workflow(name) do |b|
    b.step("echo",
           accepts: ["blazen::StartEvent"],
           emits:   ["blazen::StopEvent"]) do |evt|
      payload = (evt.data || {})["data"] || {}
      Blazen::Workflow::StepOutput.single(
        Blazen::Workflow::Event.create(
          event_type: "blazen::StopEvent",
          data: { result: payload },
        ),
      )
    end
  end
end

RSpec.describe Blazen do
  before(:all) { Blazen.init }

  # ---------------------------------------------------------------------
  # Module-level surface
  # ---------------------------------------------------------------------

  describe ".version" do
    it "returns a non-empty version string from the native library" do
      expect(Blazen.version).to be_a(String)
      expect(Blazen.version).not_to be_empty
    end

    it "matches a semver-ish pattern" do
      expect(Blazen.version).to match(/\A\d+\.\d+\.\d+/)
    end
  end

  describe "::VERSION" do
    it "exposes the gem version constant" do
      expect(Blazen::VERSION).to be_a(String)
      expect(Blazen::VERSION).not_to be_empty
    end
  end

  describe ".init / .shutdown" do
    it "completes without raising on the happy path" do
      expect { Blazen.init }.not_to raise_error
      expect { Blazen.shutdown }.not_to raise_error
    end

    it "is idempotent on .init" do
      expect { 3.times { Blazen.init } }.not_to raise_error
    end
  end

  # ---------------------------------------------------------------------
  # Error hierarchy
  # ---------------------------------------------------------------------

  describe "error hierarchy" do
    it "roots every documented exception at Blazen::Error" do
      expect(Blazen::Error).to be < StandardError
      %i[
        AuthError RateLimitError TimeoutError ValidationError
        ContentPolicyError UnsupportedError ComputeError MediaError
        ProviderError WorkflowError ToolError PeerError PersistError
        PromptError MemoryError CacheError CancelledError InternalError
      ].each do |klass|
        expect(Blazen.const_get(klass)).to be < Blazen::Error
      end
    end

    it "constructs ValidationError / AuthError without extra kwargs" do
      expect(Blazen::ValidationError.new("bad input").message).to eq("bad input")
      expect(Blazen::AuthError.new("no key").message).to eq("no key")
    end

    it "exposes retry_after_ms on RateLimitError" do
      err = Blazen::RateLimitError.new("slow down", retry_after_ms: 1_500)
      expect(err.retry_after_ms).to eq(1_500)
      expect(err.message).to eq("slow down")
    end

    it "exposes elapsed_ms on TimeoutError" do
      err = Blazen::TimeoutError.new("too long", elapsed_ms: 30_000)
      expect(err.elapsed_ms).to eq(30_000)
    end

    it "exposes structured fields on ProviderError" do
      err = Blazen::ProviderError.new(
        "openai blew up",
        kind: "OpenAIHttp",
        provider: "openai",
        status: 503,
        endpoint: "https://api.openai.com/v1/chat/completions",
        request_id: "req_abc123",
        detail: "upstream timeout",
        retry_after_ms: 5_000,
      )
      expect(err.kind).to eq("OpenAIHttp")
      expect(err.provider).to eq("openai")
      expect(err.status).to eq(503)
      expect(err.endpoint).to end_with("/chat/completions")
      expect(err.request_id).to eq("req_abc123")
      expect(err.detail).to eq("upstream timeout")
      expect(err.retry_after_ms).to eq(5_000)
    end
  end

  # ---------------------------------------------------------------------
  # Llm builders (no network)
  # ---------------------------------------------------------------------

  describe "Blazen::Llm builders" do
    it "constructs a user ChatMessage with defaults" do
      msg = Blazen::Llm.user("Hello")
      expect(msg).to be_a(Blazen::ChatMessage)
      expect(msg.role).to eq("user")
      expect(msg.content).to eq("Hello")
      expect(msg.media_parts).to eq([])
      expect(msg.tool_calls).to eq([])
      expect(msg.tool_call_id).to be_nil
      expect(msg.name).to be_nil
    end

    it "constructs a system / assistant / tool_result ChatMessage" do
      sys = Blazen::Llm.system("be brief")
      asst = Blazen::Llm.assistant("done")
      tool = Blazen::Llm.tool_result(content: "42", tool_call_id: "call_1")
      expect(sys.role).to eq("system")
      expect(asst.role).to eq("assistant")
      expect(tool.role).to eq("tool")
      expect(tool.tool_call_id).to eq("call_1")
    end

    it "builds a CompletionRequest from ChatMessages" do
      req = Blazen::Llm.completion_request(
        messages: [Blazen::Llm.system("be brief"), Blazen::Llm.user("hi")],
        temperature: 0.1,
        max_tokens: 32,
      )
      expect(req).to be_a(Blazen::CompletionRequest)
    end

    it "serializes Hash tool parameters to JSON" do
      tool = Blazen::Llm.tool(
        name: "get_weather",
        description: "Look up the weather",
        parameters: { type: "object", properties: { city: { type: "string" } } },
      )
      expect(tool).to be_a(Blazen::Tool)
      expect(tool.name).to eq("get_weather")
      expect(tool.description).to eq("Look up the weather")
      expect(JSON.parse(tool.parameters_json)).to include("type" => "object")
    end

    it "builds a TokenUsage record" do
      usage = Blazen::Llm.token_usage(prompt_tokens: 10, completion_tokens: 7, total_tokens: 17)
      expect(usage.prompt_tokens).to eq(10)
      expect(usage.completion_tokens).to eq(7)
      expect(usage.total_tokens).to eq(17)
    end
  end

  # ---------------------------------------------------------------------
  # Workflows (real, no network)
  # ---------------------------------------------------------------------

  describe "Blazen::Workflow" do
    it "round-trips a single-step echo workflow via run_blocking" do
      wf = build_echo_workflow("echo_blocking")
      result = wf.run_blocking({ hello: "world" })

      expect(result).to be_a(Blazen::Workflow::WorkflowResult)
      expect(result.event_data).to eq("result" => { "hello" => "world" })

      # Zero-token, zero-cost local workflow — assert it's a real number
      # rather than nil, but don't pin an exact value.
      expect(result.total_input_tokens).to be_a(Integer).and be >= 0
      expect(result.total_output_tokens).to be_a(Integer).and be >= 0
      expect(result.total_cost_usd).to be_a(Float).and be >= 0.0
    end

    it "round-trips via the awaitable .run path (no scheduler active)" do
      wf = build_echo_workflow("echo_run")
      result = wf.run({ hello: "ruby" })
      expect(result.event_data).to eq("result" => { "hello" => "ruby" })
    end

    it "fans out via StepOutput.multiple and reaches downstream steps" do
      collected = []
      wf = Blazen.workflow("fanout") do |b|
        b.step("expand",
               accepts: ["blazen::StartEvent"],
               emits:   ["Mid"]) do |_evt|
          events = [1, 2].map do |i|
            Blazen::Workflow::Event.create(event_type: "Mid", data: { i: i })
          end
          Blazen::Workflow::StepOutput.multiple(events)
        end
        b.step("collect",
               accepts: ["Mid"],
               emits:   ["blazen::StopEvent"]) do |evt|
          collected << evt.data["i"]
          Blazen::Workflow::StepOutput.single(
            Blazen::Workflow::Event.create(
              event_type: "blazen::StopEvent",
              data: { last: evt.data["i"] },
            ),
          )
        end
      end

      result = wf.run_blocking({})
      expect(collected.sort).to eq([1, 2])
      # Final emitted StopEvent comes from whichever Mid the engine processed
      # last, so just assert the wire shape — not the specific value. The
      # StopEvent payload is wrapped under +"result"+ by the engine.
      expect(result.event_data).to be_a(Hash)
      expect(result.event_data["result"]).to have_key("last")
    end

    it "surfaces step-handler exceptions as Blazen::WorkflowError" do
      wf = Blazen.workflow("boom") do |b|
        b.step("kaboom",
               accepts: ["blazen::StartEvent"],
               emits:   ["blazen::StopEvent"]) do |_evt|
          raise "kaboom!"
        end
      end

      # The Ruby trampoline turns any handler StandardError into the
      # documented +returned -1 without setting out_err+ path, which the
      # engine wraps as a WorkflowError.
      expect { wf.run_blocking({}) }.to raise_error(Blazen::WorkflowError, /step .* failed/)
    end

    it "rejects a workflow with no steps at build time" do
      expect {
        Blazen.workflow("empty") { |_b| }
      }.to raise_error(Blazen::WorkflowError, /at least one step/)
    end

    it "exposes step_names for each registered step (order is engine-defined)" do
      wf = Blazen.workflow("multi") do |b|
        b.step("a",
               accepts: ["blazen::StartEvent"],
               emits:   ["B"]) do |_evt|
          Blazen::Workflow::StepOutput.single(
            Blazen::Workflow::Event.create(event_type: "B", data: {}),
          )
        end
        b.step("b",
               accepts: ["B"],
               emits:   ["blazen::StopEvent"]) do |_evt|
          Blazen::Workflow::StepOutput.single(
            Blazen::Workflow::Event.create(event_type: "blazen::StopEvent", data: {}),
          )
        end
      end
      expect(wf.step_names).to match_array(%w[a b])
    end
  end

  # ---------------------------------------------------------------------
  # Async (Fiber.scheduler) interop
  # ---------------------------------------------------------------------

  describe "async / Fiber.scheduler interop" do
    it "overlaps concurrent workflow runs under Async { }" do
      unless ASYNC_AVAILABLE
        skip "install the +async+ gem to exercise Fiber.scheduler interop (gem install async)"
      end

      # Build a workflow whose step sleeps 100 ms before emitting. Three
      # concurrent runs should complete in < 300 ms wall-clock (each sleeps
      # 100 ms, and Fiber.scheduler yields during the sleep so the runs
      # genuinely overlap rather than serialising on a single thread).
      wf = Blazen.workflow("sleepy") do |b|
        b.step("sleep",
               accepts: ["blazen::StartEvent"],
               emits:   ["blazen::StopEvent"]) do |evt|
          sleep 0.1
          Blazen::Workflow::StepOutput.single(
            Blazen::Workflow::Event.create(
              event_type: "blazen::StopEvent",
              data: { from: evt.data["data"]["task"] },
            ),
          )
        end
      end

      started = Time.now
      results = Async do
        tasks = (1..3).map { |i| Async { wf.run({ task: i }) } }
        tasks.map(&:wait)
      end.wait
      elapsed = Time.now - started

      expect(results.length).to eq(3)
      results.each { |r| expect(r.event_data).to have_key("from") }
      expect(elapsed).to be < 0.3
    end
  end

  # ---------------------------------------------------------------------
  # Pipelines
  # ---------------------------------------------------------------------

  describe "Blazen::Pipeline" do
    it "chains two workflow stages sequentially" do
      wf_a = build_echo_workflow("stage_a")
      wf_b = build_echo_workflow("stage_b")
      pipe = Blazen.pipeline("pipe") do |b|
        b.stage("a", wf_a)
        b.stage("b", wf_b)
      end

      expect(pipe.stage_names).to eq(%w[a b])
      result = pipe.run_blocking({ ping: 1 })
      expect(result).to be_a(Blazen::Workflow::WorkflowResult)
      expect(result.event_data).to be_a(Hash)
    end
  end

  # ---------------------------------------------------------------------
  # Persistence (redb roundtrip — no external services)
  # ---------------------------------------------------------------------

  describe "Blazen::Persist (redb)" do
    let(:redb_path) do
      tf = Tempfile.new(["blazen-redb", ".db"])
      tf.close
      File.delete(tf.path) if File.exist?(tf.path)  # let redb create it fresh
      tf.path
    end

    after { File.delete(redb_path) if File.exist?(redb_path) }

    it "round-trips save / load / list / delete via redb" do
      store = Blazen::Persist.redb(redb_path)
      run_id = SecureRandom.uuid
      now_ms = (Time.now.to_f * 1_000).to_i

      ckpt = Blazen::Persist::WorkflowCheckpoint.new(
        workflow_name: "sample",
        run_id: run_id,
        state_json: JSON.dump(step: "first"),
        metadata_json: JSON.dump(owner: "rspec"),
        timestamp_ms: now_ms,
      )

      expect { store.save_blocking(ckpt) }.not_to raise_error

      loaded = store.load_blocking(run_id)
      expect(loaded).to be_a(Blazen::Persist::WorkflowCheckpoint)
      expect(loaded.workflow_name).to eq("sample")
      expect(loaded.run_id).to eq(run_id)
      expect(JSON.parse(loaded.state_json)).to eq("step" => "first")
      expect(loaded.timestamp_ms).to eq(now_ms)

      listing = store.list_blocking
      expect(listing.length).to be >= 1
      expect(listing.map(&:run_id)).to include(run_id)

      run_ids = store.list_run_ids_blocking
      expect(run_ids).to include(run_id)

      expect { store.delete_blocking(run_id) }.not_to raise_error
      expect(store.load_blocking(run_id)).to be_nil
      expect(store.list_run_ids_blocking).not_to include(run_id)
    end
  end

  # ---------------------------------------------------------------------
  # Streaming (no network — null-arg guard only)
  # ---------------------------------------------------------------------

  describe "Blazen::Streaming.complete" do
    it "rejects a nil request with an ArgumentError-derived guard" do
      # Without a real model + request the call short-circuits in Ruby
      # before crossing the FFI. We're not asserting which guard fires —
      # only that the now-removed +Blazen::Streaming::Unsupported+ stub no
      # longer exists.
      expect(defined?(Blazen::Streaming::Unsupported)).to be_nil
      expect { Blazen::Streaming.complete(nil, nil) }.to raise_error(StandardError)
    end
  end

  # ---------------------------------------------------------------------
  # Live LLM specs (skipped without OPENAI_API_KEY)
  # ---------------------------------------------------------------------

  describe "Blazen::Providers.openai (live)" do
    before { skip_unless_env("OPENAI_API_KEY") }

    let(:model) { Blazen::Providers.openai(api_key: ENV.fetch("OPENAI_API_KEY")) }

    it "constructs a CompletionModel" do
      expect(model).to be_a(Blazen::Llm::CompletionModel)
    end

    it "returns a non-empty content via complete_blocking" do
      req = Blazen::Llm.completion_request(
        messages: [Blazen::Llm.user("Reply with exactly the word: pong")],
        max_tokens: 8,
        temperature: 0.0,
      )
      resp = model.complete_blocking(req)
      expect(resp).to be_a(Blazen::Llm::CompletionResponse)
      expect(resp.content).to be_a(String)
      expect(resp.content).not_to be_empty
    end

    it "returns a non-empty content via the awaitable complete path" do
      req = Blazen::Llm.completion_request(
        messages: [Blazen::Llm.user("Reply with exactly the word: pong")],
        max_tokens: 8,
        temperature: 0.0,
      )
      resp = model.complete(req)
      expect(resp.content).not_to be_empty
    end
  end

  describe "Blazen::Streaming.complete (live)" do
    before { skip_unless_env("OPENAI_API_KEY") }

    it "invokes on_chunk at least once and on_done exactly once" do
      model = Blazen::Providers.openai(api_key: ENV.fetch("OPENAI_API_KEY"))
      req = Blazen::Llm.completion_request(
        messages: [Blazen::Llm.user("Count from 1 to 5, space-separated.")],
        max_tokens: 32,
        temperature: 0.0,
      )

      chunks = 0
      dones = 0
      errors = []

      Blazen::Streaming.complete(model, req) do |kind, *args|
        case kind
        when :chunk then chunks += 1
        when :done  then dones += 1
        when :error then errors << args.first
        end
      end

      expect(errors).to be_empty
      expect(chunks).to be >= 1
      expect(dones).to eq(1)
    end
  end

  describe "Blazen::Agents.new (live)" do
    before { skip_unless_env("OPENAI_API_KEY") }

    it "drives a tool-calling loop with a Ruby tool_handler" do
      model = Blazen::Providers.openai(api_key: ENV.fetch("OPENAI_API_KEY"))
      tool = Blazen::Llm.tool(
        name: "echo",
        description: "Echoes its argument back to the model verbatim.",
        parameters: {
          type: "object",
          properties: { phrase: { type: "string" } },
          required: ["phrase"],
        },
      )

      calls = []
      handler = lambda do |name, args|
        calls << [name, args]
        { echoed: args["phrase"] }
      end

      agent = Blazen::Agents.new(
        model: model,
        tools: [tool],
        tool_handler: handler,
        system_prompt: "Always call the echo tool exactly once before replying.",
        max_iterations: 4,
      )

      result = agent.run_blocking("Please echo the phrase: ruby-is-rad")
      expect(result).to be_a(Blazen::Agents::AgentResult)
      expect(result.iterations).to be >= 1
      expect(result.tool_call_count).to be >= 1
      expect(result.final_message).to be_a(String)
      expect(result.final_message).not_to be_empty
    end
  end
end
