# frozen_string_literal: true

require "spec_helper"

RSpec.describe Blazen do
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
  end

  describe "error hierarchy" do
    it "defines the documented exception classes" do
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
  end

  describe "Llm builders" do
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

    it "constructs a CompletionRequest with the message list" do
      req = Blazen::Llm.completion_request(
        messages: [Blazen::Llm.system("be brief"), Blazen::Llm.user("hi")],
        temperature: 0.1,
        max_tokens: 32,
      )
      expect(req).to be_a(Blazen::CompletionRequest)
      expect(req.messages.length).to eq(2)
      expect(req.temperature).to eq(0.1)
      expect(req.max_tokens).to eq(32)
    end

    it "serializes Hash tool parameters to JSON" do
      tool = Blazen::Llm.tool(
        name: "get_weather",
        description: "Look up the weather",
        parameters: { type: "object", properties: { city: { type: "string" } } },
      )
      expect(tool).to be_a(Blazen::Tool)
      expect(tool.name).to eq("get_weather")
      expect(JSON.parse(tool.parameters_json)).to include("type" => "object")
    end
  end

  describe "Streaming.complete" do
    it "raises Unsupported with an explanatory message" do
      expect { Blazen::Streaming.complete(nil, nil) }
        .to raise_error(Blazen::Streaming::Unsupported, /not yet supported from Ruby/)
    end
  end
end
