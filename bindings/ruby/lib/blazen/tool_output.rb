# frozen_string_literal: true

require "json"

module Blazen
  # Structured tool-handler output carrying both the user-visible +data+
  # and an optional +llm_override+ for what the LLM sees on the next turn.
  #
  # Tool handlers can return either:
  #   - a plain value (auto-wrapped as +data+ with no override), or
  #   - this +ToolOutput+ helper which produces the structured shape.
  #
  # Mirrors JsToolOutput (Node) and PyToolOutput (Python).
  #
  # @example Returning a structured ToolOutput with a text override
  #   tool_handler = lambda do |name, args|
  #     items = fetch_items(args)
  #     Blazen::ToolOutput.new(
  #       data: { items: items },
  #       llm_override: Blazen::ToolOutput.text("Found #{items.size} items.")
  #     ).to_json
  #   end
  #
  # @example Returning a structured ToolOutput with multimodal parts
  #   Blazen::ToolOutput.new(
  #     data: { ok: true },
  #     llm_override: Blazen::ToolOutput.parts([
  #       { type: "text", text: "summary line" },
  #       { type: "image", image: { source: { source_type: "url", url: image_url }, media_type: "image/png" } }
  #     ])
  #   ).to_json
  class ToolOutput
    # @param data [Object] the user-visible payload (any JSON-serialisable value)
    # @param llm_override [Hash, nil] optional override hash produced by one of
    #   the +Blazen::ToolOutput.text/json/parts/provider_raw+ class methods
    def initialize(data:, llm_override: nil)
      @data = data
      @llm_override = llm_override
    end

    attr_reader :data, :llm_override

    def to_h
      h = { "data" => @data }
      h["llm_override"] = @llm_override if @llm_override
      h
    end

    def to_json(*args)
      to_h.to_json(*args)
    end

    # Build a plain-text +llm_override+ payload.
    def self.text(text)
      { "kind" => "text", "text" => text }
    end

    # Build a structured-JSON +llm_override+ payload.
    def self.json(value)
      { "kind" => "json", "value" => value }
    end

    # Build a multi-part +llm_override+ payload. Each part is a Hash with a
    # +"type"+ discriminator (matching the core +ContentPart+ serde tag —
    # NOT +kind+!). Variants: +text+, +image+, +file+, +audio+, +video+.
    #
    # Example part hashes:
    #   { "type" => "text", "text" => "..." }
    #   { "type" => "image", "image" => { "source" => { "source_type" => "url", "url" => "..." }, "media_type" => "image/png" } }
    def self.parts(parts)
      { "kind" => "parts", "parts" => parts }
    end

    # Build a provider-specific +llm_override+ payload. +provider+ must be
    # one of +"openai"+, +"openai_compat"+, +"azure"+, +"anthropic"+,
    # +"gemini"+, +"responses"+, +"fal"+.
    def self.provider_raw(provider:, value:)
      { "kind" => "provider_raw", "provider" => provider, "value" => value }
    end
  end
end
