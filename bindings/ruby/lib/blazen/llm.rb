# frozen_string_literal: true

require "json"

module Blazen
  # Convenience builders for the LLM record types ({ChatMessage},
  # {CompletionRequest}, {Tool}, {Media}).
  #
  # The UniFFI-generated record classes require every field as a keyword
  # argument, including the fields that are usually empty (+media_parts+,
  # +tool_calls+, +tool_call_id+, +name+...). These helpers fill in sensible
  # defaults so callers only have to specify what they actually care about.
  #
  # @example Build a request from plain hashes
  #   req = Blazen::Llm.completion_request(
  #     messages: [Blazen::Llm.user("Hello")],
  #     temperature: 0.2,
  #   )
  module Llm
    module_function

    # Builds a {ChatMessage}.
    #
    # @param role [String] +"system"+, +"user"+, +"assistant"+, or +"tool"+
    # @param content [String] message text
    # @param media_parts [Array<Blazen::Media>] inline media parts
    # @param tool_calls [Array<Blazen::ToolCall>] tool-call invocations
    # @param tool_call_id [String, nil] tool-call id this message replies to
    # @param name [String, nil] author name (when distinct from +role+)
    # @return [Blazen::ChatMessage]
    def message(role:, content:, media_parts: [], tool_calls: [],
                tool_call_id: nil, name: nil)
      Blazen::ChatMessage.new(
        role: role,
        content: content,
        media_parts: media_parts,
        tool_calls: tool_calls,
        tool_call_id: tool_call_id,
        name: name,
      )
    end

    # Builds a +"system"+-role {ChatMessage}.
    #
    # @param content [String] system prompt text
    # @return [Blazen::ChatMessage]
    def system(content)
      message(role: "system", content: content)
    end

    # Builds a +"user"+-role {ChatMessage}.
    #
    # @param content [String] user message text
    # @param media_parts [Array<Blazen::Media>] inline media parts
    # @return [Blazen::ChatMessage]
    def user(content, media_parts: [])
      message(role: "user", content: content, media_parts: media_parts)
    end

    # Builds an +"assistant"+-role {ChatMessage}.
    #
    # @param content [String] assistant message text
    # @param tool_calls [Array<Blazen::ToolCall>] tool-call invocations made
    # @return [Blazen::ChatMessage]
    def assistant(content, tool_calls: [])
      message(role: "assistant", content: content, tool_calls: tool_calls)
    end

    # Builds a +"tool"+-role {ChatMessage} (a tool-execution result).
    #
    # @param content [String] tool result, serialized as a string
    # @param tool_call_id [String] id of the tool call this responds to
    # @param name [String, nil] tool name
    # @return [Blazen::ChatMessage]
    def tool_result(content:, tool_call_id:, name: nil)
      message(role: "tool", content: content, tool_call_id: tool_call_id, name: name)
    end

    # Builds a {CompletionRequest}.
    #
    # @param messages [Array<Blazen::ChatMessage>] conversation history
    # @param tools [Array<Blazen::Tool>] available tools
    # @param temperature [Float, nil] sampling temperature
    # @param max_tokens [Integer, nil] response token cap
    # @param top_p [Float, nil] nucleus-sampling cutoff
    # @param model [String, nil] override the model for this request
    # @param response_format_json [String, nil] JSON-schema string for
    #   structured output (provider-specific)
    # @param system [String, nil] system prompt override
    # @return [Blazen::CompletionRequest]
    def completion_request(messages:, tools: [], temperature: nil, max_tokens: nil,
                           top_p: nil, model: nil, response_format_json: nil, system: nil)
      Blazen::CompletionRequest.new(
        messages: messages,
        tools: tools,
        temperature: temperature,
        max_tokens: max_tokens,
        top_p: top_p,
        model: model,
        response_format_json: response_format_json,
        system: system,
      )
    end

    # Builds a {Tool} declaration.
    #
    # @param name [String] tool name (must match what the model is told)
    # @param description [String] human-readable description for the model
    # @param parameters [Hash, String] JSON-Schema for the parameters; passed
    #   as a +Hash+ it is serialized via +JSON.dump+, otherwise used verbatim
    # @return [Blazen::Tool]
    def tool(name:, description:, parameters:)
      params_json = parameters.is_a?(String) ? parameters : JSON.dump(parameters)
      Blazen::Tool.new(name: name, description: description, parameters_json: params_json)
    end

    # Builds a {Media} part (image / audio / etc.).
    #
    # @param kind [String] media kind, e.g. +"image"+, +"audio"+
    # @param mime_type [String] MIME type, e.g. +"image/png"+
    # @param data_base64 [String] base64-encoded payload
    # @return [Blazen::Media]
    def media(kind:, mime_type:, data_base64:)
      Blazen::Media.new(kind: kind, mime_type: mime_type, data_base64: data_base64)
    end
  end
end
