# frozen_string_literal: true

module Blazen
  # Convenience constructors for {Agent} (the tool-calling LLM-agent loop).
  #
  # @note Tool handlers in Ruby
  #   The {Blazen::ToolHandler} callback interface is declared with
  #   +with_foreign+ in the UniFFI surface so that Go/Swift/Kotlin can
  #   provide a host-language tool implementation. The upstream
  #   +uniffi-bindgen+ Ruby template does **not** generate foreign-callback
  #   scaffolding, so {Blazen::ToolHandler} can currently only consume
  #   handles produced on the Rust side. Agents whose tools are entirely
  #   Rust-implemented work as-is; agents that need a Ruby-defined tool are
  #   not yet supported from this gem.
  module Agents
    module_function

    # Constructs a new {Agent}.
    #
    # @param model [Blazen::CompletionModel] backing completion model
    # @param tool_handler [Blazen::ToolHandler] tool dispatcher (must be a
    #   {Blazen::ToolHandler} handle produced on the Rust side; see module docs)
    # @param tools [Array<Blazen::Tool>] declared tool list shown to the model
    # @param system_prompt [String, nil] optional system prompt
    # @param max_iterations [Integer] max tool-calling iterations before the
    #   agent gives up (default: 16)
    # @return [Blazen::Agent]
    def new(model:, tool_handler:, tools: [], system_prompt: nil, max_iterations: 16)
      Blazen.translate_errors do
        Blazen::Agent.new(model, system_prompt, tools, tool_handler, max_iterations)
      end
    end
  end
end
