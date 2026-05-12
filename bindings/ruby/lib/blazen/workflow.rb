# frozen_string_literal: true

require "json"

module Blazen
  # Helpers for working with {Workflow} and {WorkflowBuilder}.
  #
  # The UniFFI-generated {Workflow} class exposes +#run+ / +#run_blocking+
  # accepting a JSON-encoded input string. This module adds:
  #
  # * A {build} convenience that constructs a {WorkflowBuilder} and yields it
  #   for fluent configuration.
  # * A {run} helper that accepts either a JSON string or an arbitrary Ruby
  #   value (auto-encoded via +JSON.dump+).
  #
  # @note Step handlers in Ruby
  #   The +StepHandler+ callback interface in the underlying UniFFI binding is
  #   declared with +with_foreign+ so that Go/Swift/Kotlin can supply handlers
  #   implemented in their host language. The upstream +uniffi-bindgen+ Ruby
  #   template does **not** currently generate foreign-callback scaffolding,
  #   so {Blazen::StepHandler} can only consume handles produced on the Rust
  #   side. Workflows whose steps are entirely defined in Rust execute fine;
  #   workflows that require a Ruby-implemented step are not yet supported
  #   from this gem. Use the Rust crate (or Go/Swift/Kotlin bindings) for
  #   those cases until the upstream Ruby bindgen grows callback support.
  module WorkflowHelpers
    module_function

    # Creates a new {WorkflowBuilder} and yields it for configuration.
    #
    # @param name [String] workflow name
    # @yieldparam builder [Blazen::WorkflowBuilder] the builder
    # @return [Blazen::Workflow] the built workflow
    def build(name)
      builder = Blazen::WorkflowBuilder.new(name)
      yield builder if block_given?
      Blazen.translate_errors { builder.build }
    end

    # Runs a {Workflow}, accepting either a JSON string or any +JSON.dump+able
    # value as input.
    #
    # @param workflow [Blazen::Workflow]
    # @param input [String, Hash, Array, Object] workflow input
    # @return [Blazen::WorkflowResult]
    def run(workflow, input)
      json = input.is_a?(String) ? input : JSON.dump(input)
      Blazen.translate_errors { workflow.run_blocking(json) }
    end
  end

  # Convenience alias so callers can write +Blazen.workflow(name) { |b| ... }+.
  #
  # @param name [String] workflow name
  # @yieldparam builder [Blazen::WorkflowBuilder]
  # @return [Blazen::Workflow]
  def self.workflow(name, &)
    WorkflowHelpers.build(name, &)
  end

  # @!parse
  #   class WorkflowResult
  #     # @return [Hash, Array, Object, nil] +data_json+ from {#event} decoded
  #     def event_data; end
  #   end
  class WorkflowResult
    # Returns the workflow's final event payload, decoded from its JSON form.
    #
    # @return [Object, nil] decoded JSON value, or +nil+ if +event+ has no
    #   payload
    def event_data
      return nil unless event
      payload = event.data_json
      return nil if payload.nil? || payload.empty?

      JSON.parse(payload)
    end
  end
end
