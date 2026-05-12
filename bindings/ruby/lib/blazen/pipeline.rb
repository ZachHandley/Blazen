# frozen_string_literal: true

require "json"

module Blazen
  # Helpers for working with {Pipeline} and {PipelineBuilder}.
  #
  # See {WorkflowHelpers} for the analogous workflow helpers.
  module PipelineHelpers
    module_function

    # Creates a {PipelineBuilder} and yields it for configuration.
    #
    # @param name [String] pipeline name
    # @yieldparam builder [Blazen::PipelineBuilder] the builder
    # @return [Blazen::Pipeline] the built pipeline
    def build(name)
      builder = Blazen::PipelineBuilder.new(name)
      yield builder if block_given?
      Blazen.translate_errors { builder.build }
    end

    # Runs a {Pipeline}, accepting either a JSON string or any +JSON.dump+able
    # value as input.
    #
    # @param pipeline [Blazen::Pipeline]
    # @param input [String, Hash, Array, Object] pipeline input
    # @return [Blazen::WorkflowResult]
    def run(pipeline, input)
      json = input.is_a?(String) ? input : JSON.dump(input)
      Blazen.translate_errors { pipeline.run_blocking(json) }
    end
  end

  # Convenience alias so callers can write +Blazen.pipeline(name) { |b| ... }+.
  #
  # @param name [String] pipeline name
  # @yieldparam builder [Blazen::PipelineBuilder]
  # @return [Blazen::Pipeline]
  def self.pipeline(name, &)
    PipelineHelpers.build(name, &)
  end
end
