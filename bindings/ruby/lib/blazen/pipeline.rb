# frozen_string_literal: true

require "json"

module Blazen
  # Pipeline surface for the Ruby binding.
  #
  # A {Pipeline} is a sequence of {Blazen::Workflow::Instance}s (and/or
  # parallel branches of workflows) executed end-to-end with shared timeouts.
  # Wraps the cabi opaque-pointer types (+BlazenPipelineBuilder+,
  # +BlazenPipeline+) and reuses the {Blazen::Workflow::WorkflowResult} +
  # {Blazen::Workflow::Event} value types from +lib/blazen/workflow.rb+ —
  # this file MUST be required after +lib/blazen/workflow.rb+.
  #
  # @example
  #   wf_a = Blazen.workflow("a") { |b| ... }
  #   wf_b = Blazen.workflow("b") { |b| ... }
  #   pipe = Blazen.pipeline("my-pipe") do |b|
  #     b.stage("stage-a", wf_a)
  #     b.stage("stage-b", wf_b)
  #     b.total_timeout_ms(60_000)
  #   end
  #   result = pipe.run({ initial: "input" })
  module Pipeline
    # ---------------------------------------------------------------------
    # Builder — caller-owned +BlazenPipelineBuilder+ wrapper
    # ---------------------------------------------------------------------

    # Fluent builder for a {Instance}.
    class Builder
      # @param name [String] pipeline name
      def initialize(name)
        ptr = Blazen::FFI.with_cstring(name) { |n| Blazen::FFI.blazen_pipeline_builder_new(n) }
        raise Blazen::InternalError, "blazen_pipeline_builder_new returned null" if ptr.null?

        @ptr = ::FFI::AutoPointer.new(ptr, Blazen::FFI.method(:blazen_pipeline_builder_free))

        # Strong-ref jail for borrowed parallel-stage string-array buffers.
        @keepalive = []
      end

      # Appends a sequential workflow stage with an auto-generated stage
      # name. Consumes ownership of +workflow+ — after this call the
      # workflow object can no longer be used.
      #
      # @param workflow [Blazen::Workflow::Instance]
      # @return [self]
      def add_workflow(workflow)
        raise Blazen::InternalError, "builder has been consumed" if @ptr.nil?
        unless workflow.is_a?(Blazen::Workflow::Instance)
          raise ArgumentError, "workflow must be a Blazen::Workflow::Instance"
        end

        wf_ptr = workflow.consume!
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_pipeline_builder_add_workflow(@ptr, wf_ptr, out_err)
        Blazen::FFI.check_error!(out_err)
        self
      end

      # Appends a named sequential workflow stage. Consumes ownership of
      # +workflow+.
      #
      # @param name [String]
      # @param workflow [Blazen::Workflow::Instance]
      # @return [self]
      def stage(name, workflow)
        raise Blazen::InternalError, "builder has been consumed" if @ptr.nil?
        unless workflow.is_a?(Blazen::Workflow::Instance)
          raise ArgumentError, "workflow must be a Blazen::Workflow::Instance"
        end

        wf_ptr = workflow.consume!
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(name) do |n|
          Blazen::FFI.blazen_pipeline_builder_stage(@ptr, n, wf_ptr, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        self
      end

      # Appends a parallel stage running multiple workflows concurrently.
      #
      # @param name [String] stage name (the parallel stage as a whole)
      # @param branches [Hash{String=>Blazen::Workflow::Instance}] branch-name
      #   to workflow mapping; every workflow is CONSUMED by this call.
      # @param wait_all [Boolean] if true (default), every branch must
      #   complete; if false, the stage finishes as soon as the first
      #   branch produces a result.
      # @return [self]
      def parallel(name, branches, wait_all: true)
        raise Blazen::InternalError, "builder has been consumed" if @ptr.nil?
        unless branches.is_a?(Hash)
          raise ArgumentError, "branches must be a Hash of name => Workflow::Instance"
        end
        branches.each_value do |wf|
          unless wf.is_a?(Blazen::Workflow::Instance)
            raise ArgumentError, "every branch value must be a Blazen::Workflow::Instance"
          end
        end

        count = branches.length

        # Build the branch-name string array.
        branch_name_ptrs = branches.keys.map { |k| ::FFI::MemoryPointer.from_string(k.to_s) }
        @keepalive.concat(branch_name_ptrs)
        branch_names_arr = ::FFI::MemoryPointer.new(:pointer, count)
        @keepalive << branch_names_arr
        branch_name_ptrs.each_with_index { |p, i| branch_names_arr[i].write_pointer(p) }

        # Build the workflows pointer array. Every workflow is consumed by
        # the cabi (regardless of return value), so we MUST call
        # +#consume!+ on each before handing them off.
        wf_addrs = branches.values.map(&:consume!)
        workflows_arr = ::FFI::MemoryPointer.new(:pointer, count)
        @keepalive << workflows_arr
        wf_addrs.each_with_index { |p, i| workflows_arr[i].write_pointer(p) }

        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(name) do |n|
          Blazen::FFI.blazen_pipeline_builder_parallel(
            @ptr, n,
            branch_names_arr, count,
            workflows_arr,    count,
            wait_all,
            out_err,
          )
        end
        Blazen::FFI.check_error!(out_err)
        self
      end

      # Sets the per-stage timeout in milliseconds.
      # @param millis [Integer]
      # @return [self]
      def timeout_per_stage_ms(millis)
        raise Blazen::InternalError, "builder has been consumed" if @ptr.nil?

        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_pipeline_builder_timeout_per_stage_ms(@ptr, millis, out_err)
        Blazen::FFI.check_error!(out_err)
        self
      end

      # Sets the total pipeline wall-clock timeout in milliseconds.
      # @param millis [Integer]
      # @return [self]
      def total_timeout_ms(millis)
        raise Blazen::InternalError, "builder has been consumed" if @ptr.nil?

        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_pipeline_builder_total_timeout_ms(@ptr, millis, out_err)
        Blazen::FFI.check_error!(out_err)
        self
      end

      # Validates the pipeline definition and produces a runnable
      # {Instance}. The builder handle remains live but its internal state
      # is consumed; subsequent calls will fail with a validation error.
      #
      # @return [Instance]
      def build
        raise Blazen::InternalError, "builder has been consumed" if @ptr.nil?

        out_pipeline = ::FFI::MemoryPointer.new(:pointer)
        out_err      = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_pipeline_builder_build(@ptr, out_pipeline, out_err)
        Blazen::FFI.check_error!(out_err)

        pipe_ptr = out_pipeline.read_pointer
        raise Blazen::InternalError, "blazen_pipeline_builder_build returned null pipeline" if pipe_ptr.null?

        Instance.new(pipe_ptr)
      end
    end

    # ---------------------------------------------------------------------
    # Instance — caller-owned +BlazenPipeline+ wrapper
    # ---------------------------------------------------------------------

    # A runnable pipeline.
    class Instance
      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "raw_ptr is null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_pipeline_free))
      end

      # Runs the pipeline asynchronously via the cabi future surface.
      # Composes with +Fiber.scheduler+ when one is active; otherwise blocks
      # the calling thread on the cabi runtime.
      #
      # @param input [String, Object] JSON string OR JSON-dumpable input
      # @return [Blazen::Workflow::WorkflowResult]
      def run(input)
        raise Blazen::InternalError, "pipeline has been consumed" if @ptr.nil?

        input_json = input.is_a?(String) ? input : JSON.dump(input)
        fut = Blazen::FFI.with_cstring(input_json) do |i|
          Blazen::FFI.blazen_pipeline_run(@ptr, i)
        end
        raise Blazen::InternalError, "blazen_pipeline_run returned null future" if fut.nil? || fut.null?

        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_workflow_result(f, out_result, out_err)
        end
        Blazen::FFI.check_error!(out_err)

        result_ptr = out_result.read_pointer
        raise Blazen::InternalError, "blazen_future_take_workflow_result returned null" if result_ptr.null?

        Blazen::Workflow::WorkflowResult.new(result_ptr)
      end

      # Runs the pipeline synchronously on the cabi runtime, blocking the
      # calling thread until completion.
      #
      # @param input [String, Object]
      # @return [Blazen::Workflow::WorkflowResult]
      def run_blocking(input)
        raise Blazen::InternalError, "pipeline has been consumed" if @ptr.nil?

        input_json = input.is_a?(String) ? input : JSON.dump(input)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(input_json) do |i|
          Blazen::FFI.blazen_pipeline_run_blocking(@ptr, i, out_result, out_err)
        end
        Blazen::FFI.check_error!(out_err)

        result_ptr = out_result.read_pointer
        raise Blazen::InternalError, "blazen_pipeline_run_blocking returned null result" if result_ptr.null?

        Blazen::Workflow::WorkflowResult.new(result_ptr)
      end

      # @return [Array<String>] stage names, in declaration order
      def stage_names
        raise Blazen::InternalError, "pipeline has been consumed" if @ptr.nil?

        count = Blazen::FFI.blazen_pipeline_stage_names_count(@ptr)
        (0...count).map do |i|
          Blazen::FFI.consume_cstring(Blazen::FFI.blazen_pipeline_stage_names_get(@ptr, i))
        end
      end
    end

    # ---------------------------------------------------------------------
    # Module-level convenience
    # ---------------------------------------------------------------------

    module_function

    # Convenience builder: yields a fresh {Builder}, builds it, and returns
    # the resulting {Instance}.
    #
    # @param name [String]
    # @yieldparam builder [Builder]
    # @return [Instance]
    def build(name)
      builder = Builder.new(name)
      yield builder if block_given?
      builder.build
    end

    # Runs +pipeline+ with +input+, accepting either a JSON string or any
    # +JSON.dump+able value.
    #
    # @param pipeline [Instance]
    # @param input [String, Object]
    # @return [Blazen::Workflow::WorkflowResult]
    def run(pipeline, input)
      pipeline.run(input)
    end
  end

  # Top-level convenience so callers can write +Blazen.pipeline(name) { |b| ... }+.
  #
  # @param name [String] pipeline name
  # @yieldparam builder [Blazen::Pipeline::Builder]
  # @return [Blazen::Pipeline::Instance]
  def self.pipeline(name, &block)
    Pipeline.build(name, &block)
  end
end
