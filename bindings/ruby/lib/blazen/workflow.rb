# frozen_string_literal: true

require "json"

module Blazen
  # Workflow surface for the Ruby binding.
  #
  # Wraps the cabi opaque-pointer workflow types (+BlazenWorkflowBuilder+,
  # +BlazenWorkflow+, +BlazenWorkflowResult+, +BlazenEvent+, +BlazenStepOutput+)
  # with idiomatic Ruby classes, and bridges Ruby blocks across the FFI as
  # step handlers via the cabi {Blazen::FFI::BlazenStepHandlerVTable} callback
  # vtable.
  #
  # @example
  #   wf = Blazen.workflow("greet") do |b|
  #     b.step("greet",
  #            accepts: ["blazen::StartEvent"],
  #            emits:   ["blazen::StopEvent"]) do |event|
  #       name = event.data.fetch("data", {}).fetch("name", "world")
  #       Blazen::Workflow::StepOutput.single(
  #         Blazen::Workflow::Event.create(
  #           event_type: "blazen::StopEvent",
  #           data: { result: "Hello, #{name}!" },
  #         ),
  #       )
  #     end
  #   end
  #   result = wf.run({ name: "Zach" })
  #   result.event_data  # => {"result"=>"Hello, Zach!"}
  module Workflow
    # ---------------------------------------------------------------------
    # Ruby-side step-handler registry
    #
    # The C ABI hands us back an opaque +user_data+ pointer on every callback
    # invocation. We can't safely stuff a Ruby object reference into that
    # pointer (the GC moves objects and the +ffi+ gem won't trace through
    # arbitrary +c_void+s), so we use a strictly monotonic integer ID + a
    # module-level hash that the GC IS willing to trace. The ID is what we
    # cast to +void*+; the hash entry is what keeps the Ruby block alive.
    # ---------------------------------------------------------------------

    @step_registry = {}
    @step_registry_mutex = Mutex.new
    @next_step_id = 0

    # Registers a Ruby step-handler block and returns a stable integer ID to
    # use as the C-side +user_data+ pointer value.
    #
    # @param block [Proc] the step-handler block (called with an {Event})
    # @return [Integer] the registry ID
    def self.register_step_handler(block)
      @step_registry_mutex.synchronize do
        @next_step_id += 1
        id = @next_step_id
        @step_registry[id] = block
        id
      end
    end

    # Removes a step handler from the registry. Called by the cabi via the
    # +drop_user_data+ callback when the wrapping +CStepHandler+ drops.
    #
    # @param id [Integer] the registry ID
    # @return [Proc, nil] the removed block, or nil if never registered
    def self.unregister_step_handler(id)
      @step_registry_mutex.synchronize { @step_registry.delete(id) }
    end

    # Looks up a previously registered step-handler block by ID.
    #
    # @param id [Integer] the registry ID
    # @return [Proc, nil] the block, or nil if it has been dropped
    def self.lookup_step_handler(id)
      @step_registry_mutex.synchronize { @step_registry[id] }
    end

    # ---------------------------------------------------------------------
    # C-callable trampolines
    #
    # +INVOKE_FN+ is the +invoke+ slot of the step-handler vtable. The cabi
    # invokes it on a Tokio +spawn_blocking+ worker thread; the +ffi+ gem
    # reacquires the GVL before our proc runs (and +blocking: true+ tells
    # the gem to release the GVL while *we* are blocked waiting for the
    # native side, which doesn't apply for invocation but is still the
    # documented safe default for callbacks that may block the runtime).
    #
    # On success we return 0 and write a +*BlazenStepOutput+ to +out_output+.
    # On failure (Ruby exception in the block) we log to STDERR and return
    # -1 WITHOUT writing +out_err+. The Rust trampoline already handles this
    # case: it synthesises +BlazenError::Internal { "step handler returned
    # -1 without setting out_err" }+, which surfaces upstream as a
    # {Blazen::InternalError}. We can't build a +BlazenError+ from Ruby
    # (no public constructor in the cabi), so the user loses the original
    # exception's class + backtrace at the FFI boundary; STDERR captures it
    # for debugging.
    # ---------------------------------------------------------------------

    INVOKE_FN = ::FFI::Function.new(
      :int32,
      [:pointer, :pointer, :pointer, :pointer],
      proc do |user_data_ptr, event_ptr, out_output, _out_err|
        id = user_data_ptr.address
        block = lookup_step_handler(id)

        # Wrap the incoming event — the callback contractually owns it and
        # must free it (or consume it into a derivative structure) before
        # returning. Construction with owned: true wires the +AutoPointer+
        # to +blazen_event_free+; calls to +Event#consume!+ (e.g. when
        # building a StepOutput::Single) flip autorelease off and transfer
        # ownership back to the cabi.
        event = Event.new(event_ptr, owned: true)

        if block.nil?
          warn "[blazen] step handler invoked but no block registered for id=#{id}"
          # Free the event so we don't leak it.
          event.dispose!
          next(-1)
        end

        begin
          step_output = block.call(event)
          step_output = StepOutput.none if step_output.nil?

          unless step_output.is_a?(StepOutput)
            raise TypeError,
                  "step handler must return a Blazen::Workflow::StepOutput " \
                  "(or nil for None), got #{step_output.class}"
          end

          out_output.write_pointer(step_output.consume!)
          0
        rescue StandardError => e
          warn "[blazen] step handler raised: #{e.class}: #{e.message}"
          warn e.backtrace.join("\n") if e.backtrace
          -1
        end
      end,
      blocking: true,
    )

    DROP_USER_DATA_FN = ::FFI::Function.new(
      :void,
      [:pointer],
      proc do |user_data_ptr|
        id = user_data_ptr.address
        unregister_step_handler(id)
      end,
    )

    # ---------------------------------------------------------------------
    # Event — caller-owned +BlazenEvent+ wrapper
    # ---------------------------------------------------------------------

    # Wire-format workflow event ({event_type, data_json} pair).
    class Event
      # @param raw_ptr [::FFI::Pointer] underlying BlazenEvent pointer
      # @param owned [Boolean] when true, attaches an AutoPointer that calls
      #   +blazen_event_free+ on GC; when false, the pointer is borrowed and
      #   the caller manages its lifetime.
      def initialize(raw_ptr, owned: false)
        raise ArgumentError, "raw_ptr is null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = if owned
                 ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_event_free))
               else
                 raw_ptr
               end
      end

      # Constructs a new Event with the given type + payload.
      #
      # @param event_type [String]
      # @param data [String, Hash, Array, Object] JSON-encoded already, or
      #   anything +JSON.dump+able.
      # @return [Event]
      def self.create(event_type:, data:)
        data_json = data.is_a?(String) ? data : JSON.dump(data)
        ptr = Blazen::FFI.with_cstring(event_type) do |t|
          Blazen::FFI.with_cstring(data_json) { |d| Blazen::FFI.blazen_event_new(t, d) }
        end
        raise Blazen::InternalError, "blazen_event_new returned null" if ptr.null?

        new(ptr, owned: true)
      end

      # Returns the raw event-type string.
      #
      # @return [String, nil]
      def event_type
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_event_event_type(raw_ptr))
      end

      # Returns the raw JSON data payload string.
      #
      # @return [String, nil]
      def data_json
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_event_data_json(raw_ptr))
      end

      # Returns the JSON payload, decoded into a Ruby Hash/Array/scalar.
      #
      # @return [Object, nil]
      def data
        json = data_json
        return nil if json.nil? || json.empty?

        JSON.parse(json)
      end

      # Transfers ownership of the underlying pointer to the caller and
      # returns the raw +::FFI::Pointer+. After this call the Event is no
      # longer usable.
      #
      # Used internally when handing the event to {StepOutput.single} /
      # {StepOutput.multiple} or when returning it from a step callback
      # — the cabi consumes the event pointer in those paths.
      #
      # @return [::FFI::Pointer]
      def consume!
        raise Blazen::InternalError, "Event already consumed" if @ptr.nil?

        if @ptr.is_a?(::FFI::AutoPointer)
          @ptr.autorelease = false
          addr = @ptr.address
          @ptr = nil
          ::FFI::Pointer.new(addr)
        else
          p = @ptr
          @ptr = nil
          p
        end
      end

      # Explicitly frees the underlying event (idempotent). Used when the
      # callback returns an error and never builds a StepOutput from the
      # incoming event, so the +AutoPointer+ would eventually free it on GC
      # anyway — but eager release keeps the C-side memory pressure low.
      #
      # @return [void]
      def dispose!
        return if @ptr.nil?

        if @ptr.is_a?(::FFI::AutoPointer)
          @ptr.free
        else
          Blazen::FFI.blazen_event_free(@ptr)
        end
        @ptr = nil
      end

      # @api private
      # @return [::FFI::Pointer]
      def raw_ptr
        raise Blazen::InternalError, "Event has been consumed/disposed" if @ptr.nil?

        @ptr.is_a?(::FFI::AutoPointer) ? @ptr : @ptr
      end
    end

    # ---------------------------------------------------------------------
    # StepOutput — caller-owned +BlazenStepOutput+ wrapper
    # ---------------------------------------------------------------------

    # Result of a step invocation: zero, one, or many emitted events.
    class StepOutput
      # @return [StepOutput] a +None+ output (no event emitted)
      def self.none
        ptr = Blazen::FFI.blazen_step_output_new_none
        raise Blazen::InternalError, "blazen_step_output_new_none returned null" if ptr.null?

        new(ptr)
      end

      # @param event [Event]
      # @return [StepOutput] a +Single+ output wrapping +event+
      def self.single(event)
        raise ArgumentError, "event must be a Blazen::Workflow::Event" unless event.is_a?(Event)

        event_ptr = event.consume!
        ptr = Blazen::FFI.blazen_step_output_new_single(event_ptr)
        raise Blazen::InternalError, "blazen_step_output_new_single returned null" if ptr.null?

        new(ptr)
      end

      # @param events [Array<Event>]
      # @return [StepOutput] a +Multiple+ output containing every event
      def self.multiple(events)
        unless events.respond_to?(:each)
          raise ArgumentError, "events must be enumerable"
        end

        ptr = Blazen::FFI.blazen_step_output_new_multiple
        raise Blazen::InternalError, "blazen_step_output_new_multiple returned null" if ptr.null?

        events.each do |evt|
          unless evt.is_a?(Event)
            # Free the partially built output before raising.
            Blazen::FFI.blazen_step_output_free(ptr)
            raise ArgumentError, "events must all be Blazen::Workflow::Event instances"
          end
          Blazen::FFI.blazen_step_output_multiple_push(ptr, evt.consume!)
        end

        new(ptr)
      end

      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "raw_ptr is null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_step_output_free))
      end

      # Transfers ownership of the underlying pointer to the caller.
      # @return [::FFI::Pointer]
      def consume!
        raise Blazen::InternalError, "StepOutput already consumed" if @ptr.nil?

        @ptr.autorelease = false
        addr = @ptr.address
        @ptr = nil
        ::FFI::Pointer.new(addr)
      end
    end

    # ---------------------------------------------------------------------
    # WorkflowResult — caller-owned +BlazenWorkflowResult+ wrapper
    # ---------------------------------------------------------------------

    # Final result of a workflow (or pipeline) run.
    class WorkflowResult
      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "raw_ptr is null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_workflow_result_free))
      end

      # @return [Event] the terminal event (typically a +StopEvent+)
      def event
        ptr = Blazen::FFI.blazen_workflow_result_event(@ptr)
        raise Blazen::InternalError, "blazen_workflow_result_event returned null" if ptr.null?

        Event.new(ptr, owned: true)
      end

      # Decoded terminal-event payload.
      # @return [Object, nil]
      def event_data
        evt = event
        json = evt.data_json
        return nil if json.nil? || json.empty?

        JSON.parse(json)
      end

      # @return [Integer]
      def total_input_tokens
        Blazen::FFI.blazen_workflow_result_total_input_tokens(@ptr)
      end

      # @return [Integer]
      def total_output_tokens
        Blazen::FFI.blazen_workflow_result_total_output_tokens(@ptr)
      end

      # @return [Float]
      def total_cost_usd
        Blazen::FFI.blazen_workflow_result_total_cost_usd(@ptr)
      end
    end

    # ---------------------------------------------------------------------
    # Builder — caller-owned +BlazenWorkflowBuilder+ wrapper
    # ---------------------------------------------------------------------

    # Fluent builder for a {Workflow}.
    class Builder
      # @param name [String] workflow name
      def initialize(name)
        ptr = Blazen::FFI.with_cstring(name) { |n| Blazen::FFI.blazen_workflow_builder_new(n) }
        raise Blazen::InternalError, "blazen_workflow_builder_new returned null" if ptr.null?

        @ptr = ::FFI::AutoPointer.new(ptr, Blazen::FFI.method(:blazen_workflow_builder_free))

        # Strong-ref jail for the temporary +MemoryPointer+ string buffers we
        # build per +#step+ call. The cabi documents +accepts+/+emits+ array
        # strings as borrowed-for-duration-of-the-call only, but we keep the
        # backing pointers alive until +#build+ anyway so a hostile GC can't
        # munge them mid-call.
        @keepalive = []
      end

      # Adds a step to the workflow.
      #
      # @param name [String] step name (unique within the workflow)
      # @param accepts [Array<String>] event type strings this step consumes
      # @param emits   [Array<String>] event type strings this step produces
      # @yieldparam event [Event] the incoming event
      # @yieldreturn [StepOutput, nil] the step's emitted events (nil -> None)
      # @return [self]
      def step(name, accepts:, emits:, &block)
        raise ArgumentError, "step requires a block" if block.nil?
        raise Blazen::InternalError, "builder has been consumed" if @ptr.nil?

        step_id = Workflow.register_step_handler(block)

        vtable = Blazen::FFI::BlazenStepHandlerVTable.new
        # Stuff the integer step_id into the +user_data+ +void*+ slot. The
        # callbacks read it back via +ptr.address+.
        vtable[:user_data]       = ::FFI::Pointer.new(:void, step_id)
        vtable[:drop_user_data]  = Workflow::DROP_USER_DATA_FN
        vtable[:invoke]          = Workflow::INVOKE_FN

        accepts_arr = build_string_array(accepts)
        emits_arr   = build_string_array(emits)

        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(name) do |n|
          Blazen::FFI.blazen_workflow_builder_add_step(
            @ptr, n,
            accepts_arr[:ptr], accepts_arr[:count],
            emits_arr[:ptr],   emits_arr[:count],
            vtable,
            out_err,
          )
        end
        Blazen::FFI.check_error!(out_err)
        self
      end

      # Sets the per-step timeout in milliseconds.
      # @param millis [Integer]
      # @return [self]
      def step_timeout_ms(millis)
        raise Blazen::InternalError, "builder has been consumed" if @ptr.nil?

        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_workflow_builder_step_timeout_ms(@ptr, millis, out_err)
        Blazen::FFI.check_error!(out_err)
        self
      end

      # Sets the total workflow wall-clock timeout in milliseconds.
      # @param millis [Integer]
      # @return [self]
      def timeout_ms(millis)
        raise Blazen::InternalError, "builder has been consumed" if @ptr.nil?

        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_workflow_builder_timeout_ms(@ptr, millis, out_err)
        Blazen::FFI.check_error!(out_err)
        self
      end

      # Validates the workflow and produces a runnable {Workflow}. The
      # builder handle remains live but its internal state is consumed;
      # subsequent +#step+ calls will fail.
      #
      # @return [Workflow]
      def build
        raise Blazen::InternalError, "builder has been consumed" if @ptr.nil?

        out_workflow = ::FFI::MemoryPointer.new(:pointer)
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_workflow_builder_build(@ptr, out_workflow, out_err)
        Blazen::FFI.check_error!(out_err)

        wf_ptr = out_workflow.read_pointer
        raise Blazen::InternalError, "blazen_workflow_builder_build returned null workflow" if wf_ptr.null?

        Workflow::Instance.new(wf_ptr)
      end

      private

      # Builds a +::FFI::MemoryPointer+ array of NUL-terminated UTF-8 string
      # pointers from a Ruby array. Returns +{ ptr:, count: }+, both of which
      # the caller passes directly to the cabi.
      #
      # The per-string +MemoryPointer.from_string+ buffers are stashed on
      # +@keepalive+ so the Ruby GC can't reclaim them while the cabi is
      # reading them.
      def build_string_array(arr)
        arr ||= []
        return { ptr: nil, count: 0 } if arr.empty?

        str_ptrs = arr.map { |s| ::FFI::MemoryPointer.from_string(s.to_s) }
        @keepalive.concat(str_ptrs)

        arr_ptr = ::FFI::MemoryPointer.new(:pointer, str_ptrs.length)
        @keepalive << arr_ptr
        str_ptrs.each_with_index { |sp, i| arr_ptr[i].write_pointer(sp) }

        { ptr: arr_ptr, count: str_ptrs.length }
      end
    end

    # ---------------------------------------------------------------------
    # Instance — caller-owned +BlazenWorkflow+ wrapper
    #
    # Named +Instance+ rather than +Workflow+ to avoid a recursive
    # +Blazen::Workflow::Workflow+ constant. The pipeline module consumes
    # this class's underlying pointer via {#consume!}.
    # ---------------------------------------------------------------------

    # A runnable workflow.
    class Instance
      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "raw_ptr is null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_workflow_free))
      end

      # Runs the workflow asynchronously via the cabi future surface. Composes
      # with +Fiber.scheduler+ (e.g. the +async+ gem) when one is active;
      # otherwise blocks the calling thread on the cabi runtime.
      #
      # @param input [String, Object] JSON string OR JSON-dumpable input
      # @return [WorkflowResult]
      def run(input)
        raise Blazen::InternalError, "workflow has been consumed" if @ptr.nil?

        input_json = input.is_a?(String) ? input : JSON.dump(input)
        fut = Blazen::FFI.with_cstring(input_json) do |i|
          Blazen::FFI.blazen_workflow_run(@ptr, i)
        end
        raise Blazen::InternalError, "blazen_workflow_run returned null future" if fut.nil? || fut.null?

        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_workflow_result(f, out_result, out_err)
        end
        Blazen::FFI.check_error!(out_err)

        result_ptr = out_result.read_pointer
        raise Blazen::InternalError, "blazen_future_take_workflow_result returned null" if result_ptr.null?

        WorkflowResult.new(result_ptr)
      end

      # Runs the workflow synchronously on the cabi runtime, blocking the
      # calling thread until completion.
      #
      # @param input [String, Object]
      # @return [WorkflowResult]
      def run_blocking(input)
        raise Blazen::InternalError, "workflow has been consumed" if @ptr.nil?

        input_json = input.is_a?(String) ? input : JSON.dump(input)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(input_json) do |i|
          Blazen::FFI.blazen_workflow_run_blocking(@ptr, i, out_result, out_err)
        end
        Blazen::FFI.check_error!(out_err)

        result_ptr = out_result.read_pointer
        raise Blazen::InternalError, "blazen_workflow_run_blocking returned null result" if result_ptr.null?

        WorkflowResult.new(result_ptr)
      end

      # @return [Array<String>] declared step names, in declaration order
      def step_names
        raise Blazen::InternalError, "workflow has been consumed" if @ptr.nil?

        count = Blazen::FFI.blazen_workflow_step_names_count(@ptr)
        (0...count).map do |i|
          Blazen::FFI.consume_cstring(Blazen::FFI.blazen_workflow_step_names_get(@ptr, i))
        end
      end

      # Transfers ownership of the underlying +BlazenWorkflow+ pointer to the
      # caller (typically the pipeline builder, which consumes workflows when
      # adding them as stages). After this call the Instance is no longer
      # usable.
      #
      # @return [::FFI::Pointer]
      def consume!
        raise Blazen::InternalError, "workflow already consumed" if @ptr.nil?

        @ptr.autorelease = false
        addr = @ptr.address
        @ptr = nil
        ::FFI::Pointer.new(addr)
      end
    end

    # ---------------------------------------------------------------------
    # Module-level convenience
    # ---------------------------------------------------------------------

    module_function

    # Convenience builder: yields a fresh {Builder}, builds it, and returns
    # the resulting {Instance}.
    #
    # @param name [String] workflow name
    # @yieldparam builder [Builder]
    # @return [Instance]
    def build(name)
      builder = Builder.new(name)
      yield builder if block_given?
      builder.build
    end

    # Runs +workflow+ with +input+, accepting either a JSON string or any
    # +JSON.dump+able value.
    #
    # @param workflow [Instance]
    # @param input [String, Object]
    # @return [WorkflowResult]
    def run(workflow, input)
      workflow.run(input)
    end
  end

  # Top-level convenience so callers can write +Blazen.workflow(name) { |b| ... }+.
  #
  # @param name [String] workflow name
  # @yieldparam builder [Blazen::Workflow::Builder]
  # @return [Blazen::Workflow::Instance]
  def self.workflow(name, &block)
    Workflow.build(name, &block)
  end
end
