# frozen_string_literal: true

require "json"

module Blazen
  module ControlPlane
    # Worker-side handle that drives a bidi control-plane session.
    #
    # Workers connect outbound to the control-plane server, advertise
    # their capabilities and tags, and run assignments dispatched by
    # the server. The user supplies an +AssignmentHandler+ in one of
    # two forms:
    #
    # 1. A block passed to {.new} — receives a single +assignment+
    #    hash and returns a JSON-serialisable value (or raises).
    # 2. An object that responds to +#handle(assignment)+ and
    #    optionally +#on_cancel(run_id)+ / +#on_drain(immediate)+.
    #
    # @example block form
    #   worker = Blazen::ControlPlane::Worker.new(
    #     endpoint: "http://cp.example.com:7445",
    #     node_id: "ruby-worker-1",
    #     capabilities: [{ kind: "workflow:hello", version: 1 }],
    #   ) do |assignment|
    #     { greeting: "hello from #{assignment[:workflow_name]}" }
    #   end
    #   worker.run
    class Worker
      # Admission-mode integer codes (mirror the cabi convention).
      ADMISSION_FIXED        = 0
      ADMISSION_VRAM_BUDGET  = 1
      ADMISSION_REACTIVE     = 2

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # Construct (and validate) a new worker. Does NOT open a
      # network connection — the first connect happens inside {#run}.
      #
      # @param endpoint [String] gRPC URI such as
      #   +"http://cp.example.com:7445"+
      # @param node_id [String] stable identifier of this worker
      # @param capabilities [Array<Hash>] each entry must contain
      #   +:kind+ (String) and +:version+ (Integer)
      # @param tags [Hash{String=>String}] free-form tags surfaced via
      #   the worker's +Hello+ frame
      # @param admission [Symbol] one of +:fixed+, +:vram_budget+,
      #   +:reactive+ (default: +:fixed+)
      # @param admission_param [Integer] interpretation depends on
      #   +admission+:
      #   * +:fixed+ — +max_in_flight+ (default 1)
      #   * +:vram_budget+ — +max_vram_mb+
      #   * +:reactive+ — ignored
      # @param handler [Object, nil] optional handler instance with
      #   +#handle+ / optional +#on_cancel+ / +#on_drain+. Mutually
      #   exclusive with passing a block.
      # @param bearer_token [String, nil] optional bearer token attached
      #   to every RPC the worker makes (+nil+ = no token)
      # @yield [Hash, AssignmentContext] block form of the handler —
      #   receives the assignment hash (and, optionally, an
      #   {AssignmentContext} as a second argument) and returns its output
      # @return [Worker]
      # @raise [Blazen::Error] when the endpoint URI cannot be parsed
      # @raise [ArgumentError] when neither block nor handler is given
      def initialize(endpoint:, node_id:, capabilities: [], tags: {},
                     admission: :fixed, admission_param: 0,
                     mtls: nil, handler: nil, bearer_token: nil, &block)
        ControlPlane.ensure_available!

        if handler.nil? && block.nil?
          raise ArgumentError, "Worker requires either a handler: kwarg or a block"
        end
        if handler && block
          raise ArgumentError, "Worker accepts either handler: kwarg OR a block, not both"
        end

        @handler = handler || BlockHandler.new(block)
        # Persist references to the FFI::Function callbacks for the GC
        # — ::FFI::Function instances must outlive the C side that
        # references them, otherwise Ruby may collect them while a
        # foreign callback is in flight.
        @vtable_callbacks = build_vtable_callbacks(@handler)

        vtable = Blazen::FFI::BlazenAssignmentHandlerVTable.new
        vtable[:user_data]      = ::FFI::Pointer::NULL
        vtable[:drop_user_data] = @vtable_callbacks[:drop_user_data]
        vtable[:handle]         = @vtable_callbacks[:handle]
        vtable[:on_cancel]      = @vtable_callbacks[:on_cancel]
        vtable[:on_drain]       = @vtable_callbacks[:on_drain]
        @vtable = vtable

        admission_mode  = admission_code(admission)
        admission_param = admission_param_for(admission, admission_param)

        capabilities_json = JSON.generate(capabilities.map do |c|
          { kind: c[:kind].to_s, version: Integer(c[:version]) }
        end)
        tags_json = JSON.generate(tags.transform_keys(&:to_s).transform_values(&:to_s))

        out_worker = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)

        if mtls
          self.class.validate_mtls!(mtls)
          invoke_new_with_mtls_blocking(
            endpoint, node_id, capabilities_json, tags_json,
            admission_mode, admission_param, mtls, bearer_token, vtable, out_worker, out_err,
          )
        else
          invoke_new_blocking(
            endpoint, node_id, capabilities_json, tags_json,
            admission_mode, admission_param, bearer_token, vtable, out_worker, out_err,
          )
        end
        Blazen::FFI.check_error!(out_err)

        raw = out_worker.read_pointer
        if raw.nil? || raw.null?
          raise Blazen::ValidationError, "blazen_controlplane_worker_new_blocking returned a null worker"
        end

        @ptr = ::FFI::AutoPointer.new(
          raw, Blazen::FFI.method(:blazen_controlplane_worker_free),
        )
      end

      # Construct a worker with mTLS loaded from PEM files on disk.
      # Convenience over passing +mtls:+ directly to {#initialize}.
      #
      # @param cert_path [String]
      # @param key_path [String]
      # @param ca_path [String]
      # @param kwargs [Hash] forwarded verbatim to {#initialize}
      # @yield see {#initialize}
      # @return [Worker]
      def self.new_with_mtls(cert_path:, key_path:, ca_path:, **kwargs, &block)
        new(mtls: { cert_path: cert_path, key_path: key_path, ca_path: ca_path },
            **kwargs, &block)
      end

      # @api private
      def self.validate_mtls!(mtls)
        unless mtls.is_a?(Hash)
          raise ArgumentError, "mtls: must be a Hash with :cert_path / :key_path / :ca_path"
        end

        %i[cert_path key_path ca_path].each do |key|
          value = mtls[key]
          if value.nil? || value.to_s.empty?
            raise ArgumentError, "mtls[#{key.inspect}] is required and must be a non-empty path"
          end
        end
      end

      # Drive the worker until shutdown / drain / retry exhaustion.
      # Integrates with {Fiber.scheduler}.
      #
      # @return [void]
      def run
        fut = Blazen::FFI.blazen_controlplane_worker_run(@ptr)
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_controlplane_worker_run returned a null future (worker already running?)"
        end

        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_unit(f, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        nil
      end

      # Blocking-thread variant of {#run}.
      #
      # @return [void]
      def run_blocking
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_controlplane_worker_run_blocking(@ptr, out_err)
        Blazen::FFI.check_error!(out_err)
        nil
      end

      # Signal the worker to stop. No-op once {#run} has been called
      # — at that point the canonical way to terminate the worker is
      # to drop the future returned by {#run} (the bidi session loop
      # honours cancellation at every iteration).
      #
      # Idempotent.
      #
      # @return [void]
      def shutdown
        Blazen::FFI.blazen_controlplane_worker_shutdown(@ptr)
        nil
      end

      private

      # Drive +blazen_controlplane_worker_new_blocking+ with the
      # appropriate +with_cstring+ nesting. Splitting this out keeps
      # {#initialize} flat enough to also dispatch the mTLS variant.
      def invoke_new_blocking(endpoint, node_id, capabilities_json, tags_json,
                              admission_mode, admission_param, bearer_token, vtable,
                              out_worker, out_err)
        Blazen::FFI.with_cstring(endpoint.to_s) do |ep|
          Blazen::FFI.with_cstring(node_id.to_s) do |nid|
            Blazen::FFI.with_cstring(capabilities_json) do |cj|
              Blazen::FFI.with_cstring(tags_json) do |tj|
                Blazen::FFI.with_cstring(bearer_token&.to_s) do |bt|
                  Blazen::FFI.blazen_controlplane_worker_new_blocking(
                    ep, nid, cj, tj,
                    admission_mode, admission_param, bt,
                    vtable, out_worker, out_err,
                  )
                end
              end
            end
          end
        end
      end

      # mTLS variant of {#invoke_new_blocking}. Adds the three PEM
      # path arguments without ballooning {#initialize}'s nesting.
      def invoke_new_with_mtls_blocking(endpoint, node_id, capabilities_json, tags_json,
                                        admission_mode, admission_param, mtls, bearer_token, vtable,
                                        out_worker, out_err)
        Blazen::FFI.with_cstring(endpoint.to_s) do |ep|
          Blazen::FFI.with_cstring(node_id.to_s) do |nid|
            Blazen::FFI.with_cstring(capabilities_json) do |cj|
              Blazen::FFI.with_cstring(tags_json) do |tj|
                Blazen::FFI.with_cstring(mtls[:cert_path].to_s) do |cp|
                  Blazen::FFI.with_cstring(mtls[:key_path].to_s) do |kp|
                    Blazen::FFI.with_cstring(mtls[:ca_path].to_s) do |ca|
                      Blazen::FFI.with_cstring(bearer_token&.to_s) do |bt|
                        Blazen::FFI.blazen_controlplane_worker_new_with_mtls_blocking(
                          ep, nid, cj, tj,
                          admission_mode, admission_param,
                          cp, kp, ca, bt,
                          vtable, out_worker, out_err,
                        )
                      end
                    end
                  end
                end
              end
            end
          end
        end
      end

      def admission_code(symbol)
        case symbol
        when :fixed       then ADMISSION_FIXED
        when :vram_budget then ADMISSION_VRAM_BUDGET
        when :reactive    then ADMISSION_REACTIVE
        else
          raise ArgumentError,
                "unknown admission mode #{symbol.inspect}; expected :fixed, :vram_budget, or :reactive"
        end
      end

      def admission_param_for(admission, param)
        case admission
        when :fixed       then param.zero? ? 1 : Integer(param)
        when :vram_budget then Integer(param)
        else                   0
        end
      end

      def build_vtable_callbacks(handler)
        # The callbacks are plain blocks (FFI::Function wrappers) —
        # the C side never invokes +drop_user_data+ with a non-null
        # +user_data+ because we always pass +::FFI::Pointer::NULL+
        # for it. Ruby holds the handler reference directly inside
        # this Worker instance, so its lifetime is tied to the
        # surrounding Worker object's lifetime.
        drop_user_data = ::FFI::Function.new(:void, [:pointer]) { |_| }

        handle = ::FFI::Function.new(
          :int32,
          [:pointer, :pointer, :pointer, :pointer, :pointer, :pointer, :pointer],
        ) do |_user_data, run_id_ptr, workflow_ptr, input_ptr, ctx_ptr, out_json, out_err|
          dispatch_handle(handler, run_id_ptr, workflow_ptr, input_ptr, ctx_ptr, out_json, out_err)
        end

        on_cancel = ::FFI::Function.new(:void, [:pointer, :pointer]) do |_user_data, run_id_ptr|
          dispatch_on_cancel(handler, run_id_ptr)
        end

        on_drain = ::FFI::Function.new(:void, [:pointer, :bool]) do |_user_data, immediate|
          dispatch_on_drain(handler, immediate)
        end

        {
          drop_user_data: drop_user_data,
          handle:         handle,
          on_cancel:      on_cancel,
          on_drain:       on_drain,
        }
      end

      def dispatch_handle(handler, run_id_ptr, workflow_ptr, input_ptr, ctx_ptr, out_json, out_err)
        run_id    = run_id_ptr.read_string.force_encoding(Encoding::UTF_8)
        workflow  = workflow_ptr.read_string.force_encoding(Encoding::UTF_8)
        input_raw = input_ptr.read_string.force_encoding(Encoding::UTF_8)
        input     = JSON.parse(input_raw)

        assignment = {
          run_id:        run_id,
          workflow_name: workflow,
          input:         input,
        }

        # The context pointer is BORROWED — valid only for the duration
        # of this callback. The AssignmentContext wrapper must NOT be
        # retained past this method's return.
        context = AssignmentContext.new(ctx_ptr)
        output  = invoke_handler(handler, assignment, context)
        output_json = JSON.generate(output)
        # Mint the output buffer via +blazen_string_alloc+ so the
        # cabi can reclaim it via +CString::from_raw+ on the same
        # allocator. Ruby-side allocations (e.g.
        # +MemoryPointer.from_string+) are NOT wire-compatible with
        # the cabi's allocator in general.
        out_ptr = Blazen::FFI.with_cstring(output_json) do |js|
          Blazen::FFI.blazen_string_alloc(js)
        end
        out_json.write_pointer(out_ptr)
        out_err.write_pointer(::FFI::Pointer::NULL)
        0
      rescue StandardError => e
        # Failure path: synthesise a BlazenError via the JSON
        # constructor exposed by the cabi.
        err_json = JSON.generate(kind: "Internal", message: e.message)
        err_ptr = Blazen::FFI.with_cstring(err_json) do |js|
          Blazen::FFI.blazen_error_from_json(js)
        end
        out_err.write_pointer(err_ptr)
        out_json.write_pointer(::FFI::Pointer::NULL)
        -1
      end

      # Invoke the handler's +#handle+ with the assignment, passing the
      # {AssignmentContext} as a second argument when the handler's
      # +#handle+ accepts it. Handlers (and blocks) that only take a
      # single +assignment+ argument keep working unchanged.
      def invoke_handler(handler, assignment, context)
        meth = handler.method(:handle)
        # arity 1 (or 0/negative-but-not-accepting-2): legacy single-arg form.
        # arity >= 2 or splat (< -1): pass the context too.
        arity = meth.arity
        if arity == 1 || arity.zero?
          handler.handle(assignment)
        else
          handler.handle(assignment, context)
        end
      end

      def dispatch_on_cancel(handler, run_id_ptr)
        return unless handler.respond_to?(:on_cancel)

        run_id = run_id_ptr.read_string.force_encoding(Encoding::UTF_8)
        handler.on_cancel(run_id)
      rescue StandardError
        # Swallow handler errors here — the C side just notifies and
        # doesn't act on the return value.
        nil
      end

      def dispatch_on_drain(handler, immediate)
        return unless handler.respond_to?(:on_drain)

        handler.on_drain(immediate)
      rescue StandardError
        nil
      end

      # Adapts a block to the +#handle+ protocol so the dispatcher
      # can call into either form uniformly. The block may accept one
      # argument (+assignment+) or two (+assignment, context+); the
      # extra {AssignmentContext} argument is only forwarded when the
      # block's arity asks for it.
      #
      # @api private
      class BlockHandler
        def initialize(block)
          @block = block
        end

        def handle(assignment, context = nil)
          arity = @block.arity
          if arity == 1 || arity.zero?
            @block.call(assignment)
          else
            @block.call(assignment, context)
          end
        end
      end

      # Borrowed handle to a +blazen-controlplane+ +AssignmentContext+,
      # passed to the handler's +#handle+ for the duration of a single
      # assignment. Lets a running assignment emit intermediate events
      # and request human/orchestrator input.
      #
      # WARNING: the underlying pointer is only valid for the duration
      # of the +#handle+ call. Do NOT retain this object (or call its
      # methods) after +#handle+ returns.
      class AssignmentContext
        # @param ptr [::FFI::Pointer] borrowed +BlazenAssignmentContext*+
        def initialize(ptr)
          @ptr = ptr
        end

        # Emit a non-terminal event from the running assignment.
        #
        # @param event_type [String]
        # @param data [Object, nil] JSON-serialisable payload
        #   (+nil+ → JSON +null+)
        # @return [void]
        # @raise [Blazen::Error] when the worker's outbound channel is closed
        def emit_event(event_type, data = nil)
          data_json = JSON.generate(data)
          out_err = ::FFI::MemoryPointer.new(:pointer)
          Blazen::FFI.with_cstring(event_type.to_s) do |et|
            Blazen::FFI.with_cstring(data_json) do |dj|
              Blazen::FFI.blazen_assignment_context_emit_event(@ptr, et, dj, out_err)
            end
          end
          Blazen::FFI.check_error!(out_err)
          nil
        end

        # Raise an +input.request+ and block until the orchestrator
        # answers (via {Client#respond_to_input}), the assignment is
        # cancelled, or the optional timeout elapses.
        #
        # @param prompt [String]
        # @param metadata [Object, nil] JSON-serialisable metadata
        #   (+nil+ → JSON +null+)
        # @param timeout_ms [Integer, nil] optional timeout in
        #   milliseconds (+nil+ = no timeout)
        # @return [Object] the decoded JSON value the orchestrator
        #   handed back
        # @raise [Blazen::Error] when the request fails, is cancelled,
        #   or times out
        def request_input(prompt, metadata = nil, timeout_ms: nil)
          metadata_json = JSON.generate(metadata)
          timeout = timeout_ms.nil? ? 0 : Integer(timeout_ms)
          out_json = ::FFI::MemoryPointer.new(:pointer)
          out_err  = ::FFI::MemoryPointer.new(:pointer)
          Blazen::FFI.with_cstring(prompt.to_s) do |pr|
            Blazen::FFI.with_cstring(metadata_json) do |md|
              Blazen::FFI.blazen_assignment_context_request_input(
                @ptr, pr, md, timeout, out_json, out_err,
              )
            end
          end
          Blazen::FFI.check_error!(out_err)
          ptr = out_json.read_pointer
          return nil if ptr.nil? || ptr.null?

          raw = ptr.read_string.force_encoding(Encoding::UTF_8)
          Blazen::FFI.blazen_string_free(ptr)
          JSON.parse(raw)
        end
      end
    end
  end
end
