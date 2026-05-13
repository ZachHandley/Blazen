# frozen_string_literal: true

module Blazen
  # Streaming-completion support, with first-class support for
  # Ruby-implemented stream sinks via the cabi
  # +BlazenCompletionStreamSinkVTable+ trampoline.
  #
  # Two entry points are provided:
  #
  # * {Blazen::Streaming.complete} — blocking; suitable for the typical
  #   "consume the stream synchronously" Ruby use case. Backed by the cabi
  #   +blazen_complete_streaming_blocking+ call.
  # * {Blazen::Streaming.complete_async} — future-returning; composes with
  #   +Fiber.scheduler+ (the +async+ gem) so the calling fiber yields while
  #   the stream is in flight. Backed by +blazen_complete_streaming+.
  #
  # Callers can supply explicit per-event handlers via the +on_chunk:+,
  # +on_done:+, +on_error:+ kwargs, OR pass a single block that receives
  # +(event_kind, *payload)+. The block form keeps simple inline consumption
  # ergonomic:
  #
  # @example Single-block consumer
  #   Blazen::Streaming.complete(model, req) do |kind, *args|
  #     case kind
  #     when :chunk then $stdout.write(args[0].content_delta.to_s)
  #     when :done  then puts "\n[done: #{args[0]}]"
  #     when :error then warn args[0].message
  #     end
  #   end
  #
  # @example Explicit handlers
  #   Blazen::Streaming.complete(
  #     model, req,
  #     on_chunk: ->(c) { $stdout.write(c.content_delta.to_s) },
  #     on_done:  ->(finish_reason, usage) { puts "\n#{finish_reason} / #{usage.total_tokens} tokens" },
  #     on_error: ->(e) { warn "stream error: #{e.message}" },
  #   )
  module Streaming
    # ---------------------------------------------------------------------
    # Sink registry (analogous to {Blazen::Agents}'s tool-handler registry).
    # ---------------------------------------------------------------------
    #
    # Each streaming call registers its three Ruby callbacks
    # (+on_chunk+ / +on_done+ / +on_error+) in a process-wide registry keyed
    # by a small integer ID. The cabi receives the ID as the vtable's
    # opaque +user_data+ pointer; the four trampoline thunks below look the
    # handlers back up by ID on every invocation. The registry holds a
    # strong reference until +drop_user_data+ fires (which happens when
    # the inner +CStreamSink+ drops on the Rust side — after the stream
    # terminates).
    @sink_registry = {}
    @sink_registry_mutex = Mutex.new
    @next_sink_id = 0

    # @api private
    # @return [Integer] new registry ID
    def self.register_sink(handlers)
      @sink_registry_mutex.synchronize do
        @next_sink_id += 1
        id = @next_sink_id
        @sink_registry[id] = handlers
        id
      end
    end

    # @api private
    def self.unregister_sink(id)
      @sink_registry_mutex.synchronize { @sink_registry.delete(id) }
    end

    # @api private
    # @return [Hash, nil]
    def self.lookup_sink(id)
      @sink_registry_mutex.synchronize { @sink_registry[id] }
    end

    # ---------------------------------------------------------------------
    # Public StreamChunk wrapper
    # ---------------------------------------------------------------------

    # A single token-streaming event emitted by the model. Wraps
    # +BlazenStreamChunk *+.
    #
    # The chunk is caller-owned: the cabi hands a freshly-allocated
    # +*mut BlazenStreamChunk+ to {ON_CHUNK_FN}, which wraps it in a
    # {::FFI::AutoPointer} so Ruby GC frees it (via
    # +blazen_stream_chunk_free+) once the user's +on_chunk+ callback
    # returns and the wrapper goes out of scope.
    class StreamChunk
      # @api private
      # @param raw_ptr [::FFI::Pointer] caller-owned +BlazenStreamChunk *+
      def initialize(raw_ptr)
        if raw_ptr.nil? || raw_ptr.null?
          raise Blazen::InternalError, "StreamChunk: native pointer is null"
        end

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_stream_chunk_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [String, nil] the incremental content delta for this chunk
      def content_delta
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_stream_chunk_content_delta(@ptr))
      end

      # @return [Boolean] whether this is the terminal chunk of the stream
      def final?
        Blazen::FFI.blazen_stream_chunk_is_final(@ptr)
      end

      # @return [Array<Blazen::Llm::ToolCall>] tool calls emitted in this chunk
      def tool_calls
        count = Blazen::FFI.blazen_stream_chunk_tool_calls_count(@ptr)
        Array.new(count) do |i|
          raw = Blazen::FFI.blazen_stream_chunk_tool_calls_get(@ptr, i)
          Blazen::Llm::ToolCall.from_raw(raw)
        end
      end
    end

    # ---------------------------------------------------------------------
    # vtable callback thunks
    # ---------------------------------------------------------------------

    # +on_chunk+ thunk. Receives a caller-owned +*mut BlazenStreamChunk+
    # that the wrapping {StreamChunk}'s AutoPointer will free once the
    # user's handler returns. Returning +-1+ aborts the stream (Rust
    # synthesises a +InternalError("returned -1 without setting out_err")+).
    ON_CHUNK_FN = ::FFI::Function.new(
      :int32,
      %i[pointer pointer pointer],
      proc do |user_data_ptr, chunk_ptr, _out_err|
        id = user_data_ptr.address
        handlers = lookup_sink(id)
        if handlers.nil?
          warn "[blazen streaming] sink not found for id=#{id}"
          next(-1)
        end

        begin
          chunk = StreamChunk.new(chunk_ptr)
          on_chunk = handlers[:on_chunk]
          on_chunk&.call(chunk)
          0
        rescue StandardError => e
          warn "[blazen streaming on_chunk] #{e.class}: #{e.message}"
          warn e.backtrace.join("\n") if e.backtrace
          -1
        end
      end,
      blocking: true,
    )

    # +on_done+ thunk. Receives a caller-owned +*mut c_char+ for
    # +finish_reason+ (consumed via {Blazen::FFI.consume_cstring}) and a
    # caller-owned +*mut BlazenTokenUsage+ (wrapped via
    # {Blazen::Llm::TokenUsage.from_raw}).
    ON_DONE_FN = ::FFI::Function.new(
      :int32,
      %i[pointer pointer pointer pointer],
      proc do |user_data_ptr, finish_reason_ptr, usage_ptr, _out_err|
        id = user_data_ptr.address
        handlers = lookup_sink(id)
        if handlers.nil?
          warn "[blazen streaming] sink not found for id=#{id}"
          # Still consume the buffers so we don't leak — Rust handed
          # ownership across the FFI boundary regardless of whether we
          # found a handler.
          Blazen::FFI.consume_cstring(finish_reason_ptr) unless finish_reason_ptr.null?
          Blazen::FFI.blazen_token_usage_free(usage_ptr) unless usage_ptr.null?
          next(-1)
        end

        finish_reason = Blazen::FFI.consume_cstring(finish_reason_ptr)
        usage =
          if usage_ptr.nil? || usage_ptr.null?
            nil
          else
            Blazen::Llm::TokenUsage.from_raw(usage_ptr)
          end

        begin
          on_done = handlers[:on_done]
          on_done&.call(finish_reason, usage)
          0
        rescue StandardError => e
          warn "[blazen streaming on_done] #{e.class}: #{e.message}"
          warn e.backtrace.join("\n") if e.backtrace
          -1
        end
      end,
      blocking: true,
    )

    # +on_error+ thunk. Receives a caller-owned +*mut BlazenError+ which we
    # decode into the matching Ruby {Blazen::Error} subclass via
    # {Blazen.build_error_from_ptr} (that decoder consumes / frees the
    # pointer).
    ON_ERROR_FN = ::FFI::Function.new(
      :int32,
      %i[pointer pointer pointer],
      proc do |user_data_ptr, err_ptr, _out_err|
        id = user_data_ptr.address
        handlers = lookup_sink(id)
        if handlers.nil?
          warn "[blazen streaming] sink not found for id=#{id}"
          Blazen::FFI.blazen_error_free(err_ptr) unless err_ptr.null?
          next(-1)
        end

        ruby_err =
          if err_ptr.nil? || err_ptr.null?
            Blazen::InternalError.new("on_error called with null BlazenError")
          else
            Blazen.build_error_from_ptr(err_ptr)
          end

        begin
          on_error = handlers[:on_error]
          on_error&.call(ruby_err)
          0
        rescue StandardError => e
          warn "[blazen streaming on_error] #{e.class}: #{e.message}"
          warn e.backtrace.join("\n") if e.backtrace
          -1
        end
      end,
      blocking: true,
    )

    # +drop_user_data+ thunk. Drops the registry entry for the sink, called
    # exactly once when the inner +CStreamSink+ wrapper drops on the Rust
    # side (after the stream terminates, or on an early-return failure
    # path before the stream starts).
    DROP_SINK_USER_DATA_FN = ::FFI::Function.new(
      :void,
      [:pointer],
      proc do |user_data_ptr|
        unregister_sink(user_data_ptr.address)
      end,
    )

    module_function

    # ---------------------------------------------------------------------
    # Public entry points
    # ---------------------------------------------------------------------

    # Drives a streaming chat completion synchronously, dispatching events
    # to the supplied callbacks (or block). Blocks the calling thread on
    # the cabi tokio runtime until +on_done+ or +on_error+ has fired.
    #
    # Either pass explicit per-event handlers via the +on_chunk:+,
    # +on_done:+, +on_error:+ kwargs, OR pass a single block that receives
    # +(kind, *args)+:
    #
    # * +:chunk+ — +args = [chunk]+, where +chunk+ is a {StreamChunk}.
    # * +:done+  — +args = [finish_reason, usage]+, where +finish_reason+
    #   is a +String+ and +usage+ a {Blazen::Llm::TokenUsage} (or +nil+).
    # * +:error+ — +args = [err]+, where +err+ is a {Blazen::Error}
    #   subclass.
    #
    # The +request+ is consumed by the call (the cabi takes ownership of
    # the underlying +BlazenCompletionRequest *+).
    #
    # @param model [Blazen::Llm::CompletionModel] streaming-capable model
    # @param request [Blazen::Llm::CompletionRequest] consumed by the call
    # @param on_chunk [#call(chunk)] called for each {StreamChunk}
    # @param on_done [#call(finish_reason, usage)] called once when the
    #   stream completes normally
    # @param on_error [#call(err)] called once on fatal stream errors
    # @yield [kind, *args] block-form alternative to the kwargs
    # @return [void]
    def complete(model, request, on_chunk: nil, on_done: nil, on_error: nil, &block)
      handlers = build_handlers(on_chunk, on_done, on_error, block)
      sink_id = register_sink(handlers)
      registered = true

      begin
        vtable = build_vtable(sink_id)
        req_ptr = consume_request!(request)

        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_complete_streaming_blocking(model.ptr, req_ptr, vtable, out_err)

        # The cabi has now consumed the vtable's +user_data+ and will
        # eventually call +drop_user_data+ (which unregisters the sink).
        # Mark +registered+ false so our +ensure+ arm doesn't double-drop.
        registered = false
        Blazen::FFI.check_error!(out_err)
        nil
      ensure
        unregister_sink(sink_id) if registered
      end
    end

    # Asynchronous variant of {complete}. Returns when the streaming
    # future resolves; the sink callbacks fire on cabi tokio worker
    # threads while the calling fiber yields (composes with
    # +Fiber.scheduler+ — see {Blazen::FFI.await_future}).
    #
    # The +request+ is consumed by the call.
    #
    # @param model [Blazen::Llm::CompletionModel] streaming-capable model
    # @param request [Blazen::Llm::CompletionRequest] consumed by the call
    # @param on_chunk [#call(chunk)]
    # @param on_done [#call(finish_reason, usage)]
    # @param on_error [#call(err)]
    # @yield [kind, *args] block-form alternative to the kwargs
    # @return [void]
    def complete_async(model, request, on_chunk: nil, on_done: nil, on_error: nil, &block)
      handlers = build_handlers(on_chunk, on_done, on_error, block)
      sink_id = register_sink(handlers)
      registered = true

      begin
        vtable = build_vtable(sink_id)
        req_ptr = consume_request!(request)

        fut = Blazen::FFI.blazen_complete_streaming(model.ptr, req_ptr, vtable)
        # The cabi has now consumed the vtable's +user_data+ (regardless
        # of whether +fut+ is null — +blazen_complete_streaming+'s
        # null-return paths still call +drop_user_data+). Mark
        # +registered+ false to avoid double-drop.
        registered = false

        if fut.nil? || fut.null?
          raise Blazen::InternalError, "blazen_complete_streaming returned null"
        end

        Blazen::FFI.await_future(fut) do |f|
          out_err = ::FFI::MemoryPointer.new(:pointer)
          Blazen::FFI.blazen_future_take_unit(f, out_err)
          Blazen::FFI.check_error!(out_err)
        end
        nil
      ensure
        unregister_sink(sink_id) if registered
      end
    end

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    # @api private
    # Normalises the +on_chunk:+ / +on_done:+ / +on_error:+ + block argument
    # combinations into a single +{on_chunk:, on_done:, on_error:}+ hash.
    def self.build_handlers(on_chunk, on_done, on_error, block)
      if block
        on_chunk ||= ->(c)    { block.call(:chunk, c) }
        on_done  ||= ->(r, u) { block.call(:done, r, u) }
        on_error ||= ->(e)    { block.call(:error, e) }
      end
      { on_chunk: on_chunk, on_done: on_done, on_error: on_error }
    end
    private_class_method :build_handlers

    # @api private
    # Builds a fresh +BlazenCompletionStreamSinkVTable+ struct keyed to the
    # given registry +sink_id+. The struct value is passed by value into
    # the cabi (see +attach_function :blazen_complete_streaming_blocking+
    # in +ffi.rb+).
    def self.build_vtable(sink_id)
      vt = Blazen::FFI::BlazenCompletionStreamSinkVTable.new
      vt[:user_data]      = ::FFI::Pointer.new(sink_id)
      vt[:drop_user_data] = DROP_SINK_USER_DATA_FN
      vt[:on_chunk]       = ON_CHUNK_FN
      vt[:on_done]        = ON_DONE_FN
      vt[:on_error]       = ON_ERROR_FN
      vt
    end
    private_class_method :build_vtable

    # @api private
    # Consumes a +Blazen::Llm::CompletionRequest+ wrapper into a bare
    # +::FFI::Pointer+, raising +ArgumentError+ if the input isn't a
    # CompletionRequest.
    def self.consume_request!(request)
      unless request.is_a?(Blazen::Llm::CompletionRequest)
        raise ArgumentError, "request must be Blazen::Llm::CompletionRequest"
      end

      request.consume!
    end
    private_class_method :consume_request!
  end
end
