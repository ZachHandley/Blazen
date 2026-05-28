# frozen_string_literal: true

module Blazen
  # Batch-completion helpers.
  #
  # Runs many {Blazen::Llm::ModelRequest}s through a single
  # {Blazen::LlmProvider} (or per-engine provider that responds to
  # +#as_llm_provider+) with a bounded number of in-flight requests, then
  # exposes the per-request success/failure result and aggregated
  # usage/cost totals via {BatchResult} / {BatchItem}.
  module Batch
    module_function

    # Default max in-flight concurrency for {complete}.
    DEFAULT_MAX_CONCURRENCY = 4

    # Runs multiple completion requests against +provider+, bounded by
    # +max_concurrency+ in-flight requests at a time.
    #
    # +provider+ may be either a {Blazen::LlmProvider} (polymorphic
    # +BlazenLlmProvider *+ handle) OR any per-engine class that responds
    # to +#as_llm_provider+ ({Blazen::OpenAiProvider}, etc.) — the latter
    # is auto-converted via +#as_llm_provider+ before hand-off to the
    # cabi.
    #
    # Every request's underlying handle is **consumed** by this call —
    # callers MUST NOT reference the request wrappers again afterwards.
    # The default implementation drives the cabi tokio runtime via the
    # future-await pattern and composes with {Fiber.scheduler}; use
    # {complete_blocking} for an explicit thread-block.
    #
    # @param provider [Blazen::LlmProvider, #as_llm_provider]
    # @param requests [Array<Blazen::Llm::ModelRequest>]
    # @param max_concurrency [Integer]
    # @return [Blazen::Batch::BatchResult]
    def complete(provider, requests, max_concurrency: DEFAULT_MAX_CONCURRENCY)
      llm = Batch.send(:coerce_llm_provider, provider)
      reqs = Array(requests)
      array_mp, count = Batch.send(:pack_requests, reqs)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      out_err    = ::FFI::MemoryPointer.new(:pointer)

      fut = Blazen::FFI.blazen_complete_batch(llm.ptr, array_mp, count, Integer(max_concurrency))
      if fut.nil? || fut.null?
        raise Blazen::ValidationError, "blazen_complete_batch returned a null future"
      end

      Blazen::FFI.await_future(fut) do |f|
        Blazen::FFI.blazen_future_take_batch_result(f, out_result, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      BatchResult.new(out_result.read_pointer)
    end

    # Blocking-thread variant of {complete}.
    #
    # @param provider [Blazen::LlmProvider, #as_llm_provider]
    # @param requests [Array<Blazen::Llm::ModelRequest>]
    # @param max_concurrency [Integer]
    # @return [Blazen::Batch::BatchResult]
    def complete_blocking(provider, requests, max_concurrency: DEFAULT_MAX_CONCURRENCY)
      llm = Batch.send(:coerce_llm_provider, provider)
      reqs = Array(requests)
      array_mp, count = Batch.send(:pack_requests, reqs)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      out_err    = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.blazen_complete_batch_blocking(
        llm.ptr, array_mp, count, Integer(max_concurrency), out_result, out_err
      )
      Blazen::FFI.check_error!(out_err)
      BatchResult.new(out_result.read_pointer)
    end

    # Coerces +provider+ to a {Blazen::LlmProvider} polymorphic handle:
    # passes {Blazen::LlmProvider} through (calling +#as_llm_provider+
    # which yields +self+ for the abstract base, or a fresh polymorphic
    # handle for per-engine subclasses), and otherwise requires the
    # +#as_llm_provider+ duck-type.
    #
    # @api private
    # @return [Blazen::LlmProvider]
    def self.coerce_llm_provider(provider)
      if provider.is_a?(Blazen::LlmProvider)
        llm = provider.as_llm_provider
      elsif provider.respond_to?(:as_llm_provider)
        llm = provider.as_llm_provider
      else
        raise ArgumentError,
              "provider must be a Blazen::LlmProvider or respond to #as_llm_provider"
      end

      unless llm.is_a?(Blazen::LlmProvider)
        raise ArgumentError,
              "#as_llm_provider must return a Blazen::LlmProvider (got #{llm.class})"
      end

      llm
    end

    # Packs an array of ModelRequest wrappers into a +const*+ pointer
    # array suitable for the cabi. Each request's underlying handle is
    # consumed (its auto-free hook disabled) so ownership transfers to
    # the C side.
    #
    # @api private
    # @param requests [Array<Blazen::Llm::ModelRequest>]
    # @return [Array(::FFI::MemoryPointer, Integer)] +(array, count)+
    def self.pack_requests(requests)
      count = requests.length
      if count.zero?
        # Allocating a 0-length :pointer MemoryPointer returns a null-ish
        # buffer on some FFI builds; pass an explicit null pointer so the
        # cabi takes the +(null, 0)+ path documented in its safety notes.
        return [::FFI::Pointer::NULL, 0]
      end

      array_mp = ::FFI::MemoryPointer.new(:pointer, count)
      requests.each_with_index do |req, i|
        raw = req.consume!
        array_mp.put_pointer(i * ::FFI::Pointer.size, raw)
      end
      [array_mp, count]
    end
    private_class_method :pack_requests

    # Idiomatic Ruby wrapper around a +BlazenBatchResult+ handle.
    #
    # Exposes the per-request items (a heterogeneous list of success
    # responses and failure-with-message records) plus the aggregated
    # token-usage and USD-cost totals.
    class BatchResult
      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "BatchResult: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_batch_result_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [Integer] number of response items
      def size
        Blazen::FFI.blazen_batch_result_responses_count(@ptr)
      end
      alias length size

      # Fetches the per-request result at +idx+.
      #
      # @param idx [Integer]
      # @return [Blazen::Batch::BatchItem]
      def at(idx)
        raw = Blazen::FFI.blazen_batch_result_responses_get(@ptr, Integer(idx))
        raise IndexError, "BatchResult: index #{idx} out of bounds (size=#{size})" if raw.nil? || raw.null?

        BatchItem.new(raw)
      end
      alias [] at

      # @yield [Blazen::Batch::BatchItem]
      # @return [Enumerator] when no block is given
      def each
        return enum_for(:each) unless block_given?

        size.times { |i| yield at(i) }
      end

      include Enumerable

      # @return [Array<Blazen::Batch::BatchItem>]
      def items
        each.to_a
      end
      alias responses items

      # @return [Float] aggregated USD cost across all responses
      def total_cost_usd
        Blazen::FFI.blazen_batch_result_total_cost_usd(@ptr)
      end

      # @return [::FFI::Pointer, nil] raw +BlazenTokenUsage+ handle, or
      #   +nil+ when no underlying provider reported usage. Most callers
      #   should reach for the convenience aggregations on each
      #   {BatchItem#response} instead.
      def total_usage_ptr
        raw = Blazen::FFI.blazen_batch_result_total_usage(@ptr)
        raw.null? ? nil : raw
      end

      # Decoded view of the aggregated token usage.
      #
      # @return [Hash{Symbol=>Integer}, nil] +nil+ when the underlying
      #   token-usage record is absent
      def total_usage
        raw = total_usage_ptr
        return nil if raw.nil?

        usage = {
          prompt_tokens:       Blazen::FFI.blazen_token_usage_prompt_tokens(raw),
          completion_tokens:   Blazen::FFI.blazen_token_usage_completion_tokens(raw),
          total_tokens:        Blazen::FFI.blazen_token_usage_total_tokens(raw),
          cached_input_tokens: Blazen::FFI.blazen_token_usage_cached_input_tokens(raw),
          reasoning_tokens:    Blazen::FFI.blazen_token_usage_reasoning_tokens(raw),
        }
        Blazen::FFI.blazen_token_usage_free(raw)
        usage
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenBatchItem+ — one of two
    # variants:
    #
    # - +success+: the corresponding completion finished and {#response}
    #   returns a freshly-cloned {Blazen::Llm::ModelResponse}.
    # - +failure+: the request errored out and {#failure_message} returns
    #   the error message; {#response} returns +nil+.
    class BatchItem
      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "BatchItem: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_batch_item_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [Symbol] +:success+ or +:failure+
      def kind
        Blazen::FFI.blazen_batch_item_kind(@ptr) == Blazen::FFI::BATCH_ITEM_SUCCESS ? :success : :failure
      end

      # @return [Boolean]
      def success?
        kind == :success
      end

      # @return [Boolean]
      def failure?
        kind == :failure
      end

      # Returns a freshly-cloned completion response when this item is the
      # +:success+ variant, +nil+ otherwise. The clone is wrapped in
      # {Blazen::Llm::ModelResponse} when that class is available
      # (Phase R7 Agent A); otherwise the raw pointer is returned with an
      # {::FFI::AutoPointer} drop hook so callers can still free it.
      #
      # @return [Blazen::Llm::ModelResponse, ::FFI::AutoPointer, nil]
      def response
        raw = Blazen::FFI.blazen_batch_item_success_response(@ptr)
        return nil if raw.nil? || raw.null?

        if defined?(Blazen::Llm) && defined?(Blazen::Llm::ModelResponse)
          Blazen::Llm::ModelResponse.new(raw)
        else
          ::FFI::AutoPointer.new(raw, Blazen::FFI.method(:blazen_model_response_free))
        end
      end

      # @return [String, nil] failure message when {#failure?}, else +nil+
      def failure_message
        return nil unless failure?

        raw = Blazen::FFI.blazen_batch_item_failure_message(@ptr)
        return nil if raw.nil? || raw.null?

        Blazen::FFI.consume_cstring(raw)
      end
    end
  end
end
