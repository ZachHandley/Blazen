# frozen_string_literal: true

require "json"

module Blazen
  # LLM record + model wrappers around the cabi opaque handles.
  #
  # Every record (Media, ToolCall, Tool, TokenUsage, ChatMessage,
  # CompletionRequest) and every response/model (CompletionResponse,
  # EmbeddingResponse, CompletionModel, EmbeddingModel) is wrapped in a small
  # Ruby class that owns its underlying +BlazenXxx *+ via
  # {::FFI::AutoPointer}, so Ruby GC reclaims the native allocation
  # automatically.
  #
  # Record handles that get "pushed" into a parent record (e.g. media parts
  # appended to a chat message) are consumed by the +_push+ call. Use
  # {LlmHandle#consume!} to extract the raw pointer and disable auto-free
  # before passing it across; the wrapper classes already do this internally
  # for you when you construct one record from another (e.g. passing
  # +media_parts: [Media.new(...)]+ to {ChatMessage.new}).
  module Llm
    module_function

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    # Mixin providing pointer-ownership plumbing used by every wrapper class
    # below: GC-managed {::FFI::AutoPointer} wrapping, {#consume!} for
    # transferring ownership to a +_push+ call, and {#ptr} for read access.
    #
    # @api private
    module Handle
      # @return [::FFI::AutoPointer, ::FFI::Pointer] the underlying native
      #   handle. Wrapped in {::FFI::AutoPointer} while owned; replaced by a
      #   bare {::FFI::Pointer} after {#consume!}.
      attr_reader :ptr

      # Surrenders ownership of the underlying handle to a caller that takes
      # responsibility for freeing it (typically a cabi +_push+ function that
      # consumes the value). Returns the raw +::FFI::Pointer+; subsequent
      # accesses to this wrapper raise an error.
      #
      # @return [::FFI::Pointer] the raw pointer (now caller-owned)
      def consume!
        raise Blazen::InternalError, "handle already consumed" if @consumed

        @consumed = true
        raw = @ptr.is_a?(::FFI::AutoPointer) ? @ptr.address : @ptr.address
        bare = ::FFI::Pointer.new(raw)
        # Disable the AutoPointer's finalizer so it doesn't double-free.
        @ptr.autorelease = false if @ptr.is_a?(::FFI::AutoPointer)
        @ptr = bare
        bare
      end

      # @return [Boolean] true if {#consume!} has already been called
      def consumed?
        @consumed == true
      end

      private

      # Installs +raw_ptr+ as the wrapper's GC-managed handle, raising
      # {Blazen::InternalError} if +raw_ptr+ is null. +finalizer+ is the
      # cabi +blazen_<type>_free+ function bound on {Blazen::FFI}.
      def set_handle!(raw_ptr, finalizer)
        if raw_ptr.nil? || raw_ptr.null?
          raise Blazen::InternalError, "#{self.class.name}: native constructor returned null"
        end

        @consumed = false
        @ptr = ::FFI::AutoPointer.new(raw_ptr, finalizer)
      end
    end

    # ---------------------------------------------------------------------
    # Records
    # ---------------------------------------------------------------------

    # An inline media payload (image / audio / video) attached to a chat
    # message. Wraps +BlazenMedia *+.
    class Media
      include Handle

      # @param kind [String] media kind, e.g. +"image"+, +"audio"+
      # @param mime_type [String] MIME type, e.g. +"image/png"+
      # @param data_base64 [String] base64-encoded payload
      def initialize(kind:, mime_type:, data_base64:)
        raw =
          Blazen::FFI.with_cstring(kind) do |k|
            Blazen::FFI.with_cstring(mime_type) do |m|
              Blazen::FFI.with_cstring(data_base64) do |d|
                Blazen::FFI.blazen_media_new(k, m, d)
              end
            end
          end
        set_handle!(raw, Blazen::FFI.method(:blazen_media_free))
      end

      # @return [String, nil]
      def kind
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_media_kind(@ptr))
      end

      # @return [String, nil]
      def mime_type
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_media_mime_type(@ptr))
      end

      # @return [String, nil]
      def data_base64
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_media_data_base64(@ptr))
      end
    end

    # An assistant-emitted invocation of a tool/function. Wraps
    # +BlazenToolCall *+.
    class ToolCall
      include Handle

      # @param id [String] opaque id assigned by the model
      # @param name [String] tool name
      # @param arguments_json [String, Hash] JSON-encoded arguments (Hash
      #   inputs are serialized via +JSON.dump+)
      def initialize(id:, name:, arguments_json:)
        args = arguments_json.is_a?(String) ? arguments_json : JSON.dump(arguments_json)
        raw =
          Blazen::FFI.with_cstring(id) do |i|
            Blazen::FFI.with_cstring(name) do |n|
              Blazen::FFI.with_cstring(args) do |a|
                Blazen::FFI.blazen_tool_call_new(i, n, a)
              end
            end
          end
        set_handle!(raw, Blazen::FFI.method(:blazen_tool_call_free))
      end

      # Constructs a ToolCall wrapper around an already-allocated handle.
      # Used internally when reading tool calls out of a CompletionResponse.
      #
      # @api private
      # @param raw_ptr [::FFI::Pointer] caller-owned +BlazenToolCall *+
      # @return [ToolCall]
      def self.from_raw(raw_ptr)
        tc = allocate
        tc.send(:set_handle!, raw_ptr, Blazen::FFI.method(:blazen_tool_call_free))
        tc
      end

      # @return [String, nil]
      def id
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_tool_call_id(@ptr))
      end

      # @return [String, nil]
      def name
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_tool_call_name(@ptr))
      end

      # @return [String, nil] JSON-encoded arguments
      def arguments_json
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_tool_call_arguments_json(@ptr))
      end

      # @return [Hash, Array, nil] parsed arguments (returns +nil+ on a null
      #   accessor; +{}+ on an empty string; otherwise the +JSON.parse+ result)
      def arguments
        raw = arguments_json
        return nil if raw.nil?
        return {} if raw.empty?

        JSON.parse(raw)
      end
    end

    # A tool/function declaration provided to the model in a request. Wraps
    # +BlazenTool *+.
    class Tool
      include Handle

      # @param name [String] tool name (must match what the model is told)
      # @param description [String] human-readable description for the model
      # @param parameters_json [String, Hash] JSON-Schema for parameters
      #   (Hash inputs are serialized via +JSON.dump+)
      def initialize(name:, description:, parameters_json:)
        params = parameters_json.is_a?(String) ? parameters_json : JSON.dump(parameters_json)
        raw =
          Blazen::FFI.with_cstring(name) do |n|
            Blazen::FFI.with_cstring(description) do |d|
              Blazen::FFI.with_cstring(params) do |p|
                Blazen::FFI.blazen_tool_new(n, d, p)
              end
            end
          end
        set_handle!(raw, Blazen::FFI.method(:blazen_tool_free))
      end

      # @return [String, nil]
      def name
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_tool_name(@ptr))
      end

      # @return [String, nil]
      def description
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_tool_description(@ptr))
      end

      # @return [String, nil]
      def parameters_json
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_tool_parameters_json(@ptr))
      end
    end

    # Per-response token-usage counters. Wraps +BlazenTokenUsage *+.
    class TokenUsage
      include Handle

      # @param prompt_tokens [Integer]
      # @param completion_tokens [Integer]
      # @param total_tokens [Integer]
      # @param cached_input_tokens [Integer]
      # @param reasoning_tokens [Integer]
      def initialize(prompt_tokens: 0, completion_tokens: 0, total_tokens: 0,
                     cached_input_tokens: 0, reasoning_tokens: 0)
        raw = Blazen::FFI.blazen_token_usage_new(
          prompt_tokens.to_i,
          completion_tokens.to_i,
          total_tokens.to_i,
          cached_input_tokens.to_i,
          reasoning_tokens.to_i,
        )
        set_handle!(raw, Blazen::FFI.method(:blazen_token_usage_free))
      end

      # Wraps an already-allocated +BlazenTokenUsage *+ — used when reading
      # usage off a response.
      #
      # @api private
      # @param raw_ptr [::FFI::Pointer] caller-owned +BlazenTokenUsage *+
      # @return [TokenUsage]
      def self.from_raw(raw_ptr)
        tu = allocate
        tu.send(:set_handle!, raw_ptr, Blazen::FFI.method(:blazen_token_usage_free))
        tu
      end

      # @return [Integer]
      def prompt_tokens
        Blazen::FFI.blazen_token_usage_prompt_tokens(@ptr)
      end

      # @return [Integer]
      def completion_tokens
        Blazen::FFI.blazen_token_usage_completion_tokens(@ptr)
      end

      # @return [Integer]
      def total_tokens
        Blazen::FFI.blazen_token_usage_total_tokens(@ptr)
      end

      # @return [Integer]
      def cached_input_tokens
        Blazen::FFI.blazen_token_usage_cached_input_tokens(@ptr)
      end

      # @return [Integer]
      def reasoning_tokens
        Blazen::FFI.blazen_token_usage_reasoning_tokens(@ptr)
      end
    end

    # A single conversation message. Wraps +BlazenChatMessage *+.
    #
    # The constructor consumes any +Media+ / +ToolCall+ wrappers passed via
    # +media_parts+ / +tool_calls+: the cabi +_push+ calls take ownership of
    # the inner record, so we call {#consume!} on each before pushing.
    class ChatMessage
      include Handle

      # @param role [String] +"system"+, +"user"+, +"assistant"+, or +"tool"+
      # @param content [String] message text
      # @param media_parts [Array<Blazen::Llm::Media>] inline media parts
      #   (consumed)
      # @param tool_calls [Array<Blazen::Llm::ToolCall>] tool-call
      #   invocations (consumed)
      # @param tool_call_id [String, nil] tool-call id this message replies to
      # @param name [String, nil] author name (when distinct from +role+)
      def initialize(role:, content:, media_parts: [], tool_calls: [],
                     tool_call_id: nil, name: nil)
        raw =
          Blazen::FFI.with_cstring(role) do |r|
            Blazen::FFI.with_cstring(content) do |c|
              Blazen::FFI.blazen_chat_message_new(r, c)
            end
          end
        set_handle!(raw, Blazen::FFI.method(:blazen_chat_message_free))

        media_parts.each do |media|
          raise ArgumentError, "media_parts entries must be Blazen::Llm::Media" \
            unless media.is_a?(Media)

          Blazen::FFI.blazen_chat_message_media_parts_push(@ptr, media.consume!)
        end

        tool_calls.each do |tc|
          raise ArgumentError, "tool_calls entries must be Blazen::Llm::ToolCall" \
            unless tc.is_a?(ToolCall)

          Blazen::FFI.blazen_chat_message_tool_calls_push(@ptr, tc.consume!)
        end

        unless tool_call_id.nil?
          Blazen::FFI.with_cstring(tool_call_id) do |tcid|
            Blazen::FFI.blazen_chat_message_set_tool_call_id(@ptr, tcid)
          end
        end

        return if name.nil?

        Blazen::FFI.with_cstring(name) do |n|
          Blazen::FFI.blazen_chat_message_set_name(@ptr, n)
        end
      end

      # @return [String, nil]
      def role
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_chat_message_role(@ptr))
      end

      # @return [String, nil]
      def content
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_chat_message_content(@ptr))
      end

      # @return [String, nil]
      def tool_call_id
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_chat_message_tool_call_id(@ptr))
      end

      # @return [String, nil]
      def name
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_chat_message_name(@ptr))
      end

      # @return [Array<Blazen::Llm::Media>] freshly-cloned media parts
      def media_parts
        count = Blazen::FFI.blazen_chat_message_media_parts_count(@ptr)
        Array.new(count) do |i|
          raw = Blazen::FFI.blazen_chat_message_media_parts_get(@ptr, i)
          media = Media.allocate
          media.send(:set_handle!, raw, Blazen::FFI.method(:blazen_media_free))
          media
        end
      end

      # @return [Array<Blazen::Llm::ToolCall>] freshly-cloned tool calls
      def tool_calls
        count = Blazen::FFI.blazen_chat_message_tool_calls_count(@ptr)
        Array.new(count) do |i|
          raw = Blazen::FFI.blazen_chat_message_tool_calls_get(@ptr, i)
          ToolCall.from_raw(raw)
        end
      end
    end

    # A pending completion request. Wraps +BlazenCompletionRequest *+.
    #
    # Like {ChatMessage}, this consumes any +ChatMessage+ / +Tool+ wrappers
    # passed in — the cabi +_push+ calls take ownership.
    class CompletionRequest
      include Handle

      # @param messages [Array<Blazen::Llm::ChatMessage>] conversation
      #   history (consumed)
      # @param tools [Array<Blazen::Llm::Tool>] available tools (consumed)
      # @param temperature [Float, nil] sampling temperature
      # @param max_tokens [Integer, nil] response token cap
      # @param top_p [Float, nil] nucleus-sampling cutoff
      # @param model [String, nil] override the model for this request
      # @param response_format_json [String, nil] JSON-schema string for
      #   structured output (provider-specific)
      # @param system [String, nil] system prompt override
      def initialize(messages:, tools: [], temperature: nil, max_tokens: nil,
                     top_p: nil, model: nil, response_format_json: nil, system: nil)
        raw = Blazen::FFI.blazen_completion_request_new
        set_handle!(raw, Blazen::FFI.method(:blazen_completion_request_free))

        messages.each do |msg|
          raise ArgumentError, "messages entries must be Blazen::Llm::ChatMessage" \
            unless msg.is_a?(ChatMessage)

          Blazen::FFI.blazen_completion_request_messages_push(@ptr, msg.consume!)
        end

        tools.each do |tool|
          raise ArgumentError, "tools entries must be Blazen::Llm::Tool" \
            unless tool.is_a?(Tool)

          Blazen::FFI.blazen_completion_request_tools_push(@ptr, tool.consume!)
        end

        Blazen::FFI.blazen_completion_request_set_temperature(@ptr, temperature.to_f) \
          unless temperature.nil?
        Blazen::FFI.blazen_completion_request_set_max_tokens(@ptr, max_tokens.to_i) \
          unless max_tokens.nil?
        Blazen::FFI.blazen_completion_request_set_top_p(@ptr, top_p.to_f) \
          unless top_p.nil?

        unless model.nil?
          Blazen::FFI.with_cstring(model) do |m|
            Blazen::FFI.blazen_completion_request_set_model(@ptr, m)
          end
        end

        unless response_format_json.nil?
          Blazen::FFI.with_cstring(response_format_json) do |rf|
            Blazen::FFI.blazen_completion_request_set_response_format_json(@ptr, rf)
          end
        end

        return if system.nil?

        Blazen::FFI.with_cstring(system) do |s|
          Blazen::FFI.blazen_completion_request_set_system(@ptr, s)
        end
      end
    end

    # The result of a completion call. Wraps +BlazenCompletionResponse *+.
    class CompletionResponse
      include Handle

      # @api private
      # @param raw_ptr [::FFI::Pointer] caller-owned
      #   +BlazenCompletionResponse *+ produced by the cabi surface
      def initialize(raw_ptr)
        set_handle!(raw_ptr, Blazen::FFI.method(:blazen_completion_response_free))
      end

      # @return [String, nil]
      def content
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_completion_response_content(@ptr))
      end

      # @return [String, nil]
      def finish_reason
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_completion_response_finish_reason(@ptr))
      end

      # @return [String, nil]
      def model
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_completion_response_model(@ptr))
      end

      # @return [Array<Blazen::Llm::ToolCall>] freshly-cloned tool calls
      def tool_calls
        count = Blazen::FFI.blazen_completion_response_tool_calls_count(@ptr)
        Array.new(count) do |i|
          raw = Blazen::FFI.blazen_completion_response_tool_calls_get(@ptr, i)
          ToolCall.from_raw(raw)
        end
      end

      # @return [Blazen::Llm::TokenUsage, nil]
      def usage
        raw = Blazen::FFI.blazen_completion_response_usage(@ptr)
        return nil if raw.nil? || raw.null?

        TokenUsage.from_raw(raw)
      end
    end

    # The result of an embedding call. Wraps +BlazenEmbeddingResponse *+.
    class EmbeddingResponse
      include Handle

      # @api private
      # @param raw_ptr [::FFI::Pointer] caller-owned
      #   +BlazenEmbeddingResponse *+ produced by the cabi surface
      def initialize(raw_ptr)
        set_handle!(raw_ptr, Blazen::FFI.method(:blazen_embedding_response_free))
      end

      # @return [Integer] number of embedding vectors in the response
      def embeddings_count
        Blazen::FFI.blazen_embedding_response_embeddings_count(@ptr)
      end

      # @return [Integer] dimensionality of the +vec_idx+-th vector
      def embedding_dim(vec_idx)
        Blazen::FFI.blazen_embedding_response_embedding_dim(@ptr, vec_idx.to_i)
      end

      # Returns the +vec_idx+-th embedding vector as a Ruby Array of Floats.
      # Uses the bulk-copy accessor for O(1) FFI overhead.
      #
      # @param vec_idx [Integer]
      # @return [Array<Float>]
      def embedding(vec_idx)
        dim = embedding_dim(vec_idx)
        return [] if dim.zero?

        buf = ::FFI::MemoryPointer.new(:double, dim)
        Blazen::FFI.blazen_embedding_response_embedding_to_buffer(@ptr, vec_idx.to_i, buf, dim)
        buf.read_array_of_double(dim)
      end

      # @return [Array<Array<Float>>] every embedding vector in the response
      def embeddings
        Array.new(embeddings_count) { |i| embedding(i) }
      end

      # @return [String, nil]
      def model
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_embedding_response_model(@ptr))
      end

      # @return [Blazen::Llm::TokenUsage, nil]
      def usage
        raw = Blazen::FFI.blazen_embedding_response_usage(@ptr)
        return nil if raw.nil? || raw.null?

        TokenUsage.from_raw(raw)
      end
    end

    # ---------------------------------------------------------------------
    # Models
    # ---------------------------------------------------------------------

    # A configured chat-completion model bound to a specific provider. Wraps
    # +BlazenCompletionModel *+.
    #
    # Instances are constructed via {Blazen::Providers} factory methods
    # (e.g. {Blazen::Providers.openai}); user code shouldn't call +.new+
    # directly.
    class CompletionModel
      include Handle

      # @api private
      # @param raw_ptr [::FFI::Pointer] caller-owned +BlazenCompletionModel *+
      def initialize(raw_ptr)
        set_handle!(raw_ptr, Blazen::FFI.method(:blazen_completion_model_free))
      end

      # @return [String, nil] model identifier reported by the provider
      def model_id
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_completion_model_model_id(@ptr))
      end

      # Issues a synchronous completion call, blocking the current thread
      # until the provider returns. {Blazen::FFI.check_error!} translates
      # cabi failures into the matching {Blazen::Error} subclass.
      #
      # @param request [Blazen::Llm::CompletionRequest] consumed by the call
      # @return [Blazen::Llm::CompletionResponse]
      def complete_blocking(request)
        raise ArgumentError, "request must be Blazen::Llm::CompletionRequest" \
          unless request.is_a?(CompletionRequest)

        out_resp = ::FFI::MemoryPointer.new(:pointer)
        out_err = ::FFI::MemoryPointer.new(:pointer)
        req_ptr = request.consume!
        Blazen::FFI.blazen_completion_model_complete_blocking(@ptr, req_ptr, out_resp, out_err)
        Blazen::FFI.check_error!(out_err)
        CompletionResponse.new(out_resp.read_pointer)
      end

      # Issues an asynchronous completion call, returning a {Blazen::Llm::CompletionResponse}
      # when the future resolves. Composes with +Fiber.scheduler+ when one
      # is active (see {Blazen::FFI.await_future}).
      #
      # @param request [Blazen::Llm::CompletionRequest] consumed by the call
      # @return [Blazen::Llm::CompletionResponse]
      def complete(request)
        raise ArgumentError, "request must be Blazen::Llm::CompletionRequest" \
          unless request.is_a?(CompletionRequest)

        req_ptr = request.consume!
        fut = Blazen::FFI.blazen_completion_model_complete(@ptr, req_ptr)
        Blazen::FFI.await_future(fut) do |f|
          out_resp = ::FFI::MemoryPointer.new(:pointer)
          out_err = ::FFI::MemoryPointer.new(:pointer)
          Blazen::FFI.blazen_future_take_completion_response(f, out_resp, out_err)
          Blazen::FFI.check_error!(out_err)
          CompletionResponse.new(out_resp.read_pointer)
        end
      end
    end

    # A configured embedding model bound to a specific provider. Wraps
    # +BlazenEmbeddingModel *+.
    #
    # Instances are constructed via {Blazen::Providers} factory methods
    # (e.g. {Blazen::Providers.openai_embedding}).
    class EmbeddingModel
      include Handle

      # @api private
      # @param raw_ptr [::FFI::Pointer] caller-owned +BlazenEmbeddingModel *+
      def initialize(raw_ptr)
        set_handle!(raw_ptr, Blazen::FFI.method(:blazen_embedding_model_free))
      end

      # @return [String, nil] model identifier reported by the provider
      def model_id
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_embedding_model_model_id(@ptr))
      end

      # @return [Integer] declared output dimensionality (may be +0+ if
      #   the model reports it as variable)
      def dimensions
        Blazen::FFI.blazen_embedding_model_dimensions(@ptr)
      end

      # Synchronously embeds a list of input texts.
      #
      # @param texts [Array<String>] non-empty list of strings to embed
      # @return [Blazen::Llm::EmbeddingResponse]
      def embed_blocking(texts)
        text_array = Array(texts)
        raise ArgumentError, "texts must be non-empty" if text_array.empty?

        # Build the +const char *const *+ array — keep each MemoryPointer
        # alive until after the cabi call returns.
        c_strings = text_array.map { |t| ::FFI::MemoryPointer.from_string(t.to_s) }
        array_ptr = ::FFI::MemoryPointer.new(:pointer, c_strings.length)
        c_strings.each_with_index { |s, i| array_ptr[i].put_pointer(0, s) }

        out_resp = ::FFI::MemoryPointer.new(:pointer)
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_embedding_model_embed_blocking(
          @ptr, array_ptr, c_strings.length, out_resp, out_err
        )
        Blazen::FFI.check_error!(out_err)
        # Keep c_strings + array_ptr alive across the call (a no-op
        # assignment here is enough to retain the reference).
        c_strings = nil # rubocop:disable Lint/UselessAssignment
        array_ptr = nil # rubocop:disable Lint/UselessAssignment
        EmbeddingResponse.new(out_resp.read_pointer)
      end

      # Asynchronously embeds a list of input texts.
      #
      # @param texts [Array<String>]
      # @return [Blazen::Llm::EmbeddingResponse]
      def embed(texts)
        text_array = Array(texts)
        raise ArgumentError, "texts must be non-empty" if text_array.empty?

        c_strings = text_array.map { |t| ::FFI::MemoryPointer.from_string(t.to_s) }
        array_ptr = ::FFI::MemoryPointer.new(:pointer, c_strings.length)
        c_strings.each_with_index { |s, i| array_ptr[i].put_pointer(0, s) }

        fut = Blazen::FFI.blazen_embedding_model_embed(@ptr, array_ptr, c_strings.length)
        # Retain the input buffers for the duration of the future — the
        # cabi side may still be reading them while the future polls.
        Blazen::FFI.await_future(fut) do |f|
          out_resp = ::FFI::MemoryPointer.new(:pointer)
          out_err = ::FFI::MemoryPointer.new(:pointer)
          Blazen::FFI.blazen_future_take_embedding_response(f, out_resp, out_err)
          Blazen::FFI.check_error!(out_err)
          c_strings = nil # rubocop:disable Lint/UselessAssignment
          array_ptr = nil # rubocop:disable Lint/UselessAssignment
          EmbeddingResponse.new(out_resp.read_pointer)
        end
      end
    end

    # ---------------------------------------------------------------------
    # Convenience builders (public module functions)
    # ---------------------------------------------------------------------

    # Builds a {Blazen::Llm::ChatMessage}.
    #
    # @param role [String] +"system"+, +"user"+, +"assistant"+, or +"tool"+
    # @param content [String]
    # @param media_parts [Array<Blazen::Llm::Media>]
    # @param tool_calls [Array<Blazen::Llm::ToolCall>]
    # @param tool_call_id [String, nil]
    # @param name [String, nil]
    # @return [Blazen::Llm::ChatMessage]
    def message(role:, content:, media_parts: [], tool_calls: [],
                tool_call_id: nil, name: nil)
      ChatMessage.new(
        role: role,
        content: content,
        media_parts: media_parts,
        tool_calls: tool_calls,
        tool_call_id: tool_call_id,
        name: name,
      )
    end

    # Builds a +"system"+-role {Blazen::Llm::ChatMessage}.
    #
    # @param content [String]
    # @return [Blazen::Llm::ChatMessage]
    def system(content)
      message(role: "system", content: content)
    end

    # Builds a +"user"+-role {Blazen::Llm::ChatMessage}.
    #
    # @param content [String]
    # @param media_parts [Array<Blazen::Llm::Media>]
    # @return [Blazen::Llm::ChatMessage]
    def user(content, media_parts: [])
      message(role: "user", content: content, media_parts: media_parts)
    end

    # Builds an +"assistant"+-role {Blazen::Llm::ChatMessage}.
    #
    # @param content [String]
    # @param tool_calls [Array<Blazen::Llm::ToolCall>]
    # @return [Blazen::Llm::ChatMessage]
    def assistant(content, tool_calls: [])
      message(role: "assistant", content: content, tool_calls: tool_calls)
    end

    # Builds a +"tool"+-role {Blazen::Llm::ChatMessage}.
    #
    # @param content [String] tool result, serialized as a string
    # @param tool_call_id [String] id of the tool call this responds to
    # @param name [String, nil] tool name
    # @return [Blazen::Llm::ChatMessage]
    def tool_result(content:, tool_call_id:, name: nil)
      message(role: "tool", content: content, tool_call_id: tool_call_id, name: name)
    end

    # Builds a {Blazen::Llm::CompletionRequest}.
    #
    # @param messages [Array<Blazen::Llm::ChatMessage>]
    # @param tools [Array<Blazen::Llm::Tool>]
    # @param temperature [Float, nil]
    # @param max_tokens [Integer, nil]
    # @param top_p [Float, nil]
    # @param model [String, nil]
    # @param response_format_json [String, nil]
    # @param system [String, nil]
    # @return [Blazen::Llm::CompletionRequest]
    def completion_request(messages:, tools: [], temperature: nil, max_tokens: nil,
                           top_p: nil, model: nil, response_format_json: nil, system: nil)
      CompletionRequest.new(
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

    # Builds a {Blazen::Llm::Tool} declaration.
    #
    # @param name [String]
    # @param description [String]
    # @param parameters [Hash, String] JSON-Schema (Hash → +JSON.dump+'d)
    # @return [Blazen::Llm::Tool]
    def tool(name:, description:, parameters:)
      params_json = parameters.is_a?(String) ? parameters : JSON.dump(parameters)
      Tool.new(name: name, description: description, parameters_json: params_json)
    end

    # Builds a {Blazen::Llm::Media} part.
    #
    # @param kind [String] e.g. +"image"+, +"audio"+
    # @param mime_type [String]
    # @param data_base64 [String]
    # @return [Blazen::Llm::Media]
    def media(kind:, mime_type:, data_base64:)
      Media.new(kind: kind, mime_type: mime_type, data_base64: data_base64)
    end

    # Builds a {Blazen::Llm::ToolCall}.
    #
    # @param id [String]
    # @param name [String]
    # @param arguments_json [String, Hash]
    # @return [Blazen::Llm::ToolCall]
    def tool_call(id:, name:, arguments_json:)
      ToolCall.new(id: id, name: name, arguments_json: arguments_json)
    end

    # Builds a {Blazen::Llm::TokenUsage}.
    #
    # @return [Blazen::Llm::TokenUsage]
    def token_usage(prompt_tokens: 0, completion_tokens: 0, total_tokens: 0,
                    cached_input_tokens: 0, reasoning_tokens: 0)
      TokenUsage.new(
        prompt_tokens: prompt_tokens,
        completion_tokens: completion_tokens,
        total_tokens: total_tokens,
        cached_input_tokens: cached_input_tokens,
        reasoning_tokens: reasoning_tokens,
      )
    end
  end

  # Aliases at the top-level Blazen namespace mirror the legacy UniFFI-era
  # constant names so existing user code (and helper modules that haven't
  # been rewritten yet) keep working with +Blazen::ChatMessage+,
  # +Blazen::CompletionModel+, etc.
  ChatMessage        = Llm::ChatMessage        unless const_defined?(:ChatMessage, false)
  CompletionRequest  = Llm::CompletionRequest  unless const_defined?(:CompletionRequest, false)
  CompletionResponse = Llm::CompletionResponse unless const_defined?(:CompletionResponse, false)
  CompletionModel    = Llm::CompletionModel    unless const_defined?(:CompletionModel, false)
  EmbeddingResponse  = Llm::EmbeddingResponse  unless const_defined?(:EmbeddingResponse, false)
  EmbeddingModel     = Llm::EmbeddingModel     unless const_defined?(:EmbeddingModel, false)
  Media              = Llm::Media              unless const_defined?(:Media, false)
  Tool               = Llm::Tool               unless const_defined?(:Tool, false)
  ToolCall           = Llm::ToolCall           unless const_defined?(:ToolCall, false)
  TokenUsage         = Llm::TokenUsage         unless const_defined?(:TokenUsage, false)
end
