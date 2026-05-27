# frozen_string_literal: true

# Concrete per-engine embedding-provider classes (Part U).
#
# Engines:
#   * {Blazen::FastembedProvider}       — local ONNX-runtime (FastEmbed)
#   * {Blazen::TractEmbedProvider}      — local Tract (pure-Rust ONNX)
#   * {Blazen::CandleEmbedProvider}     — local Candle
#   * {Blazen::OpenAiEmbeddingProvider} — OpenAI text-embedding-3-*
#   * {Blazen::FalEmbeddingProvider}    — fal.ai-hosted embeddings
#
# The cabi result type for these providers is +BlazenEmbeddingVectors *+
# (a thin opaque carrying the embedded float arrays); see
# {Blazen::EmbeddingVectors} in +blazen/providers/base.rb+ for the Ruby
# wrapper.

module Blazen
  # @api private
  module EmbeddingProviderImpl
    private

    # Build a +const char *const *+ array from a Ruby +Array<String>+,
    # yielding +(array_ptr, length, retainers)+ to the block. The
    # caller must keep +retainers+ alive until after the cabi call.
    def with_text_array(texts)
      text_array = Array(texts)
      raise ArgumentError, "texts must be non-empty" if text_array.empty?

      c_strings = text_array.map { |t| ::FFI::MemoryPointer.from_string(t.to_s) }
      array_ptr = ::FFI::MemoryPointer.new(:pointer, c_strings.length)
      c_strings.each_with_index { |s, i| array_ptr[i].put_pointer(0, s) }
      result = yield(array_ptr, c_strings.length, c_strings)
      # Keep refs alive until after the call returns.
      c_strings = nil # rubocop:disable Lint/UselessAssignment
      array_ptr = nil # rubocop:disable Lint/UselessAssignment
      result
    end

    def embed_async_impl(texts, async_sym)
      with_text_array(texts) do |array_ptr, len, _retainers|
        fut = Blazen::FFI.public_send(async_sym, @handle, array_ptr, len)
        if fut.nil? || fut.null?
          raise Blazen::ValidationError, "#{async_sym} returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          out_result = ::FFI::MemoryPointer.new(:pointer)
          out_err    = ::FFI::MemoryPointer.new(:pointer)
          Blazen::FFI.blazen_future_take_embedding_vectors(f, out_result, out_err)
          Blazen::FFI.check_error!(out_err)
          Blazen::EmbeddingVectors.new(out_result.read_pointer)
        end
      end
    end

    def embed_blocking_impl(texts, blocking_sym)
      with_text_array(texts) do |array_ptr, len, _retainers|
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.public_send(blocking_sym, @handle, array_ptr, len, out_result, out_err)
        Blazen::FFI.check_error!(out_err)
        Blazen::EmbeddingVectors.new(out_result.read_pointer)
      end
    end
  end

  # Local FastEmbed (ONNX-runtime) embeddings.
  class FastembedProvider < EmbeddingProvider
    include EmbeddingProviderImpl

    PROVIDER_ID = "fastembed"

    # @param model_id [String, nil]
    # @param cache_dir [String, nil]
    def initialize(model_id: nil, cache_dir: nil)
      unless Blazen::FFI.respond_to?(:blazen_fastembed_provider_new)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'embed-fastembed' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(model_id) do |m|
        Blazen::FFI.with_cstring(cache_dir) do |cd|
          Blazen::FFI.blazen_fastembed_provider_new(m, cd, out_model, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_fastembed_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def embed(texts)
      embed_async_impl(texts, :blazen_fastembed_provider_embed)
    end

    def embed_blocking(texts)
      embed_blocking_impl(texts, :blazen_fastembed_provider_embed_blocking)
    end

    def dimensions
      Blazen::FFI.blazen_fastembed_provider_dimensions(@handle)
    end
  end

  # Local Tract (pure-Rust ONNX) embeddings.
  class TractEmbedProvider < EmbeddingProvider
    include EmbeddingProviderImpl

    PROVIDER_ID = "tract"

    # @param model_id [String, nil]
    # @param cache_dir [String, nil]
    def initialize(model_id: nil, cache_dir: nil)
      unless Blazen::FFI.respond_to?(:blazen_tract_embed_provider_new)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'embed-tract' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(model_id) do |m|
        Blazen::FFI.with_cstring(cache_dir) do |cd|
          Blazen::FFI.blazen_tract_embed_provider_new(m, cd, out_model, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_tract_embed_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def embed(texts)
      embed_async_impl(texts, :blazen_tract_embed_provider_embed)
    end

    def embed_blocking(texts)
      embed_blocking_impl(texts, :blazen_tract_embed_provider_embed_blocking)
    end

    def dimensions
      Blazen::FFI.blazen_tract_embed_provider_dimensions(@handle)
    end
  end

  # Local Candle embeddings.
  class CandleEmbedProvider < EmbeddingProvider
    include EmbeddingProviderImpl

    PROVIDER_ID = "candle"

    # @param model_id [String, nil]
    # @param cache_dir [String, nil]
    def initialize(model_id: nil, cache_dir: nil)
      unless Blazen::FFI.respond_to?(:blazen_candle_embed_provider_new)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'embed-candle' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(model_id) do |m|
        Blazen::FFI.with_cstring(cache_dir) do |cd|
          Blazen::FFI.blazen_candle_embed_provider_new(m, cd, out_model, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_candle_embed_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def embed(texts)
      embed_async_impl(texts, :blazen_candle_embed_provider_embed)
    end

    def embed_blocking(texts)
      embed_blocking_impl(texts, :blazen_candle_embed_provider_embed_blocking)
    end

    def dimensions
      Blazen::FFI.blazen_candle_embed_provider_dimensions(@handle)
    end
  end

  # OpenAI-hosted text embeddings.
  class OpenAiEmbeddingProvider < EmbeddingProvider
    include EmbeddingProviderImpl

    PROVIDER_ID = "openai-embedding"

    # @param api_key [String] empty string defers to +OPENAI_API_KEY+
    # @param model [String, nil] e.g. +"text-embedding-3-small"+
    def initialize(api_key:, model: nil)
      unless Blazen::FFI.respond_to?(:blazen_openai_embedding_provider_new)
        raise Blazen::UnsupportedError,
              "blazen cabi missing openai_embedding_provider_new symbol"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key.to_s) do |key|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.blazen_openai_embedding_provider_new(key, m, out_model, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer,
            Blazen::FFI.method(:blazen_openai_embedding_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def embed(texts)
      embed_async_impl(texts, :blazen_openai_embedding_provider_embed)
    end

    def embed_blocking(texts)
      embed_blocking_impl(texts, :blazen_openai_embedding_provider_embed_blocking)
    end

    def dimensions
      Blazen::FFI.blazen_openai_embedding_provider_dimensions(@handle)
    end
  end

  # fal.ai-hosted text embeddings.
  class FalEmbeddingProvider < EmbeddingProvider
    include EmbeddingProviderImpl

    PROVIDER_ID = "fal-embedding"

    # @param api_key [String]
    # @param model [String, nil] e.g. +"openai/text-embedding-3-small"+
    def initialize(api_key:, model: nil)
      unless Blazen::FFI.respond_to?(:blazen_fal_embedding_provider_new)
        raise Blazen::UnsupportedError,
              "blazen cabi missing fal_embedding_provider_new symbol"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key.to_s) do |key|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.blazen_fal_embedding_provider_new(key, m, out_model, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_fal_embedding_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def embed(texts)
      embed_async_impl(texts, :blazen_fal_embedding_provider_embed)
    end

    def embed_blocking(texts)
      embed_blocking_impl(texts, :blazen_fal_embedding_provider_embed_blocking)
    end

    def dimensions
      Blazen::FFI.blazen_fal_embedding_provider_dimensions(@handle)
    end
  end
end
