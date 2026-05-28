# frozen_string_literal: true

# Concrete per-engine chat-completion provider classes (Part U).
#
# Cloud engines (15):
#   * {Blazen::OpenAiProvider}      — OpenAI / OpenAI-compatible
#   * {Blazen::AnthropicProvider}   — Anthropic Claude
#   * {Blazen::GeminiProvider}      — Google Gemini
#   * {Blazen::AzureOpenAiProvider} — Azure OpenAI deployments
#   * {Blazen::BedrockProvider}     — AWS Bedrock
#   * {Blazen::FalLlmProvider}      — fal.ai-hosted LLMs
#   * {Blazen::MistralProvider}     — Mistral AI
#   * {Blazen::FireworksProvider}   — Fireworks AI
#   * {Blazen::DeepSeekProvider}    — DeepSeek
#   * {Blazen::PerplexityProvider}  — Perplexity
#   * {Blazen::TogetherProvider}    — Together AI
#   * {Blazen::GroqProvider}        — Groq
#   * {Blazen::OpenRouterProvider}  — OpenRouter
#   * {Blazen::CohereProvider}      — Cohere
#   * {Blazen::XaiProvider}         — xAI (Grok)
#
# Local / OpenAI-compatible engines (6):
#   * {Blazen::OpenAiCompatProvider} — generic OpenAI-protocol endpoint
#   * {Blazen::OllamaProvider}       — Ollama local server
#   * {Blazen::LmStudioProvider}     — LM Studio local server
#   * {Blazen::MistralRsProvider}    — mistral.rs native engine
#   * {Blazen::LlamaCppProvider}     — llama.cpp native engine
#   * {Blazen::CandleLlmProvider}    — Candle native engine

module Blazen
  # @api private
  module LlmProviderImpl
    # Calls the per-engine +blazen_<engine>_provider_as_llm_provider+ C
    # function to obtain a polymorphic +BlazenLlmProvider *+ handle that
    # can be passed to {Blazen::Agents.new} / {Blazen::Batch.complete}.
    # The returned {Blazen::LlmProvider} owns its own +AutoPointer+ with a
    # +blazen_llm_provider_free+ finalizer.
    #
    # @return [Blazen::LlmProvider]
    def as_llm_provider
      sym = self.class::AS_LLM_PROVIDER_SYM
      unless Blazen::FFI.respond_to?(sym)
        raise Blazen::UnsupportedError, "blazen cabi missing #{sym} symbol"
      end

      raw = Blazen::FFI.public_send(sym, @handle)
      if raw.nil? || raw.null?
        raise Blazen::InternalError, "#{sym} returned a null pointer"
      end

      Blazen::LlmProvider.new(
        ::FFI::AutoPointer.new(raw, Blazen::FFI.method(:blazen_llm_provider_free)),
      )
    end

    private

    # The completion cabi entry points consume the +BlazenModelRequest+
    # pointer (ownership transfers). We extract the raw handle via
    # +ModelRequest#consume!+ which detaches the AutoPointer's finalizer.
    def llm_complete(request, async_sym)
      unless request.is_a?(Blazen::Llm::ModelRequest)
        raise ArgumentError, "request must be a Blazen::Llm::ModelRequest"
      end
      unless Blazen::FFI.respond_to?(:blazen_future_take_model_response)
        raise Blazen::UnsupportedError,
              "blazen cabi does not export blazen_future_take_model_response; " \
                "use #complete_blocking instead"
      end

      req_ptr = request.consume!
      fut = Blazen::FFI.public_send(async_sym, @handle, req_ptr)
      if fut.nil? || fut.null?
        raise Blazen::ValidationError, "#{async_sym} returned a null future"
      end

      out_resp = ::FFI::MemoryPointer.new(:pointer)
      out_err  = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.await_future(fut) do |f|
        Blazen::FFI.blazen_future_take_model_response(f, out_resp, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      Blazen::Llm::ModelResponse.new(out_resp.read_pointer)
    end

    def llm_complete_blocking(request, blocking_sym)
      unless request.is_a?(Blazen::Llm::ModelRequest)
        raise ArgumentError, "request must be a Blazen::Llm::ModelRequest"
      end

      req_ptr  = request.consume!
      out_resp = ::FFI::MemoryPointer.new(:pointer)
      out_err  = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.public_send(blocking_sym, @handle, req_ptr, out_resp, out_err)
      Blazen::FFI.check_error!(out_err)
      Blazen::Llm::ModelResponse.new(out_resp.read_pointer)
    end

    # Drives a streaming chat completion through this provider's own
    # +blazen_<engine>_provider_complete_streaming[_blocking]+ symbol. The
    # +request+ is consumed by the call. See {Blazen::Streaming} for the
    # event/callback contract (+:chunk+ / +:done+ / +:error+).
    def llm_stream(request, sym, blocking:, on_chunk: nil, on_done: nil, on_error: nil, &block)
      unless Blazen::FFI.respond_to?(sym)
        raise Blazen::UnsupportedError, "blazen cabi missing #{sym} symbol"
      end

      Blazen::Streaming.drive_completion(
        @handle, request, sym, blocking: blocking,
        on_chunk: on_chunk, on_done: on_done, on_error: on_error, &block
      )
    end
  end

  # @api private
  # Defines a standard +api_key+/+model+/+base_url+ LLM provider class. The
  # cabi factory has signature
  # +(const char *api_key, const char *model, const char *base_url,
  # Blazen<X>Provider **out_model, BlazenError **out_err) -> int32_t+.
  def self._define_llm_provider_3str(klass_name, provider_id, factory_sym,
                                     free_sym, complete_sym, complete_blocking_sym)
    base                = factory_sym.to_s.sub(/_new\z/, "")
    stream_sym          = :"#{base}_complete_streaming"
    stream_blocking_sym = :"#{base}_complete_streaming_blocking"
    as_llm_sym          = :"#{base}_as_llm_provider"
    klass = Class.new(LlmProvider) do
      include LlmProviderImpl

      const_set(:PROVIDER_ID, provider_id)
      const_set(:AS_LLM_PROVIDER_SYM, as_llm_sym)

      define_method(:initialize) do |api_key:, model: nil, base_url: nil|
        unless Blazen::FFI.respond_to?(factory_sym)
          raise Blazen::UnsupportedError,
                "blazen cabi missing #{factory_sym} symbol"
        end

        out_model = ::FFI::MemoryPointer.new(:pointer)
        out_err   = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(api_key.to_s) do |key|
          Blazen::FFI.with_cstring(model) do |m|
            Blazen::FFI.with_cstring(base_url) do |bu|
              Blazen::FFI.public_send(factory_sym, key, m, bu, out_model, out_err)
            end
          end
        end
        Blazen::FFI.check_error!(out_err)
        super(out_model.read_pointer, Blazen::FFI.method(free_sym))
      end

      define_method(:provider_id) { self.class::PROVIDER_ID }

      define_method(:complete) { |request| llm_complete(request, complete_sym) }
      define_method(:complete_blocking) do |request|
        llm_complete_blocking(request, complete_blocking_sym)
      end

      define_method(:stream) do |request, **opts, &block|
        llm_stream(request, stream_blocking_sym, blocking: true, **opts, &block)
      end
      define_method(:stream_async) do |request, **opts, &block|
        llm_stream(request, stream_sym, blocking: false, **opts, &block)
      end
    end
    const_set(klass_name, klass)
  end

  # @api private
  # Defines a standard +api_key+/+model+ LLM provider class. The cabi
  # factory has signature
  # +(const char *api_key, const char *model, Blazen<X>Provider **out_model,
  # BlazenError **out_err) -> int32_t+.
  def self._define_llm_provider_2str(klass_name, provider_id, factory_sym,
                                     free_sym, complete_sym, complete_blocking_sym)
    base                = factory_sym.to_s.sub(/_new\z/, "")
    stream_sym          = :"#{base}_complete_streaming"
    stream_blocking_sym = :"#{base}_complete_streaming_blocking"
    as_llm_sym          = :"#{base}_as_llm_provider"
    klass = Class.new(LlmProvider) do
      include LlmProviderImpl

      const_set(:PROVIDER_ID, provider_id)
      const_set(:AS_LLM_PROVIDER_SYM, as_llm_sym)

      define_method(:initialize) do |api_key:, model: nil|
        unless Blazen::FFI.respond_to?(factory_sym)
          raise Blazen::UnsupportedError,
                "blazen cabi missing #{factory_sym} symbol"
        end

        out_model = ::FFI::MemoryPointer.new(:pointer)
        out_err   = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(api_key.to_s) do |key|
          Blazen::FFI.with_cstring(model) do |m|
            Blazen::FFI.public_send(factory_sym, key, m, out_model, out_err)
          end
        end
        Blazen::FFI.check_error!(out_err)
        super(out_model.read_pointer, Blazen::FFI.method(free_sym))
      end

      define_method(:provider_id) { self.class::PROVIDER_ID }

      define_method(:complete) { |request| llm_complete(request, complete_sym) }
      define_method(:complete_blocking) do |request|
        llm_complete_blocking(request, complete_blocking_sym)
      end

      define_method(:stream) do |request, **opts, &block|
        llm_stream(request, stream_blocking_sym, blocking: true, **opts, &block)
      end
      define_method(:stream_async) do |request, **opts, &block|
        llm_stream(request, stream_sym, blocking: false, **opts, &block)
      end
    end
    const_set(klass_name, klass)
  end

  # ----------------------------------------------------------------------
  # 3-string factories: api_key + model + base_url
  # ----------------------------------------------------------------------
  _define_llm_provider_3str(
    :OpenAiProvider, "openai",
    :blazen_openai_provider_new, :blazen_openai_provider_free,
    :blazen_openai_provider_complete, :blazen_openai_provider_complete_blocking,
  )

  _define_llm_provider_3str(
    :AnthropicProvider, "anthropic",
    :blazen_anthropic_provider_new, :blazen_anthropic_provider_free,
    :blazen_anthropic_provider_complete, :blazen_anthropic_provider_complete_blocking,
  )

  _define_llm_provider_3str(
    :GeminiProvider, "gemini",
    :blazen_gemini_provider_new, :blazen_gemini_provider_free,
    :blazen_gemini_provider_complete, :blazen_gemini_provider_complete_blocking,
  )

  _define_llm_provider_3str(
    :FalLlmProvider, "fal-llm",
    :blazen_fal_llm_provider_new, :blazen_fal_llm_provider_free,
    :blazen_fal_llm_provider_complete, :blazen_fal_llm_provider_complete_blocking,
  )

  # ----------------------------------------------------------------------
  # 2-string factories: api_key + model
  # ----------------------------------------------------------------------
  _define_llm_provider_2str(
    :MistralProvider, "mistral",
    :blazen_mistral_provider_new, :blazen_mistral_provider_free,
    :blazen_mistral_provider_complete, :blazen_mistral_provider_complete_blocking,
  )

  _define_llm_provider_2str(
    :FireworksProvider, "fireworks",
    :blazen_fireworks_provider_new, :blazen_fireworks_provider_free,
    :blazen_fireworks_provider_complete, :blazen_fireworks_provider_complete_blocking,
  )

  _define_llm_provider_2str(
    :DeepSeekProvider, "deepseek",
    :blazen_deepseek_provider_new, :blazen_deepseek_provider_free,
    :blazen_deepseek_provider_complete, :blazen_deepseek_provider_complete_blocking,
  )

  _define_llm_provider_2str(
    :PerplexityProvider, "perplexity",
    :blazen_perplexity_provider_new, :blazen_perplexity_provider_free,
    :blazen_perplexity_provider_complete, :blazen_perplexity_provider_complete_blocking,
  )

  _define_llm_provider_2str(
    :TogetherProvider, "together",
    :blazen_together_provider_new, :blazen_together_provider_free,
    :blazen_together_provider_complete, :blazen_together_provider_complete_blocking,
  )

  _define_llm_provider_2str(
    :GroqProvider, "groq",
    :blazen_groq_provider_new, :blazen_groq_provider_free,
    :blazen_groq_provider_complete, :blazen_groq_provider_complete_blocking,
  )

  _define_llm_provider_2str(
    :OpenRouterProvider, "openrouter",
    :blazen_openrouter_provider_new, :blazen_openrouter_provider_free,
    :blazen_openrouter_provider_complete, :blazen_openrouter_provider_complete_blocking,
  )

  _define_llm_provider_2str(
    :CohereProvider, "cohere",
    :blazen_cohere_provider_new, :blazen_cohere_provider_free,
    :blazen_cohere_provider_complete, :blazen_cohere_provider_complete_blocking,
  )

  _define_llm_provider_2str(
    :XaiProvider, "xai",
    :blazen_xai_provider_new, :blazen_xai_provider_free,
    :blazen_xai_provider_complete, :blazen_xai_provider_complete_blocking,
  )

  # ----------------------------------------------------------------------
  # Special factories (different argument shapes)
  # ----------------------------------------------------------------------

  # Azure OpenAI: +api_key+ + +resource_name+ + +deployment_name+ (no
  # separate model — Azure routes by deployment).
  class AzureOpenAiProvider < LlmProvider
    include LlmProviderImpl

    PROVIDER_ID = "azure-openai"
    AS_LLM_PROVIDER_SYM = :blazen_azure_openai_provider_as_llm_provider

    # @param api_key [String]
    # @param resource_name [String] forms the URL host
    #   (+<resource_name>.openai.azure.com+)
    # @param deployment_name [String] Azure deployment id
    def initialize(api_key:, resource_name:, deployment_name:)
      unless Blazen::FFI.respond_to?(:blazen_azure_openai_provider_new)
        raise Blazen::UnsupportedError,
              "blazen cabi missing azure_openai_provider_new symbol"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key.to_s) do |key|
        Blazen::FFI.with_cstring(resource_name.to_s) do |rn|
          Blazen::FFI.with_cstring(deployment_name.to_s) do |dn|
            Blazen::FFI.blazen_azure_openai_provider_new(key, rn, dn, out_model, out_err)
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer,
            Blazen::FFI.method(:blazen_azure_openai_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def complete(request)
      llm_complete(request, :blazen_azure_openai_provider_complete)
    end

    def complete_blocking(request)
      llm_complete_blocking(request, :blazen_azure_openai_provider_complete_blocking)
    end

    def stream(request, **opts, &block)
      llm_stream(request, :blazen_azure_openai_provider_complete_streaming_blocking,
                 blocking: true, **opts, &block)
    end

    def stream_async(request, **opts, &block)
      llm_stream(request, :blazen_azure_openai_provider_complete_streaming,
                 blocking: false, **opts, &block)
    end
  end

  # AWS Bedrock: +api_key+ + +region+ + +model+.
  class BedrockProvider < LlmProvider
    include LlmProviderImpl

    PROVIDER_ID = "bedrock"
    AS_LLM_PROVIDER_SYM = :blazen_bedrock_provider_as_llm_provider

    # @param api_key [String]
    # @param region [String] e.g. +"us-east-1"+
    # @param model [String, nil]
    def initialize(api_key:, region:, model: nil)
      unless Blazen::FFI.respond_to?(:blazen_bedrock_provider_new)
        raise Blazen::UnsupportedError,
              "blazen cabi missing bedrock_provider_new symbol"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key.to_s) do |key|
        Blazen::FFI.with_cstring(region.to_s) do |r|
          Blazen::FFI.with_cstring(model) do |m|
            Blazen::FFI.blazen_bedrock_provider_new(key, r, m, out_model, out_err)
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_bedrock_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def complete(request)
      llm_complete(request, :blazen_bedrock_provider_complete)
    end

    def complete_blocking(request)
      llm_complete_blocking(request, :blazen_bedrock_provider_complete_blocking)
    end

    def stream(request, **opts, &block)
      llm_stream(request, :blazen_bedrock_provider_complete_streaming_blocking,
                 blocking: true, **opts, &block)
    end

    def stream_async(request, **opts, &block)
      llm_stream(request, :blazen_bedrock_provider_complete_streaming,
                 blocking: false, **opts, &block)
    end
  end

  # @api private
  # Allocates a heap +uint32+ holding +val+ and returns the MemoryPointer, or
  # +nil+ when +val+ is +nil+ (the cabi treats a NULL +const uint32_t *+ as
  # "use the engine default"). The caller MUST keep the returned pointer
  # referenced in a local across the native call so it is not GC'd mid-flight.
  def self._opt_u32_ptr(val)
    return nil if val.nil?

    ::FFI::MemoryPointer.new(:uint32).tap { |p| p.write_uint32(Integer(val)) }
  end

  # ----------------------------------------------------------------------
  # Local / OpenAI-compatible engines (varied constructor shapes)
  # ----------------------------------------------------------------------

  # Generic OpenAI-compatible HTTP endpoint:
  # +provider_name+ + +base_url+ + +api_key+ + +model+ (all required).
  class OpenAiCompatProvider < LlmProvider
    include LlmProviderImpl

    PROVIDER_ID = "openai-compat"
    AS_LLM_PROVIDER_SYM = :blazen_openai_compat_provider_as_llm_provider

    # @param provider_name [String] label surfaced in errors / telemetry
    # @param base_url [String] OpenAI-protocol base URL
    # @param api_key [String]
    # @param model [String]
    def initialize(provider_name:, base_url:, api_key:, model:)
      unless Blazen::FFI.respond_to?(:blazen_openai_compat_provider_new)
        raise Blazen::UnsupportedError,
              "blazen cabi missing openai_compat_provider_new symbol"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(provider_name.to_s) do |pn|
        Blazen::FFI.with_cstring(base_url.to_s) do |bu|
          Blazen::FFI.with_cstring(api_key.to_s) do |ak|
            Blazen::FFI.with_cstring(model.to_s) do |m|
              Blazen::FFI.blazen_openai_compat_provider_new(pn, bu, ak, m, out_model, out_err)
            end
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer,
            Blazen::FFI.method(:blazen_openai_compat_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def complete(request)
      llm_complete(request, :blazen_openai_compat_provider_complete)
    end

    def complete_blocking(request)
      llm_complete_blocking(request, :blazen_openai_compat_provider_complete_blocking)
    end

    def stream(request, **opts, &block)
      llm_stream(request, :blazen_openai_compat_provider_complete_streaming_blocking,
                 blocking: true, **opts, &block)
    end

    def stream_async(request, **opts, &block)
      llm_stream(request, :blazen_openai_compat_provider_complete_streaming,
                 blocking: false, **opts, &block)
    end
  end

  # Ollama local server: +host+ + +port+ (uint16) + +model+.
  class OllamaProvider < LlmProvider
    include LlmProviderImpl

    PROVIDER_ID = "ollama"
    AS_LLM_PROVIDER_SYM = :blazen_ollama_provider_as_llm_provider

    # @param host [String] e.g. +"127.0.0.1"+
    # @param port [Integer] TCP port (uint16)
    # @param model [String]
    def initialize(host:, port:, model:)
      unless Blazen::FFI.respond_to?(:blazen_ollama_provider_new)
        raise Blazen::UnsupportedError,
              "blazen cabi missing ollama_provider_new symbol"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(host.to_s) do |h|
        Blazen::FFI.with_cstring(model.to_s) do |m|
          Blazen::FFI.blazen_ollama_provider_new(h, Integer(port), m, out_model, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer,
            Blazen::FFI.method(:blazen_ollama_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def complete(request)
      llm_complete(request, :blazen_ollama_provider_complete)
    end

    def complete_blocking(request)
      llm_complete_blocking(request, :blazen_ollama_provider_complete_blocking)
    end

    def stream(request, **opts, &block)
      llm_stream(request, :blazen_ollama_provider_complete_streaming_blocking,
                 blocking: true, **opts, &block)
    end

    def stream_async(request, **opts, &block)
      llm_stream(request, :blazen_ollama_provider_complete_streaming,
                 blocking: false, **opts, &block)
    end
  end

  # LM Studio local server: +host+ + +port+ (uint16) + +model+.
  class LmStudioProvider < LlmProvider
    include LlmProviderImpl

    PROVIDER_ID = "lm-studio"
    AS_LLM_PROVIDER_SYM = :blazen_lm_studio_provider_as_llm_provider

    # @param host [String] e.g. +"127.0.0.1"+
    # @param port [Integer] TCP port (uint16)
    # @param model [String]
    def initialize(host:, port:, model:)
      unless Blazen::FFI.respond_to?(:blazen_lm_studio_provider_new)
        raise Blazen::UnsupportedError,
              "blazen cabi missing lm_studio_provider_new symbol"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(host.to_s) do |h|
        Blazen::FFI.with_cstring(model.to_s) do |m|
          Blazen::FFI.blazen_lm_studio_provider_new(h, Integer(port), m, out_model, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer,
            Blazen::FFI.method(:blazen_lm_studio_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def complete(request)
      llm_complete(request, :blazen_lm_studio_provider_complete)
    end

    def complete_blocking(request)
      llm_complete_blocking(request, :blazen_lm_studio_provider_complete_blocking)
    end

    def stream(request, **opts, &block)
      llm_stream(request, :blazen_lm_studio_provider_complete_streaming_blocking,
                 blocking: true, **opts, &block)
    end

    def stream_async(request, **opts, &block)
      llm_stream(request, :blazen_lm_studio_provider_complete_streaming,
                 blocking: false, **opts, &block)
    end
  end

  # mistral.rs native engine: +model_id+ + optional +device+ / +quantization+ /
  # +context_length+ (nullable uint32) + +vision+ flag.
  class MistralRsProvider < LlmProvider
    include LlmProviderImpl

    PROVIDER_ID = "mistralrs"
    AS_LLM_PROVIDER_SYM = :blazen_mistralrs_provider_as_llm_provider

    # @param model_id [String]
    # @param device [String, nil]
    # @param quantization [String, nil]
    # @param context_length [Integer, nil]
    # @param vision [Boolean]
    def initialize(model_id:, device: nil, quantization: nil, context_length: nil, vision: false)
      unless Blazen::FFI.respond_to?(:blazen_mistralrs_provider_new)
        raise Blazen::UnsupportedError,
              "blazen cabi missing mistralrs_provider_new symbol"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      cl_ptr    = Blazen._opt_u32_ptr(context_length)
      Blazen::FFI.with_cstring(model_id.to_s) do |mid|
        Blazen::FFI.with_cstring(device) do |dev|
          Blazen::FFI.with_cstring(quantization) do |q|
            Blazen::FFI.blazen_mistralrs_provider_new(
              mid, dev, q, cl_ptr, vision ? true : false, out_model, out_err
            )
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer,
            Blazen::FFI.method(:blazen_mistralrs_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def complete(request)
      llm_complete(request, :blazen_mistralrs_provider_complete)
    end

    def complete_blocking(request)
      llm_complete_blocking(request, :blazen_mistralrs_provider_complete_blocking)
    end

    def stream(request, **opts, &block)
      llm_stream(request, :blazen_mistralrs_provider_complete_streaming_blocking,
                 blocking: true, **opts, &block)
    end

    def stream_async(request, **opts, &block)
      llm_stream(request, :blazen_mistralrs_provider_complete_streaming,
                 blocking: false, **opts, &block)
    end
  end

  # llama.cpp native engine: +model_path+ + optional +device+ / +quantization+ /
  # +context_length+ (nullable uint32) / +n_gpu_layers+ (nullable uint32).
  class LlamaCppProvider < LlmProvider
    include LlmProviderImpl

    PROVIDER_ID = "llamacpp"
    AS_LLM_PROVIDER_SYM = :blazen_llamacpp_provider_as_llm_provider

    # @param model_path [String] path to the GGUF model file
    # @param device [String, nil]
    # @param quantization [String, nil]
    # @param context_length [Integer, nil]
    # @param n_gpu_layers [Integer, nil]
    def initialize(model_path:, device: nil, quantization: nil, context_length: nil,
                   n_gpu_layers: nil)
      unless Blazen::FFI.respond_to?(:blazen_llamacpp_provider_new)
        raise Blazen::UnsupportedError,
              "blazen cabi missing llamacpp_provider_new symbol"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      cl_ptr    = Blazen._opt_u32_ptr(context_length)
      ngl_ptr   = Blazen._opt_u32_ptr(n_gpu_layers)
      Blazen::FFI.with_cstring(model_path.to_s) do |mp|
        Blazen::FFI.with_cstring(device) do |dev|
          Blazen::FFI.with_cstring(quantization) do |q|
            Blazen::FFI.blazen_llamacpp_provider_new(
              mp, dev, q, cl_ptr, ngl_ptr, out_model, out_err
            )
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer,
            Blazen::FFI.method(:blazen_llamacpp_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def complete(request)
      llm_complete(request, :blazen_llamacpp_provider_complete)
    end

    def complete_blocking(request)
      llm_complete_blocking(request, :blazen_llamacpp_provider_complete_blocking)
    end

    def stream(request, **opts, &block)
      llm_stream(request, :blazen_llamacpp_provider_complete_streaming_blocking,
                 blocking: true, **opts, &block)
    end

    def stream_async(request, **opts, &block)
      llm_stream(request, :blazen_llamacpp_provider_complete_streaming,
                 blocking: false, **opts, &block)
    end
  end

  # Candle native engine: +model_id+ + optional +device+ / +quantization+ /
  # +revision+ + +context_length+ (nullable uint32).
  class CandleLlmProvider < LlmProvider
    include LlmProviderImpl

    PROVIDER_ID = "candle"
    AS_LLM_PROVIDER_SYM = :blazen_candle_provider_as_llm_provider

    # @param model_id [String]
    # @param device [String, nil]
    # @param quantization [String, nil]
    # @param revision [String, nil] HF repo revision / branch / tag
    # @param context_length [Integer, nil]
    def initialize(model_id:, device: nil, quantization: nil, revision: nil,
                   context_length: nil)
      unless Blazen::FFI.respond_to?(:blazen_candle_provider_new)
        raise Blazen::UnsupportedError,
              "blazen cabi missing candle_provider_new symbol"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      cl_ptr    = Blazen._opt_u32_ptr(context_length)
      Blazen::FFI.with_cstring(model_id.to_s) do |mid|
        Blazen::FFI.with_cstring(device) do |dev|
          Blazen::FFI.with_cstring(quantization) do |q|
            Blazen::FFI.with_cstring(revision) do |rev|
              Blazen::FFI.blazen_candle_provider_new(
                mid, dev, q, rev, cl_ptr, out_model, out_err
              )
            end
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer,
            Blazen::FFI.method(:blazen_candle_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def complete(request)
      llm_complete(request, :blazen_candle_provider_complete)
    end

    def complete_blocking(request)
      llm_complete_blocking(request, :blazen_candle_provider_complete_blocking)
    end

    def stream(request, **opts, &block)
      llm_stream(request, :blazen_candle_provider_complete_streaming_blocking,
                 blocking: true, **opts, &block)
    end

    def stream_async(request, **opts, &block)
      llm_stream(request, :blazen_candle_provider_complete_streaming,
                 blocking: false, **opts, &block)
    end
  end
end
