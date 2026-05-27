# frozen_string_literal: true

# Concrete per-engine chat-completion provider classes (Part U).
#
# Engines (15):
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

module Blazen
  # @api private
  module LlmProviderImpl
    private

    # The completion cabi entry points consume the +BlazenModelRequest+
    # pointer (ownership transfers). We extract the raw handle via
    # +ModelRequest#consume!+ which detaches the AutoPointer's finalizer.
    def llm_complete(request, async_sym)
      unless request.is_a?(Blazen::Llm::ModelRequest)
        raise ArgumentError, "request must be a Blazen::Llm::ModelRequest"
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
  end

  # @api private
  # Defines a standard +api_key+/+model+/+base_url+ LLM provider class. The
  # cabi factory has signature
  # +(const char *api_key, const char *model, const char *base_url,
  # Blazen<X>Provider **out_model, BlazenError **out_err) -> int32_t+.
  def self._define_llm_provider_3str(klass_name, provider_id, factory_sym,
                                     free_sym, complete_sym, complete_blocking_sym)
    klass = Class.new(LlmProvider) do
      include LlmProviderImpl

      const_set(:PROVIDER_ID, provider_id)

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
    klass = Class.new(LlmProvider) do
      include LlmProviderImpl

      const_set(:PROVIDER_ID, provider_id)

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
  end

  # AWS Bedrock: +api_key+ + +region+ + +model+.
  class BedrockProvider < LlmProvider
    include LlmProviderImpl

    PROVIDER_ID = "bedrock"

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
  end
end
