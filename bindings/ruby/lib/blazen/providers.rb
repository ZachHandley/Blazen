# frozen_string_literal: true

module Blazen
  # Factory module for constructing provider-backed completion / embedding /
  # compute models on top of the +blazen-cabi+ surface.
  #
  # Every factory below:
  #
  # 1. Allocates two +::FFI::MemoryPointer+ slots — one for the resulting
  #    +BlazenXxxModel **+ out-parameter and one for the +BlazenError **+
  #    out-parameter.
  # 2. Marshals each Ruby keyword argument across the FFI boundary via
  #    {Blazen::FFI.with_cstring} (which transparently maps +nil+ → null).
  # 3. Calls the matching +blazen_<kind>_model_new_<provider>+ free function.
  # 4. Runs {Blazen::FFI.check_error!}, which decodes any populated error
  #    pointer into the matching {Blazen::Error} subclass and raises it.
  # 5. Wraps the resulting raw model pointer in the corresponding Ruby
  #    wrapper class ({Blazen::Llm::Model} /
  #    {Blazen::Llm::EmbeddingModel}, etc.). For TTS / STT / image-gen models
  #    the wrapper classes live in {Blazen::Compute}; the constants are
  #    expected to exist by the time these factories are first called (the
  #    +blazen.rb+ entrypoint requires +compute.rb+ alongside this file).
  #
  # @example OpenAI completion model
  #   model = Blazen::Providers.openai(api_key: ENV.fetch("OPENAI_API_KEY"),
  #                                     model: "gpt-4o-mini")
  #   resp = model.complete_blocking(req)
  module Providers
    module_function

    # ------------------- Completion: cloud (common shape) -------------------
    #
    # NOTE: every factory in this module now returns the matching
    # per-engine provider class ({Blazen::OpenAiProvider},
    # {Blazen::AnthropicProvider}, …) rather than the removed central
    # +Blazen::Llm::Model+ wrapper. Each per-engine class exposes
    # +#complete+ / +#complete_blocking+ / +#stream+ / +#stream_async+
    # directly + an +#as_llm_provider+ conversion for hand-off to
    # {Blazen::Agents.new} / {Blazen::Batch.complete}.

    # @return [Blazen::OpenAiProvider]
    def openai(api_key:, model: nil, base_url: nil)
      Blazen::OpenAiProvider.new(api_key: api_key, model: model, base_url: base_url)
    end

    # @return [Blazen::AnthropicProvider]
    def anthropic(api_key:, model: nil, base_url: nil)
      Blazen::AnthropicProvider.new(api_key: api_key, model: model, base_url: base_url)
    end

    # @return [Blazen::GeminiProvider]
    def gemini(api_key:, model: nil, base_url: nil)
      Blazen::GeminiProvider.new(api_key: api_key, model: model, base_url: base_url)
    end

    # @return [Blazen::OpenRouterProvider]
    def openrouter(api_key:, model: nil, base_url: nil)
      _ = base_url
      Blazen::OpenRouterProvider.new(api_key: api_key, model: model)
    end

    # @return [Blazen::GroqProvider]
    def groq(api_key:, model: nil, base_url: nil)
      _ = base_url
      Blazen::GroqProvider.new(api_key: api_key, model: model)
    end

    # @return [Blazen::TogetherProvider]
    def together(api_key:, model: nil, base_url: nil)
      _ = base_url
      Blazen::TogetherProvider.new(api_key: api_key, model: model)
    end

    # @return [Blazen::MistralProvider]
    def mistral(api_key:, model: nil, base_url: nil)
      _ = base_url
      Blazen::MistralProvider.new(api_key: api_key, model: model)
    end

    # @return [Blazen::DeepSeekProvider]
    def deepseek(api_key:, model: nil, base_url: nil)
      _ = base_url
      Blazen::DeepSeekProvider.new(api_key: api_key, model: model)
    end

    # @return [Blazen::FireworksProvider]
    def fireworks(api_key:, model: nil, base_url: nil)
      _ = base_url
      Blazen::FireworksProvider.new(api_key: api_key, model: model)
    end

    # @return [Blazen::PerplexityProvider]
    def perplexity(api_key:, model: nil, base_url: nil)
      _ = base_url
      Blazen::PerplexityProvider.new(api_key: api_key, model: model)
    end

    # @return [Blazen::XaiProvider]
    def xai(api_key:, model: nil, base_url: nil)
      _ = base_url
      Blazen::XaiProvider.new(api_key: api_key, model: model)
    end

    # @return [Blazen::CohereProvider]
    def cohere(api_key:, model: nil, base_url: nil)
      _ = base_url
      Blazen::CohereProvider.new(api_key: api_key, model: model)
    end

    # --------------- Completion: cloud (provider-specific shapes) ---------------

    # @return [Blazen::AzureOpenAiProvider]
    def azure(api_key:, resource_name:, deployment_name:, api_version: nil)
      _ = api_version
      Blazen::AzureOpenAiProvider.new(
        api_key: api_key, resource_name: resource_name,
        deployment_name: deployment_name,
      )
    end

    # @return [Blazen::BedrockProvider]
    def bedrock(api_key:, region:, model: nil, base_url: nil)
      _ = base_url
      Blazen::BedrockProvider.new(api_key: api_key, region: region, model: model)
    end

    # @return [Blazen::FalLlmProvider]
    def fal(api_key:, model: nil, endpoint: nil,
            enterprise: false, auto_route_modality: false, base_url: nil)
      _ = endpoint
      _ = enterprise
      _ = auto_route_modality
      Blazen::FalLlmProvider.new(api_key: api_key, model: model, base_url: base_url)
    end

    # @return [Blazen::OpenAiCompatProvider]
    def openai_compat(provider_name:, base_url:, api_key:, model:)
      Blazen::OpenAiCompatProvider.new(
        provider_name: provider_name, base_url: base_url,
        api_key: api_key, model: model,
      )
    end

    # @return [Blazen::OllamaProvider]
    def ollama(host:, port: 11_434, model:)
      Blazen::OllamaProvider.new(host: host, port: port, model: model)
    end

    # @return [Blazen::LmStudioProvider]
    def lm_studio(host:, port: 1234, model:)
      Blazen::LmStudioProvider.new(host: host, port: port, model: model)
    end

    # Backwards-compatible alias for the +OpenAiCompat+ shape.
    # @return [Blazen::OpenAiCompatProvider]
    def custom_with_openai_protocol(provider_id:, base_url:, model:, api_key: nil)
      Blazen::OpenAiCompatProvider.new(
        provider_name: provider_id, base_url: base_url,
        api_key: api_key.to_s, model: model,
      )
    end

    # ------------------- Completion: local -------------------

    # @return [Blazen::MistralRsProvider]
    def mistralrs(model_id:, device: nil, quantization: nil,
                  context_length: nil, vision: false)
      Blazen::MistralRsProvider.new(
        model_id: model_id, device: device, quantization: quantization,
        context_length: context_length, vision: vision,
      )
    end

    # @return [Blazen::LlamaCppProvider]
    def llamacpp(model_path:, device: nil, quantization: nil,
                 context_length: nil, n_gpu_layers: nil)
      Blazen::LlamaCppProvider.new(
        model_path: model_path, device: device, quantization: quantization,
        context_length: context_length, n_gpu_layers: n_gpu_layers,
      )
    end

    # @return [Blazen::CandleLlmProvider]
    def candle(model_id:, device: nil, quantization: nil, revision: nil,
               context_length: nil)
      Blazen::CandleLlmProvider.new(
        model_id: model_id, device: device, quantization: quantization,
        revision: revision, context_length: context_length,
      )
    end

    # ------------------- Embedding -------------------

    # @return [Blazen::OpenAiEmbeddingProvider]
    def openai_embedding(api_key:, model: nil, base_url: nil)
      _ = base_url
      Blazen::OpenAiEmbeddingProvider.new(api_key: api_key, model: model)
    end

    # @return [Blazen::FalEmbeddingProvider]
    def fal_embedding(api_key:, model: nil, dimensions: nil)
      _ = dimensions
      Blazen::FalEmbeddingProvider.new(api_key: api_key, model: model)
    end

    # @return [Blazen::FastembedProvider]
    def fastembed(model_name: nil, max_batch_size: nil, show_download_progress: true)
      _ = max_batch_size
      _ = show_download_progress
      Blazen::FastembedProvider.new(model_id: model_name)
    end

    # @return [Blazen::CandleEmbedProvider]
    def candle_embedding(model_id: nil, device: nil, revision: nil)
      _ = device
      _ = revision
      Blazen::CandleEmbedProvider.new(model_id: model_id)
    end

    # @return [Blazen::TractEmbedProvider]
    def tract(model_name: nil, max_batch_size: nil, show_download_progress: true)
      _ = max_batch_size
      _ = show_download_progress
      Blazen::TractEmbedProvider.new(model_id: model_name)
    end

    # ------------------- Compute: TTS / STT / image-gen -------------------
    #
    # These keyword-arg factories now delegate to the concrete per-engine
    # provider classes ({Blazen::FalTtsProvider}, {Blazen::PiperProvider},
    # {Blazen::FalSttProvider}, {Blazen::WhisperCppProvider},
    # {Blazen::FalImageGenProvider}, {Blazen::DiffusionProvider}) — the
    # legacy +Blazen::Compute::TtsModel+ / +SttModel+ / +ImageGenModel+
    # wrapper classes they previously returned have been removed.

    # Fal.ai TTS provider.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @return [Blazen::FalTtsProvider]
    def fal_tts(api_key:, model: nil)
      Blazen::FalTtsProvider.new(api_key: api_key, model: model)
    end

    # Fal.ai STT provider.
    #
    # @param api_key [String]
    # @return [Blazen::FalSttProvider]
    def fal_stt(api_key:, model: nil)
      _ = model # the per-engine fal STT provider has no model override
      Blazen::FalSttProvider.new(api_key: api_key)
    end

    # Fal.ai image-generation provider.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @return [Blazen::FalImageGenProvider]
    def fal_image_gen(api_key:, model: nil)
      Blazen::FalImageGenProvider.new(api_key: api_key, default_model: model)
    end

    # Local Piper TTS provider.
    #
    # @param voice_id [String] arbitrary identifier used for caching
    # @param onnx_path [String] path to the +.onnx+ Piper weights
    # @param config_path [String, nil] path to the +.json+ Piper config
    # @param default_speaker_id [Integer] multi-speaker default (-1 = first)
    # @return [Blazen::PiperProvider]
    def piper(voice_id:, onnx_path:, config_path: nil, default_speaker_id: -1)
      Blazen::PiperProvider.new(
        voice_id: voice_id, onnx_path: onnx_path,
        config_path: config_path, default_speaker_id: default_speaker_id,
      )
    end

    # Local whisper.cpp STT provider.
    #
    # @param model [String, nil]
    # @param device [String, nil]
    # @param language [String, nil]
    # @return [Blazen::WhisperCppProvider]
    def whisper(model: nil, device: nil, language: nil)
      Blazen::WhisperCppProvider.new(model: model, device: device, language: language)
    end

    # Local diffusion-rs image-generation provider.
    #
    # @param options_json [String, nil] backend-specific options as a JSON
    #   string (model id, scheduler, num steps, etc.)
    # @return [Blazen::DiffusionProvider]
    def diffusion(options_json: nil)
      Blazen::DiffusionProvider.new(options_json: options_json)
    end
  end
end

# -----------------------------------------------------------------------
# Part U — Per-engine provider classes
# -----------------------------------------------------------------------
#
# The +Blazen::Providers+ module above is the legacy factory surface that
# returns wrapped +Blazen::Llm::Model+ / +Blazen::Llm::EmbeddingModel+ /
# +Blazen::Compute::*+ handles. Part U adds *concrete per-engine
# provider classes* alongside it ({Blazen::PiperProvider},
# {Blazen::OpenAiProvider}, {Blazen::RvcProvider}, …) that wrap the new
# per-engine cabi opaques directly.
#
# Both surfaces coexist until a separate migration deletes the legacy
# factories. The per-engine classes live in
# +lib/blazen/providers/{base,tts,stt,music,vc,three_d,image,embed,llm}.rb+.

require_relative "providers/base"
require_relative "providers/tts"
require_relative "providers/stt"
require_relative "providers/music"
require_relative "providers/vc"
require_relative "providers/three_d"
require_relative "providers/image"
require_relative "providers/embed"
require_relative "providers/llm"
