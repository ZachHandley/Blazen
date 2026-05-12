# frozen_string_literal: true

module Blazen
  # Factory module for constructing provider-backed {CompletionModel},
  # {EmbeddingModel}, {ImageGenModel}, {SttModel}, and {TtsModel} instances.
  #
  # Each method wraps the corresponding UniFFI +new_*+ free function with
  # keyword arguments and forwards into the native binding, raising idiomatic
  # {Blazen::Error} subclasses on failure.
  #
  # @example OpenAI completion model
  #   model = Blazen::Providers.openai(api_key: ENV.fetch("OPENAI_API_KEY"),
  #                                     model: "gpt-4o-mini")
  #   resp = model.complete_blocking(req)
  module Providers
    module_function

    # OpenAI chat-completions model.
    #
    # @param api_key [String] OpenAI API key
    # @param model [String, nil] model name (e.g. +"gpt-4o-mini"+)
    # @param base_url [String, nil] override base URL (defaults to OpenAI)
    # @return [Blazen::CompletionModel]
    def openai(api_key:, model: nil, base_url: nil)
      Blazen.translate_errors { Blazen.new_openai_completion_model(api_key, model, base_url) }
    end

    # OpenAI embeddings model.
    #
    # @param api_key [String] OpenAI API key
    # @param model [String, nil] model name (e.g. +"text-embedding-3-small"+)
    # @param base_url [String, nil] override base URL
    # @return [Blazen::EmbeddingModel]
    def openai_embedding(api_key:, model: nil, base_url: nil)
      Blazen.translate_errors { Blazen.new_openai_embedding_model(api_key, model, base_url) }
    end

    # OpenAI-compatible completion model (LM Studio, vLLM, Ollama, etc.).
    #
    # @param provider_name [String] human-readable provider tag
    # @param base_url [String] HTTP base URL
    # @param api_key [String] API key (use any non-empty value if not required)
    # @param model [String] model identifier
    # @return [Blazen::CompletionModel]
    def openai_compat(provider_name:, base_url:, api_key:, model:)
      Blazen.translate_errors do
        Blazen.new_openai_compat_completion_model(provider_name, base_url, api_key, model)
      end
    end

    # Anthropic Claude completion model.
    #
    # @param api_key [String] Anthropic API key
    # @param model [String, nil] model name (e.g. +"claude-sonnet-4-7"+)
    # @param base_url [String, nil] override base URL
    # @return [Blazen::CompletionModel]
    def anthropic(api_key:, model: nil, base_url: nil)
      Blazen.translate_errors { Blazen.new_anthropic_completion_model(api_key, model, base_url) }
    end

    # Azure OpenAI completion model.
    #
    # @param api_key [String] Azure OpenAI API key
    # @param resource_name [String] Azure resource name
    # @param deployment_name [String] Azure deployment name
    # @param api_version [String, nil] Azure API version
    # @return [Blazen::CompletionModel]
    def azure(api_key:, resource_name:, deployment_name:, api_version: nil)
      Blazen.translate_errors do
        Blazen.new_azure_completion_model(api_key, resource_name, deployment_name, api_version)
      end
    end

    # AWS Bedrock completion model.
    #
    # @param api_key [String] AWS access key
    # @param region [String] AWS region
    # @param model [String, nil] model identifier
    # @param base_url [String, nil] override base URL
    # @return [Blazen::CompletionModel]
    def bedrock(api_key:, region:, model: nil, base_url: nil)
      Blazen.translate_errors do
        Blazen.new_bedrock_completion_model(api_key, region, model, base_url)
      end
    end

    # Cohere completion model.
    #
    # @param api_key [String] Cohere API key
    # @param model [String, nil] model name
    # @param base_url [String, nil] override base URL
    # @return [Blazen::CompletionModel]
    def cohere(api_key:, model: nil, base_url: nil)
      Blazen.translate_errors { Blazen.new_cohere_completion_model(api_key, model, base_url) }
    end

    # DeepSeek completion model.
    #
    # @param api_key [String] DeepSeek API key
    # @param model [String, nil] model name
    # @param base_url [String, nil] override base URL
    # @return [Blazen::CompletionModel]
    def deepseek(api_key:, model: nil, base_url: nil)
      Blazen.translate_errors { Blazen.new_deepseek_completion_model(api_key, model, base_url) }
    end

    # Fireworks AI completion model.
    #
    # @param api_key [String] Fireworks API key
    # @param model [String, nil] model name
    # @param base_url [String, nil] override base URL
    # @return [Blazen::CompletionModel]
    def fireworks(api_key:, model: nil, base_url: nil)
      Blazen.translate_errors { Blazen.new_fireworks_completion_model(api_key, model, base_url) }
    end

    # Google Gemini completion model.
    #
    # @param api_key [String] Google AI Studio / Vertex API key
    # @param model [String, nil] model name
    # @param base_url [String, nil] override base URL
    # @return [Blazen::CompletionModel]
    def gemini(api_key:, model: nil, base_url: nil)
      Blazen.translate_errors { Blazen.new_gemini_completion_model(api_key, model, base_url) }
    end

    # Groq completion model.
    #
    # @param api_key [String] Groq API key
    # @param model [String, nil] model name
    # @param base_url [String, nil] override base URL
    # @return [Blazen::CompletionModel]
    def groq(api_key:, model: nil, base_url: nil)
      Blazen.translate_errors { Blazen.new_groq_completion_model(api_key, model, base_url) }
    end

    # Mistral cloud completion model.
    #
    # @param api_key [String] Mistral API key
    # @param model [String, nil] model name
    # @param base_url [String, nil] override base URL
    # @return [Blazen::CompletionModel]
    def mistral(api_key:, model: nil, base_url: nil)
      Blazen.translate_errors { Blazen.new_mistral_completion_model(api_key, model, base_url) }
    end

    # OpenRouter completion model.
    #
    # @param api_key [String] OpenRouter API key
    # @param model [String, nil] model name
    # @param base_url [String, nil] override base URL
    # @return [Blazen::CompletionModel]
    def openrouter(api_key:, model: nil, base_url: nil)
      Blazen.translate_errors { Blazen.new_openrouter_completion_model(api_key, model, base_url) }
    end

    # Perplexity completion model.
    #
    # @param api_key [String] Perplexity API key
    # @param model [String, nil] model name
    # @param base_url [String, nil] override base URL
    # @return [Blazen::CompletionModel]
    def perplexity(api_key:, model: nil, base_url: nil)
      Blazen.translate_errors { Blazen.new_perplexity_completion_model(api_key, model, base_url) }
    end

    # Together AI completion model.
    #
    # @param api_key [String] Together API key
    # @param model [String, nil] model name
    # @param base_url [String, nil] override base URL
    # @return [Blazen::CompletionModel]
    def together(api_key:, model: nil, base_url: nil)
      Blazen.translate_errors { Blazen.new_together_completion_model(api_key, model, base_url) }
    end

    # xAI Grok completion model.
    #
    # @param api_key [String] xAI API key
    # @param model [String, nil] model name
    # @param base_url [String, nil] override base URL
    # @return [Blazen::CompletionModel]
    def xai(api_key:, model: nil, base_url: nil)
      Blazen.translate_errors { Blazen.new_xai_completion_model(api_key, model, base_url) }
    end

    # Fal.ai cloud completion model.
    #
    # @param api_key [String] Fal API key
    # @param model [String, nil] model name
    # @param base_url [String, nil] override base URL
    # @param endpoint [String, nil] specific endpoint path
    # @param enterprise [Boolean] use enterprise endpoint
    # @param auto_route_modality [Boolean] auto-route by request modality
    # @return [Blazen::CompletionModel]
    def fal(api_key:, model: nil, base_url: nil, endpoint: nil,
            enterprise: false, auto_route_modality: false)
      Blazen.translate_errors do
        Blazen.new_fal_completion_model(api_key, model, base_url, endpoint,
                                        enterprise, auto_route_modality)
      end
    end

    # Fal.ai embeddings model.
    #
    # @param api_key [String] Fal API key
    # @param model [String, nil] model name
    # @param dimensions [Integer, nil] target embedding dimensionality
    # @return [Blazen::EmbeddingModel]
    def fal_embedding(api_key:, model: nil, dimensions: nil)
      Blazen.translate_errors { Blazen.new_fal_embedding_model(api_key, model, dimensions) }
    end

    # Fal.ai image-generation model.
    #
    # @param api_key [String] Fal API key
    # @param model [String, nil] model name
    # @return [Blazen::ImageGenModel]
    def fal_image_gen(api_key:, model: nil)
      Blazen.translate_errors { Blazen.new_fal_image_gen_model(api_key, model) }
    end

    # Fal.ai speech-to-text model.
    #
    # @param api_key [String] Fal API key
    # @param model [String, nil] model name
    # @return [Blazen::SttModel]
    def fal_stt(api_key:, model: nil)
      Blazen.translate_errors { Blazen.new_fal_stt_model(api_key, model) }
    end

    # Fal.ai text-to-speech model.
    #
    # @param api_key [String] Fal API key
    # @param model [String, nil] model name
    # @return [Blazen::TtsModel]
    def fal_tts(api_key:, model: nil)
      Blazen.translate_errors { Blazen.new_fal_tts_model(api_key, model) }
    end

    # Local Candle (Rust-native) completion model.
    #
    # @param model_id [String] HuggingFace model ID
    # @param device [String, nil] +"cpu"+, +"cuda"+, +"metal"+
    # @param quantization [String, nil] quantization scheme
    # @param revision [String, nil] HF revision / branch
    # @param context_length [Integer, nil] context window
    # @return [Blazen::CompletionModel]
    def candle(model_id:, device: nil, quantization: nil, revision: nil, context_length: nil)
      Blazen.translate_errors do
        Blazen.new_candle_completion_model(model_id, device, quantization, revision, context_length)
      end
    end

    # Local Candle embeddings model.
    #
    # @param model_id [String, nil] HuggingFace model ID
    # @param device [String, nil] +"cpu"+, +"cuda"+, +"metal"+
    # @param revision [String, nil] HF revision / branch
    # @return [Blazen::EmbeddingModel]
    def candle_embedding(model_id: nil, device: nil, revision: nil)
      Blazen.translate_errors { Blazen.new_candle_embedding_model(model_id, device, revision) }
    end

    # FastEmbed local embeddings model.
    #
    # @param model_name [String, nil] FastEmbed model name
    # @param max_batch_size [Integer, nil] max batch size
    # @param show_download_progress [Boolean, nil] show download progress bar
    # @return [Blazen::EmbeddingModel]
    def fastembed(model_name: nil, max_batch_size: nil, show_download_progress: nil)
      Blazen.translate_errors do
        Blazen.new_fastembed_embedding_model(model_name, max_batch_size, show_download_progress)
      end
    end

    # Tract local embeddings model.
    #
    # @param model_name [String, nil] model name
    # @param max_batch_size [Integer, nil] max batch size
    # @param show_download_progress [Boolean, nil] show download progress bar
    # @return [Blazen::EmbeddingModel]
    def tract(model_name: nil, max_batch_size: nil, show_download_progress: nil)
      Blazen.translate_errors do
        Blazen.new_tract_embedding_model(model_name, max_batch_size, show_download_progress)
      end
    end

    # llama.cpp local completion model.
    #
    # @param model_path [String] filesystem path to a GGUF model
    # @param device [String, nil] +"cpu"+, +"cuda"+, +"metal"+
    # @param quantization [String, nil] quantization scheme
    # @param context_length [Integer, nil] context window
    # @param n_gpu_layers [Integer, nil] number of layers to offload to GPU
    # @return [Blazen::CompletionModel]
    def llamacpp(model_path:, device: nil, quantization: nil,
                 context_length: nil, n_gpu_layers: nil)
      Blazen.translate_errors do
        Blazen.new_llamacpp_completion_model(model_path, device, quantization,
                                             context_length, n_gpu_layers)
      end
    end

    # mistral.rs local completion model.
    #
    # @param model_id [String] HuggingFace model ID
    # @param device [String, nil] +"cpu"+, +"cuda"+, +"metal"+
    # @param quantization [String, nil] quantization scheme
    # @param context_length [Integer, nil] context window
    # @param vision [Boolean] enable vision capabilities
    # @return [Blazen::CompletionModel]
    def mistralrs(model_id:, device: nil, quantization: nil, context_length: nil, vision: false)
      Blazen.translate_errors do
        Blazen.new_mistralrs_completion_model(model_id, device, quantization, context_length, vision)
      end
    end

    # Local diffusion-based image-generation model (Candle).
    #
    # @param model_id [String, nil] HuggingFace model ID
    # @param device [String, nil] +"cpu"+, +"cuda"+, +"metal"+
    # @param width [Integer, nil] output image width in pixels
    # @param height [Integer, nil] output image height in pixels
    # @param num_inference_steps [Integer, nil] number of denoising steps
    # @param guidance_scale [Float, nil] classifier-free-guidance scale
    # @return [Blazen::ImageGenModel]
    def diffusion(model_id: nil, device: nil, width: nil, height: nil,
                  num_inference_steps: nil, guidance_scale: nil)
      Blazen.translate_errors do
        Blazen.new_diffusion_model(model_id, device, width, height, num_inference_steps, guidance_scale)
      end
    end

    # Local Whisper speech-to-text model.
    #
    # @param model [String, nil] model size / name
    # @param device [String, nil] +"cpu"+, +"cuda"+, +"metal"+
    # @param language [String, nil] expected language code (e.g. +"en"+)
    # @return [Blazen::SttModel]
    def whisper(model: nil, device: nil, language: nil)
      Blazen.translate_errors { Blazen.new_whisper_stt_model(model, device, language) }
    end

    # Local Piper text-to-speech model.
    #
    # @param model_id [String, nil] Piper voice / model identifier
    # @param speaker_id [Integer, nil] speaker index for multi-speaker voices
    # @param sample_rate [Integer, nil] output sample rate in Hz
    # @return [Blazen::TtsModel]
    def piper(model_id: nil, speaker_id: nil, sample_rate: nil)
      Blazen.translate_errors { Blazen.new_piper_tts_model(model_id, speaker_id, sample_rate) }
    end
  end
end
