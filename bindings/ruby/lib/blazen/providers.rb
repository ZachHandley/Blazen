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
  #    wrapper class ({Blazen::Llm::CompletionModel} /
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

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    # Common epilogue shared by every +Providers+ factory: pulls the model
    # pointer out of +out_model+, raises on +out_err+, and wraps the
    # pointer in +klass.new(raw_ptr)+. Returns the wrapped model instance.
    #
    # @api private
    def _wrap_model(out_model, out_err, klass)
      Blazen::FFI.check_error!(out_err)
      ptr = out_model.read_pointer
      if ptr.nil? || ptr.null?
        raise Blazen::InternalError, "#{klass.name}: native factory returned null with no error"
      end

      klass.new(ptr)
    end
    private_class_method :_wrap_model

    # ------------------- Completion: cloud (common shape) -------------------

    # OpenAI chat-completions model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::CompletionModel]
    def openai(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_completion_model_new_openai(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # Anthropic Claude completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::CompletionModel]
    def anthropic(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_completion_model_new_anthropic(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # Google Gemini completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::CompletionModel]
    def gemini(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_completion_model_new_gemini(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # OpenRouter completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::CompletionModel]
    def openrouter(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_completion_model_new_openrouter(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # Groq completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::CompletionModel]
    def groq(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_completion_model_new_groq(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # Together AI completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::CompletionModel]
    def together(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_completion_model_new_together(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # Mistral cloud completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::CompletionModel]
    def mistral(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_completion_model_new_mistral(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # DeepSeek completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::CompletionModel]
    def deepseek(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_completion_model_new_deepseek(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # Fireworks AI completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::CompletionModel]
    def fireworks(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_completion_model_new_fireworks(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # Perplexity completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::CompletionModel]
    def perplexity(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_completion_model_new_perplexity(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # xAI Grok completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::CompletionModel]
    def xai(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_completion_model_new_xai(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # Cohere completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::CompletionModel]
    def cohere(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_completion_model_new_cohere(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # --------------- Completion: cloud (provider-specific shapes) ---------------

    # Azure OpenAI completion model. +api_version+ is optional; +api_key+,
    # +resource_name+, and +deployment_name+ are required (the underlying
    # endpoint URL is derived from the latter two).
    #
    # @param api_key [String]
    # @param resource_name [String]
    # @param deployment_name [String]
    # @param api_version [String, nil]
    # @return [Blazen::Llm::CompletionModel]
    def azure(api_key:, resource_name:, deployment_name:, api_version: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(resource_name) do |r|
          Blazen::FFI.with_cstring(deployment_name) do |d|
            Blazen::FFI.with_cstring(api_version) do |v|
              Blazen::FFI.blazen_completion_model_new_azure(k, r, d, v, out_model, out_err)
            end
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # AWS Bedrock completion model. +api_key+ may be an empty string to
    # resolve the bearer token from +AWS_BEARER_TOKEN_BEDROCK+ at runtime.
    #
    # @param api_key [String]
    # @param region [String] e.g. +"us-east-1"+
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::CompletionModel]
    def bedrock(api_key:, region:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(region) do |r|
          Blazen::FFI.with_cstring(model) do |m|
            Blazen::FFI.with_cstring(base_url) do |b|
              Blazen::FFI.blazen_completion_model_new_bedrock(k, r, m, b, out_model, out_err)
            end
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # Fal.ai chat-completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param endpoint [String, nil] fal endpoint family — one of
    #   +"openai_chat"+ (default), +"openai_responses"+,
    #   +"openai_embeddings"+, +"openrouter"+, +"any_llm"+
    # @param enterprise [Boolean] use the enterprise endpoint variant
    # @param auto_route_modality [Boolean] auto-route by request modality
    # @param base_url [String, nil]
    # @return [Blazen::Llm::CompletionModel]
    def fal(api_key:, model: nil, endpoint: nil,
            enterprise: false, auto_route_modality: false, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(endpoint) do |e|
            Blazen::FFI.with_cstring(base_url) do |b|
              Blazen::FFI.blazen_completion_model_new_fal(
                k, m, e, enterprise, auto_route_modality, b, out_model, out_err
              )
            end
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # Generic OpenAI-compatible completion model (vLLM, llama-server, LM
    # Studio, etc.). All four arguments are required.
    #
    # @param provider_name [String]
    # @param base_url [String]
    # @param api_key [String]
    # @param model [String]
    # @return [Blazen::Llm::CompletionModel]
    def openai_compat(provider_name:, base_url:, api_key:, model:)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(provider_name) do |p|
        Blazen::FFI.with_cstring(base_url) do |b|
          Blazen::FFI.with_cstring(api_key) do |k|
            Blazen::FFI.with_cstring(model) do |m|
              Blazen::FFI.blazen_completion_model_new_openai_compat(
                p, b, k, m, out_model, out_err
              )
            end
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # ------------------- Completion: local -------------------

    # Local mistral.rs completion model.
    #
    # @param model_id [String] HuggingFace repo id or local GGUF path
    # @param device [String, nil] e.g. +"cpu"+, +"cuda:0"+, +"metal"+
    # @param quantization [String, nil] e.g. +"q4_k_m"+
    # @param context_length [Integer, nil] override context window
    # @param vision [Boolean] enable vision capabilities
    # @return [Blazen::Llm::CompletionModel]
    def mistralrs(model_id:, device: nil, quantization: nil,
                  context_length: nil, vision: false)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      ctx_len = context_length.nil? ? -1 : context_length.to_i
      Blazen::FFI.with_cstring(model_id) do |mid|
        Blazen::FFI.with_cstring(device) do |d|
          Blazen::FFI.with_cstring(quantization) do |q|
            Blazen::FFI.blazen_completion_model_new_mistralrs(
              mid, d, q, ctx_len, vision, out_model, out_err
            )
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # Local llama.cpp completion model.
    #
    # @param model_path [String]
    # @param device [String, nil]
    # @param quantization [String, nil]
    # @param context_length [Integer, nil]
    # @param n_gpu_layers [Integer, nil]
    # @return [Blazen::Llm::CompletionModel]
    def llamacpp(model_path:, device: nil, quantization: nil,
                 context_length: nil, n_gpu_layers: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      ctx_len = context_length.nil? ? -1 : context_length.to_i
      gpu_layers = n_gpu_layers.nil? ? -1 : n_gpu_layers.to_i
      Blazen::FFI.with_cstring(model_path) do |mp|
        Blazen::FFI.with_cstring(device) do |d|
          Blazen::FFI.with_cstring(quantization) do |q|
            Blazen::FFI.blazen_completion_model_new_llamacpp(
              mp, d, q, ctx_len, gpu_layers, out_model, out_err
            )
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # Local Candle (Rust-native) completion model.
    #
    # @param model_id [String]
    # @param device [String, nil]
    # @param quantization [String, nil]
    # @param revision [String, nil]
    # @param context_length [Integer, nil]
    # @return [Blazen::Llm::CompletionModel]
    def candle(model_id:, device: nil, quantization: nil, revision: nil,
               context_length: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      ctx_len = context_length.nil? ? -1 : context_length.to_i
      Blazen::FFI.with_cstring(model_id) do |mid|
        Blazen::FFI.with_cstring(device) do |d|
          Blazen::FFI.with_cstring(quantization) do |q|
            Blazen::FFI.with_cstring(revision) do |r|
              Blazen::FFI.blazen_completion_model_new_candle(
                mid, d, q, r, ctx_len, out_model, out_err
              )
            end
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::CompletionModel)
    end

    # ------------------- Embedding -------------------

    # OpenAI embeddings model.
    #
    # @param api_key [String]
    # @param model [String, nil] defaults to +"text-embedding-3-small"+
    # @param base_url [String, nil]
    # @return [Blazen::Llm::EmbeddingModel]
    def openai_embedding(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_embedding_model_new_openai(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::EmbeddingModel)
    end

    # Fal.ai embeddings model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param dimensions [Integer, nil] target embedding dimensionality
    # @return [Blazen::Llm::EmbeddingModel]
    def fal_embedding(api_key:, model: nil, dimensions: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      dim = dimensions.nil? ? -1 : dimensions.to_i
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.blazen_embedding_model_new_fal(k, m, dim, out_model, out_err)
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::EmbeddingModel)
    end

    # FastEmbed local embeddings model (ONNX Runtime).
    #
    # @param model_name [String, nil]
    # @param max_batch_size [Integer, nil]
    # @param show_download_progress [Boolean]
    # @return [Blazen::Llm::EmbeddingModel]
    def fastembed(model_name: nil, max_batch_size: nil, show_download_progress: true)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      max_batch = max_batch_size.nil? ? -1 : max_batch_size.to_i
      Blazen::FFI.with_cstring(model_name) do |n|
        Blazen::FFI.blazen_embedding_model_new_fastembed(
          n, max_batch, show_download_progress ? true : false, out_model, out_err
        )
      end
      _wrap_model(out_model, out_err, Blazen::Llm::EmbeddingModel)
    end

    # Local Candle text-embedding model.
    #
    # @param model_id [String, nil]
    # @param device [String, nil]
    # @param revision [String, nil]
    # @return [Blazen::Llm::EmbeddingModel]
    def candle_embedding(model_id: nil, device: nil, revision: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(model_id) do |mid|
        Blazen::FFI.with_cstring(device) do |d|
          Blazen::FFI.with_cstring(revision) do |r|
            Blazen::FFI.blazen_embedding_model_new_candle(mid, d, r, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::EmbeddingModel)
    end

    # Local tract (pure-Rust ONNX) embeddings model.
    #
    # @param model_name [String, nil]
    # @param max_batch_size [Integer, nil]
    # @param show_download_progress [Boolean]
    # @return [Blazen::Llm::EmbeddingModel]
    def tract(model_name: nil, max_batch_size: nil, show_download_progress: true)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      max_batch = max_batch_size.nil? ? -1 : max_batch_size.to_i
      Blazen::FFI.with_cstring(model_name) do |n|
        Blazen::FFI.blazen_embedding_model_new_tract(
          n, max_batch, show_download_progress ? true : false, out_model, out_err
        )
      end
      _wrap_model(out_model, out_err, Blazen::Llm::EmbeddingModel)
    end

    # ------------------- Compute: TTS / STT / image-gen -------------------
    #
    # The Ruby wrappers for these model handles
    # (+Blazen::Compute::TtsModel+ / +SttModel+ / +ImageGenModel+) are
    # defined in +lib/blazen/compute.rb+. We resolve them lazily at call
    # time so the +require+ order between +providers.rb+ and +compute.rb+
    # doesn't matter.

    # Fal.ai TTS model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @return [Blazen::Compute::TtsModel]
    def fal_tts(api_key:, model: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.blazen_tts_model_new_fal(k, m, out_model, out_err)
        end
      end
      _wrap_model(out_model, out_err, Blazen::Compute::TtsModel)
    end

    # Fal.ai STT model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @return [Blazen::Compute::SttModel]
    def fal_stt(api_key:, model: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.blazen_stt_model_new_fal(k, m, out_model, out_err)
        end
      end
      _wrap_model(out_model, out_err, Blazen::Compute::SttModel)
    end

    # Fal.ai image-generation model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @return [Blazen::Compute::ImageGenModel]
    def fal_image_gen(api_key:, model: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.blazen_image_gen_model_new_fal(k, m, out_model, out_err)
        end
      end
      _wrap_model(out_model, out_err, Blazen::Compute::ImageGenModel)
    end

    # Local Piper TTS model.
    #
    # @param model_id [String, nil] Piper voice id (e.g. +"en_US-amy-medium"+)
    # @param speaker_id [Integer, nil]
    # @param sample_rate [Integer, nil]
    # @return [Blazen::Compute::TtsModel]
    def piper(model_id: nil, speaker_id: nil, sample_rate: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      spk = speaker_id.nil? ? -1 : speaker_id.to_i
      sr  = sample_rate.nil? ? -1 : sample_rate.to_i
      Blazen::FFI.with_cstring(model_id) do |mid|
        Blazen::FFI.blazen_tts_model_new_piper(mid, spk, sr, out_model, out_err)
      end
      _wrap_model(out_model, out_err, Blazen::Compute::TtsModel)
    end

    # Local Whisper STT model.
    #
    # @param model [String, nil]
    # @param device [String, nil]
    # @param language [String, nil]
    # @return [Blazen::Compute::SttModel]
    def whisper(model: nil, device: nil, language: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(model) do |m|
        Blazen::FFI.with_cstring(device) do |d|
          Blazen::FFI.with_cstring(language) do |l|
            Blazen::FFI.blazen_stt_model_new_whisper(m, d, l, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Compute::SttModel)
    end

    # Local diffusion-rs image-generation model.
    #
    # @param model_id [String, nil]
    # @param device [String, nil]
    # @param width [Integer, nil]
    # @param height [Integer, nil]
    # @param num_inference_steps [Integer, nil]
    # @param guidance_scale [Float, nil]
    # @return [Blazen::Compute::ImageGenModel]
    def diffusion(model_id: nil, device: nil, width: nil, height: nil,
                  num_inference_steps: nil, guidance_scale: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      w  = width.nil? ? -1 : width.to_i
      h  = height.nil? ? -1 : height.to_i
      st = num_inference_steps.nil? ? -1 : num_inference_steps.to_i
      gs = guidance_scale.nil? ? -1.0 : guidance_scale.to_f
      Blazen::FFI.with_cstring(model_id) do |mid|
        Blazen::FFI.with_cstring(device) do |d|
          Blazen::FFI.blazen_image_gen_model_new_diffusion(
            mid, d, w, h, st, gs, out_model, out_err
          )
        end
      end
      _wrap_model(out_model, out_err, Blazen::Compute::ImageGenModel)
    end
  end
end
