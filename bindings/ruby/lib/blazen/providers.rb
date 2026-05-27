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
    # @return [Blazen::Llm::Model]
    def openai(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_model_new_openai(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # Anthropic Claude completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::Model]
    def anthropic(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_model_new_anthropic(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # Google Gemini completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::Model]
    def gemini(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_model_new_gemini(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # OpenRouter completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::Model]
    def openrouter(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_model_new_openrouter(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # Groq completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::Model]
    def groq(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_model_new_groq(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # Together AI completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::Model]
    def together(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_model_new_together(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # Mistral cloud completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::Model]
    def mistral(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_model_new_mistral(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # DeepSeek completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::Model]
    def deepseek(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_model_new_deepseek(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # Fireworks AI completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::Model]
    def fireworks(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_model_new_fireworks(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # Perplexity completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::Model]
    def perplexity(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_model_new_perplexity(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # xAI Grok completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::Model]
    def xai(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_model_new_xai(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # Cohere completion model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::Model]
    def cohere(api_key:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(base_url) do |b|
            Blazen::FFI.blazen_model_new_cohere(k, m, b, out_model, out_err)
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
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
    # @return [Blazen::Llm::Model]
    def azure(api_key:, resource_name:, deployment_name:, api_version: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(resource_name) do |r|
          Blazen::FFI.with_cstring(deployment_name) do |d|
            Blazen::FFI.with_cstring(api_version) do |v|
              Blazen::FFI.blazen_model_new_azure(k, r, d, v, out_model, out_err)
            end
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # AWS Bedrock completion model. +api_key+ may be an empty string to
    # resolve the bearer token from +AWS_BEARER_TOKEN_BEDROCK+ at runtime.
    #
    # @param api_key [String]
    # @param region [String] e.g. +"us-east-1"+
    # @param model [String, nil]
    # @param base_url [String, nil]
    # @return [Blazen::Llm::Model]
    def bedrock(api_key:, region:, model: nil, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(region) do |r|
          Blazen::FFI.with_cstring(model) do |m|
            Blazen::FFI.with_cstring(base_url) do |b|
              Blazen::FFI.blazen_model_new_bedrock(k, r, m, b, out_model, out_err)
            end
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
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
    # @return [Blazen::Llm::Model]
    def fal(api_key:, model: nil, endpoint: nil,
            enterprise: false, auto_route_modality: false, base_url: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key) do |k|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.with_cstring(endpoint) do |e|
            Blazen::FFI.with_cstring(base_url) do |b|
              Blazen::FFI.blazen_model_new_fal(
                k, m, e, enterprise, auto_route_modality, b, out_model, out_err
              )
            end
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # Generic OpenAI-compatible completion model (vLLM, llama-server, LM
    # Studio, etc.). All four arguments are required.
    #
    # @param provider_name [String]
    # @param base_url [String]
    # @param api_key [String]
    # @param model [String]
    # @return [Blazen::Llm::Model]
    def openai_compat(provider_name:, base_url:, api_key:, model:)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(provider_name) do |p|
        Blazen::FFI.with_cstring(base_url) do |b|
          Blazen::FFI.with_cstring(api_key) do |k|
            Blazen::FFI.with_cstring(model) do |m|
              Blazen::FFI.blazen_model_new_openai_compat(
                p, b, k, m, out_model, out_err
              )
            end
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # Model for an Ollama server.
    #
    # Convenience wrapper around {openai_compat}: builds
    # +base_url = "http://#{host}:#{port}/v1"+ with no API key.
    #
    # @param host [String] e.g. +"localhost"+, +"192.168.1.50"+
    # @param port [Integer] TCP port (Ollama default 11434)
    # @param model [String] e.g. +"llama3.1"+
    # @return [Blazen::Llm::Model]
    def ollama(host:, port: 11_434, model:)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(host) do |h|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.blazen_model_new_ollama(
            h, port.to_i, m, out_model, out_err
          )
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # Model for an LM Studio server.
    #
    # @param host [String]
    # @param port [Integer] LM Studio default 1234
    # @param model [String]
    # @return [Blazen::Llm::Model]
    def lm_studio(host:, port: 1234, model:)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(host) do |h|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.blazen_model_new_lm_studio(
            h, port.to_i, m, out_model, out_err
          )
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # Universal Model speaking the OpenAI chat-completions protocol
    # against an arbitrary base URL. Pass +api_key: nil+ for unauthenticated
    # local servers.
    #
    # @param provider_id [String]
    # @param base_url [String]
    # @param model [String]
    # @param api_key [String, nil]
    # @return [Blazen::Llm::Model]
    def custom_with_openai_protocol(provider_id:, base_url:, model:, api_key: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(provider_id) do |p|
        Blazen::FFI.with_cstring(base_url) do |b|
          Blazen::FFI.with_cstring(model) do |m|
            Blazen::FFI.with_cstring(api_key) do |k|
              Blazen::FFI.blazen_model_new_custom_with_openai_protocol(
                p, b, m, k, out_model, out_err
              )
            end
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # ------------------- Completion: local -------------------

    # Local mistral.rs completion model.
    #
    # @param model_id [String] HuggingFace repo id or local GGUF path
    # @param device [String, nil] e.g. +"cpu"+, +"cuda:0"+, +"metal"+
    # @param quantization [String, nil] e.g. +"q4_k_m"+
    # @param context_length [Integer, nil] override context window
    # @param vision [Boolean] enable vision capabilities
    # @return [Blazen::Llm::Model]
    def mistralrs(model_id:, device: nil, quantization: nil,
                  context_length: nil, vision: false)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      ctx_len = context_length.nil? ? -1 : context_length.to_i
      Blazen::FFI.with_cstring(model_id) do |mid|
        Blazen::FFI.with_cstring(device) do |d|
          Blazen::FFI.with_cstring(quantization) do |q|
            Blazen::FFI.blazen_model_new_mistralrs(
              mid, d, q, ctx_len, vision, out_model, out_err
            )
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # Local llama.cpp completion model.
    #
    # @param model_path [String]
    # @param device [String, nil]
    # @param quantization [String, nil]
    # @param context_length [Integer, nil]
    # @param n_gpu_layers [Integer, nil]
    # @return [Blazen::Llm::Model]
    def llamacpp(model_path:, device: nil, quantization: nil,
                 context_length: nil, n_gpu_layers: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      ctx_len = context_length.nil? ? -1 : context_length.to_i
      gpu_layers = n_gpu_layers.nil? ? -1 : n_gpu_layers.to_i
      Blazen::FFI.with_cstring(model_path) do |mp|
        Blazen::FFI.with_cstring(device) do |d|
          Blazen::FFI.with_cstring(quantization) do |q|
            Blazen::FFI.blazen_model_new_llamacpp(
              mp, d, q, ctx_len, gpu_layers, out_model, out_err
            )
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
    end

    # Local Candle (Rust-native) completion model.
    #
    # @param model_id [String]
    # @param device [String, nil]
    # @param quantization [String, nil]
    # @param revision [String, nil]
    # @param context_length [Integer, nil]
    # @return [Blazen::Llm::Model]
    def candle(model_id:, device: nil, quantization: nil, revision: nil,
               context_length: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      ctx_len = context_length.nil? ? -1 : context_length.to_i
      Blazen::FFI.with_cstring(model_id) do |mid|
        Blazen::FFI.with_cstring(device) do |d|
          Blazen::FFI.with_cstring(quantization) do |q|
            Blazen::FFI.with_cstring(revision) do |r|
              Blazen::FFI.blazen_model_new_candle(
                mid, d, q, r, ctx_len, out_model, out_err
              )
            end
          end
        end
      end
      _wrap_model(out_model, out_err, Blazen::Llm::Model)
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
