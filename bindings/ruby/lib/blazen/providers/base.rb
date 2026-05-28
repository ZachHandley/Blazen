# frozen_string_literal: true

# Per-engine provider classes (Part U — Provider class hierarchy).
#
# This file defines the *abstract* polymorphic base hierarchy used by the
# concrete per-engine subclasses in +blazen/providers/tts.rb+,
# +stt.rb+, +music.rb+, +vc.rb+, +three_d.rb+, +image.rb+, +embed.rb+,
# and +llm.rb+.
#
# These are the sole construction surface for the compute family of
# engines. The legacy +Blazen::Compute.*+ keyword-arg factories and the
# central +Blazen::Compute::MusicModel+ / +VcModel+ / +TtsModel+ /
# +SttModel+ / +ImageGenModel+ / +ThreeDModel+ wrappers have been
# removed; only the typed-result wrappers remain under that namespace
# (see +blazen/providers/compute_results.rb+).
#
# Each concrete subclass:
#
# * wraps a per-engine cabi opaque (e.g. +BlazenPiperProvider *+) via a
#   single +@ptr+ +::FFI::AutoPointer+ that fires the matching
#   +blazen_<engine>_provider_free+ on GC,
# * implements the capability methods (async + +_blocking+) by calling
#   the matching +blazen_<engine>_provider_<method>+ + popping the
#   typed result via +blazen_future_take_<X>+,
# * exposes a stable +PROVIDER_ID+ constant + +#provider_id+ accessor.
#
# Naming parity with the other bindings (Node, Py, WASM, Swift, Kotlin,
# Go) is intentional: every consumer-facing class name is identical
# across the 9 binding surfaces.

module Blazen
  # Abstract polymorphic root for all per-engine providers.
  #
  # Subclasses MUST:
  #
  # 1. Call +super(handle)+ to wire the +::FFI::AutoPointer+ + finalizer.
  # 2. Define +PROVIDER_ID+ (a frozen string).
  # 3. Override +#provider_id+ to return +PROVIDER_ID+.
  #
  # @abstract
  class BaseProvider
    # @return [::FFI::AutoPointer] the wrapped cabi opaque handle. Subclasses
    #   set this up via {#wrap_handle!} so GC reclaims the native side via
    #   the engine's +_free+ entry point.
    attr_reader :handle

    # @param handle [::FFI::Pointer, ::FFI::AutoPointer] caller-owned cabi
    #   opaque pointer. When +handle+ is already an {::FFI::AutoPointer} (as
    #   it is when an {LlmProvider} / {EmbeddingProvider} instance is built
    #   from a {#as_llm_provider} / {#as_embedding_provider} conversion),
    #   it is adopted verbatim and +free_fn+ is ignored.
    # @param free_fn [Method, Proc, nil] the matching
    #   +blazen_<engine>_provider_free+ callable (required when +handle+ is
    #   a raw {::FFI::Pointer})
    def initialize(handle, free_fn = nil)
      if handle.is_a?(::FFI::AutoPointer)
        @handle = handle
        return
      end

      if handle.nil? || handle.null?
        raise ArgumentError, "#{self.class.name}: native pointer must be non-null"
      end

      raise ArgumentError, "#{self.class.name}: free_fn required for raw pointers" if free_fn.nil?

      @handle = ::FFI::AutoPointer.new(handle, free_fn)
    end

    # Backwards-compat alias for the existing +Blazen::Compute+ wrapper
    # APIs, which expose the handle as +ptr+. New code should prefer
    # {#handle}.
    #
    # @return [::FFI::AutoPointer]
    def ptr
      @handle
    end

    # Returns the stable provider identifier (e.g. +"piper"+, +"openai"+).
    #
    # @return [String]
    def provider_id
      raise NotImplementedError, "#{self.class.name} must override provider_id"
    end
  end

  # Abstract base for text-to-speech providers (Piper, Kokoro, VibeVoice,
  # Qwen3-TTS, Spark-TTS, Bark, F5, fal-tts).
  #
  # @abstract
  class TtsProvider < BaseProvider
    # @param text [String]
    # @param voice [String, nil]
    # @param language [String, nil]
    # @return [Blazen::Compute::TtsResult]
    def synthesize(text, voice: nil, language: nil)
      raise NotImplementedError, "#{self.class.name} must override synthesize"
    end

    # @param text [String]
    # @param voice [String, nil]
    # @param language [String, nil]
    # @return [Blazen::Compute::TtsResult]
    def synthesize_blocking(text, voice: nil, language: nil)
      raise NotImplementedError, "#{self.class.name} must override synthesize_blocking"
    end
  end

  # Abstract base for speech-to-text providers (whisper.cpp, faster-whisper,
  # whisper-streaming, fal-stt).
  #
  # @abstract
  class SttProvider < BaseProvider
    # @param audio_source [String]
    # @param language [String, nil]
    # @return [Blazen::Compute::SttResult]
    def transcribe(audio_source, language: nil)
      raise NotImplementedError, "#{self.class.name} must override transcribe"
    end

    # @param audio_source [String]
    # @param language [String, nil]
    # @return [Blazen::Compute::SttResult]
    def transcribe_blocking(audio_source, language: nil)
      raise NotImplementedError, "#{self.class.name} must override transcribe_blocking"
    end
  end

  # Abstract base for music-generation providers (MusicGen, AudioGen,
  # Stable Audio, fal-music).
  #
  # @abstract
  class MusicProvider < BaseProvider
    # @param prompt [String]
    # @param duration_seconds [Float]
    # @return [Blazen::Compute::MusicResult]
    def generate_music(prompt, duration_seconds:)
      raise NotImplementedError, "#{self.class.name} must override generate_music"
    end

    # @param prompt [String]
    # @param duration_seconds [Float]
    # @return [Blazen::Compute::MusicResult]
    def generate_music_blocking(prompt, duration_seconds:)
      raise NotImplementedError,
            "#{self.class.name} must override generate_music_blocking"
    end

    # @param prompt [String]
    # @param duration_seconds [Float]
    # @return [Blazen::Compute::MusicResult]
    def generate_sfx(prompt, duration_seconds:)
      raise NotImplementedError, "#{self.class.name} must override generate_sfx"
    end

    # @param prompt [String]
    # @param duration_seconds [Float]
    # @return [Blazen::Compute::MusicResult]
    def generate_sfx_blocking(prompt, duration_seconds:)
      raise NotImplementedError, "#{self.class.name} must override generate_sfx_blocking"
    end
  end

  # Abstract base for voice-conversion providers (RVC, fal-vc).
  #
  # @abstract
  class VcProvider < BaseProvider
    # @param input_path [String]
    # @param target_voice_id [String]
    # @return [Blazen::Compute::VcResult]
    def convert_voice(input_path, target_voice_id:)
      raise NotImplementedError, "#{self.class.name} must override convert_voice"
    end

    # @param input_path [String]
    # @param target_voice_id [String]
    # @return [Blazen::Compute::VcResult]
    def convert_voice_blocking(input_path, target_voice_id:)
      raise NotImplementedError,
            "#{self.class.name} must override convert_voice_blocking"
    end

    # @param voice_id [String]
    # @param reference_path [String]
    # @return [void]
    def clone_voice(voice_id:, reference_path:)
      raise NotImplementedError, "#{self.class.name} must override clone_voice"
    end

    # @param voice_id [String]
    # @param reference_path [String]
    # @return [void]
    def clone_voice_blocking(voice_id:, reference_path:)
      raise NotImplementedError, "#{self.class.name} must override clone_voice_blocking"
    end

    # @return [Array<Blazen::Compute::TargetVoice>]
    def list_target_voices
      raise NotImplementedError, "#{self.class.name} must override list_target_voices"
    end

    # @return [Array<Blazen::Compute::TargetVoice>]
    def list_target_voices_blocking
      raise NotImplementedError,
            "#{self.class.name} must override list_target_voices_blocking"
    end
  end

  # Abstract base for image-to-3D providers (TripoSR).
  #
  # @abstract
  class ThreeDProvider < BaseProvider
    # @param image_bytes [String] encoded PNG / JPEG buffer
    # @param mesh_resolution [Integer]
    # @return [Blazen::Compute::ThreeDGenerateResult]
    def generate_from_image(image_bytes, mesh_resolution: 256)
      raise NotImplementedError,
            "#{self.class.name} must override generate_from_image"
    end

    # @param image_bytes [String]
    # @param mesh_resolution [Integer]
    # @return [Blazen::Compute::ThreeDGenerateResult]
    def generate_from_image_blocking(image_bytes, mesh_resolution: 256)
      raise NotImplementedError,
            "#{self.class.name} must override generate_from_image_blocking"
    end
  end

  # Abstract base for image-generation providers (diffusion, fal-image-gen).
  #
  # @abstract
  class ImageGenProvider < BaseProvider
    # @param prompt [String]
    # @param width [Integer, nil] +nil+ defers to backend default
    # @param height [Integer, nil]
    # @return [Blazen::Compute::ImageGenResult]
    def generate_image(prompt, width: nil, height: nil)
      raise NotImplementedError, "#{self.class.name} must override generate_image"
    end

    # @param prompt [String]
    # @param width [Integer, nil]
    # @param height [Integer, nil]
    # @return [Blazen::Compute::ImageGenResult]
    def generate_image_blocking(prompt, width: nil, height: nil)
      raise NotImplementedError,
            "#{self.class.name} must override generate_image_blocking"
    end
  end

  # Abstract base for embedding providers (FastEmbed, Tract, Candle,
  # OpenAI-embedding, fal-embedding), AND the concrete polymorphic handle
  # returned by per-engine +#as_embedding_provider+ conversions.
  #
  # An instance constructed directly via {.new}(handle) wraps a
  # +BlazenEmbeddingProvider *+ opaque (the cabi-side
  # +Arc<dyn EmbeddingProvider>+). Per-engine subclasses extend this with
  # engine-specific embed / dimensions / conversion methods.
  class EmbeddingProvider < BaseProvider
    # Constructs an {EmbeddingProvider} that wraps a
    # +BlazenEmbeddingProvider *+ opaque, taking ownership of the handle.
    # The handle is freed via +blazen_embedding_provider_free+ on GC.
    def initialize(handle, free_fn = nil)
      if free_fn.nil? && !handle.is_a?(::FFI::AutoPointer) &&
         Blazen::FFI.respond_to?(:blazen_embedding_provider_free)
        free_fn = Blazen::FFI.method(:blazen_embedding_provider_free)
      end
      super
    end

    # Returns an {EmbeddingProvider} (a thin wrapper over
    # +BlazenEmbeddingProvider *+). Default implementation on the abstract
    # base returns +self+; per-engine subclasses override this to call the
    # matching +blazen_<engine>_provider_as_embedding_provider+ C function.
    #
    # @return [EmbeddingProvider]
    def as_embedding_provider
      self
    end

    # @param texts [Array<String>]
    # @return [Blazen::EmbeddingVectors] wrapper around
    #   +BlazenEmbeddingVectors *+ (lazy float array accessors)
    def embed(texts)
      raise NotImplementedError, "#{self.class.name} must override embed"
    end

    # @param texts [Array<String>]
    # @return [Blazen::EmbeddingVectors]
    def embed_blocking(texts)
      raise NotImplementedError, "#{self.class.name} must override embed_blocking"
    end

    # @return [Integer] vector dimensionality
    def dimensions
      raise NotImplementedError, "#{self.class.name} must override dimensions"
    end
  end

  # Abstract base for chat-completion providers (OpenAI, Anthropic, Gemini,
  # Azure-OpenAI, Bedrock, fal-LLM, Mistral, Fireworks, DeepSeek, Perplexity,
  # Together, Groq, OpenRouter, Cohere, xAI), AND the concrete polymorphic
  # handle returned by per-engine +#as_llm_provider+ conversions.
  #
  # An instance constructed directly via {.new}(handle) wraps a
  # +BlazenLlmProvider *+ opaque (the cabi-side +Arc<dyn LlmProvider>+) and
  # is consumed by polymorphic entry points like {Blazen::Agents.new} and
  # {Blazen::Batch.complete}. Subclasses (per-engine providers) extend
  # this with engine-specific completion, streaming, and conversion
  # methods.
  class LlmProvider < BaseProvider
    # Constructs an {LlmProvider} that wraps a +BlazenLlmProvider *+
    # opaque, taking ownership of the handle. The handle is freed via
    # +blazen_llm_provider_free+ on GC.
    #
    # Per-engine subclasses bypass this default constructor and pass a
    # different +free_fn+ to +BaseProvider#initialize+; user code typically
    # builds an instance of this class through {#as_llm_provider} on a
    # per-engine provider rather than calling +.new+ directly.
    def initialize(handle, free_fn = nil)
      if free_fn.nil? && !handle.is_a?(::FFI::AutoPointer) &&
         Blazen::FFI.respond_to?(:blazen_llm_provider_free)
        free_fn = Blazen::FFI.method(:blazen_llm_provider_free)
      end
      super
    end

    # Returns an {LlmProvider} (a thin wrapper over +BlazenLlmProvider *+)
    # suitable for hand-off to {Blazen::Agents.new} / {Blazen::Batch.complete}.
    #
    # The default implementation on the abstract base returns +self+ — used
    # by callers who already hold an {LlmProvider}-typed handle. Per-engine
    # subclasses override this to call the matching
    # +blazen_<engine>_provider_as_llm_provider+ C function and wrap the
    # returned opaque in a fresh {LlmProvider} (with its own
    # +blazen_llm_provider_free+ finalizer).
    #
    # @return [LlmProvider]
    def as_llm_provider
      self
    end

    # @param request [Blazen::Llm::ModelRequest]
    # @return [Blazen::Llm::ModelResponse]
    def complete(request)
      raise NotImplementedError, "#{self.class.name} must override complete"
    end

    # @param request [Blazen::Llm::ModelRequest]
    # @return [Blazen::Llm::ModelResponse]
    def complete_blocking(request)
      raise NotImplementedError, "#{self.class.name} must override complete_blocking"
    end
  end

  # Idiomatic Ruby wrapper around a +BlazenEmbeddingVectors+ handle (the
  # typed result of +EmbeddingProvider#embed+). Exposes the embedded
  # vectors lazily via {#count}, {#dim}, and {#at}.
  class EmbeddingVectors
    # @param raw_ptr [::FFI::Pointer]
    def initialize(raw_ptr)
      if raw_ptr.nil? || raw_ptr.null?
        raise ArgumentError, "EmbeddingVectors: pointer must be non-null"
      end

      @ptr = ::FFI::AutoPointer.new(
        raw_ptr, Blazen::FFI.method(:blazen_embedding_vectors_free),
      )
    end

    # @return [::FFI::AutoPointer]
    attr_reader :ptr

    # @return [Integer] number of embedded vectors
    def count
      Blazen::FFI.blazen_embedding_vectors_count(@ptr)
    end
    alias length count
    alias size count

    # @param vec_idx [Integer]
    # @return [Integer] dimensionality of vector +vec_idx+
    def dim(vec_idx = 0)
      Blazen::FFI.blazen_embedding_vectors_dim(@ptr, Integer(vec_idx))
    end

    # @param vec_idx [Integer]
    # @param dim_idx [Integer]
    # @return [Float]
    def value_at(vec_idx, dim_idx)
      Blazen::FFI.blazen_embedding_vectors_get(
        @ptr, Integer(vec_idx), Integer(dim_idx),
      )
    end

    # @param vec_idx [Integer]
    # @return [Array<Float>]
    def at(vec_idx)
      idx = Integer(vec_idx)
      d   = dim(idx)
      Array.new(d) { |j| Blazen::FFI.blazen_embedding_vectors_get(@ptr, idx, j) }
    end
    alias [] at

    # @return [Array<Array<Float>>]
    def to_a
      Array.new(count) { |i| at(i) }
    end
  end
end
