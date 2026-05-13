# frozen_string_literal: true

module Blazen
  # Helpers and wrapper classes for the compute family of models:
  # speech-to-text ({SttModel}), text-to-speech ({TtsModel}), and image
  # generation ({ImageGenModel}).
  #
  # Provides:
  #
  # - keyword-arg module functions ({transcribe}, {synthesize}, {generate})
  #   that delegate to the blocking model methods,
  # - keyword-arg factories for the fal.ai cloud backend and (when the
  #   underlying native library was built with the matching feature) the
  #   local Piper / Whisper / diffusion backends,
  # - the {TtsModel}, {SttModel}, {ImageGenModel} wrapper classes (each
  #   exposing both async and +_blocking+ method variants),
  # - the {TtsResult}, {SttResult}, {ImageGenResult} result wrappers.
  module Compute
    # ----------------------------------------------------------------
    # Module-level shorthand functions
    # ----------------------------------------------------------------

    module_function

    # Transcribes audio with an {SttModel}.
    #
    # @param model [Blazen::Compute::SttModel]
    # @param audio_source [String] base64-encoded audio, file path, or URL
    #   — interpretation depends on the underlying STT backend
    # @param language [String, nil] expected ISO-639-1 language code
    # @return [Blazen::Compute::SttResult]
    def transcribe(model, audio_source, language: nil)
      model.transcribe_blocking(audio_source, language: language)
    end

    # Synthesizes speech with a {TtsModel}.
    #
    # @param model [Blazen::Compute::TtsModel]
    # @param text [String] text to speak
    # @param voice [String, nil] voice identifier
    # @param language [String, nil] language code
    # @return [Blazen::Compute::TtsResult]
    def synthesize(model, text, voice: nil, language: nil)
      model.synthesize_blocking(text, voice: voice, language: language)
    end

    # Generates images with an {ImageGenModel}.
    #
    # @param model [Blazen::Compute::ImageGenModel]
    # @param prompt [String] image prompt
    # @param negative_prompt [String] negative prompt (empty string when
    #   omitted — the cabi requires a non-null buffer)
    # @param width [Integer, nil] override width in pixels
    # @param height [Integer, nil] override height in pixels
    # @param num_images [Integer, nil] number of images to produce
    # @param model_override [String, nil] override the model identifier
    # @return [Blazen::Compute::ImageGenResult]
    def generate(model, prompt:, negative_prompt: "", width: nil, height: nil,
                 num_images: nil, model_override: nil)
      model.generate_blocking(
        prompt:          prompt,
        negative_prompt: negative_prompt,
        width:           width,
        height:          height,
        num_images:      num_images,
        model_override:  model_override,
      )
    end

    # ----------------------------------------------------------------
    # Provider factories
    # ----------------------------------------------------------------

    # Build a fal.ai-backed text-to-speech model.
    #
    # @param api_key [String] fal.ai API key (empty string resolves
    #   +FAL_KEY+ from the environment upstream)
    # @param model [String, nil] override the default TTS endpoint
    # @return [Blazen::Compute::TtsModel]
    def self.fal_tts(api_key:, model: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key.to_s) do |key|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.blazen_tts_model_new_fal(key, m, out_model, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      TtsModel.new(out_model.read_pointer)
    end

    # Build a fal.ai-backed speech-to-text model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @return [Blazen::Compute::SttModel]
    def self.fal_stt(api_key:, model: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key.to_s) do |key|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.blazen_stt_model_new_fal(key, m, out_model, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      SttModel.new(out_model.read_pointer)
    end

    # Build a fal.ai-backed image-generation model.
    #
    # @param api_key [String]
    # @param model [String, nil]
    # @return [Blazen::Compute::ImageGenModel]
    def self.fal_image_gen(api_key:, model: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key.to_s) do |key|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.blazen_image_gen_model_new_fal(key, m, out_model, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      ImageGenModel.new(out_model.read_pointer)
    end

    # Build a local Piper text-to-speech model (when the native library is
    # built with the +piper+ feature).
    #
    # @param model_id [String, nil] Piper voice id (e.g.
    #   +"en_US-amy-medium"+); +nil+ leaves the upstream default
    # @param speaker_id [Integer, nil]
    # @param sample_rate [Integer, nil]
    # @return [Blazen::Compute::TtsModel]
    # @raise [Blazen::UnsupportedError] when the +piper+ feature is missing
    def self.piper_tts(model_id: nil, speaker_id: nil, sample_rate: nil)
      unless Blazen::FFI.respond_to?(:blazen_tts_model_new_piper)
        raise Blazen::UnsupportedError, "blazen was built without the 'piper' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      sid = speaker_id.nil? ? -1 : Integer(speaker_id)
      sr  = sample_rate.nil? ? -1 : Integer(sample_rate)
      Blazen::FFI.with_cstring(model_id) do |mid|
        Blazen::FFI.blazen_tts_model_new_piper(mid, sid, sr, out_model, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      TtsModel.new(out_model.read_pointer)
    end

    # Build a local whisper.cpp speech-to-text model (when the native
    # library is built with the +whispercpp+ feature).
    #
    # @param model [String, nil] whisper variant (e.g. +"small"+); +nil+
    #   defaults to +"small"+ upstream
    # @param device [String, nil] +"cpu"+ / +"cuda"+ / +"metal"+ / etc.
    # @param language [String, nil] default ISO-639-1 hint
    # @return [Blazen::Compute::SttModel]
    # @raise [Blazen::UnsupportedError] when the +whispercpp+ feature is missing
    def self.whisper_stt(model: nil, device: nil, language: nil)
      unless Blazen::FFI.respond_to?(:blazen_stt_model_new_whisper)
        raise Blazen::UnsupportedError, "blazen was built without the 'whispercpp' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(model) do |m|
        Blazen::FFI.with_cstring(device) do |dev|
          Blazen::FFI.with_cstring(language) do |lang|
            Blazen::FFI.blazen_stt_model_new_whisper(m, dev, lang, out_model, out_err)
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      SttModel.new(out_model.read_pointer)
    end

    # Build a local diffusion-rs image-generation model (when the native
    # library is built with the +diffusion+ feature).
    #
    # @param model_id [String, nil] Hugging Face repo id
    # @param device [String, nil]
    # @param width [Integer, nil]
    # @param height [Integer, nil]
    # @param num_inference_steps [Integer, nil]
    # @param guidance_scale [Float, nil] +nil+ encodes to NaN upstream
    # @return [Blazen::Compute::ImageGenModel]
    # @raise [Blazen::UnsupportedError] when the +diffusion+ feature is
    #   missing
    def self.diffusion(model_id: nil, device: nil, width: nil, height: nil,
                       num_inference_steps: nil, guidance_scale: nil)
      unless Blazen::FFI.respond_to?(:blazen_image_gen_model_new_diffusion)
        raise Blazen::UnsupportedError, "blazen was built without the 'diffusion' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      w  = width.nil? ? -1 : Integer(width)
      h  = height.nil? ? -1 : Integer(height)
      ns = num_inference_steps.nil? ? -1 : Integer(num_inference_steps)
      gs = guidance_scale.nil? ? Float::NAN : Float(guidance_scale)
      Blazen::FFI.with_cstring(model_id) do |mid|
        Blazen::FFI.with_cstring(device) do |dev|
          Blazen::FFI.blazen_image_gen_model_new_diffusion(
            mid, dev, w, h, ns, gs, out_model, out_err,
          )
        end
      end
      Blazen::FFI.check_error!(out_err)
      ImageGenModel.new(out_model.read_pointer)
    end

    # ----------------------------------------------------------------
    # Wrapper classes
    # ----------------------------------------------------------------

    # Idiomatic Ruby wrapper around a +BlazenTtsModel+ handle.
    class TtsModel
      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "TtsModel: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_tts_model_free))
      end

      # Synthesizes +text+ asynchronously (integrates with
      # {Fiber.scheduler}).
      #
      # @param text [String]
      # @param voice [String, nil]
      # @param language [String, nil]
      # @return [Blazen::Compute::TtsResult]
      def synthesize(text, voice: nil, language: nil)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        fut = Blazen::FFI.with_cstring(text.to_s) do |t|
          Blazen::FFI.with_cstring(voice) do |v|
            Blazen::FFI.with_cstring(language) do |lang|
              Blazen::FFI.blazen_tts_model_synthesize(@ptr, t, v, lang)
            end
          end
        end
        if fut.nil? || fut.null?
          raise Blazen::ValidationError, "blazen_tts_model_synthesize returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_tts_result(f, out_result, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        TtsResult.new(out_result.read_pointer)
      end

      # Blocking-thread variant of {#synthesize}.
      #
      # @param text [String]
      # @param voice [String, nil]
      # @param language [String, nil]
      # @return [Blazen::Compute::TtsResult]
      def synthesize_blocking(text, voice: nil, language: nil)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(text.to_s) do |t|
          Blazen::FFI.with_cstring(voice) do |v|
            Blazen::FFI.with_cstring(language) do |lang|
              Blazen::FFI.blazen_tts_model_synthesize_blocking(@ptr, t, v, lang, out_result, out_err)
            end
          end
        end
        Blazen::FFI.check_error!(out_err)
        TtsResult.new(out_result.read_pointer)
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenSttModel+ handle.
    class SttModel
      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "SttModel: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_stt_model_free))
      end

      # Transcribes +audio_source+ asynchronously.
      #
      # @param audio_source [String]
      # @param language [String, nil]
      # @return [Blazen::Compute::SttResult]
      def transcribe(audio_source, language: nil)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        fut = Blazen::FFI.with_cstring(audio_source.to_s) do |src|
          Blazen::FFI.with_cstring(language) do |lang|
            Blazen::FFI.blazen_stt_model_transcribe(@ptr, src, lang)
          end
        end
        if fut.nil? || fut.null?
          raise Blazen::ValidationError, "blazen_stt_model_transcribe returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_stt_result(f, out_result, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        SttResult.new(out_result.read_pointer)
      end

      # Blocking-thread variant of {#transcribe}.
      #
      # @param audio_source [String]
      # @param language [String, nil]
      # @return [Blazen::Compute::SttResult]
      def transcribe_blocking(audio_source, language: nil)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(audio_source.to_s) do |src|
          Blazen::FFI.with_cstring(language) do |lang|
            Blazen::FFI.blazen_stt_model_transcribe_blocking(@ptr, src, lang, out_result, out_err)
          end
        end
        Blazen::FFI.check_error!(out_err)
        SttResult.new(out_result.read_pointer)
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenImageGenModel+ handle.
    class ImageGenModel
      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "ImageGenModel: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_image_gen_model_free))
      end

      # Asynchronously generates one or more images.
      #
      # @param prompt [String]
      # @param negative_prompt [String] empty string when omitted
      # @param width [Integer, nil]
      # @param height [Integer, nil]
      # @param num_images [Integer, nil]
      # @param model_override [String, nil]
      # @return [Blazen::Compute::ImageGenResult]
      def generate(prompt:, negative_prompt: "", width: nil, height: nil,
                   num_images: nil, model_override: nil)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        w = width.nil?     ? -1 : Integer(width)
        h = height.nil?    ? -1 : Integer(height)
        n = num_images.nil? ? -1 : Integer(num_images)
        fut = Blazen::FFI.with_cstring(prompt.to_s) do |p|
          Blazen::FFI.with_cstring(negative_prompt.to_s) do |np|
            Blazen::FFI.with_cstring(model_override) do |mo|
              Blazen::FFI.blazen_image_gen_model_generate(@ptr, p, np, w, h, n, mo)
            end
          end
        end
        if fut.nil? || fut.null?
          raise Blazen::ValidationError, "blazen_image_gen_model_generate returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_image_gen_result(f, out_result, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        ImageGenResult.new(out_result.read_pointer)
      end

      # Blocking-thread variant of {#generate}.
      #
      # @param prompt [String]
      # @param negative_prompt [String]
      # @param width [Integer, nil]
      # @param height [Integer, nil]
      # @param num_images [Integer, nil]
      # @param model_override [String, nil]
      # @return [Blazen::Compute::ImageGenResult]
      def generate_blocking(prompt:, negative_prompt: "", width: nil, height: nil,
                            num_images: nil, model_override: nil)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        w = width.nil?     ? -1 : Integer(width)
        h = height.nil?    ? -1 : Integer(height)
        n = num_images.nil? ? -1 : Integer(num_images)
        Blazen::FFI.with_cstring(prompt.to_s) do |p|
          Blazen::FFI.with_cstring(negative_prompt.to_s) do |np|
            Blazen::FFI.with_cstring(model_override) do |mo|
              Blazen::FFI.blazen_image_gen_model_generate_blocking(
                @ptr, p, np, w, h, n, mo, out_result, out_err,
              )
            end
          end
        end
        Blazen::FFI.check_error!(out_err)
        ImageGenResult.new(out_result.read_pointer)
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenTtsResult+ handle.
    class TtsResult
      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "TtsResult: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_tts_result_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [String] base64-encoded audio bytes (empty string when the
      #   provider only returned a URL)
      def audio_base64
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_tts_result_audio_base64(@ptr))
      end

      # @return [String]
      def mime_type
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_tts_result_mime_type(@ptr))
      end

      # @return [Integer]
      def duration_ms
        Blazen::FFI.blazen_tts_result_duration_ms(@ptr)
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenSttResult+ handle.
    class SttResult
      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "SttResult: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_stt_result_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [String]
      def transcript
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_stt_result_transcript(@ptr))
      end

      # @return [String]
      def language
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_stt_result_language(@ptr))
      end

      # @return [Integer]
      def duration_ms
        Blazen::FFI.blazen_stt_result_duration_ms(@ptr)
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenImageGenResult+ handle.
    #
    # Exposes the generated media artefacts via {#images}, each wrapped in
    # {Blazen::Llm::Media} when that class is available (post-R7 Agent A)
    # or as an {::FFI::AutoPointer} drop hook otherwise.
    class ImageGenResult
      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "ImageGenResult: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_image_gen_result_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [Integer] number of generated images
      def size
        Blazen::FFI.blazen_image_gen_result_images_count(@ptr)
      end
      alias length size

      # @param idx [Integer]
      # @return [Blazen::Llm::Media, ::FFI::AutoPointer]
      def at(idx)
        raw = Blazen::FFI.blazen_image_gen_result_images_get(@ptr, Integer(idx))
        raise IndexError, "ImageGenResult: index #{idx} out of bounds (size=#{size})" if raw.nil? || raw.null?

        if defined?(Blazen::Llm) && defined?(Blazen::Llm::Media)
          Blazen::Llm::Media.new(raw)
        else
          ::FFI::AutoPointer.new(raw, Blazen::FFI.method(:blazen_media_free))
        end
      end
      alias [] at

      # @return [Array<Blazen::Llm::Media>]
      def images
        Array.new(size) { |i| at(i) }
      end
    end
  end
end
