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

    # Build a fal.ai-backed music-generation model.
    #
    # @param api_key [String] fal.ai API key (empty string resolves
    #   +FAL_KEY+ from the environment upstream)
    # @param model [String, nil] override the default fal music / SFX
    #   endpoint identifier
    # @return [Blazen::Compute::MusicModel]
    def self.fal_music(api_key:, model: nil)
      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key.to_s) do |key|
        Blazen::FFI.with_cstring(model) do |m|
          Blazen::FFI.blazen_music_model_new_fal(key, m, out_model, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      MusicModel.new(out_model.read_pointer)
    end

    # Build a native MusicGen-backed music model (when the cabi library is
    # built with the +music-musicgen+ feature).
    #
    # @param variant [String, nil] +"small"+ / +"medium"+ / +"large"+
    #   (case-insensitive); +nil+ defaults to +"small"+ upstream
    # @param device [String, nil] +"cpu"+ / +"cuda"+ / +"cuda:N"+ /
    #   +"metal"+ / +"metal:N"+; +nil+ defers to backend auto-detection
    # @param cache_dir [String, nil] override the Hugging Face Hub cache dir
    # @param max_duration_seconds [Float, nil] +nil+ defaults to 30 s upstream
    # @return [Blazen::Compute::MusicModel]
    # @raise [Blazen::UnsupportedError] when the +music-musicgen+ feature is missing
    def self.musicgen(variant: nil, device: nil, cache_dir: nil,
                      max_duration_seconds: nil)
      unless Blazen::FFI.respond_to?(:blazen_music_model_new_musicgen)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'music-musicgen' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      max_dur   = max_duration_seconds.nil? ? Float::NAN : Float(max_duration_seconds)
      Blazen::FFI.with_cstring(variant) do |v|
        Blazen::FFI.with_cstring(device) do |dev|
          Blazen::FFI.with_cstring(cache_dir) do |cd|
            Blazen::FFI.blazen_music_model_new_musicgen(
              v, dev, cd, max_dur, out_model, out_err,
            )
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      MusicModel.new(out_model.read_pointer)
    end

    # Build a native Stable Audio Open-backed music model (when the cabi
    # library is built with the +music-stable-audio+ feature).
    #
    # @param tokenizer_path [String] REQUIRED — path to the T5
    #   +tokenizer.json+ shipped with the Stable Audio Open repo
    # @param variant [String, nil] +"small"+ / +"open-1.0"+ /
    #   +"open1.0"+ (case-insensitive); +nil+ defaults to +"small"+
    # @param device [String, nil]
    # @param max_duration_seconds [Float, nil] +nil+ uses the variant's
    #   internal default
    # @return [Blazen::Compute::MusicModel]
    # @raise [Blazen::UnsupportedError] when the +music-stable-audio+ feature is missing
    def self.stable_audio(tokenizer_path:, variant: nil, device: nil,
                          max_duration_seconds: nil)
      unless Blazen::FFI.respond_to?(:blazen_music_model_new_stable_audio)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'music-stable-audio' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      max_dur   = max_duration_seconds.nil? ? Float::NAN : Float(max_duration_seconds)
      Blazen::FFI.with_cstring(variant) do |v|
        Blazen::FFI.with_cstring(tokenizer_path.to_s) do |tok|
          Blazen::FFI.with_cstring(device) do |dev|
            Blazen::FFI.blazen_music_model_new_stable_audio(
              v, tok, dev, max_dur, out_model, out_err,
            )
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      MusicModel.new(out_model.read_pointer)
    end

    # Build a native AudioGen-backed music / SFX model (when the cabi
    # library is built with the +music-audiogen+ feature).
    #
    # @param repo_id [String, nil] HF repo override (default
    #   +"facebook/audiogen-medium"+)
    # @param revision [String, nil] commit / tag pin
    # @param device [String, nil]
    # @param cache_dir [String, nil]
    # @param max_duration_seconds [Float, nil]
    # @return [Blazen::Compute::MusicModel]
    # @raise [Blazen::UnsupportedError] when the +music-audiogen+ feature is missing
    def self.audiogen(repo_id: nil, revision: nil, device: nil, cache_dir: nil,
                      max_duration_seconds: nil)
      unless Blazen::FFI.respond_to?(:blazen_music_model_new_audiogen)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'music-audiogen' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      max_dur   = max_duration_seconds.nil? ? Float::NAN : Float(max_duration_seconds)
      Blazen::FFI.with_cstring(repo_id) do |r|
        Blazen::FFI.with_cstring(revision) do |rev|
          Blazen::FFI.with_cstring(device) do |dev|
            Blazen::FFI.with_cstring(cache_dir) do |cd|
              Blazen::FFI.blazen_music_model_new_audiogen(
                r, rev, dev, cd, max_dur, out_model, out_err,
              )
            end
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      MusicModel.new(out_model.read_pointer)
    end

    # Build a native RVC-backed voice-conversion model (when the cabi
    # library is built with the +audio-vc-rvc+ feature).
    #
    # The RVC pipeline reads its voice directory + device lazily on the
    # first conversion call, so constructing the model up-front before
    # spinning off threads is safe.
    #
    # @param voice_dir [String, nil] override the per-process
    #   +BLAZEN_RVC_VOICE_DIR+ environment variable
    # @param device [String, nil] +"cpu"+ / +"cuda"+ / +"cuda:N"+ /
    #   +"metal"+ / +"metal:N"+; +nil+ defers to CPU
    # @return [Blazen::Compute::VcModel]
    # @raise [Blazen::UnsupportedError] when the +audio-vc-rvc+ feature
    #   is missing
    def self.rvc(voice_dir: nil, device: nil)
      unless Blazen::FFI.respond_to?(:blazen_vc_model_new_rvc)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'audio-vc-rvc' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(voice_dir) do |vd|
        Blazen::FFI.with_cstring(device) do |dev|
          Blazen::FFI.blazen_vc_model_new_rvc(vd, dev, out_model, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      VcModel.new(out_model.read_pointer)
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

    # Idiomatic Ruby wrapper around a +BlazenMusicModel+ handle.
    #
    # Exposes the four music-flavored entry points across two surface
    # variants (async + blocking, music + SFX, non-streaming + streaming).
    # The non-streaming variants resolve to a {MusicResult} (URL or
    # inline bytes); the streaming variants drive a {Blazen::Streaming}
    # sink that receives each {MusicChunk} as it's emitted.
    class MusicModel
      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "MusicModel: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_music_model_free))
      end

      # Asynchronously generate +duration_seconds+ of music conditioned
      # on +prompt+. Composes with +Fiber.scheduler+.
      #
      # @param prompt [String]
      # @param duration_seconds [Float]
      # @return [Blazen::Compute::MusicResult]
      def generate_music(prompt, duration_seconds)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        dur        = Float(duration_seconds)
        fut = Blazen::FFI.with_cstring(prompt.to_s) do |p|
          Blazen::FFI.blazen_music_model_generate_music(@ptr, p, dur)
        end
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_music_model_generate_music returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_music_result(f, out_result, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        MusicResult.new(out_result.read_pointer)
      end

      # Blocking-thread variant of {#generate_music}.
      #
      # @param prompt [String]
      # @param duration_seconds [Float]
      # @return [Blazen::Compute::MusicResult]
      def generate_music_blocking(prompt, duration_seconds)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        dur        = Float(duration_seconds)
        Blazen::FFI.with_cstring(prompt.to_s) do |p|
          Blazen::FFI.blazen_music_model_generate_music_blocking(
            @ptr, p, dur, out_result, out_err,
          )
        end
        Blazen::FFI.check_error!(out_err)
        MusicResult.new(out_result.read_pointer)
      end

      # Asynchronously generate +duration_seconds+ of SFX audio
      # conditioned on +prompt+.
      #
      # @param prompt [String]
      # @param duration_seconds [Float]
      # @return [Blazen::Compute::MusicResult]
      def generate_sfx(prompt, duration_seconds)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        dur        = Float(duration_seconds)
        fut = Blazen::FFI.with_cstring(prompt.to_s) do |p|
          Blazen::FFI.blazen_music_model_generate_sfx(@ptr, p, dur)
        end
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_music_model_generate_sfx returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_music_result(f, out_result, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        MusicResult.new(out_result.read_pointer)
      end

      # Blocking-thread variant of {#generate_sfx}.
      #
      # @param prompt [String]
      # @param duration_seconds [Float]
      # @return [Blazen::Compute::MusicResult]
      def generate_sfx_blocking(prompt, duration_seconds)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        dur        = Float(duration_seconds)
        Blazen::FFI.with_cstring(prompt.to_s) do |p|
          Blazen::FFI.blazen_music_model_generate_sfx_blocking(
            @ptr, p, dur, out_result, out_err,
          )
        end
        Blazen::FFI.check_error!(out_err)
        MusicResult.new(out_result.read_pointer)
      end

      # Drive a streaming music generation, dispatching each {MusicChunk}
      # to the supplied callbacks (or block). Blocks the calling thread
      # on the cabi tokio runtime until +on_done+ or +on_error+ fires.
      #
      # See {Blazen::Streaming.stream_music} for the full kwargs / block
      # contract.
      #
      # @param prompt [String]
      # @param duration_seconds [Float]
      # @return [void]
      def stream_generate_music(prompt, duration_seconds,
                                on_chunk: nil, on_done: nil, on_error: nil, &block)
        Blazen::Streaming.stream_music(
          self, prompt, duration_seconds,
          mode: :music, blocking: true,
          on_chunk: on_chunk, on_done: on_done, on_error: on_error,
          &block
        )
      end

      # Async variant of {#stream_generate_music} — composes with
      # +Fiber.scheduler+ via {Blazen::FFI.await_future}.
      #
      # @param prompt [String]
      # @param duration_seconds [Float]
      # @return [void]
      def stream_generate_music_async(prompt, duration_seconds,
                                      on_chunk: nil, on_done: nil, on_error: nil, &block)
        Blazen::Streaming.stream_music(
          self, prompt, duration_seconds,
          mode: :music, blocking: false,
          on_chunk: on_chunk, on_done: on_done, on_error: on_error,
          &block
        )
      end

      # Drive a streaming SFX generation. See {#stream_generate_music}.
      #
      # @param prompt [String]
      # @param duration_seconds [Float]
      # @return [void]
      def stream_generate_sfx(prompt, duration_seconds,
                              on_chunk: nil, on_done: nil, on_error: nil, &block)
        Blazen::Streaming.stream_music(
          self, prompt, duration_seconds,
          mode: :sfx, blocking: true,
          on_chunk: on_chunk, on_done: on_done, on_error: on_error,
          &block
        )
      end

      # Async variant of {#stream_generate_sfx}.
      #
      # @param prompt [String]
      # @param duration_seconds [Float]
      # @return [void]
      def stream_generate_sfx_async(prompt, duration_seconds,
                                    on_chunk: nil, on_done: nil, on_error: nil, &block)
        Blazen::Streaming.stream_music(
          self, prompt, duration_seconds,
          mode: :sfx, blocking: false,
          on_chunk: on_chunk, on_done: on_done, on_error: on_error,
          &block
        )
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenMusicChunk+ handle.
    #
    # Caller-owned: the cabi hands ownership of the chunk to the music
    # stream sink's +on_chunk+ callback, which wraps it in an
    # {::FFI::AutoPointer} so Ruby GC frees the chunk (via
    # +blazen_music_chunk_free+) once the user's handler returns.
    class MusicChunk
      # @api private
      # @param raw_ptr [::FFI::Pointer] caller-owned +BlazenMusicChunk *+
      def initialize(raw_ptr)
        if raw_ptr.nil? || raw_ptr.null?
          raise Blazen::InternalError, "MusicChunk: native pointer is null"
        end

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_music_chunk_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [Array<Float>] PCM sample slice (f32) for this chunk
      def samples
        len_ptr = ::FFI::MemoryPointer.new(:size_t)
        base    = Blazen::FFI.blazen_music_chunk_samples(@ptr, len_ptr)
        len     = len_ptr.read(:size_t)
        return [] if base.nil? || base.null? || len.zero?

        base.read_array_of_float(len)
      end

      # @return [Boolean] whether this is the final chunk of the stream
      def final?
        Blazen::FFI.blazen_music_chunk_is_final(@ptr)
      end

      # @return [Float, nil] per-chunk latency in seconds, or +nil+ when
      #   the backend did not report a measurement (NaN sentinel upstream)
      def latency_seconds
        v = Blazen::FFI.blazen_music_chunk_latency_seconds(@ptr)
        v.nan? ? nil : v
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenMusicResult+ handle.
    class MusicResult
      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "MusicResult: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_music_result_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [String] the encoded audio bytes (binary string). Empty
      #   when the upstream provider only returned a URL (check {#url}).
      def bytes
        len_ptr = ::FFI::MemoryPointer.new(:size_t)
        base    = Blazen::FFI.blazen_music_result_bytes(@ptr, len_ptr)
        len     = len_ptr.read(:size_t)
        return String.new(encoding: Encoding::ASCII_8BIT) if base.nil? || base.null? || len.zero?

        base.read_bytes(len).force_encoding(Encoding::ASCII_8BIT)
      end

      # @return [String, nil] IANA MIME type of the encoded audio
      def mime_type
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_music_result_mime_type(@ptr))
      end

      # @return [Integer] sample rate in Hz (+0+ when not reported)
      def sample_rate
        Blazen::FFI.blazen_music_result_sample_rate(@ptr)
      end

      # @return [Integer] channel count (1 = mono, 2 = stereo;
      #   +0+ when not reported)
      def channels
        Blazen::FFI.blazen_music_result_channels(@ptr)
      end

      # @return [Float, nil] duration in seconds, or +nil+ when the
      #   upstream provider did not report a duration (encoded as +0.0+
      #   on the wire)
      def duration_seconds
        v = Blazen::FFI.blazen_music_result_duration_seconds(@ptr)
        v.zero? ? nil : v
      end

      # @return [String, nil] URL of the audio asset when the upstream
      #   provider returned a link rather than inline bytes; +nil+ for
      #   inline-bytes results (cabi returns the empty string in that
      #   case, which we normalise to +nil+)
      def url
        s = Blazen::FFI.consume_cstring(Blazen::FFI.blazen_music_result_url(@ptr))
        return nil if s.nil? || s.empty?

        s
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenVcModel+ handle.
    #
    # Exposes the voice-conversion entry points across async + blocking
    # variants: file-to-result conversion, target-voice listing /
    # registration, and streaming PCM-to-PCM conversion (which drives a
    # {Blazen::Streaming} sink that receives each {VcChunk} as it's
    # emitted).
    class VcModel
      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "VcModel: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_vc_model_free))
      end

      # Asynchronously convert the source utterance at +input_path+ into
      # the voice of registered target speaker +target_voice_id+.
      # Composes with +Fiber.scheduler+.
      #
      # @param input_path [String]
      # @param target_voice_id [String]
      # @return [Blazen::Compute::VcResult]
      def convert_voice(input_path, target_voice_id)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        fut = Blazen::FFI.with_cstring(input_path.to_s) do |ip|
          Blazen::FFI.with_cstring(target_voice_id.to_s) do |v|
            Blazen::FFI.blazen_vc_model_convert_voice(@ptr, ip, v)
          end
        end
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_vc_model_convert_voice returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_vc_result(f, out_result, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        VcResult.new(out_result.read_pointer)
      end

      # Blocking-thread variant of {#convert_voice}.
      #
      # @param input_path [String]
      # @param target_voice_id [String]
      # @return [Blazen::Compute::VcResult]
      def convert_voice_blocking(input_path, target_voice_id)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(input_path.to_s) do |ip|
          Blazen::FFI.with_cstring(target_voice_id.to_s) do |v|
            Blazen::FFI.blazen_vc_model_convert_voice_blocking(
              @ptr, ip, v, out_result, out_err,
            )
          end
        end
        Blazen::FFI.check_error!(out_err)
        VcResult.new(out_result.read_pointer)
      end

      # Asynchronously list the target voices this backend can render.
      #
      # @return [Array<Blazen::Compute::TargetVoice>]
      def list_target_voices
        out_list = ::FFI::MemoryPointer.new(:pointer)
        out_err  = ::FFI::MemoryPointer.new(:pointer)
        fut = Blazen::FFI.blazen_vc_model_list_target_voices(@ptr)
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_vc_model_list_target_voices returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_target_voice_list(f, out_list, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        self.class._take_target_voice_list(out_list.read_pointer)
      end

      # Blocking-thread variant of {#list_target_voices}.
      #
      # @return [Array<Blazen::Compute::TargetVoice>]
      def list_target_voices_blocking
        out_list = ::FFI::MemoryPointer.new(:pointer)
        out_err  = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_vc_model_list_target_voices_blocking(
          @ptr, out_list, out_err,
        )
        Blazen::FFI.check_error!(out_err)
        self.class._take_target_voice_list(out_list.read_pointer)
      end

      # Asynchronously register a new target voice for the backend,
      # sourcing its identity from the reference utterance at +ref_path+.
      #
      # @param voice_id [String]
      # @param ref_path [String]
      # @return [void]
      def register_target_voice(voice_id, ref_path)
        fut = Blazen::FFI.with_cstring(voice_id.to_s) do |v|
          Blazen::FFI.with_cstring(ref_path.to_s) do |rp|
            Blazen::FFI.blazen_vc_model_register_target_voice(@ptr, v, rp)
          end
        end
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_vc_model_register_target_voice returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          out_err = ::FFI::MemoryPointer.new(:pointer)
          Blazen::FFI.blazen_future_take_unit(f, out_err)
          Blazen::FFI.check_error!(out_err)
        end
        nil
      end

      # Blocking-thread variant of {#register_target_voice}.
      #
      # @param voice_id [String]
      # @param ref_path [String]
      # @return [void]
      def register_target_voice_blocking(voice_id, ref_path)
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(voice_id.to_s) do |v|
          Blazen::FFI.with_cstring(ref_path.to_s) do |rp|
            Blazen::FFI.blazen_vc_model_register_target_voice_blocking(
              @ptr, v, rp, out_err,
            )
          end
        end
        Blazen::FFI.check_error!(out_err)
        nil
      end

      # Drive a streaming voice conversion across +pcm_samples+ (an Array
      # of f32 PCM samples), dispatching each {VcChunk} to the supplied
      # callbacks (or block). Blocks the calling thread on the cabi
      # tokio runtime until +on_done+ or +on_error+ fires.
      #
      # See {Blazen::Streaming.stream_convert} for the full kwargs /
      # block contract.
      #
      # @param pcm_samples [Array<Float>, #to_a]
      # @param target_voice_id [String]
      # @return [void]
      def stream_convert_pcm(pcm_samples, target_voice_id,
                             on_chunk: nil, on_done: nil, on_error: nil, &block)
        Blazen::Streaming.stream_convert(
          self, pcm_samples, target_voice_id,
          blocking: true,
          on_chunk: on_chunk, on_done: on_done, on_error: on_error,
          &block
        )
      end

      # Async variant of {#stream_convert_pcm} — composes with
      # +Fiber.scheduler+ via {Blazen::FFI.await_future}.
      #
      # @param pcm_samples [Array<Float>, #to_a]
      # @param target_voice_id [String]
      # @return [void]
      def stream_convert_pcm_async(pcm_samples, target_voice_id,
                                   on_chunk: nil, on_done: nil, on_error: nil, &block)
        Blazen::Streaming.stream_convert(
          self, pcm_samples, target_voice_id,
          blocking: false,
          on_chunk: on_chunk, on_done: on_done, on_error: on_error,
          &block
        )
      end

      # @api private
      # Drains a freshly-popped +BlazenTargetVoiceList *+ into an array
      # of {TargetVoice} wrappers, then frees the list. Used by the
      # list_target_voices entry points above.
      def self._take_target_voice_list(list_ptr)
        return [] if list_ptr.nil? || list_ptr.null?

        begin
          len = Blazen::FFI.blazen_target_voice_list_len(list_ptr)
          Array.new(len) do |i|
            raw = Blazen::FFI.blazen_target_voice_list_take(list_ptr, i)
            TargetVoice.new(raw)
          end
        ensure
          Blazen::FFI.blazen_target_voice_list_free(list_ptr)
        end
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenVcChunk+ handle.
    #
    # Caller-owned: the cabi hands ownership of the chunk to the vc
    # stream sink's +on_chunk+ callback, which wraps it in an
    # {::FFI::AutoPointer} so Ruby GC frees the chunk (via
    # +blazen_vc_chunk_free+) once the user's handler returns.
    class VcChunk
      # @api private
      # @param raw_ptr [::FFI::Pointer] caller-owned +BlazenVcChunk *+
      def initialize(raw_ptr)
        if raw_ptr.nil? || raw_ptr.null?
          raise Blazen::InternalError, "VcChunk: native pointer is null"
        end

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_vc_chunk_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [Array<Float>] PCM sample slice (f32) for this chunk
      def samples
        len_ptr = ::FFI::MemoryPointer.new(:size_t)
        base    = Blazen::FFI.blazen_vc_chunk_samples(@ptr, len_ptr)
        len     = len_ptr.read(:size_t)
        return [] if base.nil? || base.null? || len.zero?

        base.read_array_of_float(len)
      end

      # @return [Boolean] whether this is the final chunk of the stream
      def final?
        Blazen::FFI.blazen_vc_chunk_is_final(@ptr)
      end

      # @return [Float, nil] per-chunk latency in seconds, or +nil+ when
      #   the backend did not report a measurement (NaN sentinel upstream)
      def latency_seconds
        v = Blazen::FFI.blazen_vc_chunk_latency_seconds(@ptr)
        v.nan? ? nil : v
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenVcResult+ handle.
    class VcResult
      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "VcResult: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_vc_result_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [String] the encoded WAV bytes (binary string). Empty
      #   when the backend produced no audio.
      def bytes
        len_ptr = ::FFI::MemoryPointer.new(:size_t)
        base    = Blazen::FFI.blazen_vc_result_bytes(@ptr, len_ptr)
        len     = len_ptr.read(:size_t)
        return String.new(encoding: Encoding::ASCII_8BIT) if base.nil? || base.null? || len.zero?

        base.read_bytes(len).force_encoding(Encoding::ASCII_8BIT)
      end

      # @return [String, nil] IANA MIME type of the encoded audio
      def mime_type
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_vc_result_mime_type(@ptr))
      end

      # @return [Integer] sample rate in Hz (+0+ when not reported)
      def sample_rate
        Blazen::FFI.blazen_vc_result_sample_rate(@ptr)
      end

      # @return [Float, nil] duration in seconds, or +nil+ when the
      #   backend did not report a duration (encoded as +0.0+ on the
      #   wire)
      def duration_seconds
        v = Blazen::FFI.blazen_vc_result_duration_seconds(@ptr)
        v.zero? ? nil : v
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenTargetVoice+ handle.
    class TargetVoice
      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        if raw_ptr.nil? || raw_ptr.null?
          raise ArgumentError, "TargetVoice: pointer must be non-null"
        end

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_target_voice_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [String] stable voice identifier
      def id
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_target_voice_id(@ptr))
      end

      # @return [String, nil] human-readable label (cabi returns the
      #   empty string when no label is set, which we normalise to +nil+)
      def label
        s = Blazen::FFI.consume_cstring(Blazen::FFI.blazen_target_voice_label(@ptr))
        return nil if s.nil? || s.empty?

        s
      end

      # @return [Integer] sample rate in Hz the voice was trained at
      def sample_rate_hz
        Blazen::FFI.blazen_target_voice_sample_rate_hz(@ptr)
      end
    end
  end
end
