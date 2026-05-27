# frozen_string_literal: true

# Concrete per-engine TTS provider classes (Part U — Provider class hierarchy).
#
# Engines:
#   * {Blazen::PiperProvider}      — local onnxruntime + Piper voice bundles
#   * {Blazen::KokoroProvider}     — local Kokoro-82M
#   * {Blazen::VibeVoiceProvider}  — local VibeVoice (Microsoft)
#   * {Blazen::Qwen3TtsProvider}   — local Qwen3-TTS
#   * {Blazen::SparkTtsProvider}   — local Spark-TTS
#   * {Blazen::BarkProvider}       — local Bark
#   * {Blazen::F5Provider}         — local F5-TTS
#   * {Blazen::FalTtsProvider}     — fal.ai-hosted TTS
#
# Each class wraps a per-engine cabi opaque
# (+BlazenPiperProvider *+, +BlazenKokoroProvider *+, …) and forwards
# +synthesize+ / +synthesize_blocking+ through the matching cabi entry
# point. The future-await + result-pop logic mirrors the pattern used by
# +Blazen::Compute::TtsModel+ (see +blazen/compute.rb+).

module Blazen
  # @api private
  # Shared synthesize/synthesize_blocking implementations factored out so
  # each engine subclass body stays a one-liner. Mixed into every
  # +TtsProvider+ subclass below.
  module TtsProviderImpl
    private

    # @api private
    def tts_synthesize(text, voice, language, async_sym, free_required: true)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      out_err    = ::FFI::MemoryPointer.new(:pointer)
      fut = Blazen::FFI.with_cstring(text.to_s) do |t|
        Blazen::FFI.with_cstring(voice) do |v|
          Blazen::FFI.with_cstring(language) do |lang|
            Blazen::FFI.public_send(async_sym, @handle, t, v, lang)
          end
        end
      end
      if fut.nil? || fut.null?
        raise Blazen::ValidationError, "#{async_sym} returned a null future"
      end

      Blazen::FFI.await_future(fut) do |f|
        Blazen::FFI.blazen_future_take_tts_result(f, out_result, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      _ = free_required # silence rubocop
      Blazen::Compute::TtsResult.new(out_result.read_pointer)
    end

    # @api private
    def tts_synthesize_blocking(text, voice, language, blocking_sym)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      out_err    = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(text.to_s) do |t|
        Blazen::FFI.with_cstring(voice) do |v|
          Blazen::FFI.with_cstring(language) do |lang|
            Blazen::FFI.public_send(
              blocking_sym, @handle, t, v, lang, out_result, out_err,
            )
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      Blazen::Compute::TtsResult.new(out_result.read_pointer)
    end
  end

  # Local Piper TTS (onnxruntime). Requires a Piper voice bundle (+.onnx+
  # weights + matching +.json+ config); see https://github.com/rhasspy/piper
  # for the published voice set.
  class PiperProvider < TtsProvider
    include TtsProviderImpl

    PROVIDER_ID = "piper"

    # @param voice_id [String] arbitrary identifier used for caching
    # @param onnx_path [String] path to the +.onnx+ Piper weights
    # @param config_path [String, nil] path to the +.json+ Piper config
    # @param default_speaker_id [Integer] multi-speaker default (-1 = first)
    def initialize(voice_id:, onnx_path:, config_path: nil, default_speaker_id: -1)
      unless Blazen::FFI.respond_to?(:blazen_piper_provider_new)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'audio-tts-piper' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(voice_id.to_s) do |vid|
        Blazen::FFI.with_cstring(onnx_path.to_s) do |op|
          Blazen::FFI.with_cstring(config_path) do |cp|
            Blazen::FFI.blazen_piper_provider_new(
              vid, op, cp, Integer(default_speaker_id), out_model, out_err,
            )
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_piper_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def synthesize(text, voice: nil, language: nil)
      tts_synthesize(text, voice, language, :blazen_piper_provider_synthesize)
    end

    def synthesize_blocking(text, voice: nil, language: nil)
      tts_synthesize_blocking(text, voice, language, :blazen_piper_provider_synthesize_blocking)
    end
  end

  # Local Kokoro-82M TTS.
  class KokoroProvider < TtsProvider
    include TtsProviderImpl

    PROVIDER_ID = "kokoro"

    # @param voice [String, nil]
    # @param language [String, nil]
    # @param sample_rate [Integer, nil] negative / nil = engine default
    def initialize(voice: nil, language: nil, sample_rate: -1)
      unless Blazen::FFI.respond_to?(:blazen_kokoro_provider_new)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'audio-tts-kokoro' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      sr = sample_rate.nil? ? -1 : Integer(sample_rate)
      Blazen::FFI.with_cstring(voice) do |v|
        Blazen::FFI.with_cstring(language) do |lang|
          Blazen::FFI.blazen_kokoro_provider_new(v, lang, sr, out_model, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_kokoro_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def synthesize(text, voice: nil, language: nil)
      tts_synthesize(text, voice, language, :blazen_kokoro_provider_synthesize)
    end

    def synthesize_blocking(text, voice: nil, language: nil)
      tts_synthesize_blocking(text, voice, language, :blazen_kokoro_provider_synthesize_blocking)
    end
  end

  # Local VibeVoice (Microsoft) TTS.
  class VibeVoiceProvider < TtsProvider
    include TtsProviderImpl

    PROVIDER_ID = "vibevoice"

    # @param voice [String, nil]
    # @param language [String, nil]
    # @param sample_rate [Integer, nil]
    def initialize(voice: nil, language: nil, sample_rate: -1)
      unless Blazen::FFI.respond_to?(:blazen_vibevoice_provider_new)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'audio-tts-vibevoice' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      sr = sample_rate.nil? ? -1 : Integer(sample_rate)
      Blazen::FFI.with_cstring(voice) do |v|
        Blazen::FFI.with_cstring(language) do |lang|
          Blazen::FFI.blazen_vibevoice_provider_new(v, lang, sr, out_model, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_vibevoice_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def synthesize(text, voice: nil, language: nil)
      tts_synthesize(text, voice, language, :blazen_vibevoice_provider_synthesize)
    end

    def synthesize_blocking(text, voice: nil, language: nil)
      tts_synthesize_blocking(text, voice, language, :blazen_vibevoice_provider_synthesize_blocking)
    end
  end

  # Local Qwen3-TTS.
  class Qwen3TtsProvider < TtsProvider
    include TtsProviderImpl

    PROVIDER_ID = "qwen3-tts"

    # @param voice [String, nil]
    # @param language [String, nil]
    # @param sample_rate [Integer, nil]
    def initialize(voice: nil, language: nil, sample_rate: -1)
      unless Blazen::FFI.respond_to?(:blazen_qwen3_tts_provider_new)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'audio-tts-qwen3' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      sr = sample_rate.nil? ? -1 : Integer(sample_rate)
      Blazen::FFI.with_cstring(voice) do |v|
        Blazen::FFI.with_cstring(language) do |lang|
          Blazen::FFI.blazen_qwen3_tts_provider_new(v, lang, sr, out_model, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_qwen3_tts_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def synthesize(text, voice: nil, language: nil)
      tts_synthesize(text, voice, language, :blazen_qwen3_tts_provider_synthesize)
    end

    def synthesize_blocking(text, voice: nil, language: nil)
      tts_synthesize_blocking(text, voice, language, :blazen_qwen3_tts_provider_synthesize_blocking)
    end
  end

  # Local Spark-TTS. The cabi constructor is infallible and returns a
  # raw handle pointer.
  class SparkTtsProvider < TtsProvider
    include TtsProviderImpl

    PROVIDER_ID = "spark-tts"

    # @param model_id [String, nil] HF repo id (defaults upstream)
    # @param model_dir [String, nil] pre-resolved local checkpoint dir
    # @param revision [String, nil] HF revision pin
    def initialize(model_id: nil, model_dir: nil, revision: nil)
      unless Blazen::FFI.respond_to?(:blazen_spark_tts_provider_new)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'audio-tts-spark' feature"
      end

      raw = Blazen::FFI.with_cstring(model_id) do |mid|
        Blazen::FFI.with_cstring(model_dir) do |md|
          Blazen::FFI.with_cstring(revision) do |rev|
            Blazen::FFI.blazen_spark_tts_provider_new(mid, md, rev)
          end
        end
      end
      if raw.nil? || raw.null?
        raise Blazen::ValidationError,
              "blazen_spark_tts_provider_new returned null (non-UTF-8 input?)"
      end

      super(raw, Blazen::FFI.method(:blazen_spark_tts_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def synthesize(text, voice: nil, language: nil)
      tts_synthesize(text, voice, language, :blazen_spark_tts_provider_synthesize)
    end

    def synthesize_blocking(text, voice: nil, language: nil)
      tts_synthesize_blocking(text, voice, language, :blazen_spark_tts_provider_synthesize_blocking)
    end
  end

  # Local Bark TTS. Infallible constructor; takes no arguments.
  class BarkProvider < TtsProvider
    include TtsProviderImpl

    PROVIDER_ID = "bark"

    def initialize
      unless Blazen::FFI.respond_to?(:blazen_bark_provider_new)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'audio-tts-bark' feature"
      end

      raw = Blazen::FFI.blazen_bark_provider_new
      if raw.nil? || raw.null?
        raise Blazen::InternalError, "blazen_bark_provider_new returned null"
      end

      super(raw, Blazen::FFI.method(:blazen_bark_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def synthesize(text, voice: nil, language: nil)
      tts_synthesize(text, voice, language, :blazen_bark_provider_synthesize)
    end

    def synthesize_blocking(text, voice: nil, language: nil)
      tts_synthesize_blocking(text, voice, language, :blazen_bark_provider_synthesize_blocking)
    end
  end

  # Local F5-TTS. Infallible constructor; takes no arguments.
  class F5Provider < TtsProvider
    include TtsProviderImpl

    PROVIDER_ID = "f5"

    def initialize
      unless Blazen::FFI.respond_to?(:blazen_f5_provider_new)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'audio-tts-f5' feature"
      end

      raw = Blazen::FFI.blazen_f5_provider_new
      if raw.nil? || raw.null?
        raise Blazen::InternalError, "blazen_f5_provider_new returned null"
      end

      super(raw, Blazen::FFI.method(:blazen_f5_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def synthesize(text, voice: nil, language: nil)
      tts_synthesize(text, voice, language, :blazen_f5_provider_synthesize)
    end

    def synthesize_blocking(text, voice: nil, language: nil)
      tts_synthesize_blocking(text, voice, language, :blazen_f5_provider_synthesize_blocking)
    end
  end

  # fal.ai-hosted TTS. Empty +api_key+ defers to +FAL_KEY+ env var.
  class FalTtsProvider < TtsProvider
    include TtsProviderImpl

    PROVIDER_ID = "fal-tts"

    # @param api_key [String]
    # @param model [String, nil] optional fal TTS model id override
    #   (e.g. +"fal-ai/dia-tts"+); uses +_with_model+ variant when supplied
    def initialize(api_key:, model: nil)
      unless Blazen::FFI.respond_to?(:blazen_fal_tts_provider_new)
        raise Blazen::UnsupportedError, "blazen cabi missing fal_tts_provider_new symbol"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key.to_s) do |key|
        if model.nil?
          Blazen::FFI.blazen_fal_tts_provider_new(key, out_model, out_err)
        else
          Blazen::FFI.with_cstring(model) do |m|
            Blazen::FFI.blazen_fal_tts_provider_new_with_model(key, m, out_model, out_err)
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_fal_tts_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def synthesize(text, voice: nil, language: nil)
      tts_synthesize(text, voice, language, :blazen_fal_tts_provider_synthesize)
    end

    def synthesize_blocking(text, voice: nil, language: nil)
      tts_synthesize_blocking(text, voice, language, :blazen_fal_tts_provider_synthesize_blocking)
    end
  end
end
