# frozen_string_literal: true

# Concrete per-engine STT provider classes (Part U).
#
# Engines:
#   * {Blazen::WhisperCppProvider}      — local whisper.cpp
#   * {Blazen::FasterWhisperProvider}   — local faster-whisper (CTranslate2)
#   * {Blazen::WhisperStreamingProvider}— local whisper streaming
#   * {Blazen::FalSttProvider}          — fal.ai-hosted STT
#
# The cabi-side names use +whispercpp+ (single word) and the engines are
# all gated under the +audio-stt-*+ Cargo features.

module Blazen
  # @api private
  module SttProviderImpl
    private

    def stt_transcribe(audio_source, language, async_sym)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      out_err    = ::FFI::MemoryPointer.new(:pointer)
      fut = Blazen::FFI.with_cstring(audio_source.to_s) do |src|
        Blazen::FFI.with_cstring(language) do |lang|
          Blazen::FFI.public_send(async_sym, @handle, src, lang)
        end
      end
      if fut.nil? || fut.null?
        raise Blazen::ValidationError, "#{async_sym} returned a null future"
      end

      Blazen::FFI.await_future(fut) do |f|
        Blazen::FFI.blazen_future_take_stt_result(f, out_result, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      Blazen::Compute::SttResult.new(out_result.read_pointer)
    end

    def stt_transcribe_blocking(audio_source, language, blocking_sym)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      out_err    = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(audio_source.to_s) do |src|
        Blazen::FFI.with_cstring(language) do |lang|
          Blazen::FFI.public_send(blocking_sym, @handle, src, lang, out_result, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      Blazen::Compute::SttResult.new(out_result.read_pointer)
    end
  end

  # Local whisper.cpp STT.
  class WhisperCppProvider < SttProvider
    include SttProviderImpl

    PROVIDER_ID = "whispercpp"

    # @param model [String, nil] HF repo id / local path; nil = engine default
    # @param device [String, nil] +"cpu"+ / +"cuda"+ / +"metal"+
    # @param language [String, nil] ISO-639-1 default
    def initialize(model: nil, device: nil, language: nil)
      unless Blazen::FFI.respond_to?(:blazen_whispercpp_provider_new)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'audio-stt-whispercpp' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(model) do |m|
        Blazen::FFI.with_cstring(device) do |d|
          Blazen::FFI.with_cstring(language) do |lang|
            Blazen::FFI.blazen_whispercpp_provider_new(m, d, lang, out_model, out_err)
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_whispercpp_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def transcribe(audio_source, language: nil)
      stt_transcribe(audio_source, language, :blazen_whispercpp_provider_transcribe)
    end

    def transcribe_blocking(audio_source, language: nil)
      stt_transcribe_blocking(
        audio_source, language, :blazen_whispercpp_provider_transcribe_blocking,
      )
    end
  end

  # Local faster-whisper STT (CTranslate2). Infallible constructor.
  class FasterWhisperProvider < SttProvider
    include SttProviderImpl

    PROVIDER_ID = "faster-whisper"

    # @param model_id [String, nil] HF repo id (default
    #   +"Systran/faster-whisper-tiny"+)
    # @param model_dir [String, nil] pre-resolved CTranslate2 bundle dir
    # @param revision [String, nil] HF revision pin (default +"main"+)
    def initialize(model_id: nil, model_dir: nil, revision: nil)
      unless Blazen::FFI.respond_to?(:blazen_faster_whisper_provider_new)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'audio-stt-faster-whisper' feature"
      end

      raw = Blazen::FFI.with_cstring(model_id) do |mid|
        Blazen::FFI.with_cstring(model_dir) do |md|
          Blazen::FFI.with_cstring(revision) do |rev|
            Blazen::FFI.blazen_faster_whisper_provider_new(mid, md, rev)
          end
        end
      end
      if raw.nil? || raw.null?
        raise Blazen::ValidationError,
              "blazen_faster_whisper_provider_new returned null (non-UTF-8 input?)"
      end

      super(raw, Blazen::FFI.method(:blazen_faster_whisper_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def transcribe(audio_source, language: nil)
      stt_transcribe(audio_source, language, :blazen_faster_whisper_provider_transcribe)
    end

    def transcribe_blocking(audio_source, language: nil)
      stt_transcribe_blocking(
        audio_source, language, :blazen_faster_whisper_provider_transcribe_blocking,
      )
    end
  end

  # Local whisper-streaming STT. Infallible constructor; +chunk_seconds+ /
  # +chunk_overlap_seconds+ accept +nil+ (NaN sentinel upstream).
  class WhisperStreamingProvider < SttProvider
    include SttProviderImpl

    PROVIDER_ID = "whisper-streaming"

    # @param model_id [String, nil]
    # @param vad_model_path [String, nil]
    # @param chunk_seconds [Float, nil]
    # @param chunk_overlap_seconds [Float, nil]
    def initialize(model_id: nil, vad_model_path: nil,
                   chunk_seconds: nil, chunk_overlap_seconds: nil)
      unless Blazen::FFI.respond_to?(:blazen_whisper_streaming_provider_new)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'audio-stt-whisper-streaming' feature"
      end

      cs = chunk_seconds.nil? ? Float::NAN : Float(chunk_seconds)
      co = chunk_overlap_seconds.nil? ? Float::NAN : Float(chunk_overlap_seconds)
      raw = Blazen::FFI.with_cstring(model_id) do |mid|
        Blazen::FFI.with_cstring(vad_model_path) do |vad|
          Blazen::FFI.blazen_whisper_streaming_provider_new(mid, vad, cs, co)
        end
      end
      if raw.nil? || raw.null?
        raise Blazen::ValidationError,
              "blazen_whisper_streaming_provider_new returned null (non-UTF-8 input?)"
      end

      super(raw, Blazen::FFI.method(:blazen_whisper_streaming_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def transcribe(audio_source, language: nil)
      stt_transcribe(audio_source, language, :blazen_whisper_streaming_provider_transcribe)
    end

    def transcribe_blocking(audio_source, language: nil)
      stt_transcribe_blocking(
        audio_source, language, :blazen_whisper_streaming_provider_transcribe_blocking,
      )
    end
  end

  # fal.ai-hosted STT.
  class FalSttProvider < SttProvider
    include SttProviderImpl

    PROVIDER_ID = "fal-stt"

    # @param api_key [String]
    def initialize(api_key:)
      unless Blazen::FFI.respond_to?(:blazen_fal_stt_provider_new)
        raise Blazen::UnsupportedError, "blazen cabi missing fal_stt_provider_new symbol"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key.to_s) do |key|
        Blazen::FFI.blazen_fal_stt_provider_new(key, out_model, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_fal_stt_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def transcribe(audio_source, language: nil)
      stt_transcribe(audio_source, language, :blazen_fal_stt_provider_transcribe)
    end

    def transcribe_blocking(audio_source, language: nil)
      stt_transcribe_blocking(
        audio_source, language, :blazen_fal_stt_provider_transcribe_blocking,
      )
    end
  end
end
