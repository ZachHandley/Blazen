# frozen_string_literal: true

# Concrete per-engine music-generation provider classes (Part U).
#
# Engines:
#   * {Blazen::MusicGenProvider}   — local Meta MusicGen
#   * {Blazen::AudioGenProvider}   — local Meta AudioGen (SFX-only upstream)
#   * {Blazen::StableAudioProvider}— local Stability AI Stable Audio Open
#   * {Blazen::FalMusicProvider}   — fal.ai-hosted music / SFX
#
# Note: AudioGen's +generate_music+ surface raises Unsupported upstream
# (AudioGen is SFX-only); MusicGen's +generate_sfx+ surface raises
# Unsupported (MusicGen is music-only). The Ruby classes still expose
# both methods for API parity — the cabi backend surfaces the upstream
# error.

module Blazen
  # @api private
  module MusicProviderImpl
    private

    def music_generate(prompt, duration_seconds, async_sym)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      out_err    = ::FFI::MemoryPointer.new(:pointer)
      dur        = Float(duration_seconds)
      fut = Blazen::FFI.with_cstring(prompt.to_s) do |p|
        Blazen::FFI.public_send(async_sym, @handle, p, dur)
      end
      if fut.nil? || fut.null?
        raise Blazen::ValidationError, "#{async_sym} returned a null future"
      end

      Blazen::FFI.await_future(fut) do |f|
        Blazen::FFI.blazen_future_take_music_result(f, out_result, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      Blazen::Compute::MusicResult.new(out_result.read_pointer)
    end

    def music_generate_blocking(prompt, duration_seconds, blocking_sym)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      out_err    = ::FFI::MemoryPointer.new(:pointer)
      dur        = Float(duration_seconds)
      Blazen::FFI.with_cstring(prompt.to_s) do |p|
        Blazen::FFI.public_send(blocking_sym, @handle, p, dur, out_result, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      Blazen::Compute::MusicResult.new(out_result.read_pointer)
    end
  end

  # Local Meta MusicGen.
  class MusicGenProvider < MusicProvider
    include MusicProviderImpl

    PROVIDER_ID = "musicgen"

    # @param variant [String, nil] +"small"+ / +"medium"+ / +"large"+
    # @param device [String, nil] +"cpu"+ / +"cuda"+ / +"metal"+
    # @param cache_dir [String, nil]
    # @param max_duration_seconds [Float, nil]
    def initialize(variant: nil, device: nil, cache_dir: nil, max_duration_seconds: nil)
      unless Blazen::FFI.respond_to?(:blazen_musicgen_provider_new)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'music-musicgen' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      max_dur   = max_duration_seconds.nil? ? Float::NAN : Float(max_duration_seconds)
      Blazen::FFI.with_cstring(variant) do |v|
        Blazen::FFI.with_cstring(device) do |dev|
          Blazen::FFI.with_cstring(cache_dir) do |cd|
            Blazen::FFI.blazen_musicgen_provider_new(
              v, dev, cd, max_dur, out_model, out_err,
            )
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_musicgen_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def generate_music(prompt, duration_seconds:)
      music_generate(prompt, duration_seconds, :blazen_musicgen_provider_generate_music)
    end

    def generate_music_blocking(prompt, duration_seconds:)
      music_generate_blocking(
        prompt, duration_seconds, :blazen_musicgen_provider_generate_music_blocking,
      )
    end

    def generate_sfx(prompt, duration_seconds:)
      music_generate(prompt, duration_seconds, :blazen_musicgen_provider_generate_sfx)
    end

    def generate_sfx_blocking(prompt, duration_seconds:)
      music_generate_blocking(
        prompt, duration_seconds, :blazen_musicgen_provider_generate_sfx_blocking,
      )
    end
  end

  # Local Meta AudioGen (SFX-only — +generate_music+ raises Unsupported upstream).
  class AudioGenProvider < MusicProvider
    include MusicProviderImpl

    PROVIDER_ID = "audiogen"

    # @param repo_id [String, nil]
    # @param revision [String, nil]
    # @param device [String, nil]
    # @param cache_dir [String, nil]
    # @param max_duration_seconds [Float, nil]
    def initialize(repo_id: nil, revision: nil, device: nil, cache_dir: nil,
                   max_duration_seconds: nil)
      unless Blazen::FFI.respond_to?(:blazen_audiogen_provider_new)
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
              Blazen::FFI.blazen_audiogen_provider_new(
                r, rev, dev, cd, max_dur, out_model, out_err,
              )
            end
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_audiogen_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def generate_music(prompt, duration_seconds:)
      music_generate(prompt, duration_seconds, :blazen_audiogen_provider_generate_music)
    end

    def generate_music_blocking(prompt, duration_seconds:)
      music_generate_blocking(
        prompt, duration_seconds, :blazen_audiogen_provider_generate_music_blocking,
      )
    end

    def generate_sfx(prompt, duration_seconds:)
      music_generate(prompt, duration_seconds, :blazen_audiogen_provider_generate_sfx)
    end

    def generate_sfx_blocking(prompt, duration_seconds:)
      music_generate_blocking(
        prompt, duration_seconds, :blazen_audiogen_provider_generate_sfx_blocking,
      )
    end
  end

  # Local Stability AI Stable Audio Open.
  class StableAudioProvider < MusicProvider
    include MusicProviderImpl

    PROVIDER_ID = "stable-audio"

    # @param tokenizer_path [String] REQUIRED — path to the T5 +tokenizer.json+
    # @param variant [String, nil] +"small"+ / +"open-1.0"+
    # @param device [String, nil]
    # @param max_duration_seconds [Float, nil]
    def initialize(tokenizer_path:, variant: nil, device: nil, max_duration_seconds: nil)
      unless Blazen::FFI.respond_to?(:blazen_stable_audio_provider_new)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'music-stable-audio' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      max_dur   = max_duration_seconds.nil? ? Float::NAN : Float(max_duration_seconds)
      Blazen::FFI.with_cstring(variant) do |v|
        Blazen::FFI.with_cstring(tokenizer_path.to_s) do |tp|
          Blazen::FFI.with_cstring(device) do |dev|
            Blazen::FFI.blazen_stable_audio_provider_new(
              v, tp, dev, max_dur, out_model, out_err,
            )
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_stable_audio_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def generate_music(prompt, duration_seconds:)
      music_generate(prompt, duration_seconds, :blazen_stable_audio_provider_generate_music)
    end

    def generate_music_blocking(prompt, duration_seconds:)
      music_generate_blocking(
        prompt, duration_seconds, :blazen_stable_audio_provider_generate_music_blocking,
      )
    end

    def generate_sfx(prompt, duration_seconds:)
      music_generate(prompt, duration_seconds, :blazen_stable_audio_provider_generate_sfx)
    end

    def generate_sfx_blocking(prompt, duration_seconds:)
      music_generate_blocking(
        prompt, duration_seconds, :blazen_stable_audio_provider_generate_sfx_blocking,
      )
    end
  end

  # fal.ai-hosted music / SFX.
  class FalMusicProvider < MusicProvider
    include MusicProviderImpl

    PROVIDER_ID = "fal-music"

    # @param api_key [String]
    def initialize(api_key:)
      unless Blazen::FFI.respond_to?(:blazen_fal_music_provider_new)
        raise Blazen::UnsupportedError, "blazen cabi missing fal_music_provider_new symbol"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key.to_s) do |key|
        Blazen::FFI.blazen_fal_music_provider_new(key, out_model, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_fal_music_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def generate_music(prompt, duration_seconds:)
      music_generate(prompt, duration_seconds, :blazen_fal_music_provider_generate_music)
    end

    def generate_music_blocking(prompt, duration_seconds:)
      music_generate_blocking(
        prompt, duration_seconds, :blazen_fal_music_provider_generate_music_blocking,
      )
    end

    def generate_sfx(prompt, duration_seconds:)
      music_generate(prompt, duration_seconds, :blazen_fal_music_provider_generate_sfx)
    end

    def generate_sfx_blocking(prompt, duration_seconds:)
      music_generate_blocking(
        prompt, duration_seconds, :blazen_fal_music_provider_generate_sfx_blocking,
      )
    end
  end
end
