# frozen_string_literal: true

require "spec_helper"

# Spec for the +Blazen::Compute+ music surface (MusicModel + MusicChunk +
# MusicResult, plus +Blazen::Streaming.stream_music+).
#
# The construction / wrapper specs run unconditionally because the fal
# factory is always present in the cabi build. The streaming + live-API
# specs are gated behind +BLAZEN_RUN_LIVE_FAL_MUSIC=1+ so CI doesn't
# burn API credit on every run.

RSpec.describe Blazen::Compute, "music surface" do
  before(:all) { Blazen.init }

  describe ".fal_music" do
    it "constructs a MusicModel handle" do
      model = described_class.fal_music(api_key: "sk-test")
      expect(model).to be_a(Blazen::Compute::MusicModel)
      expect(model.ptr).not_to be_null
    end

    it "accepts an optional model override" do
      model = described_class.fal_music(
        api_key: "sk-test",
        model: "fal-ai/musicgen-medium",
      )
      expect(model.ptr).not_to be_null
    end

    it "accepts an empty api_key (defers to FAL_KEY env upstream)" do
      # Without FAL_KEY in the environment the cabi raises AuthError;
      # when present the call succeeds. Either outcome confirms the
      # FFI bridge correctly forwards the empty-string sentinel.
      if ENV["FAL_KEY"].to_s.empty?
        expect { described_class.fal_music(api_key: "") }
          .to raise_error(Blazen::Error)
      else
        model = described_class.fal_music(api_key: "")
        expect(model.ptr).not_to be_null
      end
    end

    it "lets GC reclaim the model via FFI::AutoPointer" do
      # Two rounds of allocation + GC should not crash; the AutoPointer
      # release path calls +blazen_music_model_free+.
      10.times { described_class.fal_music(api_key: "sk-test") }
      GC.start
      expect(true).to be(true)
    end
  end

  describe ".musicgen" do
    if Blazen::FFI.respond_to?(:blazen_music_model_new_musicgen)
      it "constructs a MusicGen-backed model when the feature is present" do
        # The factory does not eagerly load weights — only the call to
        # +generate_music_blocking+ would. So a successful construction
        # is the assertion here. (A real backend-init failure would
        # surface here; today the factory is lazy.)
        model = described_class.musicgen(
          variant: "small",
          device: "cpu",
          cache_dir: File.expand_path("~/.cache/blazen-music-rspec/musicgen-noexist"),
        )
        expect(model).to be_a(Blazen::Compute::MusicModel)
        expect(model.ptr).not_to be_null
      end
    else
      it "raises UnsupportedError when the music-musicgen feature is missing" do
        expect { described_class.musicgen }
          .to raise_error(Blazen::UnsupportedError, /music-musicgen/)
      end
    end
  end

  describe ".stable_audio" do
    if Blazen::FFI.respond_to?(:blazen_music_model_new_stable_audio)
      it "requires the tokenizer_path argument" do
        expect { described_class.stable_audio }
          .to raise_error(ArgumentError)
      end
    else
      it "raises UnsupportedError when the music-stable-audio feature is missing" do
        expect { described_class.stable_audio(tokenizer_path: "/nope") }
          .to raise_error(Blazen::UnsupportedError, /music-stable-audio/)
      end
    end
  end

  describe ".audiogen" do
    if Blazen::FFI.respond_to?(:blazen_music_model_new_audiogen)
      it "constructs an AudioGen-backed model when the feature is present" do
        model = described_class.audiogen(
          device: "cpu",
          cache_dir: File.expand_path("~/.cache/blazen-music-rspec/audiogen-noexist"),
        )
        expect(model).to be_a(Blazen::Compute::MusicModel)
        expect(model.ptr).not_to be_null
      end
    else
      it "raises UnsupportedError when the music-audiogen feature is missing" do
        expect { described_class.audiogen }
          .to raise_error(Blazen::UnsupportedError, /music-audiogen/)
      end
    end
  end

  describe Blazen::Compute::MusicModel do
    let(:model) { Blazen::Compute.fal_music(api_key: "sk-test") }

    it "exposes async + blocking variants for music + SFX" do
      expect(model).to respond_to(:generate_music)
      expect(model).to respond_to(:generate_music_blocking)
      expect(model).to respond_to(:generate_sfx)
      expect(model).to respond_to(:generate_sfx_blocking)
    end

    it "exposes streaming entry points for music + SFX" do
      expect(model).to respond_to(:stream_generate_music)
      expect(model).to respond_to(:stream_generate_music_async)
      expect(model).to respond_to(:stream_generate_sfx)
      expect(model).to respond_to(:stream_generate_sfx_async)
    end

    it "rejects a nil pointer at construction" do
      expect { Blazen::Compute::MusicModel.new(nil) }
        .to raise_error(ArgumentError, /pointer must be non-null/)
    end
  end

  describe Blazen::Compute::MusicChunk do
    it "rejects a nil pointer at construction" do
      expect { described_class.new(nil) }
        .to raise_error(Blazen::InternalError, /native pointer is null/)
    end
  end

  describe Blazen::Compute::MusicResult do
    it "rejects a nil pointer at construction" do
      expect { described_class.new(nil) }
        .to raise_error(ArgumentError, /pointer must be non-null/)
    end
  end

  describe ".stream_music routing", :aggregate_failures do
    let(:model) { Blazen::Compute.fal_music(api_key: "sk-test") }

    it "rejects a non-MusicModel argument" do
      expect do
        Blazen::Streaming.stream_music(Object.new, "p", 1.0)
      end.to raise_error(ArgumentError, /MusicModel/)
    end

    it "rejects an unknown mode" do
      expect do
        Blazen::Streaming.stream_music(model, "p", 1.0, mode: :bogus)
      end.to raise_error(ArgumentError, /must be :music or :sfx/)
    end
  end

  # ---------------------------------------------------------------------
  # Live-API specs (require +FAL_KEY+ + +BLAZEN_RUN_LIVE_FAL_MUSIC=1+).
  # ---------------------------------------------------------------------
  if ENV["BLAZEN_RUN_LIVE_FAL_MUSIC"] == "1" && !ENV["FAL_KEY"].to_s.empty?
    describe "live fal music generation" do
      let(:model) { Blazen::Compute.fal_music(api_key: ENV.fetch("FAL_KEY")) }

      it "round-trips a short music clip via the blocking surface" do
        result = model.generate_music_blocking("piano cinematic intro", 5.0)
        expect(result).to be_a(Blazen::Compute::MusicResult)
        # fal returns either inline bytes or a URL — assert that at least
        # one of the two is populated.
        expect(result.bytes.bytesize.positive? || !result.url.nil?).to be(true)
        expect(result.mime_type).to be_a(String) unless result.mime_type.nil?
      end

      it "round-trips a short SFX clip via the blocking surface" do
        result = model.generate_sfx_blocking("dog barking", 3.0)
        expect(result).to be_a(Blazen::Compute::MusicResult)
        expect(result.bytes.bytesize.positive? || !result.url.nil?).to be(true)
      end
    end
  end
end
