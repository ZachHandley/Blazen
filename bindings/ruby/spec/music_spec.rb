# frozen_string_literal: true

require "spec_helper"

# Spec for the music surface: the per-engine provider classes
# (+Blazen::FalMusicProvider+, +Blazen::MusicGenProvider+,
# +Blazen::AudioGenProvider+, +Blazen::StableAudioProvider+) for
# construction + non-streaming generation, plus the typed-result
# wrappers ({Blazen::Compute::MusicChunk} / {Blazen::Compute::MusicResult})
# returned by their capability methods.
#
# The construction specs run unconditionally because the fal provider
# factory is always present in the cabi build. Live-API specs are gated
# behind +BLAZEN_RUN_LIVE_FAL_MUSIC=1+ so CI doesn't burn API credit on
# every run.

RSpec.describe "music surface" do
  before(:all) { Blazen.init }

  describe Blazen::FalMusicProvider do
    it "constructs a fal music provider handle" do
      model = described_class.new(api_key: "sk-test")
      expect(model).to be_a(Blazen::FalMusicProvider)
      expect(model.handle).not_to be_null
    end

    it "accepts an empty api_key (defers to FAL_KEY env upstream)" do
      # The per-engine fal provider constructor does not eagerly validate
      # the key — it forwards the empty-string sentinel so the cabi can
      # resolve +FAL_KEY+ from the environment on first use. Construction
      # succeeds regardless; the assertion confirms the FFI bridge accepts
      # the sentinel.
      model = described_class.new(api_key: "")
      expect(model.handle).not_to be_null
    end

    it "lets GC reclaim the model via FFI::AutoPointer" do
      # Two rounds of allocation + GC should not crash; the AutoPointer
      # release path calls +blazen_fal_music_provider_free+.
      10.times { described_class.new(api_key: "sk-test") }
      GC.start
      expect(true).to be(true)
    end
  end

  describe Blazen::MusicGenProvider do
    if Blazen::FFI.respond_to?(:blazen_musicgen_provider_new)
      it "constructs a MusicGen-backed provider when the feature is present" do
        # The constructor does not eagerly load weights — only the call to
        # +generate_music_blocking+ would. A successful construction is the
        # assertion here.
        model = described_class.new(
          variant: "small",
          device: "cpu",
          cache_dir: File.expand_path("~/.cache/blazen-music-rspec/musicgen-noexist"),
        )
        expect(model).to be_a(Blazen::MusicGenProvider)
        expect(model.handle).not_to be_null
      end
    else
      it "raises UnsupportedError when the music-musicgen feature is missing" do
        expect { described_class.new }
          .to raise_error(Blazen::UnsupportedError, /music-musicgen/)
      end
    end
  end

  describe Blazen::StableAudioProvider do
    if Blazen::FFI.respond_to?(:blazen_stable_audio_provider_new)
      it "requires the tokenizer_path argument" do
        expect { described_class.new }
          .to raise_error(ArgumentError)
      end
    else
      it "raises UnsupportedError when the music-stable-audio feature is missing" do
        expect { described_class.new(tokenizer_path: "/nope") }
          .to raise_error(Blazen::UnsupportedError, /music-stable-audio/)
      end
    end
  end

  describe Blazen::AudioGenProvider do
    if Blazen::FFI.respond_to?(:blazen_audiogen_provider_new)
      it "constructs an AudioGen-backed provider when the feature is present" do
        model = described_class.new(
          device: "cpu",
          cache_dir: File.expand_path("~/.cache/blazen-music-rspec/audiogen-noexist"),
        )
        expect(model).to be_a(Blazen::AudioGenProvider)
        expect(model.handle).not_to be_null
      end
    else
      it "raises UnsupportedError when the music-audiogen feature is missing" do
        expect { described_class.new }
          .to raise_error(Blazen::UnsupportedError, /music-audiogen/)
      end
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

  # ---------------------------------------------------------------------
  # Live-API specs (require +FAL_KEY+ + +BLAZEN_RUN_LIVE_FAL_MUSIC=1+).
  # ---------------------------------------------------------------------
  if ENV["BLAZEN_RUN_LIVE_FAL_MUSIC"] == "1" && !ENV["FAL_KEY"].to_s.empty?
    describe "live fal music generation" do
      let(:model) { Blazen::FalMusicProvider.new(api_key: ENV.fetch("FAL_KEY")) }

      it "round-trips a short music clip via the blocking surface" do
        result = model.generate_music_blocking("piano cinematic intro", duration_seconds: 5.0)
        expect(result).to be_a(Blazen::Compute::MusicResult)
        # fal returns either inline bytes or a URL — assert that at least
        # one of the two is populated.
        expect(result.bytes.bytesize.positive? || !result.url.nil?).to be(true)
        expect(result.mime_type).to be_a(String) unless result.mime_type.nil?
      end

      it "round-trips a short SFX clip via the blocking surface" do
        result = model.generate_sfx_blocking("dog barking", duration_seconds: 3.0)
        expect(result).to be_a(Blazen::Compute::MusicResult)
        expect(result.bytes.bytesize.positive? || !result.url.nil?).to be(true)
      end
    end
  end
end
