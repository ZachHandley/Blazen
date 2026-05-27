# frozen_string_literal: true

require "spec_helper"
require "fileutils"

# Spec for the voice-conversion (RVC) surface: the per-engine
# +Blazen::RvcProvider+ class for construction + non-streaming
# conversion, plus the streaming-handle class +Blazen::Compute::VcModel+
# (+ VcChunk + VcResult + TargetVoice) that drives
# +Blazen::Streaming.stream_convert+.
#
# Hermetic specs (construction / null-pointer rejection / streaming
# routing) run unconditionally. Live specs that exercise the real RVC
# pipeline are gated behind +BLAZEN_RUN_RVC_TESTS=1+ plus a populated
# voice directory and reference input — the pipeline is expensive (loads
# RVC + HuBERT + rmvpe weights) and CI doesn't carry those models by
# default.

RSpec.describe "voice-conversion (RVC) surface" do
  before(:all) { Blazen.init }

  let(:scratch_dir) do
    dir = File.expand_path("~/.cache/blazen-rvc-rspec")
    FileUtils.mkdir_p(dir)
    dir
  end

  describe Blazen::RvcProvider do
    if Blazen::FFI.respond_to?(:blazen_rvc_provider_new)
      it "constructs an RvcProvider handle when the feature is present" do
        # The constructor does not eagerly load weights — only conversion /
        # listing would. A successful construction is the assertion. Voice
        # weights are loaded lazily from +$BLAZEN_RVC_VOICE_DIR/<id>/+.
        model = described_class.new
        expect(model).to be_a(Blazen::RvcProvider)
        expect(model.handle).not_to be_null
      end

      it "lets GC reclaim the model via FFI::AutoPointer" do
        # Two rounds of allocation + GC should not crash; the
        # AutoPointer release path calls +blazen_rvc_provider_free+.
        10.times { described_class.new }
        GC.start
        expect(true).to be(true)
      end
    else
      it "raises UnsupportedError when the audio-vc-rvc feature is missing" do
        expect { described_class.new }
          .to raise_error(Blazen::UnsupportedError, /audio-vc-rvc/)
      end
    end
  end

  describe Blazen::Compute::VcModel do
    it "rejects a nil pointer at construction" do
      expect { described_class.new(nil) }
        .to raise_error(ArgumentError, /pointer must be non-null/)
    end
  end

  describe Blazen::Compute::VcChunk do
    it "rejects a nil pointer at construction" do
      expect { described_class.new(nil) }
        .to raise_error(Blazen::InternalError, /native pointer is null/)
    end
  end

  describe Blazen::Compute::VcResult do
    it "rejects a nil pointer at construction" do
      expect { described_class.new(nil) }
        .to raise_error(ArgumentError, /pointer must be non-null/)
    end
  end

  describe Blazen::Compute::TargetVoice do
    it "rejects a nil pointer at construction" do
      expect { described_class.new(nil) }
        .to raise_error(ArgumentError, /pointer must be non-null/)
    end
  end

  describe ".stream_convert routing", :aggregate_failures do
    it "rejects a non-VcModel argument" do
      expect do
        Blazen::Streaming.stream_convert(Object.new, [0.0, 0.0], "voice-1")
      end.to raise_error(ArgumentError, /VcModel/)
    end
  end

  describe "#list_target_voices_blocking on an empty voice dir" do
    if Blazen::FFI.respond_to?(:blazen_vc_model_new_rvc)
      it "returns [] or raises a Blazen::Error (both acceptable)" do
        empty_dir = File.join(scratch_dir, "voices-empty")
        FileUtils.mkdir_p(empty_dir)
        model = Blazen::Compute::VcModel.rvc(voice_dir: empty_dir, device: "cpu")
        begin
          voices = model.list_target_voices_blocking
          expect(voices).to be_an(Array)
        rescue Blazen::Error => e
          # The backend may surface a "no voices registered" error or
          # similar — both outcomes confirm the FFI bridge is alive.
          expect(e).to be_a(Blazen::Error)
        end
      end
    else
      it "is skipped (audio-vc-rvc feature missing)" do
        skip "blazen was built without the 'audio-vc-rvc' feature"
      end
    end
  end

  # ---------------------------------------------------------------------
  # Live RVC specs (require BLAZEN_RUN_RVC_TESTS=1 + a populated voice
  # directory + a registered voice id + a reference input WAV).
  # ---------------------------------------------------------------------
  if ENV["BLAZEN_RUN_RVC_TESTS"] == "1" &&
     !ENV["BLAZEN_RVC_VOICE_DIR"].to_s.empty? &&
     !ENV["BLAZEN_RVC_VOICE_ID"].to_s.empty? &&
     !ENV["BLAZEN_RVC_INPUT_WAV"].to_s.empty?
    describe "live RVC conversion" do
      let(:model) do
        Blazen::Compute::VcModel.rvc(
          voice_dir: ENV.fetch("BLAZEN_RVC_VOICE_DIR"),
          device:    ENV["BLAZEN_RVC_DEVICE"], # nil falls back to CPU
        )
      end

      let(:voice_id)  { ENV.fetch("BLAZEN_RVC_VOICE_ID") }
      let(:input_wav) { ENV.fetch("BLAZEN_RVC_INPUT_WAV") }

      it "lists the registered target voice" do
        voices = model.list_target_voices_blocking
        expect(voices).to be_an(Array)
        expect(voices.map(&:id)).to include(voice_id)
      end

      it "converts via the blocking surface" do
        result = model.convert_voice_blocking(input_wav, voice_id)
        expect(result).to be_a(Blazen::Compute::VcResult)
        expect(result.bytes.bytesize.positive?).to be(true)
        expect(result.mime_type).to match(%r{audio/})
        expect(result.sample_rate).to be > 0
      end

      it "streams via stream_convert_pcm block-form" do
        # Read the input WAV's PCM into f32 samples. We use the cabi's
        # own blocking conversion to bootstrap a result, decode its raw
        # bytes' tail (skipping the 44-byte canonical WAV header) into
        # f32, and feed that back as the streaming input. This avoids a
        # WAV-decoder dependency in the spec while still exercising the
        # streaming pipeline end-to-end.
        prepared = model.convert_voice_blocking(input_wav, voice_id)
        wav_bytes = prepared.bytes
        skip "blocking surface returned empty audio" if wav_bytes.bytesize <= 44

        pcm_i16 = wav_bytes.byteslice(44..).unpack("s<*")
        pcm_f32 = pcm_i16.map { |s| s / 32_768.0 }

        seen = { chunks: 0, done: 0, error: nil }
        model.stream_convert_pcm(pcm_f32, voice_id) do |kind, *args|
          case kind
          when :chunk then seen[:chunks] += 1
          when :done  then seen[:done]   += 1
          when :error then seen[:error]  = args[0]
          end
        end

        expect(seen[:error]).to be_nil
        expect(seen[:chunks]).to be >= 1
        expect(seen[:done]).to be >= 1
      end
    end
  end
end
