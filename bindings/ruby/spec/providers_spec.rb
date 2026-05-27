# frozen_string_literal: true

require "spec_helper"
require "fileutils"

# Smoke specs for the per-engine provider classes (Part U).
#
# Network- / weights-gated tests live in
# +spec/providers_network_spec.rb+. These specs run unconditionally and
# only exercise behaviour that doesn't need external infrastructure:
#
#   * Construction with valid args succeeds (or surfaces the expected
#     error class for engines that immediately load weights).
#   * +provider_id+ returns the expected stable literal.
#   * +UnsupportedError+ fires when the cabi factory symbol is absent.
#   * +ArgumentError+ fires for null-pointer wrappers.
#   * The GC finalizer doesn't crash when handles are reclaimed.

RSpec.describe "Blazen per-engine provider classes" do
  before(:all) { Blazen.init }

  # ----------------------------------------------------------------
  # BaseProvider / abstract hierarchy
  # ----------------------------------------------------------------
  describe Blazen::BaseProvider do
    it "rejects a null handle at construction" do
      expect { described_class.new(nil, ->(_) {}) }
        .to raise_error(ArgumentError, /native pointer must be non-null/)
    end

    it "raises NotImplementedError on the abstract #provider_id" do
      # Build a fake non-null pointer so the constructor accepts it. Use
      # a dummy free fn that does nothing — it'll be called on GC.
      handle = ::FFI::Pointer.new(0xdeadbeef)
      free_fn = ->(_p) {}
      instance = described_class.new(handle, free_fn)
      expect { instance.provider_id }.to raise_error(NotImplementedError)
    end
  end

  # Capability bases all live as direct subclasses of BaseProvider.
  describe "capability bases" do
    [
      Blazen::TtsProvider, Blazen::SttProvider, Blazen::MusicProvider,
      Blazen::VcProvider, Blazen::ThreeDProvider, Blazen::ImageGenProvider,
      Blazen::EmbeddingProvider, Blazen::LlmProvider
    ].each do |base|
      it "#{base.name} is a Blazen::BaseProvider subclass" do
        expect(base.ancestors).to include(Blazen::BaseProvider)
      end
    end
  end

  # ----------------------------------------------------------------
  # TTS engines (8)
  # ----------------------------------------------------------------
  describe Blazen::FalTtsProvider do
    if Blazen::FFI.respond_to?(:blazen_fal_tts_provider_new)
      it "constructs and reports provider_id" do
        model = described_class.new(api_key: "sk-test")
        expect(model).to be_a(Blazen::TtsProvider)
        expect(model).to be_a(Blazen::BaseProvider)
        expect(model.provider_id).to eq("fal-tts")
        expect(model.handle).not_to be_null
      end

      it "accepts an optional model override" do
        model = described_class.new(api_key: "sk-test", model: "fal-ai/dia-tts")
        expect(model.handle).not_to be_null
      end

      it "survives GC of multiple instances" do
        10.times { described_class.new(api_key: "sk-test") }
        GC.start
        expect(true).to be(true)
      end
    else
      it "raises UnsupportedError when the cabi symbol is absent (pre-rebuild)" do
        expect { described_class.new(api_key: "sk-test") }
          .to raise_error(Blazen::UnsupportedError)
      end
    end
  end

  # Feature-gated TTS engines: each test branches on the cabi symbol
  # being present (full build) vs. absent (minimal build).
  [
    [Blazen::PiperProvider,    "piper",     :blazen_piper_provider_new,    "audio-tts-piper",     [{ voice_id: "test", onnx_path: "/nope.onnx" }]],
    [Blazen::KokoroProvider,   "kokoro",    :blazen_kokoro_provider_new,   "audio-tts-kokoro",    [{}]],
    [Blazen::VibeVoiceProvider, "vibevoice", :blazen_vibevoice_provider_new, "audio-tts-vibevoice", [{}]],
    [Blazen::Qwen3TtsProvider, "qwen3-tts", :blazen_qwen3_tts_provider_new, "audio-tts-qwen3",     [{}]],
    [Blazen::SparkTtsProvider, "spark-tts", :blazen_spark_tts_provider_new, "audio-tts-spark",     [{}]],
    [Blazen::BarkProvider,     "bark",      :blazen_bark_provider_new,      "audio-tts-bark",      []],
    [Blazen::F5Provider,       "f5",        :blazen_f5_provider_new,        "audio-tts-f5",        []],
  ].each do |(klass, provider_id, factory_sym, feature, ctor_args)|
    describe klass do
      if Blazen::FFI.respond_to?(factory_sym)
        it "exposes provider_id == #{provider_id.inspect}" do
          # Some engines (Piper) refuse to construct without real
          # weights; rescue + skip on construct failure.
          begin
            model = ctor_args.empty? ? klass.new : klass.new(**ctor_args.first)
          rescue Blazen::Error => e
            skip "#{klass.name} construction needs real weights: #{e.message}"
          end
          expect(model.provider_id).to eq(provider_id)
          expect(model).to be_a(Blazen::TtsProvider)
        end
      else
        it "raises UnsupportedError when the #{feature} feature is missing" do
          expect do
            ctor_args.empty? ? klass.new : klass.new(**ctor_args.first)
          end.to raise_error(Blazen::UnsupportedError)
        end
      end
    end
  end

  # ----------------------------------------------------------------
  # STT engines (4)
  # ----------------------------------------------------------------
  describe Blazen::FalSttProvider do
    if Blazen::FFI.respond_to?(:blazen_fal_stt_provider_new)
      it "constructs and reports provider_id" do
        model = described_class.new(api_key: "sk-test")
        expect(model).to be_a(Blazen::SttProvider)
        expect(model.provider_id).to eq("fal-stt")
        expect(model.handle).not_to be_null
      end
    else
      it "raises UnsupportedError when the cabi symbol is absent (pre-rebuild)" do
        expect { described_class.new(api_key: "sk-test") }
          .to raise_error(Blazen::UnsupportedError)
      end
    end
  end

  [
    [Blazen::WhisperCppProvider,       "whispercpp",        :blazen_whispercpp_provider_new,        "audio-stt-whispercpp"],
    [Blazen::FasterWhisperProvider,    "faster-whisper",    :blazen_faster_whisper_provider_new,    "audio-stt-faster-whisper"],
    [Blazen::WhisperStreamingProvider, "whisper-streaming", :blazen_whisper_streaming_provider_new, "audio-stt-whisper-streaming"],
  ].each do |(klass, provider_id, factory_sym, feature)|
    describe klass do
      if Blazen::FFI.respond_to?(factory_sym)
        it "exposes provider_id == #{provider_id.inspect}" do
          begin
            model = klass.new
          rescue Blazen::Error => e
            skip "#{klass.name} construction needs real weights: #{e.message}"
          end
          expect(model.provider_id).to eq(provider_id)
          expect(model).to be_a(Blazen::SttProvider)
        end
      else
        it "raises UnsupportedError when the #{feature} feature is missing" do
          expect { klass.new }.to raise_error(Blazen::UnsupportedError)
        end
      end
    end
  end

  # ----------------------------------------------------------------
  # Music engines (4)
  # ----------------------------------------------------------------
  describe Blazen::FalMusicProvider do
    if Blazen::FFI.respond_to?(:blazen_fal_music_provider_new)
      it "constructs and reports provider_id" do
        model = described_class.new(api_key: "sk-test")
        expect(model).to be_a(Blazen::MusicProvider)
        expect(model.provider_id).to eq("fal-music")
        expect(model.handle).not_to be_null
      end
    else
      it "raises UnsupportedError when the cabi symbol is absent (pre-rebuild)" do
        expect { described_class.new(api_key: "sk-test") }
          .to raise_error(Blazen::UnsupportedError)
      end
    end
  end

  [
    [Blazen::MusicGenProvider, "musicgen",     :blazen_musicgen_provider_new, "music-musicgen", {}],
    [Blazen::AudioGenProvider, "audiogen",     :blazen_audiogen_provider_new, "music-audiogen", {}],
  ].each do |(klass, provider_id, factory_sym, feature, ctor_args)|
    describe klass do
      if Blazen::FFI.respond_to?(factory_sym)
        it "exposes provider_id == #{provider_id.inspect}" do
          begin
            model = klass.new(**ctor_args)
          rescue Blazen::Error => e
            skip "#{klass.name} construction needs real weights: #{e.message}"
          end
          expect(model.provider_id).to eq(provider_id)
        end
      else
        it "raises UnsupportedError when the #{feature} feature is missing" do
          expect { klass.new(**ctor_args) }.to raise_error(Blazen::UnsupportedError)
        end
      end
    end
  end

  describe Blazen::StableAudioProvider do
    factory_sym = :blazen_stable_audio_provider_new
    if Blazen::FFI.respond_to?(factory_sym)
      it "exposes provider_id == \"stable-audio\"" do
        begin
          model = described_class.new(tokenizer_path: "/nope")
        rescue Blazen::Error => e
          skip "StableAudioProvider construction needs real tokenizer: #{e.message}"
        end
        expect(model.provider_id).to eq("stable-audio")
      end
    else
      it "raises UnsupportedError when the music-stable-audio feature is missing" do
        expect { described_class.new(tokenizer_path: "/nope") }
          .to raise_error(Blazen::UnsupportedError)
      end
    end
  end

  # ----------------------------------------------------------------
  # VC engines (2)
  # ----------------------------------------------------------------
  describe Blazen::FalVcProvider do
    if Blazen::FFI.respond_to?(:blazen_fal_vc_provider_new)
      it "constructs and reports provider_id" do
        model = described_class.new(api_key: "sk-test")
        expect(model).to be_a(Blazen::VcProvider)
        expect(model.provider_id).to eq("fal-vc")
        expect(model.handle).not_to be_null
      end
    else
      it "raises UnsupportedError when the cabi symbol is absent (pre-rebuild)" do
        expect { described_class.new(api_key: "sk-test") }
          .to raise_error(Blazen::UnsupportedError)
      end
    end
  end

  describe Blazen::RvcProvider do
    if Blazen::FFI.respond_to?(:blazen_rvc_provider_new)
      it "constructs (lazy weights load) and reports provider_id" do
        model = described_class.new
        expect(model.provider_id).to eq("rvc")
        expect(model).to be_a(Blazen::VcProvider)
      end
    else
      it "raises UnsupportedError when the audio-vc-rvc feature is missing" do
        expect { described_class.new }.to raise_error(Blazen::UnsupportedError)
      end
    end
  end

  # ----------------------------------------------------------------
  # 3D engines (1)
  # ----------------------------------------------------------------
  describe Blazen::TripoSrProvider do
    if Blazen::FFI.respond_to?(:blazen_triposr_provider_new)
      it "constructs (lazy weights load) and reports provider_id" do
        # Constructor returns a handle without downloading; expect either
        # success or a clean cabi error (no panic / crash).
        begin
          model = described_class.new
        rescue Blazen::Error => e
          skip "TripoSrProvider needs real weights: #{e.message}"
        end
        expect(model.provider_id).to eq("triposr")
        expect(model).to be_a(Blazen::ThreeDProvider)
      end
    else
      it "raises UnsupportedError when the triposr feature is missing" do
        expect { described_class.new }.to raise_error(Blazen::UnsupportedError)
      end
    end
  end

  # ----------------------------------------------------------------
  # ImageGen engines (2)
  # ----------------------------------------------------------------
  describe Blazen::FalImageGenProvider do
    if Blazen::FFI.respond_to?(:blazen_fal_image_gen_provider_new)
      it "constructs and reports provider_id" do
        model = described_class.new(api_key: "sk-test")
        expect(model).to be_a(Blazen::ImageGenProvider)
        expect(model.provider_id).to eq("fal-image-gen")
      end

      it "accepts default_model + base_url overrides" do
        model = described_class.new(
          api_key: "sk-test",
          default_model: "fal-ai/flux/schnell",
          base_url: "https://queue.fal.run",
        )
        expect(model.handle).not_to be_null
      end
    else
      it "raises UnsupportedError when the cabi symbol is absent (pre-rebuild)" do
        expect { described_class.new(api_key: "sk-test") }
          .to raise_error(Blazen::UnsupportedError)
      end
    end
  end

  describe Blazen::DiffusionProvider do
    if Blazen::FFI.respond_to?(:blazen_diffusion_provider_new)
      it "exposes provider_id == \"diffusion\"" do
        begin
          model = described_class.new(options_json: "{}")
        rescue Blazen::Error => e
          skip "DiffusionProvider construction needs real options: #{e.message}"
        end
        expect(model.provider_id).to eq("diffusion")
      end
    else
      it "raises UnsupportedError when the diffusion feature is missing" do
        expect { described_class.new }.to raise_error(Blazen::UnsupportedError)
      end
    end
  end

  # ----------------------------------------------------------------
  # Embed engines (5)
  # ----------------------------------------------------------------
  describe Blazen::OpenAiEmbeddingProvider do
    if Blazen::FFI.respond_to?(:blazen_openai_embedding_provider_new)
      it "constructs and reports provider_id" do
        model = described_class.new(api_key: "sk-test")
        expect(model).to be_a(Blazen::EmbeddingProvider)
        expect(model.provider_id).to eq("openai-embedding")
      end
    else
      it "raises UnsupportedError when the cabi symbol is absent" do
        expect { described_class.new(api_key: "sk-test") }
          .to raise_error(Blazen::UnsupportedError)
      end
    end
  end

  describe Blazen::FalEmbeddingProvider do
    if Blazen::FFI.respond_to?(:blazen_fal_embedding_provider_new)
      it "constructs and reports provider_id" do
        model = described_class.new(api_key: "sk-test")
        expect(model.provider_id).to eq("fal-embedding")
      end
    else
      it "raises UnsupportedError when the cabi symbol is absent" do
        expect { described_class.new(api_key: "sk-test") }
          .to raise_error(Blazen::UnsupportedError)
      end
    end
  end

  [
    [Blazen::FastembedProvider,  "fastembed", :blazen_fastembed_provider_new,  "embed-fastembed"],
    [Blazen::TractEmbedProvider, "tract",     :blazen_tract_embed_provider_new, "embed-tract"],
    [Blazen::CandleEmbedProvider, "candle",   :blazen_candle_embed_provider_new, "embed-candle"],
  ].each do |(klass, provider_id, factory_sym, feature)|
    describe klass do
      if Blazen::FFI.respond_to?(factory_sym)
        it "exposes provider_id == #{provider_id.inspect}" do
          begin
            model = klass.new
          rescue Blazen::Error => e
            skip "#{klass.name} construction needs real model: #{e.message}"
          end
          expect(model.provider_id).to eq(provider_id)
          expect(model).to be_a(Blazen::EmbeddingProvider)
        end
      else
        it "raises UnsupportedError when the #{feature} feature is missing" do
          expect { klass.new }.to raise_error(Blazen::UnsupportedError)
        end
      end
    end
  end

  # ----------------------------------------------------------------
  # LLM engines (15)
  # ----------------------------------------------------------------
  THREE_STR_LLM_ENGINES = [
    [Blazen::OpenAiProvider,    "openai",    :blazen_openai_provider_new],
    [Blazen::AnthropicProvider, "anthropic", :blazen_anthropic_provider_new],
    [Blazen::GeminiProvider,    "gemini",    :blazen_gemini_provider_new],
    [Blazen::FalLlmProvider,    "fal-llm",   :blazen_fal_llm_provider_new],
  ].freeze

  TWO_STR_LLM_ENGINES = [
    [Blazen::MistralProvider,    "mistral",    :blazen_mistral_provider_new],
    [Blazen::FireworksProvider,  "fireworks",  :blazen_fireworks_provider_new],
    [Blazen::DeepSeekProvider,   "deepseek",   :blazen_deepseek_provider_new],
    [Blazen::PerplexityProvider, "perplexity", :blazen_perplexity_provider_new],
    [Blazen::TogetherProvider,   "together",   :blazen_together_provider_new],
    [Blazen::GroqProvider,       "groq",       :blazen_groq_provider_new],
    [Blazen::OpenRouterProvider, "openrouter", :blazen_openrouter_provider_new],
    [Blazen::CohereProvider,     "cohere",     :blazen_cohere_provider_new],
    [Blazen::XaiProvider,        "xai",        :blazen_xai_provider_new],
  ].freeze

  THREE_STR_LLM_ENGINES.each do |(klass, provider_id, factory_sym)|
    describe klass do
      if Blazen::FFI.respond_to?(factory_sym)
        it "constructs and exposes provider_id == #{provider_id.inspect}" do
          model = klass.new(api_key: "sk-test")
          expect(model).to be_a(Blazen::LlmProvider)
          expect(model.provider_id).to eq(provider_id)
          expect(model.handle).not_to be_null
        end

        it "accepts model + base_url overrides" do
          model = klass.new(api_key: "sk-test", model: "x", base_url: "https://example.com")
          expect(model.handle).not_to be_null
        end
      else
        it "raises UnsupportedError when the cabi symbol is absent" do
          expect { klass.new(api_key: "sk-test") }.to raise_error(Blazen::UnsupportedError)
        end
      end
    end
  end

  TWO_STR_LLM_ENGINES.each do |(klass, provider_id, factory_sym)|
    describe klass do
      if Blazen::FFI.respond_to?(factory_sym)
        it "constructs and exposes provider_id == #{provider_id.inspect}" do
          model = klass.new(api_key: "sk-test")
          expect(model).to be_a(Blazen::LlmProvider)
          expect(model.provider_id).to eq(provider_id)
        end

        it "accepts a model override" do
          model = klass.new(api_key: "sk-test", model: "x")
          expect(model.handle).not_to be_null
        end
      else
        it "raises UnsupportedError when the cabi symbol is absent" do
          expect { klass.new(api_key: "sk-test") }.to raise_error(Blazen::UnsupportedError)
        end
      end
    end
  end

  describe Blazen::AzureOpenAiProvider do
    factory_sym = :blazen_azure_openai_provider_new
    if Blazen::FFI.respond_to?(factory_sym)
      it "constructs with (api_key, resource_name, deployment_name)" do
        model = described_class.new(
          api_key: "sk-test",
          resource_name: "my-resource",
          deployment_name: "gpt-4o-deployment",
        )
        expect(model.provider_id).to eq("azure-openai")
        expect(model).to be_a(Blazen::LlmProvider)
      end
    else
      it "raises UnsupportedError when the cabi symbol is absent" do
        expect do
          described_class.new(
            api_key: "sk-test", resource_name: "r", deployment_name: "d",
          )
        end.to raise_error(Blazen::UnsupportedError)
      end
    end
  end

  describe Blazen::BedrockProvider do
    factory_sym = :blazen_bedrock_provider_new
    if Blazen::FFI.respond_to?(factory_sym)
      it "constructs with (api_key, region, model?)" do
        model = described_class.new(api_key: "sk-test", region: "us-east-1")
        expect(model.provider_id).to eq("bedrock")
        expect(model).to be_a(Blazen::LlmProvider)
      end
    else
      it "raises UnsupportedError when the cabi symbol is absent" do
        expect { described_class.new(api_key: "sk-test", region: "us-east-1") }
          .to raise_error(Blazen::UnsupportedError)
      end
    end
  end

  # ----------------------------------------------------------------
  # Per-engine streaming surface
  # ----------------------------------------------------------------
  #
  # Method-definition checks only (no construction / no cabi-symbol
  # dependency) — proves every per-engine class carries its streaming
  # entry points regardless of which features the loaded cabi was built
  # with. The actual wire behaviour is covered by the env-gated live specs
  # in music_spec / vc_spec / blazen_spec.
  describe "per-engine streaming surface", :aggregate_failures do
    llm_classes = [
      Blazen::OpenAiProvider, Blazen::AnthropicProvider, Blazen::GeminiProvider,
      Blazen::FalLlmProvider, Blazen::MistralProvider, Blazen::FireworksProvider,
      Blazen::DeepSeekProvider, Blazen::PerplexityProvider, Blazen::TogetherProvider,
      Blazen::GroqProvider, Blazen::OpenRouterProvider, Blazen::CohereProvider,
      Blazen::XaiProvider, Blazen::AzureOpenAiProvider, Blazen::BedrockProvider,
      Blazen::OpenAiCompatProvider, Blazen::OllamaProvider, Blazen::LmStudioProvider,
      Blazen::MistralRsProvider, Blazen::LlamaCppProvider, Blazen::CandleLlmProvider,
    ]
    llm_classes.each do |klass|
      it "#{klass.name.split('::').last} defines #stream / #stream_async" do
        expect(klass.instance_methods).to include(:stream, :stream_async)
      end
    end

    music_classes = [
      Blazen::MusicGenProvider, Blazen::AudioGenProvider,
      Blazen::StableAudioProvider, Blazen::FalMusicProvider,
    ]
    music_classes.each do |klass|
      it "#{klass.name.split('::').last} defines streaming music + SFX entry points" do
        expect(klass.instance_methods).to include(
          :stream_generate_music, :stream_generate_music_async,
          :stream_generate_sfx, :stream_generate_sfx_async,
        )
      end
    end

    [Blazen::RvcProvider, Blazen::FalVcProvider].each do |klass|
      it "#{klass.name.split('::').last} defines #stream_convert_pcm[_async]" do
        expect(klass.instance_methods).to include(
          :stream_convert_pcm, :stream_convert_pcm_async,
        )
      end
    end
  end

  # ----------------------------------------------------------------
  # GC finalizer hygiene
  # ----------------------------------------------------------------
  describe "GC hygiene" do
    {
      Blazen::FalTtsProvider      => :blazen_fal_tts_provider_new,
      Blazen::FalSttProvider      => :blazen_fal_stt_provider_new,
      Blazen::FalMusicProvider    => :blazen_fal_music_provider_new,
      Blazen::FalVcProvider       => :blazen_fal_vc_provider_new,
      Blazen::FalImageGenProvider => :blazen_fal_image_gen_provider_new,
    }.each do |klass, factory_sym|
      if Blazen::FFI.respond_to?(factory_sym)
        it "reclaims #{klass.name.split('::').last} handles cleanly across rounds" do
          30.times { klass.new(api_key: "sk-test") }
          GC.start
          expect(true).to be(true)
        end
      end
    end
  end
end
