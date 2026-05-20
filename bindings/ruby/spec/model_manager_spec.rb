# frozen_string_literal: true

require "spec_helper"

RSpec.describe Blazen::ModelManager do
  before(:all) { Blazen.init }

  describe ".new" do
    it "constructs without raising (u64::MAX budgets)" do
      expect { described_class.new }.not_to raise_error
    end
  end

  describe ".with_budgets_gb" do
    it "constructs without raising (explicit budgets)" do
      expect {
        described_class.with_budgets_gb(cpu_ram_gb: 8.0, gpu_vram_gb: 4.0)
      }.not_to raise_error
    end

    it "reflects the budgets in the pool snapshot" do
      mgr = described_class.with_budgets_gb(cpu_ram_gb: 2.0, gpu_vram_gb: 1.0)
      pools = mgr.pools
      cpu = pools.find { |p| p.pool == "cpu" }
      gpu = pools.find { |p| p.pool == "gpu:0" }
      expect(cpu.budget_bytes).to eq((2.0 * 1024**3).to_i)
      expect(gpu.budget_bytes).to eq((1.0 * 1024**3).to_i)
    end
  end

  describe "#status" do
    it "returns [] on an empty manager" do
      mgr = described_class.new
      expect(mgr.status).to eq([])
    end
  end

  describe "#pools" do
    let(:mgr) { described_class.new }

    it "returns the two default pools (cpu, gpu:0)" do
      labels = mgr.pools.map(&:pool)
      expect(labels).to contain_exactly("cpu", "gpu:0")
    end

    it "reports zero used_bytes and zero loaded_models on an empty manager" do
      mgr.pools.each do |pool|
        expect(pool.used_bytes).to eq(0)
        expect(pool.loaded_models).to eq(0)
      end
    end
  end

  describe "#used_bytes / #available_bytes" do
    let(:mgr) { described_class.with_budgets_gb(cpu_ram_gb: 4.0, gpu_vram_gb: 2.0) }

    it "returns 0 used + full budget available on an empty cpu pool" do
      expect(mgr.used_bytes("cpu")).to eq(0)
      expect(mgr.available_bytes("cpu")).to eq((4.0 * 1024**3).to_i)
    end

    it "raises ValidationError on an unknown pool label" do
      expect { mgr.used_bytes("not-a-pool") }.to raise_error(Blazen::ValidationError)
      expect { mgr.available_bytes("gpu:99") }.to raise_error(Blazen::ValidationError)
    end
  end

  describe "#load" do
    it "raises a Blazen::Error subclass for a nonexistent model" do
      mgr = described_class.new
      expect { mgr.load("nonexistent") }.to raise_error(Blazen::Error)
    end
  end

  describe "#loaded?" do
    it "returns false for an unregistered model id" do
      mgr = described_class.new
      expect(mgr.loaded?("nonexistent")).to eq(false)
    end
  end

  describe "#load_adapter" do
    it "raises a Blazen::Error subclass when the base model is not registered" do
      mgr = described_class.new
      expect {
        mgr.load_adapter("nonexistent", "/tmp/fake", Blazen::AdapterOptions.new)
      }.to raise_error(Blazen::Error)
    end
  end

  describe "#load_from_hf" do
    it "raises a Blazen::Error subclass for a nonexistent repo" do
      mgr = described_class.new
      expect {
        mgr.load_from_hf(
          "ghost",
          "blazen-test/does-not-exist-#{Process.pid}-#{rand(1 << 32)}",
        )
      }.to raise_error(Blazen::Error)
    end

    it "accepts default HfLoadOptions without crashing the FFI layer" do
      mgr = described_class.new
      expect {
        mgr.load_from_hf(
          "ghost2",
          "blazen-test/also-missing-#{Process.pid}-#{rand(1 << 32)}",
          Blazen::HfLoadOptions.new,
        )
      }.to raise_error(Blazen::Error)
    end
  end

  describe Blazen::BackendHint do
    it "exposes stable string labels" do
      expect(described_class::MISTRALRS).to eq("mistralrs")
      expect(described_class::CANDLE).to eq("candle")
      expect(described_class::LLAMACPP).to eq("llamacpp")
    end

    it "maps labels to the cabi int32 sentinels" do
      expect(described_class.to_cabi("mistralrs")).to eq(Blazen::FFI::BLAZEN_BACKEND_HINT_MISTRALRS)
      expect(described_class.to_cabi("candle")).to eq(Blazen::FFI::BLAZEN_BACKEND_HINT_CANDLE)
      expect(described_class.to_cabi("llamacpp")).to eq(Blazen::FFI::BLAZEN_BACKEND_HINT_LLAMACPP)
      expect(described_class.to_cabi(nil)).to eq(Blazen::FFI::BLAZEN_BACKEND_HINT_NONE)
      expect(described_class.to_cabi("garbage")).to eq(Blazen::FFI::BLAZEN_BACKEND_HINT_NONE)
    end
  end

  describe Blazen::HfLoadOptions do
    it "defaults every field to nil" do
      opts = described_class.new
      expect(opts.backend_hint).to be_nil
      expect(opts.revision).to be_nil
      expect(opts.hf_token).to be_nil
      expect(opts.cache_dir).to be_nil
      expect(opts.device).to be_nil
      expect(opts.gguf_file).to be_nil
      expect(opts.memory_estimate_bytes).to be_nil
      expect(opts.pool).to be_nil
    end

    it "round-trips every keyword arg" do
      opts = described_class.new(
        backend_hint: Blazen::BackendHint::CANDLE,
        revision: "main",
        hf_token: "hf_xxx",
        cache_dir: "/tmp/hf-cache",
        device: "cpu",
        gguf_file: "model.Q4_K_M.gguf",
        memory_estimate_bytes: 8_000_000_000,
        pool: "cpu",
      )
      expect(opts.backend_hint).to eq("candle")
      expect(opts.revision).to eq("main")
      expect(opts.hf_token).to eq("hf_xxx")
      expect(opts.cache_dir).to eq("/tmp/hf-cache")
      expect(opts.device).to eq("cpu")
      expect(opts.gguf_file).to eq("model.Q4_K_M.gguf")
      expect(opts.memory_estimate_bytes).to eq(8_000_000_000)
      expect(opts.pool).to eq("cpu")
    end
  end

  describe "#register_local" do
    it "raises Blazen::UnsupportedError with the documented message" do
      mgr = described_class.new
      fake_model = Object.new
      expect { mgr.register_local(fake_model) }.to raise_error(
        Blazen::UnsupportedError,
        /foreign-callback trampoline.*not yet wired through the C-ABI/,
      )
    end
  end

  describe Blazen::AdapterOptions do
    it "defaults scale to 1.0" do
      expect(described_class.new.scale).to eq(1.0)
    end

    it "accepts a custom scale" do
      expect(described_class.new(scale: 0.5).scale).to eq(0.5)
    end
  end

  describe Blazen::SchedulerKind do
    it "exposes string constants matching the cabi convention" do
      expect(described_class::CONSTANT).to eq("constant")
      expect(described_class::LINEAR).to eq("linear")
      expect(described_class::COSINE).to eq("cosine")
    end

    it "maps each label to the cabi int32 sentinel" do
      expect(described_class.to_cabi("constant")).to eq(Blazen::FFI::BLAZEN_SCHEDULER_CONSTANT)
      expect(described_class.to_cabi("linear")).to eq(Blazen::FFI::BLAZEN_SCHEDULER_LINEAR)
      expect(described_class.to_cabi("cosine")).to eq(Blazen::FFI::BLAZEN_SCHEDULER_COSINE)
    end

    it "raises ValidationError on an unknown label" do
      expect { described_class.to_cabi("nope") }.to raise_error(Blazen::ValidationError)
    end
  end

  describe Blazen::MixedPrecision do
    it "exposes string constants matching the cabi convention" do
      expect(described_class::NONE).to eq("none")
      expect(described_class::BF16).to eq("bf16")
    end
  end

  describe Blazen::TrainConfig do
    it "has reasonable defaults for optional fields" do
      cfg = described_class.new(
        base_model_repo: "blazen-test/base",
        output_dir: "/tmp/blazen-out",
      )
      expect(cfg.base_model_repo).to eq("blazen-test/base")
      expect(cfg.output_dir).to eq("/tmp/blazen-out")
      expect(cfg.max_steps).to eq(100)
      expect(cfg.batch_size).to eq(1)
      expect(cfg.gradient_accumulation_steps).to eq(1)
      expect(cfg.max_seq_len).to eq(2048)
      expect(cfg.eval_steps).to be_nil
      expect(cfg.save_steps).to be_nil
      expect(cfg.seed).to eq(42)
      expect(cfg.mixed_precision).to eq(Blazen::MixedPrecision::NONE)
      expect(cfg.device).to be_nil
      expect(cfg.lora).to be_a(Blazen::LoraConfig)
      expect(cfg.optim).to be_a(Blazen::OptimConfig)
      expect(cfg.scheduler).to be_a(Blazen::SchedulerConfig)
    end

    it "LoraConfig defaults to rank 16 / alpha 32 / standard attention modules" do
      lora = Blazen::LoraConfig.new
      expect(lora.rank).to eq(16)
      expect(lora.alpha).to eq(32.0)
      expect(lora.dropout).to eq(0.0)
      expect(lora.target_modules).to eq(%w[q_proj k_proj v_proj o_proj])
    end

    it "OptimConfig#gradient_clip can be set to nil to disable clipping" do
      optim = Blazen::OptimConfig.new(gradient_clip: nil)
      expect(optim.gradient_clip).to be_nil
    end
  end

  describe Blazen::JsonlDataset do
    describe ".from_path" do
      it "raises a Blazen::Error subclass on a nonexistent file" do
        expect {
          described_class.from_path(
            "/nonexistent/blazen-train-#{Process.pid}-#{rand(1 << 32)}.jsonl",
            tokenizer_path: "/nonexistent/tokenizer.json",
          )
        }.to raise_error(Blazen::Error)
      end
    end
  end

  describe "#train_lora" do
    it "raises Blazen::ValidationError when max_steps is 0 (invalid config)" do
      mgr = described_class.new
      cfg = Blazen::TrainConfig.new(
        base_model_repo: "blazen-test/base",
        output_dir: File.join(Dir.home, ".cache", "blazen-train-ruby-wrap", "out"),
        max_steps: 0,
      )
      fake_dataset = Blazen::JsonlDataset.allocate
      fake_dataset.instance_variable_set(:@ptr, ::FFI::Pointer::NULL)
      expect { mgr.train_lora(cfg, fake_dataset) }.to raise_error(Blazen::ValidationError)
    end
  end

  describe Blazen::TrainCoreConfig do
    it "has reasonable defaults for optional fields" do
      core = described_class.new(
        base_model_repo: "blazen-test/base",
        output_dir: "/tmp/blazen-out",
      )
      expect(core.base_model_repo).to eq("blazen-test/base")
      expect(core.base_model_revision).to be_nil
      expect(core.output_dir).to eq("/tmp/blazen-out")
      expect(core.max_steps).to eq(100)
      expect(core.batch_size).to eq(1)
      expect(core.gradient_accumulation_steps).to eq(1)
      expect(core.max_seq_len).to eq(2048)
      expect(core.eval_steps).to be_nil
      expect(core.save_steps).to be_nil
      expect(core.seed).to eq(42)
      expect(core.mixed_precision).to eq(Blazen::MixedPrecision::NONE)
      expect(core.device).to be_nil
      expect(core.optim).to be_a(Blazen::OptimConfig)
      expect(core.scheduler).to be_a(Blazen::SchedulerConfig)
    end
  end

  describe Blazen::DpoConfig do
    let(:core) do
      Blazen::TrainCoreConfig.new(
        base_model_repo: "blazen-test/base",
        output_dir: "/tmp/blazen-dpo-out",
      )
    end

    it "defaults beta=0.1, label_smoothing=0.0, reference_model_*=nil" do
      cfg = described_class.new(core: core)
      expect(cfg.beta).to eq(0.1)
      expect(cfg.label_smoothing).to eq(0.0)
      expect(cfg.reference_model_repo).to be_nil
      expect(cfg.reference_model_revision).to be_nil
      expect(cfg.lora).to be_a(Blazen::LoraConfig)
      expect(cfg.core).to equal(core)
    end

    it "accepts a custom reference model repo + revision" do
      cfg = described_class.new(
        core: core,
        reference_model_repo: "blazen-test/ref",
        reference_model_revision: "main",
      )
      expect(cfg.reference_model_repo).to eq("blazen-test/ref")
      expect(cfg.reference_model_revision).to eq("main")
    end
  end

  describe Blazen::OrpoConfig do
    it "defaults lambda to 0.1" do
      core = Blazen::TrainCoreConfig.new(
        base_model_repo: "blazen-test/base",
        output_dir: "/tmp/blazen-orpo-out",
      )
      cfg = described_class.new(core: core)
      expect(cfg.lambda).to eq(0.1)
      expect(cfg.lora).to be_a(Blazen::LoraConfig)
    end
  end

  describe Blazen::SimpoConfig do
    it "defaults beta=2.0, gamma=1.0" do
      core = Blazen::TrainCoreConfig.new(
        base_model_repo: "blazen-test/base",
        output_dir: "/tmp/blazen-simpo-out",
      )
      cfg = described_class.new(core: core)
      expect(cfg.beta).to eq(2.0)
      expect(cfg.gamma).to eq(1.0)
      expect(cfg.lora).to be_a(Blazen::LoraConfig)
    end
  end

  describe Blazen::KtoConfig do
    it "defaults beta=0.1, lambda_d=1.0, lambda_u=1.0, reference_model_*=nil" do
      core = Blazen::TrainCoreConfig.new(
        base_model_repo: "blazen-test/base",
        output_dir: "/tmp/blazen-kto-out",
      )
      cfg = described_class.new(core: core)
      expect(cfg.beta).to eq(0.1)
      expect(cfg.lambda_d).to eq(1.0)
      expect(cfg.lambda_u).to eq(1.0)
      expect(cfg.reference_model_repo).to be_nil
      expect(cfg.reference_model_revision).to be_nil
      expect(cfg.lora).to be_a(Blazen::LoraConfig)
    end
  end

  describe Blazen::FullFineTuneConfig do
    it "defaults gradient_checkpointing to false" do
      core = Blazen::TrainCoreConfig.new(
        base_model_repo: "blazen-test/base",
        output_dir: "/tmp/blazen-ft-out",
      )
      cfg = described_class.new(core: core)
      expect(cfg.gradient_checkpointing).to eq(false)
      expect(cfg.core).to equal(core)
    end

    it "coerces gradient_checkpointing to a strict bool" do
      core = Blazen::TrainCoreConfig.new(
        base_model_repo: "blazen-test/base",
        output_dir: "/tmp/blazen-ft-out",
      )
      cfg = described_class.new(core: core, gradient_checkpointing: "yes")
      expect(cfg.gradient_checkpointing).to eq(true)
    end
  end

  describe Blazen::PreferenceJsonlDataset do
    describe ".from_path" do
      it "raises a Blazen::Error subclass on a nonexistent file" do
        expect {
          described_class.from_path(
            "/nonexistent/blazen-pref-#{Process.pid}-#{rand(1 << 32)}.jsonl",
            tokenizer_path: "/nonexistent/tokenizer.json",
          )
        }.to raise_error(Blazen::Error)
      end
    end
  end

  describe Blazen::RatedJsonlDataset do
    describe ".from_path" do
      it "raises a Blazen::Error subclass on a nonexistent file" do
        expect {
          described_class.from_path(
            "/nonexistent/blazen-rated-#{Process.pid}-#{rand(1 << 32)}.jsonl",
            tokenizer_path: "/nonexistent/tokenizer.json",
          )
        }.to raise_error(Blazen::Error)
      end
    end
  end

  describe Blazen::FullFineTuneResult do
    it "exposes output_dir / final_loss / steps_completed as keyword fields" do
      res = described_class.new(
        output_dir: "/tmp/blazen-ft-result",
        final_loss: 0.5,
        steps_completed: 100,
      )
      expect(res.output_dir).to eq("/tmp/blazen-ft-result")
      expect(res.final_loss).to eq(0.5)
      expect(res.steps_completed).to eq(100)
    end
  end

  describe "#train_dpo" do
    it "is defined and accepts (config, dataset)" do
      expect(described_class.instance_method(:train_dpo).arity).to eq(2)
    end

    it "raises Blazen::ValidationError when config isn't a DpoConfig" do
      mgr = described_class.new
      fake_dataset = Blazen::PreferenceJsonlDataset.allocate
      fake_dataset.instance_variable_set(:@ptr, ::FFI::Pointer::NULL)
      expect {
        mgr.train_dpo(Object.new, fake_dataset)
      }.to raise_error(Blazen::ValidationError)
    end
  end

  describe "#train_orpo" do
    it "is defined and accepts (config, dataset)" do
      expect(described_class.instance_method(:train_orpo).arity).to eq(2)
    end
  end

  describe "#train_simpo" do
    it "is defined and accepts (config, dataset)" do
      expect(described_class.instance_method(:train_simpo).arity).to eq(2)
    end
  end

  describe "#train_kto" do
    it "is defined and accepts (config, dataset)" do
      expect(described_class.instance_method(:train_kto).arity).to eq(2)
    end

    it "raises Blazen::ValidationError when config isn't a KtoConfig" do
      mgr = described_class.new
      fake_dataset = Blazen::RatedJsonlDataset.allocate
      fake_dataset.instance_variable_set(:@ptr, ::FFI::Pointer::NULL)
      expect {
        mgr.train_kto(Object.new, fake_dataset)
      }.to raise_error(Blazen::ValidationError)
    end
  end

  describe "#fine_tune" do
    it "is defined and accepts (config, dataset)" do
      expect(described_class.instance_method(:fine_tune).arity).to eq(2)
    end

    it "raises Blazen::ValidationError when config isn't a FullFineTuneConfig" do
      mgr = described_class.new
      fake_dataset = Blazen::JsonlDataset.allocate
      fake_dataset.instance_variable_set(:@ptr, ::FFI::Pointer::NULL)
      expect {
        mgr.fine_tune(Object.new, fake_dataset)
      }.to raise_error(Blazen::ValidationError)
    end
  end
end
