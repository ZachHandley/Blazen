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
end
