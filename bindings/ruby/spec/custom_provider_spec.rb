# frozen_string_literal: true

require "spec_helper"

# Specs covering the Phase B-Pivot Ruby +CustomProvider+ surface:
#
# * Module-level factory presets (+Blazen.ollama+, +Blazen.lm_studio+) that
#   build OpenAI-protocol +CustomProviderHandle+ instances on the cabi side.
# * The +Blazen::CustomProvider+ base class — its default typed methods all
#   raise +Blazen::UnsupportedError+ so unimplemented modalities surface a
#   clear error instead of a silent +nil+.
# * +Blazen::CustomProvider.from_subclass+ — wraps a Ruby subclass in a cabi
#   vtable and returns a usable handle.
#
# V1 LIMITATION
# -------------
# The Ruby cabi surface does not yet expose result-record constructors
# (BlazenCompletion, BlazenTtsAudio, BlazenEmbeddings, ...) callable from
# Ruby, so subclasses cannot today return a success value back across the
# FFI vtable — only +raise+-style overrides round-trip cleanly. The specs
# below verify the install path and the raise-propagation path; full
# success-value round-trips through the vtable are intentionally deferred
# and are covered by an explicit +pending+ example near the end of the file.

RSpec.describe Blazen::CustomProvider do
  before(:all) { Blazen.init }

  describe "Blazen.ollama" do
    it "constructs a CustomProviderHandle with provider_id == 'ollama'" do
      handle = Blazen.ollama(model: "llama3")
      expect(handle).to be_a(Blazen::CustomProviderHandle)
      expect(handle.provider_id).to eq("ollama")
      expect(handle.model_id).to eq("llama3")
    end
  end

  describe "Blazen.lm_studio" do
    it "constructs a CustomProviderHandle with provider_id == 'lm_studio'" do
      handle = Blazen.lm_studio(model: "qwen3")
      expect(handle).to be_a(Blazen::CustomProviderHandle)
      expect(handle.provider_id).to eq("lm_studio")
      expect(handle.model_id).to eq("qwen3")
    end
  end

  describe "default typed-method behaviour" do
    # An empty subclass — every typed method should fall through to the
    # base class default and raise +Blazen::UnsupportedError+.
    class StubBareProvider < Blazen::CustomProvider; end

    it "raises Blazen::UnsupportedError from #text_to_speech by default" do
      expect { StubBareProvider.new.text_to_speech(nil) }
        .to raise_error(Blazen::UnsupportedError, /text_to_speech/)
    end

    it "raises Blazen::UnsupportedError from #complete by default" do
      expect { StubBareProvider.new.complete(nil) }
        .to raise_error(Blazen::UnsupportedError, /complete/)
    end

    it "derives a snake_case provider_id from the class name" do
      # StubBareProvider → "stub_bare_provider"
      expect(StubBareProvider.new.provider_id).to eq("stub_bare_provider")
    end
  end

  describe "Blazen::CustomProvider.from_subclass" do
    # A subclass with an explicit raise override: the install path runs,
    # and calling +text_to_speech+ directly (in Ruby) propagates the custom
    # error message without crossing the cabi vtable. Cross-FFI propagation
    # is exercised by the higher-level integration suites; this spec only
    # checks the Ruby-side surface.
    class StubRaisingTts < Blazen::CustomProvider
      def text_to_speech(_request)
        raise Blazen::UnsupportedError, "StubRaisingTts: custom message"
      end
    end

    it "returns a handle whose provider_id reflects the subclass name" do
      handle = Blazen::CustomProvider.from_subclass(StubRaisingTts.new)
      expect(handle).to be_a(Blazen::CustomProviderHandle)
      expect(handle.provider_id).to match(/stub/i)
    end

    it "propagates the overridden raise when called directly in Ruby" do
      instance = StubRaisingTts.new
      expect { instance.text_to_speech(nil) }
        .to raise_error(Blazen::UnsupportedError, /custom message/)
    end

    it "rejects non-CustomProvider instances with ArgumentError" do
      expect { Blazen::CustomProvider.from_subclass(Object.new) }
        .to raise_error(ArgumentError, /CustomProvider/)
    end

    # V1 limitation — see file-level comment. The cabi does not yet expose
    # Ruby-callable result-record constructors, so a subclass cannot today
    # return a populated +BlazenTtsAudio+ (or similar) from across the
    # vtable. When that surface lands, replace this +pending+ with a real
    # round-trip assertion.
    it "round-trips a success value returned from a subclass override across the FFI vtable" do
      pending "cabi result-record constructors not yet exposed Ruby-side (V1 limitation)"
      raise "unreached"
    end
  end
end
