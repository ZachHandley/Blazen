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
# * The V2 typed-result surface — Ruby subclasses can now return a +Hash+
#   matching the matching record's serde schema, a +Blazen::*Result+
#   wrapper instance, or raise a +Blazen::*Error+. All three paths round
#   trip through the cabi vtable as typed values.

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

  end

  describe "V2 typed returns through the FFI vtable" do
    # The trampoline +*_FN+ constants on +Blazen::CustomProvider+ are
    # +::FFI::Function+ values bound to the cabi vtable. We invoke them
    # directly with a user_data pointer whose +address+ matches the
    # registry id +Blazen::CustomProvider.register+ handed back, fake
    # +out_response+ / +out_err+ memory slots, and a null request pointer
    # — the V2 overrides under test don't read the request, and the
    # trampoline's +ensure+ block only frees when the matching
    # +blazen_*_request_free+ symbol is actually attached on +Blazen::FFI+.
    #
    # +blazen_voice_handle_free+ is currently a cabi-side-only function
    # (no +attach_function+ in +lib/blazen/ffi.rb+ — the +VoiceHandle+
    # wrapper's finalizer references it). For the list_voices spec we
    # attach it ad-hoc on the +Blazen::FFI+ module so the test can free
    # the two voice handles it allocates.
    before(:all) do
      Blazen::FFI.module_eval do
        unless respond_to?(:blazen_voice_handle_free)
          attach_function :blazen_voice_handle_free, [:pointer], :void
        end
      end
    end

    # Minimum-viable +AudioResult+ JSON shape — matches
    # +blazen_llm::compute::AudioResult+ exactly. +MediaType+ is an
    # internally-tagged enum (+#[serde(tag = "type")]+), so the encoded
    # form is +{"type" => "mp3"}+, not the bare string +"mp3"+.
    let(:audio_result_hash) do
      {
        "audio" => [
          {
            "media" => {
              "base64"     => "ZGF0YQ==",
              "media_type" => { "type" => "mp3" },
              "metadata"   => {},
            },
          },
        ],
        "timing"        => { "total_ms" => 0 },
        "metadata"      => {},
        "cost"          => 0.0123,
        "audio_seconds" => 1.5,
      }
    end

    # Minimum-viable +ImageResult+ JSON shape — see +AudioResult+ above
    # for the +media_type+ encoding rationale.
    let(:image_result_hash) do
      {
        "images" => [
          {
            "media" => {
              "url"        => "https://example.com/x.png",
              "media_type" => { "type" => "png" },
              "metadata"   => {},
            },
          },
        ],
        "timing"   => { "total_ms" => 0 },
        "metadata" => {},
        "cost"     => 0.0042,
      }
    end

    # Invokes +fn+ (one of the +Blazen::CustomProvider::*_FN+ constants)
    # against +instance+ with a null request pointer. Returns the
    # +[status, response_ptr_or_nil, error_ptr_or_nil]+ triple the
    # trampoline wrote. Always +unregister+s the instance afterwards so
    # the per-test registry id doesn't leak across examples.
    def drive_typed_trampoline(instance, fn)
      id = Blazen::CustomProvider.register(instance, [])
      user_data_ptr = ::FFI::Pointer.new(:void, id)
      out_response  = ::FFI::MemoryPointer.new(:pointer)
      out_err       = ::FFI::MemoryPointer.new(:pointer)
      begin
        status = fn.call(user_data_ptr, ::FFI::Pointer::NULL,
                         out_response, out_err)
        resp = out_response.read_pointer
        err  = out_err.read_pointer
        [status, resp.null? ? nil : resp, err.null? ? nil : err]
      ensure
        Blazen::CustomProvider.unregister(id)
      end
    end

    it "round-trips a Hash-returned text_to_speech across the FFI vtable" do
      stub_class = Class.new(Blazen::CustomProvider) do
        define_method(:text_to_speech) do |_req|
          {
            "audio" => [
              {
                "media" => {
                  "base64"     => "ZGF0YQ==",
                  "media_type" => { "type" => "mp3" },
                  "metadata"   => {},
                },
              },
            ],
            "timing"        => { "total_ms" => 0 },
            "metadata"      => {},
            "cost"          => 0.0123,
            "audio_seconds" => 1.5,
          }
        end
      end

      status, response, error = drive_typed_trampoline(
        stub_class.new, Blazen::CustomProvider::TEXT_TO_SPEECH_FN
      )
      begin
        expect(status).to eq(0)
        expect(error).to be_nil
        expect(response).not_to be_nil
      ensure
        Blazen::FFI.blazen_audio_result_free(response) if response
      end
    end

    it "round-trips a Blazen::AudioResult wrapper returned from text_to_speech" do
      hash = audio_result_hash
      stub_class = Class.new(Blazen::CustomProvider) do
        define_method(:text_to_speech) { |_req| Blazen::AudioResult.new(hash) }
      end

      status, response, error = drive_typed_trampoline(
        stub_class.new, Blazen::CustomProvider::TEXT_TO_SPEECH_FN
      )
      begin
        expect(status).to eq(0)
        expect(error).to be_nil
        expect(response).not_to be_nil
      ensure
        Blazen::FFI.blazen_audio_result_free(response) if response
      end
    end

    it "round-trips a Hash-returned generate_image across the FFI vtable" do
      hash = image_result_hash
      stub_class = Class.new(Blazen::CustomProvider) do
        define_method(:generate_image) { |_req| hash }
      end

      status, response, error = drive_typed_trampoline(
        stub_class.new, Blazen::CustomProvider::GENERATE_IMAGE_FN
      )
      begin
        expect(status).to eq(0)
        expect(error).to be_nil
        expect(response).not_to be_nil
      ensure
        Blazen::FFI.blazen_image_result_free(response) if response
      end
    end

    it "propagates Blazen::UnsupportedError from an overridden complete to a typed Unsupported BlazenError" do
      stub_class = Class.new(Blazen::CustomProvider) do
        def complete(_req)
          raise Blazen::UnsupportedError, "no LLM here"
        end
      end

      status, response, error = drive_typed_trampoline(
        stub_class.new, Blazen::CustomProvider::COMPLETE_FN
      )
      begin
        expect(status).to eq(-1)
        expect(response).to be_nil
        expect(error).not_to be_nil
        expect(Blazen::FFI.blazen_error_kind(error))
          .to eq(Blazen::FFI::ERROR_KIND_UNSUPPORTED)
        expect(Blazen::FFI.consume_cstring(Blazen::FFI.blazen_error_message(error)))
          .to eq("no LLM here")
      ensure
        Blazen::FFI.blazen_error_free(error) if error
      end
    end

    it "propagates a generic StandardError as Internal" do
      stub_class = Class.new(Blazen::CustomProvider) do
        def text_to_speech(_req)
          raise "generic boom"
        end
      end

      status, response, error = drive_typed_trampoline(
        stub_class.new, Blazen::CustomProvider::TEXT_TO_SPEECH_FN
      )
      begin
        expect(status).to eq(-1)
        expect(response).to be_nil
        expect(error).not_to be_nil
        expect(Blazen::FFI.blazen_error_kind(error))
          .to eq(Blazen::FFI::ERROR_KIND_INTERNAL)
        expect(Blazen::FFI.consume_cstring(Blazen::FFI.blazen_error_message(error)))
          .to eq("generic boom")
      ensure
        Blazen::FFI.blazen_error_free(error) if error
      end
    end

    it "returns 0 for delete_voice (void return) when override returns nil" do
      stub_class = Class.new(Blazen::CustomProvider) do
        def delete_voice(_voice)
          nil
        end
      end

      id = Blazen::CustomProvider.register(stub_class.new, [])
      user_data_ptr = ::FFI::Pointer.new(:void, id)
      out_err = ::FFI::MemoryPointer.new(:pointer)
      begin
        # +DELETE_VOICE_FN+ has the +(user_data, voice_ptr, out_err)+
        # signature — no +out_response+ slot.
        status = Blazen::CustomProvider::DELETE_VOICE_FN.call(
          user_data_ptr, ::FFI::Pointer::NULL, out_err
        )
        expect(status).to eq(0)
        expect(out_err.read_pointer).to be_null
      ensure
        Blazen::CustomProvider.unregister(id)
      end
    end

    it "round-trips an Array-of-Hashes list_voices via the array surface" do
      stub_class = Class.new(Blazen::CustomProvider) do
        def list_voices
          [
            { "id" => "v1", "name" => "Alice", "provider" => "test", "metadata" => {} },
            { "id" => "v2", "name" => "Bob",   "provider" => "test", "metadata" => {} },
          ]
        end
      end

      id = Blazen::CustomProvider.register(stub_class.new, [])
      user_data_ptr = ::FFI::Pointer.new(:void, id)
      out_array = ::FFI::MemoryPointer.new(:pointer)
      out_count = ::FFI::MemoryPointer.new(:size_t)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      voice_ptrs = []
      begin
        status = Blazen::CustomProvider::LIST_VOICES_FN.call(
          user_data_ptr, out_array, out_count, out_err
        )
        expect(status).to eq(0)
        expect(out_err.read_pointer).to be_null

        count = out_count.read(:size_t)
        expect(count).to eq(2)

        buf_ptr = out_array.read_pointer
        expect(buf_ptr).not_to be_null

        # Read +count+ +BlazenVoiceHandle *+ entries out of the contiguous
        # buffer. Each one is a caller-owned handle.
        count.times do |i|
          entry = buf_ptr.get_pointer(i * ::FFI::Pointer.size)
          expect(entry).not_to be_null
          voice_ptrs << entry
        end
      ensure
        voice_ptrs.each do |vp|
          Blazen::FFI.blazen_voice_handle_free(vp) unless vp.nil? || vp.null?
        end
        Blazen::CustomProvider.unregister(id)
      end
    end
  end
end
