# frozen_string_literal: true

# Concrete per-engine voice-conversion provider classes (Part U).
#
# Engines:
#   * {Blazen::RvcProvider}   — local RVC (audio-vc-rvc feature)
#   * {Blazen::FalVcProvider} — fal.ai-hosted voice conversion (clone_voice
#     and list_target_voices return Unsupported / empty list upstream)

module Blazen
  # @api private
  module VcProviderImpl
    private

    def vc_convert(input_path, target_voice_id, async_sym)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      out_err    = ::FFI::MemoryPointer.new(:pointer)
      fut = Blazen::FFI.with_cstring(input_path.to_s) do |ip|
        Blazen::FFI.with_cstring(target_voice_id.to_s) do |v|
          Blazen::FFI.public_send(async_sym, @handle, ip, v)
        end
      end
      if fut.nil? || fut.null?
        raise Blazen::ValidationError, "#{async_sym} returned a null future"
      end

      Blazen::FFI.await_future(fut) do |f|
        Blazen::FFI.blazen_future_take_vc_result(f, out_result, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      Blazen::Compute::VcResult.new(out_result.read_pointer)
    end

    def vc_convert_blocking(input_path, target_voice_id, blocking_sym)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      out_err    = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(input_path.to_s) do |ip|
        Blazen::FFI.with_cstring(target_voice_id.to_s) do |v|
          Blazen::FFI.public_send(blocking_sym, @handle, ip, v, out_result, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      Blazen::Compute::VcResult.new(out_result.read_pointer)
    end

    def vc_clone(voice_id, reference_path, async_sym)
      fut = Blazen::FFI.with_cstring(voice_id.to_s) do |v|
        Blazen::FFI.with_cstring(reference_path.to_s) do |rp|
          Blazen::FFI.public_send(async_sym, @handle, v, rp)
        end
      end
      if fut.nil? || fut.null?
        raise Blazen::ValidationError, "#{async_sym} returned a null future"
      end

      Blazen::FFI.await_future(fut) do |f|
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_future_take_unit(f, out_err)
        Blazen::FFI.check_error!(out_err)
      end
      nil
    end

    def vc_clone_blocking(voice_id, reference_path, blocking_sym)
      out_err = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(voice_id.to_s) do |v|
        Blazen::FFI.with_cstring(reference_path.to_s) do |rp|
          Blazen::FFI.public_send(blocking_sym, @handle, v, rp, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      nil
    end

    def vc_list_target_voices(async_sym)
      fut = Blazen::FFI.public_send(async_sym, @handle)
      if fut.nil? || fut.null?
        raise Blazen::ValidationError, "#{async_sym} returned a null future"
      end

      out_list = ::FFI::MemoryPointer.new(:pointer)
      out_err  = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.await_future(fut) do |f|
        Blazen::FFI.blazen_future_take_target_voice_list(f, out_list, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      Blazen::Compute::VcModel._take_target_voice_list(out_list.read_pointer)
    end

    def vc_list_target_voices_blocking(blocking_sym)
      out_list = ::FFI::MemoryPointer.new(:pointer)
      out_err  = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.public_send(blocking_sym, @handle, out_list, out_err)
      Blazen::FFI.check_error!(out_err)
      Blazen::Compute::VcModel._take_target_voice_list(out_list.read_pointer)
    end
  end

  # Local RVC (Retrieval-based Voice Conversion). Infallible constructor;
  # voice weights are loaded lazily from +$BLAZEN_RVC_VOICE_DIR/<voice_id>/+
  # on the first +convert_voice+ call.
  class RvcProvider < VcProvider
    include VcProviderImpl

    PROVIDER_ID = "rvc"

    def initialize
      unless Blazen::FFI.respond_to?(:blazen_rvc_provider_new)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'audio-vc-rvc' feature"
      end

      raw = Blazen::FFI.blazen_rvc_provider_new
      if raw.nil? || raw.null?
        raise Blazen::InternalError, "blazen_rvc_provider_new returned null"
      end

      super(raw, Blazen::FFI.method(:blazen_rvc_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def convert_voice(input_path, target_voice_id:)
      vc_convert(input_path, target_voice_id, :blazen_rvc_provider_convert_voice)
    end

    def convert_voice_blocking(input_path, target_voice_id:)
      vc_convert_blocking(
        input_path, target_voice_id, :blazen_rvc_provider_convert_voice_blocking,
      )
    end

    def clone_voice(voice_id:, reference_path:)
      vc_clone(voice_id, reference_path, :blazen_rvc_provider_clone_voice)
    end

    def clone_voice_blocking(voice_id:, reference_path:)
      vc_clone_blocking(
        voice_id, reference_path, :blazen_rvc_provider_clone_voice_blocking,
      )
    end

    def list_target_voices
      vc_list_target_voices(:blazen_rvc_provider_list_target_voices)
    end

    def list_target_voices_blocking
      vc_list_target_voices_blocking(:blazen_rvc_provider_list_target_voices_blocking)
    end
  end

  # fal.ai-hosted voice conversion. +clone_voice+ and +list_target_voices+
  # are exposed for parity but resolve to Unsupported / empty list upstream.
  class FalVcProvider < VcProvider
    include VcProviderImpl

    PROVIDER_ID = "fal-vc"

    # @param api_key [String]
    def initialize(api_key:)
      unless Blazen::FFI.respond_to?(:blazen_fal_vc_provider_new)
        raise Blazen::UnsupportedError, "blazen cabi missing fal_vc_provider_new symbol"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key.to_s) do |key|
        Blazen::FFI.blazen_fal_vc_provider_new(key, out_model, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_fal_vc_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def convert_voice(input_path, target_voice_id:)
      vc_convert(input_path, target_voice_id, :blazen_fal_vc_provider_convert_voice)
    end

    def convert_voice_blocking(input_path, target_voice_id:)
      vc_convert_blocking(
        input_path, target_voice_id, :blazen_fal_vc_provider_convert_voice_blocking,
      )
    end

    def clone_voice(voice_id:, reference_path:)
      vc_clone(voice_id, reference_path, :blazen_fal_vc_provider_clone_voice)
    end

    def clone_voice_blocking(voice_id:, reference_path:)
      vc_clone_blocking(
        voice_id, reference_path, :blazen_fal_vc_provider_clone_voice_blocking,
      )
    end

    def list_target_voices
      vc_list_target_voices(:blazen_fal_vc_provider_list_target_voices)
    end

    def list_target_voices_blocking
      vc_list_target_voices_blocking(:blazen_fal_vc_provider_list_target_voices_blocking)
    end
  end
end
