# frozen_string_literal: true

# Concrete per-engine image-to-3D provider classes (Part U).
#
# Engines:
#   * {Blazen::TripoSrProvider} — local TripoSR (single-image → 3D mesh)

module Blazen
  # Local TripoSR image-to-3D.
  class TripoSrProvider < ThreeDProvider
    PROVIDER_ID = "triposr"

    # @param hf_repo_id [String, nil] HF repo id (defaults upstream)
    # @param revision [String, nil] HF revision pin
    # @param weights_path [String, nil] pre-resolved local weights file
    def initialize(hf_repo_id: nil, revision: nil, weights_path: nil)
      unless Blazen::FFI.respond_to?(:blazen_triposr_provider_new)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'triposr' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(hf_repo_id) do |r|
        Blazen::FFI.with_cstring(revision) do |rev|
          Blazen::FFI.with_cstring(weights_path) do |wp|
            Blazen::FFI.blazen_triposr_provider_new(r, rev, wp, out_model, out_err)
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_triposr_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def generate_from_image(image_bytes, mesh_resolution: 256)
      bytes      = image_bytes.to_s.dup.force_encoding(Encoding::ASCII_8BIT)
      buf        = ::FFI::MemoryPointer.new(:uint8, bytes.bytesize)
      buf.write_bytes(bytes) unless bytes.empty?
      out_result = ::FFI::MemoryPointer.new(:pointer)
      out_err    = ::FFI::MemoryPointer.new(:pointer)
      res        = Integer(mesh_resolution)

      fut = Blazen::FFI.blazen_triposr_provider_generate_from_image(
        @handle, buf, bytes.bytesize, res,
      )
      if fut.nil? || fut.null?
        raise Blazen::ValidationError,
              "blazen_triposr_provider_generate_from_image returned a null future"
      end

      Blazen::FFI.await_future(fut) do |f|
        Blazen::FFI.blazen_future_take_three_d_generate_result(f, out_result, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      Blazen::Compute::ThreeDGenerateResult.new(out_result.read_pointer)
    end

    def generate_from_image_blocking(image_bytes, mesh_resolution: 256)
      bytes      = image_bytes.to_s.dup.force_encoding(Encoding::ASCII_8BIT)
      buf        = ::FFI::MemoryPointer.new(:uint8, bytes.bytesize)
      buf.write_bytes(bytes) unless bytes.empty?
      out_result = ::FFI::MemoryPointer.new(:pointer)
      out_err    = ::FFI::MemoryPointer.new(:pointer)
      res        = Integer(mesh_resolution)

      Blazen::FFI.blazen_triposr_provider_generate_from_image_blocking(
        @handle, buf, bytes.bytesize, res, out_result, out_err,
      )
      Blazen::FFI.check_error!(out_err)
      Blazen::Compute::ThreeDGenerateResult.new(out_result.read_pointer)
    end
  end
end
