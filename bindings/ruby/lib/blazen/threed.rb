# frozen_string_literal: true

require "json"

module Blazen
  # Ruby wrapper for the +Compat3dProvider+ HTTP-proxy 3D pipeline
  # exposed by +blazen-3d+ via the +blazen-cabi+ surface (built behind the
  # +threed-compat-proxy+ feature flag — see
  # +crates/blazen-cabi/src/threed.rs+).
  #
  # The provider speaks +multipart/form-data+ to a configurable upstream
  # 3D server and exposes four async verbs:
  #
  # * {Compat3dProvider#texturize} — apply / generate a PBR texture for
  #   an input mesh GLB
  # * {Compat3dProvider#rig} — auto-rig a mesh (armature + optional
  #   skin weights)
  # * {Compat3dProvider#refine} — decimate, hole-fill, UV-unwrap,
  #   retopologize, smooth a mesh
  # * {Compat3dProvider#animate} — produce an animated GLB from a text
  #   prompt, MP4 driving video, or BVH motion clip
  #
  # All four methods compose with +Fiber.scheduler+ when one is active
  # (via {Blazen::FFI.await_future}) so they cooperate with the +async+
  # gem; the matching +*_blocking+ siblings drive the future on the
  # cabi tokio runtime when no scheduler is in scope.
  #
  # @example basic usage
  #   provider = Blazen::ThreeD::Compat3dProvider.new(
  #     base_url: "https://3d.example.com",
  #     api_key: ENV["THREED_API_KEY"],
  #   )
  #   mesh = File.binread("input.glb")
  #   result = provider.texturize(
  #     mesh,
  #     prompt: "weathered bronze",
  #     pbr: true,
  #   )
  #   File.binwrite("textured.glb", result.glb_bytes)
  module ThreeD
    # Raised when the native library was built without the
    # +threed-compat-proxy+ feature so the Compat3d symbols are missing.
    class FeatureMissing < Blazen::UnsupportedError
      def initialize(msg = nil)
        super(msg || "Blazen::ThreeD requires libblazen_cabi built with the " \
                     "`threed-compat-proxy` feature; current build omits it.")
      end
    end

    # @return [Boolean] whether the native lib supports the Compat3d
    #   surface in this build.
    def self.available?
      Blazen::FFI.respond_to?(:threed_available?) && Blazen::FFI.threed_available?
    end

    # Result of any of the four pipeline stages. Wraps the opaque
    # +BlazenCompat3dResult *+ pointer; all stage-specific accessors
    # gracefully return +nil+ / empty collections when called on the
    # wrong stage's result.
    class Result
      # @return [::FFI::AutoPointer] underlying +BlazenCompat3dResult *+
      attr_reader :ptr
      # @return [Symbol] one of +:texturize+, +:rig+, +:refine+,
      #   +:animate+ — set by the calling verb so stage-specific
      #   accessors can short-circuit on the wrong variant.
      attr_reader :stage

      # @api private
      def initialize(raw_ptr, stage)
        if raw_ptr.nil? || raw_ptr.null?
          raise Blazen::InternalError,
                "Blazen::ThreeD::Result: native handle is null"
        end

        @stage = stage
        @ptr = ::FFI::AutoPointer.new(
          raw_ptr, Blazen::FFI.method(:blazen_compat3d_result_free),
        )
      end

      # @return [String] the produced GLB bytes (binary-encoded)
      def glb_bytes
        out_ptr = ::FFI::MemoryPointer.new(:pointer)
        out_len = ::FFI::MemoryPointer.new(:size_t)
        rc = Blazen::FFI.blazen_compat3d_result_glb_bytes(@ptr, out_ptr, out_len)
        raise Blazen::InternalError, "glb_bytes accessor returned #{rc}" unless rc.zero?

        data_ptr = out_ptr.read_pointer
        length   = out_len.read(:size_t)
        return String.new(encoding: Encoding::BINARY) if data_ptr.nil? || data_ptr.null? || length.zero?

        data_ptr.read_bytes(length).force_encoding(Encoding::BINARY)
      end

      # @return [String] the MIME type of {#glb_bytes}; always
      #   +"model/gltf-binary"+ from this backend.
      def mime_type
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_compat3d_result_mime_type(@ptr)) || ""
      end

      # @return [Boolean] +true+ when this is a +texturize+ result and
      #   the upstream returned an out-of-band PBR-map bundle.
      def pbr_maps?
        Blazen::FFI.blazen_compat3d_result_has_pbr_maps(@ptr) == 1
      end

      # Returns the bytes of one PBR channel. Returns +nil+ for any
      # other stage / missing channel.
      #
      # @param channel [Symbol] one of +:albedo+, +:normal+,
      #   +:roughness+, +:metallic+
      # @return [String, nil]
      def pbr_map(channel)
        ch = pbr_channel_id(channel)
        out_ptr = ::FFI::MemoryPointer.new(:pointer)
        out_len = ::FFI::MemoryPointer.new(:size_t)
        rc = Blazen::FFI.blazen_compat3d_result_pbr_map_bytes(@ptr, ch, out_ptr, out_len)
        return nil unless rc.zero?

        length   = out_len.read(:size_t)
        data_ptr = out_ptr.read_pointer
        return nil if data_ptr.nil? || data_ptr.null? || length.zero?

        data_ptr.read_bytes(length).force_encoding(Encoding::BINARY)
      end

      # @return [Array<String>] bone names for a +rig+ result (in
      #   depth-first traversal order); empty array for any other stage.
      def bone_names
        len = Blazen::FFI.blazen_compat3d_result_bone_names_count(@ptr)
        return [] if len.zero?

        Array.new(len) do |i|
          Blazen::FFI.consume_cstring(
            Blazen::FFI.blazen_compat3d_result_bone_name_get(@ptr, i),
          ) || ""
        end
      end

      # @return [Hash{Symbol => Integer, nil}] +{input_tri_count:,
      #   output_tri_count:, uv_chart_count:}+ for a +refine+ result;
      #   +uv_chart_count+ is +nil+ when UV unwrapping wasn't requested.
      #   Returns +nil+ for any other stage.
      def refine_stats
        return nil unless stage == :refine

        uv = Blazen::FFI.blazen_compat3d_result_refine_uv_chart_count(@ptr)
        {
          input_tri_count: Blazen::FFI.blazen_compat3d_result_refine_input_tri_count(@ptr),
          output_tri_count: Blazen::FFI.blazen_compat3d_result_refine_output_tri_count(@ptr),
          uv_chart_count: uv.negative? ? nil : uv,
        }
      end

      # @return [Float, nil] the produced animation duration in seconds
      #   for an +animate+ result; +nil+ for any other stage.
      def duration_seconds
        return nil unless stage == :animate

        Blazen::FFI.blazen_compat3d_result_animate_duration_seconds(@ptr)
      end

      # @return [Integer, nil] the produced animation framerate for an
      #   +animate+ result; +nil+ for any other stage.
      def fps
        return nil unless stage == :animate

        Blazen::FFI.blazen_compat3d_result_animate_fps(@ptr)
      end

      private

      def pbr_channel_id(channel)
        case channel
        when :albedo    then Blazen::FFI::PBR_MAP_ALBEDO
        when :normal    then Blazen::FFI::PBR_MAP_NORMAL
        when :roughness then Blazen::FFI::PBR_MAP_ROUGHNESS
        when :metallic  then Blazen::FFI::PBR_MAP_METALLIC
        else
          raise Blazen::ValidationError,
                "unknown PBR channel: #{channel.inspect} " \
                "(expected :albedo, :normal, :roughness, :metallic)"
        end
      end
    end

    # HTTP-proxy 3D provider. See {Blazen::ThreeD} for usage.
    class Compat3dProvider
      # @return [::FFI::AutoPointer] underlying +BlazenCompat3dProvider *+
      attr_reader :ptr

      # @param base_url [String] upstream root URL
      #   (e.g. +"https://3d.example.com"+)
      # @param api_key [String, nil] optional bearer token attached as
      #   +Authorization: Bearer ...+
      # @param timeout_secs [Integer, nil] optional per-request timeout
      #   in seconds. +nil+ / +0+ → 10-minute default.
      def initialize(base_url:, api_key: nil, timeout_secs: nil)
        unless Blazen::ThreeD.available?
          raise FeatureMissing
        end
        raise Blazen::ValidationError, "base_url must be a non-empty string" \
          if base_url.nil? || base_url.to_s.empty?

        raw = Blazen::FFI.with_cstring(base_url.to_s) do |base|
          Blazen::FFI.with_cstring(api_key) do |key|
            Blazen::FFI.blazen_compat3d_provider_new(
              base, key, Integer(timeout_secs || 0),
            )
          end
        end
        if raw.nil? || raw.null?
          raise Blazen::InternalError,
                "blazen_compat3d_provider_new returned null"
        end

        @ptr = ::FFI::AutoPointer.new(
          raw, Blazen::FFI.method(:blazen_compat3d_provider_free),
        )
      end

      # Apply or generate a texture / material for an input mesh GLB.
      #
      # @param mesh_glb [String] input mesh as GLB bytes (binary)
      # @param prompt [String, nil] text-guided texture prompt
      # @param reference_image [String, nil] PNG/JPEG reference bytes
      # @param style [String, nil] backend-specific style preset
      # @param resolution [Integer, nil] target square texture
      #   resolution in pixels
      # @param pbr [Boolean] request a full PBR material bundle
      # @return [Blazen::ThreeD::Result]
      # @raise [Blazen::Error] on any provider / network failure
      def texturize(mesh_glb, prompt: nil, reference_image: nil, style: nil,
                    resolution: nil, pbr: false)
        json = compact_json(
          prompt: prompt, style: style, resolution: resolution, pbr: pbr,
        )
        with_byte_buffer(mesh_glb) do |mesh_ptr, mesh_len|
          with_byte_buffer(reference_image) do |ref_ptr, ref_len|
            Blazen::FFI.with_cstring(json) do |json_ptr|
              fut = Blazen::FFI.blazen_compat3d_texturize(
                @ptr, mesh_ptr, mesh_len, ref_ptr, ref_len, json_ptr,
              )
              take_future_result(fut, :texturize)
            end
          end
        end
      end

      # Auto-rig a 3D mesh.
      #
      # @param mesh_glb [String]
      # @param template [String, nil] +"humanoid"+ / +"quadruped"+ /
      #   +"auto"+ (backend-specific)
      # @param skin [Boolean] apply skin-weight painting after armature
      # @param pose_hint [String, nil] +"t-pose"+ / +"a-pose"+ / backend
      #   JSON
      # @return [Blazen::ThreeD::Result]
      def rig(mesh_glb, template: nil, skin: false, pose_hint: nil)
        json = compact_json(template: template, skin: skin, pose_hint: pose_hint)
        with_byte_buffer(mesh_glb) do |mesh_ptr, mesh_len|
          Blazen::FFI.with_cstring(json) do |json_ptr|
            fut = Blazen::FFI.blazen_compat3d_rig(
              @ptr, mesh_ptr, mesh_len, json_ptr,
            )
            take_future_result(fut, :rig)
          end
        end
      end

      # Refine a mesh: decimate / hole-fill / UV-unwrap / retopologize /
      # smooth.
      #
      # @param mesh_glb [String]
      # @param decimate_target_tris [Integer, nil]
      # @param fill_holes [Boolean]
      # @param unwrap_uvs [Boolean]
      # @param retopologize [Boolean]
      # @param smooth_iterations [Integer, nil]
      # @return [Blazen::ThreeD::Result]
      def refine(mesh_glb, decimate_target_tris: nil, fill_holes: false,
                 unwrap_uvs: false, retopologize: false, smooth_iterations: nil)
        json = compact_json(
          decimate_target_tris: decimate_target_tris,
          fill_holes: fill_holes,
          unwrap_uvs: unwrap_uvs,
          retopologize: retopologize,
          smooth_iterations: smooth_iterations,
        )
        with_byte_buffer(mesh_glb) do |mesh_ptr, mesh_len|
          Blazen::FFI.with_cstring(json) do |json_ptr|
            fut = Blazen::FFI.blazen_compat3d_refine(
              @ptr, mesh_ptr, mesh_len, json_ptr,
            )
            take_future_result(fut, :refine)
          end
        end
      end

      # Animate a rigged mesh from a text prompt, MP4 video, or BVH clip.
      #
      # @param rigged_glb [String] rigged input mesh bytes
      # @param prompt [String, nil] text motion prompt
      # @param driving_video [String, nil] MP4 driving video bytes
      # @param bvh_motion [String, nil] BVH motion-capture clip bytes
      # @param duration_seconds [Float, nil]
      # @param fps [Integer, nil]
      # @param loop_animation [Boolean] mark output as a seamless loop
      # @return [Blazen::ThreeD::Result]
      def animate(rigged_glb, prompt: nil, driving_video: nil, bvh_motion: nil,
                  duration_seconds: nil, fps: nil, loop_animation: false)
        json = compact_json(
          prompt: prompt, duration_seconds: duration_seconds, fps: fps,
          loop_animation: loop_animation,
        )
        with_byte_buffer(rigged_glb) do |mesh_ptr, mesh_len|
          with_byte_buffer(driving_video) do |drv_ptr, drv_len|
            with_byte_buffer(bvh_motion) do |bvh_ptr, bvh_len|
              Blazen::FFI.with_cstring(json) do |json_ptr|
                fut = Blazen::FFI.blazen_compat3d_animate(
                  @ptr, mesh_ptr, mesh_len, drv_ptr, drv_len,
                  bvh_ptr, bvh_len, json_ptr,
                )
                take_future_result(fut, :animate)
              end
            end
          end
        end
      end

      private

      # Builds a JSON string with +nil+ values dropped and +false+
      # booleans normalised. Mirrors the cbindgen wire-format
      # contract — see +crates/blazen-cabi/src/threed.rs+ for the
      # accepted shape per stage.
      def compact_json(fields)
        compact = fields.reject { |_, v| v.nil? }
        JSON.generate(compact)
      end

      # Yields a +(MemoryPointer, length)+ pair for +bytes+. Yields
      # +(nil, 0)+ when +bytes+ is nil or empty. The buffer's strong
      # ref is held in the block scope so GC can't reclaim it
      # mid-call.
      def with_byte_buffer(bytes)
        if bytes.nil? || bytes.empty?
          return yield(nil, 0)
        end

        raw = bytes.is_a?(String) ? bytes.b : bytes.pack("C*")
        len = raw.bytesize
        buf = ::FFI::MemoryPointer.new(:uint8, len)
        buf.write_bytes(raw, 0, len)
        yield(buf, len)
      end

      def take_future_result(fut, stage)
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_compat3d_#{stage} rejected the call (null inputs?)"
        end

        Blazen::FFI.await_future(fut) do |f|
          out = ::FFI::MemoryPointer.new(:pointer)
          out_err = ::FFI::MemoryPointer.new(:pointer)
          Blazen::FFI.blazen_future_take_compat3d_result(f, out, out_err)
          Blazen::FFI.check_error!(out_err)
          Result.new(out.read_pointer, stage)
        end
      end
    end
  end
end
