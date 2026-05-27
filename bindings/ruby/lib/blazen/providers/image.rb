# frozen_string_literal: true

# Concrete per-engine image-generation provider classes (Part U).
#
# Engines:
#   * {Blazen::DiffusionProvider}   — local Stable-Diffusion-family
#   * {Blazen::FalImageGenProvider} — fal.ai-hosted image generation

module Blazen
  # @api private
  module ImageGenProviderImpl
    private

    def image_generate(prompt, width, height, async_sym)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      out_err    = ::FFI::MemoryPointer.new(:pointer)
      # cabi uses uint32; 0 is the "None" sentinel
      w = width.nil?  ? 0 : Integer(width)
      h = height.nil? ? 0 : Integer(height)
      fut = Blazen::FFI.with_cstring(prompt.to_s) do |p|
        Blazen::FFI.public_send(async_sym, @handle, p, w, h)
      end
      if fut.nil? || fut.null?
        raise Blazen::ValidationError, "#{async_sym} returned a null future"
      end

      Blazen::FFI.await_future(fut) do |f|
        Blazen::FFI.blazen_future_take_image_gen_result(f, out_result, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      Blazen::Compute::ImageGenResult.new(out_result.read_pointer)
    end

    def image_generate_blocking(prompt, width, height, blocking_sym)
      out_result = ::FFI::MemoryPointer.new(:pointer)
      out_err    = ::FFI::MemoryPointer.new(:pointer)
      w = width.nil?  ? 0 : Integer(width)
      h = height.nil? ? 0 : Integer(height)
      Blazen::FFI.with_cstring(prompt.to_s) do |p|
        Blazen::FFI.public_send(blocking_sym, @handle, p, w, h, out_result, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      Blazen::Compute::ImageGenResult.new(out_result.read_pointer)
    end
  end

  # Local Stable-Diffusion-family image generator.
  class DiffusionProvider < ImageGenProvider
    include ImageGenProviderImpl

    PROVIDER_ID = "diffusion"

    # @param options_json [String, nil] backend-specific options as a JSON
    #   string (model id, scheduler, num steps, etc.)
    def initialize(options_json: nil)
      unless Blazen::FFI.respond_to?(:blazen_diffusion_provider_new)
        raise Blazen::UnsupportedError,
              "blazen was built without the 'diffusion' feature"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(options_json) do |opts|
        Blazen::FFI.blazen_diffusion_provider_new(opts, out_model, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_diffusion_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def generate_image(prompt, width: nil, height: nil)
      image_generate(prompt, width, height, :blazen_diffusion_provider_generate_image)
    end

    def generate_image_blocking(prompt, width: nil, height: nil)
      image_generate_blocking(
        prompt, width, height, :blazen_diffusion_provider_generate_image_blocking,
      )
    end
  end

  # fal.ai-hosted image generation.
  class FalImageGenProvider < ImageGenProvider
    include ImageGenProviderImpl

    PROVIDER_ID = "fal-image-gen"

    # @param api_key [String]
    # @param default_model [String, nil] e.g. +"fal-ai/flux/schnell"+
    # @param base_url [String, nil] override fal queue URL (proxies / staging)
    def initialize(api_key:, default_model: nil, base_url: nil)
      unless Blazen::FFI.respond_to?(:blazen_fal_image_gen_provider_new)
        raise Blazen::UnsupportedError,
              "blazen cabi missing fal_image_gen_provider_new symbol"
      end

      out_model = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(api_key.to_s) do |key|
        Blazen::FFI.with_cstring(default_model) do |dm|
          Blazen::FFI.with_cstring(base_url) do |bu|
            Blazen::FFI.blazen_fal_image_gen_provider_new(key, dm, bu, out_model, out_err)
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      super(out_model.read_pointer, Blazen::FFI.method(:blazen_fal_image_gen_provider_free))
    end

    def provider_id
      PROVIDER_ID
    end

    def generate_image(prompt, width: nil, height: nil)
      image_generate(prompt, width, height, :blazen_fal_image_gen_provider_generate_image)
    end

    def generate_image_blocking(prompt, width: nil, height: nil)
      image_generate_blocking(
        prompt, width, height, :blazen_fal_image_gen_provider_generate_image_blocking,
      )
    end
  end
end
