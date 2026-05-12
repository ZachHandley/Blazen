# frozen_string_literal: true

module Blazen
  # Helpers for the compute family of models: speech-to-text ({SttModel}),
  # text-to-speech ({TtsModel}), and image generation ({ImageGenModel}).
  #
  # These wrappers translate keyword arguments to the positional UniFFI
  # signatures and route exceptions through {Blazen.translate_errors}.
  module Compute
    module_function

    # Transcribes audio with an {SttModel}.
    #
    # @param model [Blazen::SttModel]
    # @param audio_source [String] base64-encoded audio payload, file path,
    #   or URL — interpretation depends on the underlying STT backend
    # @param language [String, nil] expected language code (e.g. +"en"+)
    # @return [Blazen::SttResult]
    def transcribe(model, audio_source, language: nil)
      Blazen.translate_errors { model.transcribe_blocking(audio_source, language) }
    end

    # Synthesizes speech with a {TtsModel}.
    #
    # @param model [Blazen::TtsModel]
    # @param text [String] text to speak
    # @param voice [String, nil] voice identifier
    # @param language [String, nil] language code
    # @return [Blazen::TtsResult]
    def synthesize(model, text, voice: nil, language: nil)
      Blazen.translate_errors { model.synthesize_blocking(text, voice, language) }
    end

    # Generates images with an {ImageGenModel}.
    #
    # @param model [Blazen::ImageGenModel]
    # @param prompt [String] image prompt
    # @param negative_prompt [String, nil] negative prompt
    # @param width [Integer, nil] override width in pixels
    # @param height [Integer, nil] override height in pixels
    # @param num_images [Integer, nil] number of images to produce
    # @param model_override [String, nil] override the model identifier
    # @return [Blazen::ImageGenResult]
    def generate(model, prompt, negative_prompt: nil, width: nil, height: nil,
                 num_images: nil, model_override: nil)
      Blazen.translate_errors do
        model.generate_blocking(prompt, negative_prompt, width, height, num_images, model_override)
      end
    end
  end
end
