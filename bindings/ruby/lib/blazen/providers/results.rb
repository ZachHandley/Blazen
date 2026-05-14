# frozen_string_literal: true

require "json"
require_relative "../ffi"

module Blazen
  # User-constructible typed-result wrappers used by Ruby +CustomProvider+
  # subclasses (V2). Each class accepts a Ruby +Hash+ matching the matching
  # serde schema of its Rust counterpart (+blazen_llm::compute::AudioResult+,
  # +blazen_llm::compute::ImageResult+, …), serialises to JSON, and asks the
  # cabi to materialise a heap-owned +*mut Blazen<X>+ handle.
  #
  # ## Two uses
  #
  # 1. As the return value of a +CustomProvider#text_to_speech+ /
  #    +#generate_image+ / etc. override. The trampoline calls
  #    {#take_ptr!} to steal ownership of the cabi handle and writes it
  #    through the vtable's +out_response+ slot. After that, the Ruby-side
  #    finalizer is disabled so the cabi side is the sole owner.
  # 2. As a standalone constructor — useful for tests or for hosts that
  #    want to materialise a typed result from JSON-shaped data without
  #    going through a vtable.
  #
  # On cabi-side parse / shape failure the constructor raises
  # +Blazen::InternalError+ (or whatever the cabi-side
  # +blazen_error_from_json+ analog produced — typically +Internal+).
  #
  # ## Naming note
  #
  # The two response wrappers below are intentionally named
  # +Blazen::CompletionResult+ / +Blazen::EmbeddingResult+ rather than
  # +CompletionResponse+ / +EmbeddingResponse+ — the latter are accessor
  # classes living in +blazen/llm.rb+ that wrap caller-owned handles
  # produced by +complete_blocking+ / +embed_blocking+. The +Result+
  # variants here are *constructors* a Ruby override hands back, and they
  # share a base class with the media-result wrappers
  # (+AudioResult+, +ImageResult+, …). Keeping the names disjoint avoids
  # ambiguity at call sites and mirrors the cabi-side
  # +blazen_*_response_from_json+ vs +blazen_*_response_<accessor>+ split.
  module Providers
    # Shared lifecycle + take-ownership semantics for every +Blazen::*Result+
    # wrapper below. Subclasses set +@ptr+ in +#initialize+ and arrange a
    # finalizer; users (typically the +CustomProvider+ trampoline) call
    # {#take_ptr!} to surrender ownership to the cabi.
    module ResultHandle
      # The underlying +*mut Blazen<X>+, or +nil+ after {#take_ptr!}.
      # @return [::FFI::Pointer, nil]
      attr_reader :ptr

      # Surrenders ownership of the underlying cabi handle. After this call,
      # the Ruby-side finalizer is removed and +@ptr+ becomes +nil+ — the
      # caller is responsible for ensuring the handle ends up freed (e.g.
      # by writing it through a vtable +out_response+ slot whose receiver
      # will free it).
      #
      # Idempotent: second + subsequent calls return +nil+.
      #
      # @return [::FFI::Pointer, nil]
      def take_ptr!
        ptr = @ptr
        @ptr = nil
        ObjectSpace.undefine_finalizer(self) if ptr
        ptr
      end
    end

    # @api private
    # Builds a typed-result handle from a Ruby Hash via the matching
    # +blazen_*_from_json+ cabi entry point. Raises
    # +Blazen::InternalError+ on parse / shape failure.
    #
    # @param hash [Hash] schema-shaped record
    # @param from_json_sym [Symbol] e.g. +:blazen_audio_result_from_json+
    # @param fn_label [String] human-readable label used in fallback errors
    # @return [::FFI::Pointer] caller-owned cabi handle
    def self.construct_from_json(hash, from_json_sym, fn_label)
      unless hash.is_a?(::Hash)
        raise Blazen::InternalError,
              "#{fn_label}: expected a Hash, got #{hash.class}"
      end

      json = JSON.dump(hash)
      json_ptr = ::FFI::MemoryPointer.from_string(json)
      err_holder = ::FFI::MemoryPointer.new(:pointer)
      ptr = Blazen::FFI.send(from_json_sym, json_ptr, err_holder)
      return ptr unless ptr.nil? || ptr.null?

      # cabi populated *err_holder; surface its message and free it.
      err = err_holder.read_pointer
      msg =
        if err.null?
          "#{fn_label}: returned null (no error reported)"
        else
          Blazen::FFI.consume_cstring(Blazen::FFI.blazen_error_message(err))
        end
      Blazen::FFI.blazen_error_free(err) unless err.null?
      raise Blazen::InternalError, msg || "#{fn_label}: returned null"
    end
  end

  # Audio output for +CustomProvider#text_to_speech+ / +#generate_music+ /
  # +#generate_sfx+ overrides. Wraps +BlazenAudioResult *+.
  class AudioResult
    include Providers::ResultHandle

    # @param data [Hash] schema-shaped audio-result record
    def initialize(data)
      @ptr = Providers.construct_from_json(
        data, :blazen_audio_result_from_json, "blazen_audio_result_from_json"
      )
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    # @api private
    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_audio_result_free(ptr) unless ptr.nil? || ptr.null? }
    end
  end

  # Image output for +CustomProvider#generate_image+ / +#upscale_image+ /
  # +#remove_background+ overrides. Wraps +BlazenImageResult *+.
  class ImageResult
    include Providers::ResultHandle

    # @param data [Hash] schema-shaped image-result record
    def initialize(data)
      @ptr = Providers.construct_from_json(
        data, :blazen_image_result_from_json, "blazen_image_result_from_json"
      )
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    # @api private
    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_image_result_free(ptr) unless ptr.nil? || ptr.null? }
    end
  end

  # Video output for +CustomProvider#text_to_video+ / +#image_to_video+
  # overrides. Wraps +BlazenVideoResult *+.
  class VideoResult
    include Providers::ResultHandle

    # @param data [Hash] schema-shaped video-result record
    def initialize(data)
      @ptr = Providers.construct_from_json(
        data, :blazen_video_result_from_json, "blazen_video_result_from_json"
      )
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    # @api private
    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_video_result_free(ptr) unless ptr.nil? || ptr.null? }
    end
  end

  # 3D-model output for +CustomProvider#generate_3d+ overrides. Wraps
  # +BlazenThreeDResult *+.
  class ThreeDResult
    include Providers::ResultHandle

    # @param data [Hash] schema-shaped three-d-result record
    def initialize(data)
      @ptr = Providers.construct_from_json(
        data, :blazen_three_d_result_from_json, "blazen_three_d_result_from_json"
      )
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    # @api private
    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_three_d_result_free(ptr) unless ptr.nil? || ptr.null? }
    end
  end

  # Transcription output for +CustomProvider#transcribe+ overrides. Wraps
  # +BlazenTranscriptionResult *+.
  class TranscriptionResult
    include Providers::ResultHandle

    # @param data [Hash] schema-shaped transcription-result record
    def initialize(data)
      @ptr = Providers.construct_from_json(
        data, :blazen_transcription_result_from_json, "blazen_transcription_result_from_json"
      )
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    # @api private
    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_transcription_result_free(ptr) unless ptr.nil? || ptr.null? }
    end
  end

  # Voice-handle output for +CustomProvider#clone_voice+ overrides (and an
  # element of the +#list_voices+ array). Wraps +BlazenVoiceHandle *+.
  class VoiceHandle
    include Providers::ResultHandle

    # @param data [Hash] schema-shaped voice-handle record
    def initialize(data)
      @ptr = Providers.construct_from_json(
        data, :blazen_voice_handle_from_json, "blazen_voice_handle_from_json"
      )
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    # @api private
    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_voice_handle_free(ptr) unless ptr.nil? || ptr.null? }
    end
  end

  # Completion output for +CustomProvider#complete+ overrides. Wraps
  # +BlazenCompletionResponse *+. Disambiguated from
  # +Blazen::CompletionResponse+ (accessor for caller-owned handles
  # returned from +complete_blocking+) — see the +Blazen::Providers+
  # docstring for rationale.
  class CompletionResult
    include Providers::ResultHandle

    # @param data [Hash] schema-shaped completion-response record
    def initialize(data)
      @ptr = Providers.construct_from_json(
        data, :blazen_completion_response_from_json,
        "blazen_completion_response_from_json"
      )
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    # @api private
    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_completion_response_free(ptr) unless ptr.nil? || ptr.null? }
    end
  end

  # Embedding output for +CustomProvider#embed+ overrides. Wraps
  # +BlazenEmbeddingResponse *+. Disambiguated from
  # +Blazen::EmbeddingResponse+ (accessor for caller-owned handles) — see
  # the +Blazen::Providers+ docstring for rationale.
  class EmbeddingResult
    include Providers::ResultHandle

    # @param data [Hash] schema-shaped embedding-response record
    def initialize(data)
      @ptr = Providers.construct_from_json(
        data, :blazen_embedding_response_from_json,
        "blazen_embedding_response_from_json"
      )
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    # @api private
    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_embedding_response_free(ptr) unless ptr.nil? || ptr.null? }
    end
  end
end
