# frozen_string_literal: true

module Blazen
  # Shared typed-result wrappers used by the per-engine provider classes
  # (+lib/blazen/providers/{tts,stt,music,vc,three_d,image}.rb+).
  #
  # ## History
  #
  # The +Blazen::Compute+ namespace previously also carried a set of
  # keyword-arg module-factory functions (+Compute.fal_tts+,
  # +Compute.musicgen+, +Compute.rvc+, …) plus the +TtsModel+ / +SttModel+
  # / +ImageGenModel+ / +ThreeDModel+ / +MusicModel+ / +VcModel+ wrapper
  # classes. Those were a 1:1 match for the concrete per-engine provider
  # classes ({Blazen::PiperProvider}, {Blazen::FalTtsProvider},
  # {Blazen::MusicGenProvider}, {Blazen::RvcProvider}, …) and have been
  # removed. The per-engine classes are the sole construction surface now
  # and own their own streaming entry points
  # ({Blazen::MusicGenProvider#stream_generate_music},
  # {Blazen::RvcProvider#stream_convert_pcm}, …).
  #
  # What remains here are the pieces the per-engine classes still depend on:
  #
  # * the typed-result wrappers ({TtsResult}, {SttResult},
  #   {ImageGenResult}, {MusicResult}, {ThreeDGenerateResult},
  #   {VcResult}) returned by the provider capability methods,
  # * the streaming-chunk wrappers ({MusicChunk}, {VcChunk}) handed to the
  #   per-engine streaming +on_chunk+ callbacks,
  # * the {TargetVoice} wrapper + +TargetVoice._take_list+ helper used by
  #   {Blazen::VcProvider#list_target_voices}.
  module Compute
    # Idiomatic Ruby wrapper around a +BlazenTtsResult+ handle.
    class TtsResult
      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "TtsResult: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_tts_result_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [String] base64-encoded audio bytes (empty string when the
      #   provider only returned a URL)
      def audio_base64
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_tts_result_audio_base64(@ptr))
      end

      # @return [String]
      def mime_type
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_tts_result_mime_type(@ptr))
      end

      # @return [Integer]
      def duration_ms
        Blazen::FFI.blazen_tts_result_duration_ms(@ptr)
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenSttResult+ handle.
    class SttResult
      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "SttResult: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_stt_result_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [String]
      def transcript
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_stt_result_transcript(@ptr))
      end

      # @return [String]
      def language
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_stt_result_language(@ptr))
      end

      # @return [Integer]
      def duration_ms
        Blazen::FFI.blazen_stt_result_duration_ms(@ptr)
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenImageGenResult+ handle.
    #
    # Exposes the generated media artefacts via {#images}, each wrapped in
    # {Blazen::Llm::Media} when that class is available (post-R7 Agent A)
    # or as an {::FFI::AutoPointer} drop hook otherwise.
    class ImageGenResult
      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "ImageGenResult: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_image_gen_result_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [Integer] number of generated images
      def size
        Blazen::FFI.blazen_image_gen_result_images_count(@ptr)
      end
      alias length size

      # @param idx [Integer]
      # @return [Blazen::Llm::Media, ::FFI::AutoPointer]
      def at(idx)
        raw = Blazen::FFI.blazen_image_gen_result_images_get(@ptr, Integer(idx))
        raise IndexError, "ImageGenResult: index #{idx} out of bounds (size=#{size})" if raw.nil? || raw.null?

        if defined?(Blazen::Llm) && defined?(Blazen::Llm::Media)
          Blazen::Llm::Media.new(raw)
        else
          ::FFI::AutoPointer.new(raw, Blazen::FFI.method(:blazen_media_free))
        end
      end
      alias [] at

      # @return [Array<Blazen::Llm::Media>]
      def images
        Array.new(size) { |i| at(i) }
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenMusicChunk+ handle.
    #
    # Caller-owned: the cabi hands ownership of the chunk to the music
    # stream sink's +on_chunk+ callback, which wraps it in an
    # {::FFI::AutoPointer} so Ruby GC frees the chunk (via
    # +blazen_music_chunk_free+) once the user's handler returns.
    class MusicChunk
      # @api private
      # @param raw_ptr [::FFI::Pointer] caller-owned +BlazenMusicChunk *+
      def initialize(raw_ptr)
        if raw_ptr.nil? || raw_ptr.null?
          raise Blazen::InternalError, "MusicChunk: native pointer is null"
        end

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_music_chunk_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [Array<Float>] PCM sample slice (f32) for this chunk
      def samples
        len_ptr = ::FFI::MemoryPointer.new(:size_t)
        base    = Blazen::FFI.blazen_music_chunk_samples(@ptr, len_ptr)
        len     = len_ptr.read(:size_t)
        return [] if base.nil? || base.null? || len.zero?

        base.read_array_of_float(len)
      end

      # @return [Boolean] whether this is the final chunk of the stream
      def final?
        Blazen::FFI.blazen_music_chunk_is_final(@ptr)
      end

      # @return [Float, nil] per-chunk latency in seconds, or +nil+ when
      #   the backend did not report a measurement (NaN sentinel upstream)
      def latency_seconds
        v = Blazen::FFI.blazen_music_chunk_latency_seconds(@ptr)
        v.nan? ? nil : v
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenMusicResult+ handle.
    class MusicResult
      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "MusicResult: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_music_result_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [String] the encoded audio bytes (binary string). Empty
      #   when the upstream provider only returned a URL (check {#url}).
      def bytes
        len_ptr = ::FFI::MemoryPointer.new(:size_t)
        base    = Blazen::FFI.blazen_music_result_bytes(@ptr, len_ptr)
        len     = len_ptr.read(:size_t)
        return String.new(encoding: Encoding::ASCII_8BIT) if base.nil? || base.null? || len.zero?

        base.read_bytes(len).force_encoding(Encoding::ASCII_8BIT)
      end

      # @return [String, nil] IANA MIME type of the encoded audio
      def mime_type
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_music_result_mime_type(@ptr))
      end

      # @return [Integer] sample rate in Hz (+0+ when not reported)
      def sample_rate
        Blazen::FFI.blazen_music_result_sample_rate(@ptr)
      end

      # @return [Integer] channel count (1 = mono, 2 = stereo;
      #   +0+ when not reported)
      def channels
        Blazen::FFI.blazen_music_result_channels(@ptr)
      end

      # @return [Float, nil] duration in seconds, or +nil+ when the
      #   upstream provider did not report a duration (encoded as +0.0+
      #   on the wire)
      def duration_seconds
        v = Blazen::FFI.blazen_music_result_duration_seconds(@ptr)
        v.zero? ? nil : v
      end

      # @return [String, nil] URL of the audio asset when the upstream
      #   provider returned a link rather than inline bytes; +nil+ for
      #   inline-bytes results (cabi returns the empty string in that
      #   case, which we normalise to +nil+)
      def url
        s = Blazen::FFI.consume_cstring(Blazen::FFI.blazen_music_result_url(@ptr))
        return nil if s.nil? || s.empty?

        s
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenThreeDGenerateResult+ handle.
    #
    # Produced by {Blazen::TripoSrProvider#generate_from_image} +
    # +#generate_from_image_blocking+. Carries the encoded model bytes
    # (typically GLB / gltf-binary) and a MIME type string.
    class ThreeDGenerateResult
      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        if raw_ptr.nil? || raw_ptr.null?
          raise ArgumentError, "ThreeDGenerateResult: pointer must be non-null"
        end

        @ptr = ::FFI::AutoPointer.new(
          raw_ptr, Blazen::FFI.method(:blazen_three_d_generate_result_free),
        )
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [String] encoded 3D model bytes (binary string; typically
      #   GLB / gltf-binary container)
      def model_bytes
        len_ptr = ::FFI::MemoryPointer.new(:size_t)
        base    = Blazen::FFI.blazen_three_d_generate_result_model_bytes(@ptr, len_ptr)
        len     = len_ptr.read(:size_t)
        return String.new(encoding: Encoding::ASCII_8BIT) if base.nil? || base.null? || len.zero?

        base.read_bytes(len).force_encoding(Encoding::ASCII_8BIT)
      end

      # @return [String, nil] IANA MIME type of the encoded model
      #   (typically +"model/gltf-binary"+)
      def mime_type
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_three_d_generate_result_mime_type(@ptr))
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenVcChunk+ handle.
    #
    # Caller-owned: the cabi hands ownership of the chunk to the vc
    # stream sink's +on_chunk+ callback, which wraps it in an
    # {::FFI::AutoPointer} so Ruby GC frees the chunk (via
    # +blazen_vc_chunk_free+) once the user's handler returns.
    class VcChunk
      # @api private
      # @param raw_ptr [::FFI::Pointer] caller-owned +BlazenVcChunk *+
      def initialize(raw_ptr)
        if raw_ptr.nil? || raw_ptr.null?
          raise Blazen::InternalError, "VcChunk: native pointer is null"
        end

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_vc_chunk_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [Array<Float>] PCM sample slice (f32) for this chunk
      def samples
        len_ptr = ::FFI::MemoryPointer.new(:size_t)
        base    = Blazen::FFI.blazen_vc_chunk_samples(@ptr, len_ptr)
        len     = len_ptr.read(:size_t)
        return [] if base.nil? || base.null? || len.zero?

        base.read_array_of_float(len)
      end

      # @return [Boolean] whether this is the final chunk of the stream
      def final?
        Blazen::FFI.blazen_vc_chunk_is_final(@ptr)
      end

      # @return [Float, nil] per-chunk latency in seconds, or +nil+ when
      #   the backend did not report a measurement (NaN sentinel upstream)
      def latency_seconds
        v = Blazen::FFI.blazen_vc_chunk_latency_seconds(@ptr)
        v.nan? ? nil : v
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenVcResult+ handle.
    class VcResult
      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "VcResult: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_vc_result_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [String] the encoded WAV bytes (binary string). Empty
      #   when the backend produced no audio.
      def bytes
        len_ptr = ::FFI::MemoryPointer.new(:size_t)
        base    = Blazen::FFI.blazen_vc_result_bytes(@ptr, len_ptr)
        len     = len_ptr.read(:size_t)
        return String.new(encoding: Encoding::ASCII_8BIT) if base.nil? || base.null? || len.zero?

        base.read_bytes(len).force_encoding(Encoding::ASCII_8BIT)
      end

      # @return [String, nil] IANA MIME type of the encoded audio
      def mime_type
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_vc_result_mime_type(@ptr))
      end

      # @return [Integer] sample rate in Hz (+0+ when not reported)
      def sample_rate
        Blazen::FFI.blazen_vc_result_sample_rate(@ptr)
      end

      # @return [Float, nil] duration in seconds, or +nil+ when the
      #   backend did not report a duration (encoded as +0.0+ on the
      #   wire)
      def duration_seconds
        v = Blazen::FFI.blazen_vc_result_duration_seconds(@ptr)
        v.zero? ? nil : v
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenTargetVoice+ handle.
    class TargetVoice
      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        if raw_ptr.nil? || raw_ptr.null?
          raise ArgumentError, "TargetVoice: pointer must be non-null"
        end

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_target_voice_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [String] stable voice identifier
      def id
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_target_voice_id(@ptr))
      end

      # @return [String, nil] human-readable label (cabi returns the
      #   empty string when no label is set, which we normalise to +nil+)
      def label
        s = Blazen::FFI.consume_cstring(Blazen::FFI.blazen_target_voice_label(@ptr))
        return nil if s.nil? || s.empty?

        s
      end

      # @return [Integer] sample rate in Hz the voice was trained at
      def sample_rate_hz
        Blazen::FFI.blazen_target_voice_sample_rate_hz(@ptr)
      end

      # @api private
      # Drains a freshly-popped +BlazenTargetVoiceList *+ into an array of
      # {TargetVoice} wrappers, then frees the list. Used by the
      # per-engine {Blazen::VcProvider} subclasses
      # ({Blazen::RvcProvider}, {Blazen::FalVcProvider}) to materialise
      # the result of +list_target_voices[_blocking]+.
      def self._take_list(list_ptr)
        return [] if list_ptr.nil? || list_ptr.null?

        begin
          len = Blazen::FFI.blazen_target_voice_list_len(list_ptr)
          Array.new(len) do |i|
            raw = Blazen::FFI.blazen_target_voice_list_take(list_ptr, i)
            new(raw)
          end
        ensure
          Blazen::FFI.blazen_target_voice_list_free(list_ptr)
        end
      end
    end
  end
end
