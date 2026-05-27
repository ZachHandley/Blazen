# frozen_string_literal: true

module Blazen
  # Shared typed-result wrappers and the streaming-capable music / voice
  # conversion model handles used by the per-engine provider classes
  # (+lib/blazen/providers/{tts,stt,music,vc,three_d,image}.rb+) and by the
  # streaming surface (+lib/blazen/streaming.rb+).
  #
  # ## History
  #
  # The +Blazen::Compute+ namespace previously also carried a set of
  # keyword-arg module-factory functions (+Compute.fal_tts+,
  # +Compute.musicgen+, +Compute.rvc+, …) plus the +TtsModel+ / +SttModel+
  # / +ImageGenModel+ / +ThreeDModel+ wrapper classes. Those were a 1:1
  # match for the concrete per-engine provider classes
  # ({Blazen::PiperProvider}, {Blazen::FalTtsProvider},
  # {Blazen::MusicGenProvider}, {Blazen::RvcProvider}, …) and have been
  # removed. The per-engine classes are the sole construction surface now.
  #
  # What remains here are the pieces the per-engine classes and the
  # streaming surface still depend on:
  #
  # * the typed-result wrappers ({TtsResult}, {SttResult},
  #   {ImageGenResult}, {MusicResult}, {ThreeDGenerateResult},
  #   {VcResult}) returned by the provider capability methods,
  # * the streaming-chunk wrappers ({MusicChunk}, {VcChunk}) handed to the
  #   +on_chunk+ callbacks of {Blazen::Streaming.stream_music} /
  #   {Blazen::Streaming.stream_convert},
  # * the {TargetVoice} wrapper + {VcModel._take_target_voice_list} helper
  #   used by {Blazen::VcProvider#list_target_voices},
  # * the {MusicModel} / {VcModel} streaming-handle classes — the only
  #   surface that drives the +blazen_music_model_stream_*+ /
  #   +blazen_vc_model_stream_*+ cabi entry points (the per-engine
  #   provider opaques do not carry a streaming entry point).
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

    # Idiomatic Ruby wrapper around a +BlazenMusicModel+ handle.
    #
    # The per-engine music provider classes ({Blazen::MusicGenProvider},
    # {Blazen::AudioGenProvider}, {Blazen::StableAudioProvider},
    # {Blazen::FalMusicProvider}) are the construction surface for
    # non-streaming generation. {MusicModel} remains the handle the
    # streaming surface ({Blazen::Streaming.stream_music}) drives, since
    # the per-engine opaques do not carry a streaming entry point.
    class MusicModel
      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "MusicModel: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_music_model_free))
      end

      # Build a fal.ai-backed streaming-capable music model.
      #
      # @param api_key [String] fal.ai API key (empty string resolves
      #   +FAL_KEY+ from the environment upstream)
      # @param model [String, nil] override the default fal music / SFX
      #   endpoint identifier
      # @return [Blazen::Compute::MusicModel]
      def self.fal(api_key:, model: nil)
        out_model = ::FFI::MemoryPointer.new(:pointer)
        out_err   = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(api_key.to_s) do |key|
          Blazen::FFI.with_cstring(model) do |m|
            Blazen::FFI.blazen_music_model_new_fal(key, m, out_model, out_err)
          end
        end
        Blazen::FFI.check_error!(out_err)
        new(out_model.read_pointer)
      end

      # Build a native MusicGen-backed streaming-capable music model.
      #
      # @raise [Blazen::UnsupportedError] when the +music-musicgen+ feature is missing
      # @return [Blazen::Compute::MusicModel]
      def self.musicgen(variant: nil, device: nil, cache_dir: nil, max_duration_seconds: nil)
        unless Blazen::FFI.respond_to?(:blazen_music_model_new_musicgen)
          raise Blazen::UnsupportedError,
                "blazen was built without the 'music-musicgen' feature"
        end

        out_model = ::FFI::MemoryPointer.new(:pointer)
        out_err   = ::FFI::MemoryPointer.new(:pointer)
        max_dur   = max_duration_seconds.nil? ? Float::NAN : Float(max_duration_seconds)
        Blazen::FFI.with_cstring(variant) do |v|
          Blazen::FFI.with_cstring(device) do |dev|
            Blazen::FFI.with_cstring(cache_dir) do |cd|
              Blazen::FFI.blazen_music_model_new_musicgen(v, dev, cd, max_dur, out_model, out_err)
            end
          end
        end
        Blazen::FFI.check_error!(out_err)
        new(out_model.read_pointer)
      end

      # Build a native Stable Audio Open-backed streaming-capable music model.
      #
      # @raise [Blazen::UnsupportedError] when the +music-stable-audio+ feature is missing
      # @return [Blazen::Compute::MusicModel]
      def self.stable_audio(tokenizer_path:, variant: nil, device: nil, max_duration_seconds: nil)
        unless Blazen::FFI.respond_to?(:blazen_music_model_new_stable_audio)
          raise Blazen::UnsupportedError,
                "blazen was built without the 'music-stable-audio' feature"
        end

        out_model = ::FFI::MemoryPointer.new(:pointer)
        out_err   = ::FFI::MemoryPointer.new(:pointer)
        max_dur   = max_duration_seconds.nil? ? Float::NAN : Float(max_duration_seconds)
        Blazen::FFI.with_cstring(variant) do |v|
          Blazen::FFI.with_cstring(tokenizer_path.to_s) do |tok|
            Blazen::FFI.with_cstring(device) do |dev|
              Blazen::FFI.blazen_music_model_new_stable_audio(v, tok, dev, max_dur, out_model, out_err)
            end
          end
        end
        Blazen::FFI.check_error!(out_err)
        new(out_model.read_pointer)
      end

      # Build a native AudioGen-backed streaming-capable music / SFX model.
      #
      # @raise [Blazen::UnsupportedError] when the +music-audiogen+ feature is missing
      # @return [Blazen::Compute::MusicModel]
      def self.audiogen(repo_id: nil, revision: nil, device: nil, cache_dir: nil, max_duration_seconds: nil)
        unless Blazen::FFI.respond_to?(:blazen_music_model_new_audiogen)
          raise Blazen::UnsupportedError,
                "blazen was built without the 'music-audiogen' feature"
        end

        out_model = ::FFI::MemoryPointer.new(:pointer)
        out_err   = ::FFI::MemoryPointer.new(:pointer)
        max_dur   = max_duration_seconds.nil? ? Float::NAN : Float(max_duration_seconds)
        Blazen::FFI.with_cstring(repo_id) do |r|
          Blazen::FFI.with_cstring(revision) do |rev|
            Blazen::FFI.with_cstring(device) do |dev|
              Blazen::FFI.with_cstring(cache_dir) do |cd|
                Blazen::FFI.blazen_music_model_new_audiogen(r, rev, dev, cd, max_dur, out_model, out_err)
              end
            end
          end
        end
        Blazen::FFI.check_error!(out_err)
        new(out_model.read_pointer)
      end

      # Asynchronously generate +duration_seconds+ of music conditioned
      # on +prompt+. Composes with +Fiber.scheduler+.
      #
      # @param prompt [String]
      # @param duration_seconds [Float]
      # @return [Blazen::Compute::MusicResult]
      def generate_music(prompt, duration_seconds)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        dur        = Float(duration_seconds)
        fut = Blazen::FFI.with_cstring(prompt.to_s) do |p|
          Blazen::FFI.blazen_music_model_generate_music(@ptr, p, dur)
        end
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_music_model_generate_music returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_music_result(f, out_result, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        MusicResult.new(out_result.read_pointer)
      end

      # Blocking-thread variant of {#generate_music}.
      #
      # @param prompt [String]
      # @param duration_seconds [Float]
      # @return [Blazen::Compute::MusicResult]
      def generate_music_blocking(prompt, duration_seconds)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        dur        = Float(duration_seconds)
        Blazen::FFI.with_cstring(prompt.to_s) do |p|
          Blazen::FFI.blazen_music_model_generate_music_blocking(@ptr, p, dur, out_result, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        MusicResult.new(out_result.read_pointer)
      end

      # Asynchronously generate +duration_seconds+ of SFX audio
      # conditioned on +prompt+.
      #
      # @param prompt [String]
      # @param duration_seconds [Float]
      # @return [Blazen::Compute::MusicResult]
      def generate_sfx(prompt, duration_seconds)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        dur        = Float(duration_seconds)
        fut = Blazen::FFI.with_cstring(prompt.to_s) do |p|
          Blazen::FFI.blazen_music_model_generate_sfx(@ptr, p, dur)
        end
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_music_model_generate_sfx returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_music_result(f, out_result, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        MusicResult.new(out_result.read_pointer)
      end

      # Blocking-thread variant of {#generate_sfx}.
      #
      # @param prompt [String]
      # @param duration_seconds [Float]
      # @return [Blazen::Compute::MusicResult]
      def generate_sfx_blocking(prompt, duration_seconds)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        dur        = Float(duration_seconds)
        Blazen::FFI.with_cstring(prompt.to_s) do |p|
          Blazen::FFI.blazen_music_model_generate_sfx_blocking(@ptr, p, dur, out_result, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        MusicResult.new(out_result.read_pointer)
      end

      # Drive a streaming music generation, dispatching each {MusicChunk}
      # to the supplied callbacks (or block). See
      # {Blazen::Streaming.stream_music} for the full kwargs / block
      # contract.
      #
      # @return [void]
      def stream_generate_music(prompt, duration_seconds,
                                on_chunk: nil, on_done: nil, on_error: nil, &block)
        Blazen::Streaming.stream_music(
          self, prompt, duration_seconds,
          mode: :music, blocking: true,
          on_chunk: on_chunk, on_done: on_done, on_error: on_error,
          &block
        )
      end

      # Async variant of {#stream_generate_music}.
      #
      # @return [void]
      def stream_generate_music_async(prompt, duration_seconds,
                                      on_chunk: nil, on_done: nil, on_error: nil, &block)
        Blazen::Streaming.stream_music(
          self, prompt, duration_seconds,
          mode: :music, blocking: false,
          on_chunk: on_chunk, on_done: on_done, on_error: on_error,
          &block
        )
      end

      # Drive a streaming SFX generation. See {#stream_generate_music}.
      #
      # @return [void]
      def stream_generate_sfx(prompt, duration_seconds,
                              on_chunk: nil, on_done: nil, on_error: nil, &block)
        Blazen::Streaming.stream_music(
          self, prompt, duration_seconds,
          mode: :sfx, blocking: true,
          on_chunk: on_chunk, on_done: on_done, on_error: on_error,
          &block
        )
      end

      # Async variant of {#stream_generate_sfx}.
      #
      # @return [void]
      def stream_generate_sfx_async(prompt, duration_seconds,
                                    on_chunk: nil, on_done: nil, on_error: nil, &block)
        Blazen::Streaming.stream_music(
          self, prompt, duration_seconds,
          mode: :sfx, blocking: false,
          on_chunk: on_chunk, on_done: on_done, on_error: on_error,
          &block
        )
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

    # Idiomatic Ruby wrapper around a +BlazenVcModel+ handle.
    #
    # The per-engine voice-conversion provider classes
    # ({Blazen::RvcProvider}, {Blazen::FalVcProvider}) are the construction
    # surface for non-streaming conversion. {VcModel} remains the handle
    # the streaming surface ({Blazen::Streaming.stream_convert}) drives,
    # since the per-engine opaques do not carry a streaming entry point.
    class VcModel
      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "VcModel: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_vc_model_free))
      end

      # Build a native RVC-backed streaming-capable voice-conversion model.
      #
      # @param voice_dir [String, nil] override the per-process
      #   +BLAZEN_RVC_VOICE_DIR+ environment variable
      # @param device [String, nil] +"cpu"+ / +"cuda"+ / +"cuda:N"+ /
      #   +"metal"+ / +"metal:N"+; +nil+ defers to CPU
      # @raise [Blazen::UnsupportedError] when the +audio-vc-rvc+ feature is missing
      # @return [Blazen::Compute::VcModel]
      def self.rvc(voice_dir: nil, device: nil)
        unless Blazen::FFI.respond_to?(:blazen_vc_model_new_rvc)
          raise Blazen::UnsupportedError,
                "blazen was built without the 'audio-vc-rvc' feature"
        end

        out_model = ::FFI::MemoryPointer.new(:pointer)
        out_err   = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(voice_dir) do |vd|
          Blazen::FFI.with_cstring(device) do |dev|
            Blazen::FFI.blazen_vc_model_new_rvc(vd, dev, out_model, out_err)
          end
        end
        Blazen::FFI.check_error!(out_err)
        new(out_model.read_pointer)
      end

      # Asynchronously convert the source utterance at +input_path+ into
      # the voice of registered target speaker +target_voice_id+.
      #
      # @param input_path [String]
      # @param target_voice_id [String]
      # @return [Blazen::Compute::VcResult]
      def convert_voice(input_path, target_voice_id)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        fut = Blazen::FFI.with_cstring(input_path.to_s) do |ip|
          Blazen::FFI.with_cstring(target_voice_id.to_s) do |v|
            Blazen::FFI.blazen_vc_model_convert_voice(@ptr, ip, v)
          end
        end
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_vc_model_convert_voice returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_vc_result(f, out_result, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        VcResult.new(out_result.read_pointer)
      end

      # Blocking-thread variant of {#convert_voice}.
      #
      # @param input_path [String]
      # @param target_voice_id [String]
      # @return [Blazen::Compute::VcResult]
      def convert_voice_blocking(input_path, target_voice_id)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(input_path.to_s) do |ip|
          Blazen::FFI.with_cstring(target_voice_id.to_s) do |v|
            Blazen::FFI.blazen_vc_model_convert_voice_blocking(@ptr, ip, v, out_result, out_err)
          end
        end
        Blazen::FFI.check_error!(out_err)
        VcResult.new(out_result.read_pointer)
      end

      # Asynchronously list the target voices this backend can render.
      #
      # @return [Array<Blazen::Compute::TargetVoice>]
      def list_target_voices
        out_list = ::FFI::MemoryPointer.new(:pointer)
        out_err  = ::FFI::MemoryPointer.new(:pointer)
        fut = Blazen::FFI.blazen_vc_model_list_target_voices(@ptr)
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_vc_model_list_target_voices returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_target_voice_list(f, out_list, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        self.class._take_target_voice_list(out_list.read_pointer)
      end

      # Blocking-thread variant of {#list_target_voices}.
      #
      # @return [Array<Blazen::Compute::TargetVoice>]
      def list_target_voices_blocking
        out_list = ::FFI::MemoryPointer.new(:pointer)
        out_err  = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_vc_model_list_target_voices_blocking(@ptr, out_list, out_err)
        Blazen::FFI.check_error!(out_err)
        self.class._take_target_voice_list(out_list.read_pointer)
      end

      # Asynchronously register a new target voice for the backend,
      # sourcing its identity from the reference utterance at +ref_path+.
      #
      # @param voice_id [String]
      # @param ref_path [String]
      # @return [void]
      def register_target_voice(voice_id, ref_path)
        fut = Blazen::FFI.with_cstring(voice_id.to_s) do |v|
          Blazen::FFI.with_cstring(ref_path.to_s) do |rp|
            Blazen::FFI.blazen_vc_model_register_target_voice(@ptr, v, rp)
          end
        end
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_vc_model_register_target_voice returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          out_err = ::FFI::MemoryPointer.new(:pointer)
          Blazen::FFI.blazen_future_take_unit(f, out_err)
          Blazen::FFI.check_error!(out_err)
        end
        nil
      end

      # Blocking-thread variant of {#register_target_voice}.
      #
      # @param voice_id [String]
      # @param ref_path [String]
      # @return [void]
      def register_target_voice_blocking(voice_id, ref_path)
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(voice_id.to_s) do |v|
          Blazen::FFI.with_cstring(ref_path.to_s) do |rp|
            Blazen::FFI.blazen_vc_model_register_target_voice_blocking(@ptr, v, rp, out_err)
          end
        end
        Blazen::FFI.check_error!(out_err)
        nil
      end

      # Drive a streaming voice conversion across +pcm_samples+ (an Array
      # of f32 PCM samples). See {Blazen::Streaming.stream_convert} for the
      # full kwargs / block contract.
      #
      # @param pcm_samples [Array<Float>, #to_a]
      # @param target_voice_id [String]
      # @return [void]
      def stream_convert_pcm(pcm_samples, target_voice_id,
                             on_chunk: nil, on_done: nil, on_error: nil, &block)
        Blazen::Streaming.stream_convert(
          self, pcm_samples, target_voice_id,
          blocking: true,
          on_chunk: on_chunk, on_done: on_done, on_error: on_error,
          &block
        )
      end

      # Async variant of {#stream_convert_pcm}.
      #
      # @param pcm_samples [Array<Float>, #to_a]
      # @param target_voice_id [String]
      # @return [void]
      def stream_convert_pcm_async(pcm_samples, target_voice_id,
                                   on_chunk: nil, on_done: nil, on_error: nil, &block)
        Blazen::Streaming.stream_convert(
          self, pcm_samples, target_voice_id,
          blocking: false,
          on_chunk: on_chunk, on_done: on_done, on_error: on_error,
          &block
        )
      end

      # @api private
      # Drains a freshly-popped +BlazenTargetVoiceList *+ into an array
      # of {TargetVoice} wrappers, then frees the list. Used by the
      # list_target_voices entry points above and by the per-engine
      # {Blazen::VcProvider} subclasses.
      def self._take_target_voice_list(list_ptr)
        return [] if list_ptr.nil? || list_ptr.null?

        begin
          len = Blazen::FFI.blazen_target_voice_list_len(list_ptr)
          Array.new(len) do |i|
            raw = Blazen::FFI.blazen_target_voice_list_take(list_ptr, i)
            TargetVoice.new(raw)
          end
        ensure
          Blazen::FFI.blazen_target_voice_list_free(list_ptr)
        end
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
    end
  end
end
