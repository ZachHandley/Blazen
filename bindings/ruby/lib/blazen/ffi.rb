# frozen_string_literal: true

require "ffi"
require "rbconfig"

module Blazen
  # Hand-written FFI bridge to +libblazen_cabi+.
  #
  # Mirrors every symbol declared in +ext/blazen/blazen.h+ (cbindgen-emitted
  # from the Rust +blazen-cabi+ crate). All foreign opaque types cross the FFI
  # boundary as +:pointer+ — typed accessors live in the per-type Ruby helper
  # modules in +lib/blazen/*.rb+. Three by-value vtable structs
  # ({BlazenStepHandlerVTable}, {BlazenToolHandlerVTable},
  # {BlazenCompletionStreamSinkVTable}) are mapped to {::FFI::Struct} subclasses
  # below so foreign callback hosts can fill them out in Ruby and pass them in
  # by value.
  module FFI
    extend ::FFI::Library

    # Per-arch subdirectory for prebuilt native libs produced by
    # +scripts/build-uniffi-lib.sh+. We probe both the per-arch path
    # (+ext/blazen/linux_amd64/libblazen_cabi.so+) and the flat path
    # (+ext/blazen/libblazen_cabi.so+) at load time.
    #
    # @return [String] the per-arch subdirectory ("" on unknown platforms)
    def self.arch_subdir
      host_os = RbConfig::CONFIG["host_os"]
      host_cpu = RbConfig::CONFIG["host_cpu"]
      case host_os
      when /linux/
        case host_cpu
        when "x86_64" then "linux_amd64"
        when /aarch64|arm64/ then "linux_arm64"
        else ""
        end
      when /darwin/
        case host_cpu
        when "x86_64" then "darwin_amd64"
        when /arm64|aarch64/ then "darwin_arm64"
        else ""
        end
      when /mingw|mswin/
        case host_cpu
        when "x86_64" then "windows_amd64"
        else ""
        end
      else
        ""
      end
    end

    # Native library filename for the current platform.
    LIB_NAME = case RbConfig::CONFIG["host_os"]
               when /darwin/ then "libblazen_cabi.dylib"
               when /mingw|mswin|cygwin/ then "libblazen_cabi.dll"
               else "libblazen_cabi.so"
               end

    # Candidate paths probed in order. The first existing file wins; the
    # bare name is the final fallback (delegated to the loader's search path).
    LIB_SEARCH_PATHS = [
      File.expand_path("../../ext/blazen/#{arch_subdir}/#{LIB_NAME}", __dir__),
      File.expand_path("../../ext/blazen/#{LIB_NAME}", __dir__),
    ].uniq

    lib_path = LIB_SEARCH_PATHS.find { |p| File.exist?(p) } || LIB_NAME
    ffi_lib lib_path

    # -------------------------------------------------------------------
    # Error-kind constants (mirror BLAZEN_ERROR_KIND_* from blazen.h)
    # -------------------------------------------------------------------
    ERROR_KIND_AUTH           = 1
    ERROR_KIND_RATE_LIMIT     = 2
    ERROR_KIND_TIMEOUT        = 3
    ERROR_KIND_VALIDATION     = 4
    ERROR_KIND_CONTENT_POLICY = 5
    ERROR_KIND_UNSUPPORTED    = 6
    ERROR_KIND_COMPUTE        = 7
    ERROR_KIND_MEDIA          = 8
    ERROR_KIND_PROVIDER       = 9
    ERROR_KIND_WORKFLOW       = 10
    ERROR_KIND_TOOL           = 11
    ERROR_KIND_PEER           = 12
    ERROR_KIND_PERSIST        = 13
    ERROR_KIND_PROMPT         = 14
    ERROR_KIND_MEMORY         = 15
    ERROR_KIND_CACHE          = 16
    ERROR_KIND_CANCELLED      = 17
    ERROR_KIND_INTERNAL       = 18

    # -------------------------------------------------------------------
    # BatchItem / StepOutput tag constants
    # -------------------------------------------------------------------
    BATCH_ITEM_SUCCESS = 0
    BATCH_ITEM_FAILURE = 1

    STEP_OUTPUT_NONE     = 0
    STEP_OUTPUT_SINGLE   = 1
    STEP_OUTPUT_MULTIPLE = 2

    # -------------------------------------------------------------------
    # Callback typedefs (used by the vtable structs that follow)
    # -------------------------------------------------------------------

    # StepHandler vtable callbacks
    callback :step_handler_drop_user_data, [:pointer], :void
    callback :step_handler_invoke,
             [:pointer, :pointer, :pointer, :pointer],
             :int32

    # ToolHandler vtable callbacks
    callback :tool_handler_drop_user_data, [:pointer], :void
    callback :tool_handler_execute,
             [:pointer, :pointer, :pointer, :pointer, :pointer],
             :int32

    # CompletionStreamSink vtable callbacks
    callback :stream_sink_drop_user_data, [:pointer], :void
    callback :stream_sink_on_chunk,
             [:pointer, :pointer, :pointer],
             :int32
    callback :stream_sink_on_done,
             [:pointer, :pointer, :pointer, :pointer],
             :int32
    callback :stream_sink_on_error,
             [:pointer, :pointer, :pointer],
             :int32

    # MusicStreamSink vtable callbacks (distinct from completion-stream:
    # music chunks carry PCM samples + a final-flag and the +on_done+
    # signal has no finish-reason / token-usage payload — see
    # +BlazenMusicStreamSinkVTable+ in +blazen.h+).
    callback :music_stream_sink_drop_user_data, [:pointer], :void
    callback :music_stream_sink_on_chunk,
             [:pointer, :pointer, :pointer],
             :int32
    callback :music_stream_sink_on_done,
             [:pointer, :pointer],
             :int32
    callback :music_stream_sink_on_error,
             [:pointer, :pointer, :pointer],
             :int32

    # VcStreamSink vtable callbacks (mirror +BlazenVcStreamSinkVTable+ in
    # +blazen.h+). Distinct from the music sink: voice-conversion chunks
    # carry PCM samples + a final-flag and the +on_done+ signal has no
    # auxiliary payload (matches the music sink shape — but the chunk
    # handle type is +BlazenVcChunk+).
    callback :vc_stream_sink_drop_user_data, [:pointer], :void
    callback :vc_stream_sink_on_chunk,
             [:pointer, :pointer, :pointer], :int32
    callback :vc_stream_sink_on_done,
             [:pointer, :pointer], :int32
    callback :vc_stream_sink_on_error,
             [:pointer, :pointer, :pointer], :int32

    # CustomProvider vtable callbacks — 16 typed-method fn-pointers + 2 lifecycle.
    # Every typed callback receives a request pointer (caller-owned, the callback
    # must free or consume it before returning) and writes either a success
    # result pointer into +out_*+ or a +BlazenError *+ into +out_err+. Returning
    # +-1+ without writing +out_err+ surfaces upstream as a synthetic
    # +InternalError+ (see the cabi adapter docs).
    callback :custom_provider_drop_user_data, [:pointer], :void
    callback :custom_provider_complete,
             [:pointer, :pointer, :pointer, :pointer],
             :int32
    callback :custom_provider_stream,
             [:pointer, :pointer, :pointer, :pointer],
             :int32
    callback :custom_provider_embed,
             [:pointer, :pointer, :size_t, :pointer, :pointer],
             :int32
    callback :custom_provider_text_to_speech,
             [:pointer, :pointer, :pointer, :pointer],
             :int32
    callback :custom_provider_generate_music,
             [:pointer, :pointer, :pointer, :pointer],
             :int32
    callback :custom_provider_generate_sfx,
             [:pointer, :pointer, :pointer, :pointer],
             :int32
    callback :custom_provider_clone_voice,
             [:pointer, :pointer, :pointer, :pointer],
             :int32
    callback :custom_provider_list_voices,
             [:pointer, :pointer, :pointer, :pointer],
             :int32
    callback :custom_provider_delete_voice,
             [:pointer, :pointer, :pointer],
             :int32
    callback :custom_provider_generate_image,
             [:pointer, :pointer, :pointer, :pointer],
             :int32
    callback :custom_provider_upscale_image,
             [:pointer, :pointer, :pointer, :pointer],
             :int32
    callback :custom_provider_text_to_video,
             [:pointer, :pointer, :pointer, :pointer],
             :int32
    callback :custom_provider_image_to_video,
             [:pointer, :pointer, :pointer, :pointer],
             :int32
    callback :custom_provider_transcribe,
             [:pointer, :pointer, :pointer, :pointer],
             :int32
    callback :custom_provider_generate_3d,
             [:pointer, :pointer, :pointer, :pointer],
             :int32
    callback :custom_provider_remove_background,
             [:pointer, :pointer, :pointer, :pointer],
             :int32

    # -------------------------------------------------------------------
    # By-value vtable structs (mirror layouts in blazen.h)
    # -------------------------------------------------------------------

    # Vtable a foreign caller fills in to implement a workflow step handler.
    class BlazenStepHandlerVTable < ::FFI::Struct
      layout :user_data,       :pointer,
             :drop_user_data,  :step_handler_drop_user_data,
             :invoke,          :step_handler_invoke
    end

    # Vtable a foreign caller fills in to implement an agent tool handler.
    class BlazenToolHandlerVTable < ::FFI::Struct
      layout :user_data,       :pointer,
             :drop_user_data,  :tool_handler_drop_user_data,
             :execute,         :tool_handler_execute
    end

    # Vtable a foreign caller fills in to receive streaming-completion events.
    class BlazenCompletionStreamSinkVTable < ::FFI::Struct
      layout :user_data,       :pointer,
             :drop_user_data,  :stream_sink_drop_user_data,
             :on_chunk,        :stream_sink_on_chunk,
             :on_done,         :stream_sink_on_done,
             :on_error,        :stream_sink_on_error
    end

    # Vtable a foreign caller fills in to receive streaming-music events
    # (mirrors +BlazenMusicStreamSinkVTable+ in +blazen.h+). Distinct from
    # +BlazenCompletionStreamSinkVTable+ — music's +on_done+ carries no
    # payload, and +on_chunk+ receives a +BlazenMusicChunk+ rather than a
    # +BlazenStreamChunk+.
    class BlazenMusicStreamSinkVTable < ::FFI::Struct
      layout :user_data,       :pointer,
             :drop_user_data,  :music_stream_sink_drop_user_data,
             :on_chunk,        :music_stream_sink_on_chunk,
             :on_done,         :music_stream_sink_on_done,
             :on_error,        :music_stream_sink_on_error
    end

    # Vtable a foreign caller fills in to receive streaming
    # voice-conversion events (mirrors +BlazenVcStreamSinkVTable+ in
    # +blazen.h+). Same shape as the music sink — chunks carry PCM
    # samples and +on_done+ is payload-free — but the chunk handle type
    # is +BlazenVcChunk+ rather than +BlazenMusicChunk+.
    class BlazenVcStreamSinkVTable < ::FFI::Struct
      layout :user_data,       :pointer,
             :drop_user_data,  :vc_stream_sink_drop_user_data,
             :on_chunk,        :vc_stream_sink_on_chunk,
             :on_done,         :vc_stream_sink_on_done,
             :on_error,        :vc_stream_sink_on_error
    end

    # Vtable a foreign caller fills in to implement a custom provider.
    # 18 fields: opaque +user_data+, +drop_user_data+ callback, two C-string
    # metadata pointers (+provider_id+, +model_id+) owned by the vtable, and
    # 16 typed-method fn-pointers (one per +InnerCustomProviderTrait+ method).
    class BlazenCustomProviderVTable < ::FFI::Struct
      layout :user_data,         :pointer,
             :drop_user_data,    :custom_provider_drop_user_data,
             :provider_id,       :pointer,
             :model_id,          :pointer,
             :complete,          :custom_provider_complete,
             :stream,            :custom_provider_stream,
             :embed,             :custom_provider_embed,
             :text_to_speech,    :custom_provider_text_to_speech,
             :generate_music,    :custom_provider_generate_music,
             :generate_sfx,      :custom_provider_generate_sfx,
             :clone_voice,       :custom_provider_clone_voice,
             :list_voices,       :custom_provider_list_voices,
             :delete_voice,      :custom_provider_delete_voice,
             :generate_image,    :custom_provider_generate_image,
             :upscale_image,     :custom_provider_upscale_image,
             :text_to_video,     :custom_provider_text_to_video,
             :image_to_video,    :custom_provider_image_to_video,
             :transcribe,        :custom_provider_transcribe,
             :generate_3d,       :custom_provider_generate_3d,
             :remove_background, :custom_provider_remove_background
    end

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------
    attach_function :blazen_version,           [],          :pointer
    attach_function :blazen_init,              [],          :int32
    attach_function :blazen_shutdown,          [],          :int32
    attach_function :blazen_shutdown_telemetry, [],         :void
    attach_function :blazen_string_free,       [:pointer],  :void
    attach_function :blazen_string_alloc,      [:pointer],  :pointer
    attach_function :blazen_string_array_free, [:pointer, :size_t], :void

    # -------------------------------------------------------------------
    # Error
    # -------------------------------------------------------------------
    attach_function :blazen_error_kind,             [:pointer], :uint32
    attach_function :blazen_error_message,          [:pointer], :pointer
    attach_function :blazen_error_retry_after_ms,   [:pointer], :int64
    attach_function :blazen_error_elapsed_ms,       [:pointer], :uint64
    attach_function :blazen_error_status,           [:pointer], :int32
    attach_function :blazen_error_provider,         [:pointer], :pointer
    attach_function :blazen_error_endpoint,         [:pointer], :pointer
    attach_function :blazen_error_request_id,       [:pointer], :pointer
    attach_function :blazen_error_detail,           [:pointer], :pointer
    attach_function :blazen_error_subkind,          [:pointer], :pointer
    attach_function :blazen_error_name,             [:pointer], :pointer
    attach_function :blazen_error_properties_json,  [:pointer], :pointer
    attach_function :blazen_error_from_json,        [:pointer], :pointer
    attach_function :blazen_error_free,             [:pointer], :void

    # -------------------------------------------------------------------
    # Future
    # -------------------------------------------------------------------
    attach_function :blazen_future_fd,    [:pointer], :int64
    attach_function :blazen_future_poll,  [:pointer], :int32
    attach_function :blazen_future_wait,  [:pointer], :int32, blocking: true
    attach_function :blazen_future_free,  [:pointer], :void

    # Typed future-take functions (one per output type)
    attach_function :blazen_future_take_agent_result,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_batch_result,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_tts_result,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_stt_result,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_image_gen_result,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_model_response,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_embedding_response,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_workflow_result,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_unit,
                    [:pointer, :pointer], :int32
    attach_function :blazen_future_take_string,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_workflow_checkpoint_option,
                    [:pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_workflow_checkpoint_list,
                    [:pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_string_list,
                    [:pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_trained_adapter,
                    [:pointer, :pointer, :pointer], :int32

    # -------------------------------------------------------------------
    # Agent (opaque object methods)
    # -------------------------------------------------------------------
    attach_function :blazen_agent_run_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_agent_run, [:pointer, :pointer], :pointer
    attach_function :blazen_agent_free, [:pointer], :void
    attach_function :blazen_agent_new,
                    [:pointer, :pointer, :pointer, :size_t,
                     BlazenToolHandlerVTable.by_value, :uint32,
                     :pointer, :pointer],
                    :int32

    # AgentResult accessors
    attach_function :blazen_agent_result_final_message,   [:pointer], :pointer
    attach_function :blazen_agent_result_iterations,      [:pointer], :uint32
    attach_function :blazen_agent_result_tool_call_count, [:pointer], :uint32
    attach_function :blazen_agent_result_total_usage,     [:pointer], :pointer
    attach_function :blazen_agent_result_total_cost_usd,  [:pointer], :double
    attach_function :blazen_agent_result_free,            [:pointer], :void

    # -------------------------------------------------------------------
    # Batch
    # -------------------------------------------------------------------
    attach_function :blazen_complete_batch_blocking,
                    [:pointer, :pointer, :size_t, :uint32, :pointer, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_complete_batch,
                    [:pointer, :pointer, :size_t, :uint32],
                    :pointer

    # BatchItem accessors
    attach_function :blazen_batch_item_kind,              [:pointer], :uint32
    attach_function :blazen_batch_item_success_response,  [:pointer], :pointer
    attach_function :blazen_batch_item_failure_message,   [:pointer], :pointer
    attach_function :blazen_batch_item_free,              [:pointer], :void

    # BatchResult accessors
    attach_function :blazen_batch_result_responses_count, [:pointer], :size_t
    attach_function :blazen_batch_result_responses_get,   [:pointer, :size_t], :pointer
    attach_function :blazen_batch_result_total_usage,     [:pointer], :pointer
    attach_function :blazen_batch_result_total_cost_usd,  [:pointer], :double
    attach_function :blazen_batch_result_free,            [:pointer], :void

    # -------------------------------------------------------------------
    # Compute — TTS / STT / ImageGen models and results
    # -------------------------------------------------------------------
    attach_function :blazen_tts_model_synthesize_blocking,
                    [:pointer, :pointer, :pointer, :pointer, :pointer, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_tts_model_synthesize,
                    [:pointer, :pointer, :pointer, :pointer],
                    :pointer
    attach_function :blazen_tts_model_free, [:pointer], :void

    attach_function :blazen_stt_model_transcribe_blocking,
                    [:pointer, :pointer, :pointer, :pointer, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_stt_model_transcribe,
                    [:pointer, :pointer, :pointer],
                    :pointer
    attach_function :blazen_stt_model_free, [:pointer], :void

    attach_function :blazen_image_gen_model_generate_blocking,
                    [:pointer, :pointer, :pointer, :int32, :int32, :int32,
                     :pointer, :pointer, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_image_gen_model_generate,
                    [:pointer, :pointer, :pointer, :int32, :int32, :int32, :pointer],
                    :pointer
    attach_function :blazen_image_gen_model_free, [:pointer], :void

    # Compute provider factories
    attach_function :blazen_tts_model_new_fal,
                    [:pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_stt_model_new_fal,
                    [:pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_image_gen_model_new_fal,
                    [:pointer, :pointer, :pointer, :pointer], :int32
    # Feature-gated symbols: tolerate FFI::NotFoundError so the loader
    # works against any cabi build, full-features or minimal. Downstream
    # callers in compute.rb / providers.rb already guard via
    # +Blazen::FFI.respond_to?+ before invoking each.
    begin
      attach_function :blazen_tts_model_new_piper,
                      [:pointer, :int32, :int32, :pointer, :pointer], :int32
    rescue ::FFI::NotFoundError
      # Optional: the local TTS factory was renamed away from "piper" in
      # PR6; this symbol is kept attached only for older release builds
      # that still export it.
    end
    begin
      attach_function :blazen_stt_model_new_whisper,
                      [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    rescue ::FFI::NotFoundError
      # Optional: only present when libblazen_cabi was built with `whispercpp`.
    end
    begin
      attach_function :blazen_tts_model_new_spark,
                      [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    rescue ::FFI::NotFoundError
      # Optional: only present when libblazen_cabi was built with `audio-tts-spark`.
    end
    begin
      attach_function :blazen_stt_model_new_faster_whisper,
                      [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    rescue ::FFI::NotFoundError
      # Optional: only present when libblazen_cabi was built with `audio-stt-faster-whisper`.
    end
    begin
      attach_function :blazen_image_gen_model_new_diffusion,
                      [:pointer, :pointer, :int32, :int32, :int32, :float,
                       :pointer, :pointer],
                      :int32
    rescue ::FFI::NotFoundError
      # Optional: only present when libblazen_cabi was built with `diffusion`.
    end

    # TtsResult accessors
    attach_function :blazen_tts_result_audio_base64, [:pointer], :pointer
    attach_function :blazen_tts_result_mime_type,    [:pointer], :pointer
    attach_function :blazen_tts_result_duration_ms,  [:pointer], :uint64
    attach_function :blazen_tts_result_free,         [:pointer], :void

    # SttResult accessors
    attach_function :blazen_stt_result_transcript,   [:pointer], :pointer
    attach_function :blazen_stt_result_language,     [:pointer], :pointer
    attach_function :blazen_stt_result_duration_ms,  [:pointer], :uint64
    attach_function :blazen_stt_result_free,         [:pointer], :void

    # ImageGenResult accessors
    attach_function :blazen_image_gen_result_images_count, [:pointer], :size_t
    attach_function :blazen_image_gen_result_images_get,   [:pointer, :size_t], :pointer
    attach_function :blazen_image_gen_result_free,         [:pointer], :void

    # -------------------------------------------------------------------
    # Compute — Music / SFX (MusicModel + MusicChunk + MusicResult)
    # -------------------------------------------------------------------
    #
    # The fal factory is always present (no feature gate). The three native
    # factories (+musicgen+ / +stable_audio+ / +audiogen+) are feature-gated
    # in the cabi build; we rescue +::FFI::NotFoundError+ so a minimal
    # libblazen_cabi loads cleanly and downstream callers can probe via
    # +Blazen::FFI.respond_to?+.
    attach_function :blazen_music_model_new_fal,
                    [:pointer, :pointer, :pointer, :pointer], :int32
    begin
      attach_function :blazen_music_model_new_musicgen,
                      [:pointer, :pointer, :pointer, :float, :pointer, :pointer], :int32
    rescue ::FFI::NotFoundError
      # Optional: only present when libblazen_cabi was built with the
      # +music-musicgen+ feature.
    end
    begin
      attach_function :blazen_music_model_new_stable_audio,
                      [:pointer, :pointer, :pointer, :float, :pointer, :pointer], :int32
    rescue ::FFI::NotFoundError
      # Optional: only present when libblazen_cabi was built with the
      # +music-stable-audio+ feature.
    end
    begin
      attach_function :blazen_music_model_new_audiogen,
                      [:pointer, :pointer, :pointer, :pointer, :float, :pointer, :pointer],
                      :int32
    rescue ::FFI::NotFoundError
      # Optional: only present when libblazen_cabi was built with the
      # +music-audiogen+ feature.
    end

    # Lifecycle + non-streaming generate
    attach_function :blazen_music_model_free, [:pointer], :void
    attach_function :blazen_music_model_generate_music_blocking,
                    [:pointer, :pointer, :float, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_music_model_generate_music,
                    [:pointer, :pointer, :float], :pointer
    attach_function :blazen_music_model_generate_sfx_blocking,
                    [:pointer, :pointer, :float, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_music_model_generate_sfx,
                    [:pointer, :pointer, :float], :pointer
    attach_function :blazen_future_take_music_result,
                    [:pointer, :pointer, :pointer], :int32

    # Streaming pumps (4: music + SFX, each in blocking / async flavor).
    # +sink+ is passed by value so the cabi consumes the +user_data+ even
    # on early-return failure paths.
    attach_function :blazen_music_model_stream_generate_music_blocking,
                    [:pointer, :pointer, :float,
                     BlazenMusicStreamSinkVTable.by_value, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_music_model_stream_generate_music,
                    [:pointer, :pointer, :float,
                     BlazenMusicStreamSinkVTable.by_value],
                    :pointer
    attach_function :blazen_music_model_stream_generate_sfx_blocking,
                    [:pointer, :pointer, :float,
                     BlazenMusicStreamSinkVTable.by_value, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_music_model_stream_generate_sfx,
                    [:pointer, :pointer, :float,
                     BlazenMusicStreamSinkVTable.by_value],
                    :pointer

    # MusicChunk accessors (caller-owned chunk handed to +on_chunk+ sink)
    attach_function :blazen_music_chunk_samples,
                    [:pointer, :pointer], :pointer
    attach_function :blazen_music_chunk_is_final,
                    [:pointer], :bool
    attach_function :blazen_music_chunk_latency_seconds,
                    [:pointer], :float
    attach_function :blazen_music_chunk_free,
                    [:pointer], :void

    # MusicResult accessors
    attach_function :blazen_music_result_bytes,
                    [:pointer, :pointer], :pointer
    attach_function :blazen_music_result_mime_type,
                    [:pointer], :pointer
    attach_function :blazen_music_result_sample_rate,
                    [:pointer], :uint32
    attach_function :blazen_music_result_channels,
                    [:pointer], :uint32
    attach_function :blazen_music_result_duration_seconds,
                    [:pointer], :float
    attach_function :blazen_music_result_url,
                    [:pointer], :pointer
    attach_function :blazen_music_result_free,
                    [:pointer], :void

    # -------------------------------------------------------------------
    # Compute — Voice Conversion (RVC) (VcModel + VcChunk + VcResult +
    # TargetVoice + TargetVoiceList)
    # -------------------------------------------------------------------
    #
    # The RVC factory (+blazen_vc_model_new_rvc+) is feature-gated under
    # +audio-vc-rvc+ in the cabi build, so we rescue +FFI::NotFoundError+
    # for it and surface the absence via +Blazen::FFI.respond_to?+ in
    # +compute.rb+. All accessor / lifecycle / streaming / future-taker
    # entry points are always present.
    begin
      attach_function :blazen_vc_model_new_rvc,
                      [:pointer, :pointer, :pointer, :pointer], :int32
    rescue ::FFI::NotFoundError
      # Optional: only present when libblazen_cabi was built with the
      # +audio-vc-rvc+ feature.
    end

    attach_function :blazen_vc_model_free, [:pointer], :void

    attach_function :blazen_vc_model_convert_voice_blocking,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_vc_model_convert_voice,
                    [:pointer, :pointer, :pointer], :pointer

    attach_function :blazen_vc_model_list_target_voices_blocking,
                    [:pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_vc_model_list_target_voices,
                    [:pointer], :pointer

    attach_function :blazen_vc_model_register_target_voice_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_vc_model_register_target_voice,
                    [:pointer, :pointer, :pointer], :pointer

    attach_function :blazen_future_take_vc_result,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_target_voice_list,
                    [:pointer, :pointer, :pointer], :int32

    # Streaming pumps (2: blocking / async). +sink+ is passed by value so
    # the cabi consumes the +user_data+ even on early-return failure
    # paths.
    attach_function :blazen_vc_model_stream_convert_pcm_to_sink_blocking,
                    [:pointer, :pointer, :size_t, :pointer,
                     BlazenVcStreamSinkVTable.by_value, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_vc_model_stream_convert_pcm_to_sink,
                    [:pointer, :pointer, :size_t, :pointer,
                     BlazenVcStreamSinkVTable.by_value],
                    :pointer

    # VcChunk accessors (caller-owned chunk handed to +on_chunk+ sink)
    attach_function :blazen_vc_chunk_samples,         [:pointer, :pointer], :pointer
    attach_function :blazen_vc_chunk_is_final,        [:pointer], :bool
    attach_function :blazen_vc_chunk_latency_seconds, [:pointer], :float
    attach_function :blazen_vc_chunk_free,            [:pointer], :void

    # VcResult accessors
    attach_function :blazen_vc_result_bytes,            [:pointer, :pointer], :pointer
    attach_function :blazen_vc_result_mime_type,        [:pointer], :pointer
    attach_function :blazen_vc_result_sample_rate,      [:pointer], :uint32
    attach_function :blazen_vc_result_duration_seconds, [:pointer], :float
    attach_function :blazen_vc_result_free,             [:pointer], :void

    # TargetVoice accessors
    attach_function :blazen_target_voice_id,             [:pointer], :pointer
    attach_function :blazen_target_voice_label,          [:pointer], :pointer
    attach_function :blazen_target_voice_sample_rate_hz, [:pointer], :uint32
    attach_function :blazen_target_voice_free,           [:pointer], :void

    # TargetVoiceList lifecycle / iteration
    attach_function :blazen_target_voice_list_len,  [:pointer], :size_t
    attach_function :blazen_target_voice_list_get,  [:pointer, :size_t], :pointer
    attach_function :blazen_target_voice_list_take, [:pointer, :size_t], :pointer
    attach_function :blazen_target_voice_list_free, [:pointer], :void

    # -------------------------------------------------------------------
    # LLM — Model / EmbeddingModel
    # -------------------------------------------------------------------
    attach_function :blazen_model_model_id, [:pointer], :pointer
    attach_function :blazen_model_complete_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_model_complete,
                    [:pointer, :pointer], :pointer
    attach_function :blazen_model_free, [:pointer], :void

    attach_function :blazen_embedding_model_model_id,   [:pointer], :pointer
    attach_function :blazen_embedding_model_dimensions, [:pointer], :uint32
    attach_function :blazen_embedding_model_embed_blocking,
                    [:pointer, :pointer, :size_t, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_embedding_model_embed,
                    [:pointer, :pointer, :size_t], :pointer
    attach_function :blazen_embedding_model_free, [:pointer], :void

    # -------------------------------------------------------------------
    # LLM records — Media / ToolCall / Tool / TokenUsage / ChatMessage /
    #               ModelRequest / ModelResponse / EmbeddingResponse
    # -------------------------------------------------------------------
    attach_function :blazen_media_new,
                    [:pointer, :pointer, :pointer], :pointer
    attach_function :blazen_media_kind,        [:pointer], :pointer
    attach_function :blazen_media_mime_type,   [:pointer], :pointer
    attach_function :blazen_media_data_base64, [:pointer], :pointer
    attach_function :blazen_media_free,        [:pointer], :void

    attach_function :blazen_tool_call_new,
                    [:pointer, :pointer, :pointer], :pointer
    attach_function :blazen_tool_call_id,             [:pointer], :pointer
    attach_function :blazen_tool_call_name,           [:pointer], :pointer
    attach_function :blazen_tool_call_arguments_json, [:pointer], :pointer
    attach_function :blazen_tool_call_free,           [:pointer], :void

    attach_function :blazen_tool_new,
                    [:pointer, :pointer, :pointer], :pointer
    attach_function :blazen_tool_name,            [:pointer], :pointer
    attach_function :blazen_tool_description,     [:pointer], :pointer
    attach_function :blazen_tool_parameters_json, [:pointer], :pointer
    attach_function :blazen_tool_free,            [:pointer], :void

    attach_function :blazen_token_usage_new,
                    [:uint64, :uint64, :uint64, :uint64, :uint64], :pointer
    attach_function :blazen_token_usage_prompt_tokens,       [:pointer], :uint64
    attach_function :blazen_token_usage_completion_tokens,   [:pointer], :uint64
    attach_function :blazen_token_usage_total_tokens,        [:pointer], :uint64
    attach_function :blazen_token_usage_cached_input_tokens, [:pointer], :uint64
    attach_function :blazen_token_usage_reasoning_tokens,    [:pointer], :uint64
    attach_function :blazen_token_usage_free,                [:pointer], :void

    attach_function :blazen_chat_message_new,
                    [:pointer, :pointer], :pointer
    attach_function :blazen_chat_message_role,             [:pointer], :pointer
    attach_function :blazen_chat_message_content,          [:pointer], :pointer
    attach_function :blazen_chat_message_media_parts_push, [:pointer, :pointer], :void
    attach_function :blazen_chat_message_media_parts_count, [:pointer], :size_t
    attach_function :blazen_chat_message_media_parts_get,   [:pointer, :size_t], :pointer
    attach_function :blazen_chat_message_tool_calls_push,   [:pointer, :pointer], :void
    attach_function :blazen_chat_message_tool_calls_count,  [:pointer], :size_t
    attach_function :blazen_chat_message_tool_calls_get,    [:pointer, :size_t], :pointer
    attach_function :blazen_chat_message_set_tool_call_id,  [:pointer, :pointer], :void
    attach_function :blazen_chat_message_tool_call_id,      [:pointer], :pointer
    attach_function :blazen_chat_message_set_name,          [:pointer, :pointer], :void
    attach_function :blazen_chat_message_name,              [:pointer], :pointer
    attach_function :blazen_chat_message_free,              [:pointer], :void

    attach_function :blazen_model_request_new, [], :pointer
    attach_function :blazen_model_request_messages_push,
                    [:pointer, :pointer], :void
    attach_function :blazen_model_request_tools_push,
                    [:pointer, :pointer], :void
    attach_function :blazen_model_request_set_temperature,
                    [:pointer, :double], :void
    attach_function :blazen_model_request_clear_temperature,
                    [:pointer], :void
    attach_function :blazen_model_request_set_max_tokens,
                    [:pointer, :uint32], :void
    attach_function :blazen_model_request_clear_max_tokens,
                    [:pointer], :void
    attach_function :blazen_model_request_set_top_p,
                    [:pointer, :double], :void
    attach_function :blazen_model_request_clear_top_p,
                    [:pointer], :void
    attach_function :blazen_model_request_set_model,
                    [:pointer, :pointer], :void
    attach_function :blazen_model_request_set_response_format_json,
                    [:pointer, :pointer], :void
    attach_function :blazen_model_request_set_system,
                    [:pointer, :pointer], :void
    attach_function :blazen_model_request_free, [:pointer], :void

    attach_function :blazen_model_response_content,          [:pointer], :pointer
    attach_function :blazen_model_response_finish_reason,    [:pointer], :pointer
    attach_function :blazen_model_response_model,            [:pointer], :pointer
    attach_function :blazen_model_response_tool_calls_count, [:pointer], :size_t
    attach_function :blazen_model_response_tool_calls_get,
                    [:pointer, :size_t], :pointer
    attach_function :blazen_model_response_usage,            [:pointer], :pointer
    attach_function :blazen_model_response_free,             [:pointer], :void

    attach_function :blazen_embedding_response_embeddings_count, [:pointer], :size_t
    attach_function :blazen_embedding_response_embedding_dim,
                    [:pointer, :size_t], :size_t
    attach_function :blazen_embedding_response_embedding_get,
                    [:pointer, :size_t, :size_t], :double
    attach_function :blazen_embedding_response_embedding_to_buffer,
                    [:pointer, :size_t, :pointer, :size_t], :size_t
    attach_function :blazen_embedding_response_model, [:pointer], :pointer
    attach_function :blazen_embedding_response_usage, [:pointer], :pointer
    attach_function :blazen_embedding_response_free,  [:pointer], :void

    # -------------------------------------------------------------------
    # Typed-result _from_json constructors (sub-wave 3a)
    #
    # These are the entry points the V2 +CustomProvider+ trampolines use to
    # materialise a cabi-side handle from a Ruby Hash. The Ruby override
    # serialises its return value to JSON; the trampoline calls one of these
    # to produce an owned +*mut Blazen<X>+ and writes it through the cabi
    # vtable's +out_response+ slot. On parse / shape failure the cabi side
    # populates +*out_err+ with a fresh +BlazenError+ which the trampoline
    # propagates through to the cabi vtable's +out_err+ slot.
    #
    # +blazen_error_from_json+ is the inverse path: when a Ruby override
    # raises, the trampoline encodes a small JSON sentinel
    # (+{ "kind": "...", "message": "..." }+) and asks the cabi to build a
    # matching +BlazenError+ handle (never null for non-null input — falls
    # back to +Internal+ on malformed JSON).
    #
    # The voice-handle array helpers exist because +list_voices+ returns a
    # variable-length array via +out_array+/+out_count+; the cabi exposes
    # +_array_from_json+ to parse a JSON array, +_array_len+ to query its
    # length, +_array_take(idx)+ to pop entries one-at-a-time (canonical
    # drain pattern is +_take(0)+ until +_len+ returns 0), and +_array_free+
    # to release the remaining array.
    # -------------------------------------------------------------------
    attach_function :blazen_audio_result_from_json,         [:pointer, :pointer], :pointer
    attach_function :blazen_image_result_from_json,         [:pointer, :pointer], :pointer
    attach_function :blazen_video_result_from_json,         [:pointer, :pointer], :pointer
    attach_function :blazen_three_d_result_from_json,       [:pointer, :pointer], :pointer
    attach_function :blazen_transcription_result_from_json, [:pointer, :pointer], :pointer
    attach_function :blazen_voice_handle_from_json,         [:pointer, :pointer], :pointer
    attach_function :blazen_voice_handle_array_from_json,   [:pointer, :pointer], :pointer
    attach_function :blazen_voice_handle_array_len,         [:pointer],           :size_t
    attach_function :blazen_voice_handle_array_take,        [:pointer, :size_t],  :pointer
    attach_function :blazen_voice_handle_array_free,        [:pointer],           :void
    attach_function :blazen_model_response_from_json,  [:pointer, :pointer], :pointer
    attach_function :blazen_embedding_response_from_json,   [:pointer, :pointer], :pointer
    attach_function :blazen_error_from_json,                [:pointer],           :pointer

    # Companion +_free+ functions for the typed result handles. These are
    # the same functions the rest of the cabi already uses internally, but
    # we need them attached here so the Ruby +Blazen::*Result+ finalizers
    # (in +providers/results.rb+) can release handles the user constructs
    # for testing but never hands across the FFI.
    attach_function :blazen_audio_result_free,              [:pointer], :void
    attach_function :blazen_image_result_free,              [:pointer], :void
    attach_function :blazen_video_result_free,              [:pointer], :void
    attach_function :blazen_three_d_result_free,            [:pointer], :void
    attach_function :blazen_transcription_result_free,      [:pointer], :void
    attach_function :blazen_voice_handle_free,              [:pointer], :void

    # -------------------------------------------------------------------
    # Model provider factories
    # -------------------------------------------------------------------
    attach_function :blazen_model_new_openai,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_model_new_anthropic,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_model_new_gemini,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_model_new_openrouter,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_model_new_groq,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_model_new_together,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_model_new_mistral,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_model_new_deepseek,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_model_new_fireworks,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_model_new_perplexity,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_model_new_xai,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_model_new_cohere,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_model_new_azure,
                    [:pointer, :pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_model_new_bedrock,
                    [:pointer, :pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_model_new_fal,
                    [:pointer, :pointer, :pointer, :bool, :bool, :pointer,
                     :pointer, :pointer],
                    :int32
    attach_function :blazen_model_new_openai_compat,
                    [:pointer, :pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_model_new_ollama,
                    [:pointer, :uint16, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_model_new_lm_studio,
                    [:pointer, :uint16, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_model_new_custom_with_openai_protocol,
                    [:pointer, :pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_model_new_mistralrs,
                    [:pointer, :pointer, :pointer, :int32, :bool, :pointer, :pointer],
                    :int32
    attach_function :blazen_model_new_llamacpp,
                    [:pointer, :pointer, :pointer, :int32, :int32, :pointer, :pointer],
                    :int32
    attach_function :blazen_model_new_candle,
                    [:pointer, :pointer, :pointer, :pointer, :int32, :pointer, :pointer],
                    :int32

    # EmbeddingModel provider factories
    attach_function :blazen_embedding_model_new_openai,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_embedding_model_new_fal,
                    [:pointer, :pointer, :int32, :pointer, :pointer], :int32
    attach_function :blazen_embedding_model_new_fastembed,
                    [:pointer, :int32, :bool, :pointer, :pointer], :int32
    attach_function :blazen_embedding_model_new_candle,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_embedding_model_new_tract,
                    [:pointer, :int32, :bool, :pointer, :pointer], :int32

    # -------------------------------------------------------------------
    # ApiProtocol — OpenAI vs custom selector for CustomProvider transport
    # -------------------------------------------------------------------
    attach_function :blazen_api_protocol_openai, [:pointer], :pointer
    attach_function :blazen_api_protocol_custom, [],         :pointer
    attach_function :blazen_api_protocol_kind,   [:pointer], :pointer
    attach_function :blazen_api_protocol_config, [:pointer], :pointer
    attach_function :blazen_api_protocol_free,   [:pointer], :void

    # -------------------------------------------------------------------
    # OpenAiCompatConfig — config bundle for OpenAI-protocol custom providers
    # -------------------------------------------------------------------
    attach_function :blazen_openai_compat_config_new,
                    [:pointer, :pointer, :pointer, :pointer, :uint32, :pointer, :bool],
                    :pointer
    attach_function :blazen_openai_compat_config_push_extra_header,
                    [:pointer, :pointer, :pointer], :void
    attach_function :blazen_openai_compat_config_push_query_param,
                    [:pointer, :pointer, :pointer], :void
    attach_function :blazen_openai_compat_config_provider_name, [:pointer], :pointer
    attach_function :blazen_openai_compat_config_base_url,      [:pointer], :pointer
    attach_function :blazen_openai_compat_config_api_key,       [:pointer], :pointer
    attach_function :blazen_openai_compat_config_default_model, [:pointer], :pointer
    attach_function :blazen_openai_compat_config_auth_code,     [:pointer], :uint32
    attach_function :blazen_openai_compat_config_free,          [:pointer], :void

    # -------------------------------------------------------------------
    # CustomProvider — provider-of-providers handle, optionally backed by a
    # foreign vtable (host-implemented in Ruby) or by one of the cabi's
    # built-in OpenAI-protocol presets (ollama / lm_studio / openai_compat).
    # -------------------------------------------------------------------
    attach_function :blazen_custom_provider_from_vtable,
                    [BlazenCustomProviderVTable.by_value], :pointer
    attach_function :blazen_custom_provider_ollama,
                    [:pointer, :pointer, :uint16], :pointer
    attach_function :blazen_custom_provider_lm_studio,
                    [:pointer, :pointer, :uint16], :pointer
    attach_function :blazen_custom_provider_openai_compat,
                    [:pointer, :pointer], :pointer
    attach_function :blazen_custom_provider_provider_id, [:pointer], :pointer
    attach_function :blazen_custom_provider_model_id,    [:pointer], :pointer
    attach_function :blazen_custom_provider_as_base_provider, [:pointer], :pointer
    attach_function :blazen_custom_provider_free,        [:pointer], :void

    # -------------------------------------------------------------------
    # BaseProviderDefaults — provider-role-agnostic defaults handle
    # -------------------------------------------------------------------
    attach_function :blazen_base_provider_defaults_new, [], :pointer
    attach_function :blazen_base_provider_defaults_has_before_request,
                    [:pointer], :bool
    attach_function :blazen_base_provider_defaults_free, [:pointer], :void

    # -------------------------------------------------------------------
    # ProviderDefaults
    # -------------------------------------------------------------------
    attach_function :blazen_completion_provider_defaults_new, [], :pointer
    attach_function :blazen_completion_provider_defaults_set_system_prompt,
                    [:pointer, :pointer], :void
    attach_function :blazen_completion_provider_defaults_system_prompt,
                    [:pointer], :pointer
    attach_function :blazen_completion_provider_defaults_set_tools_json,
                    [:pointer, :pointer], :void
    attach_function :blazen_completion_provider_defaults_tools_json,
                    [:pointer], :pointer
    attach_function :blazen_completion_provider_defaults_set_response_format_json,
                    [:pointer, :pointer], :void
    attach_function :blazen_completion_provider_defaults_response_format_json,
                    [:pointer], :pointer
    attach_function :blazen_completion_provider_defaults_set_base,
                    [:pointer, :pointer], :void
    attach_function :blazen_completion_provider_defaults_base,
                    [:pointer], :pointer
    attach_function :blazen_provider_defaults_has_before_model,
                    [:pointer], :bool
    attach_function :blazen_completion_provider_defaults_free, [:pointer], :void

    # -------------------------------------------------------------------
    # EmbeddingProviderDefaults
    # -------------------------------------------------------------------
    attach_function :blazen_embedding_provider_defaults_new, [], :pointer
    attach_function :blazen_embedding_provider_defaults_set_base,
                    [:pointer, :pointer], :void
    attach_function :blazen_embedding_provider_defaults_base,
                    [:pointer], :pointer
    attach_function :blazen_embedding_provider_defaults_free, [:pointer], :void

    # -------------------------------------------------------------------
    # AudioSpeechProviderDefaults (TTS)
    # -------------------------------------------------------------------
    attach_function :blazen_audio_speech_provider_defaults_new, [], :pointer
    attach_function :blazen_audio_speech_provider_defaults_set_base,
                    [:pointer, :pointer], :void
    attach_function :blazen_audio_speech_provider_defaults_base,
                    [:pointer], :pointer
    attach_function :blazen_audio_speech_provider_defaults_has_before,
                    [:pointer], :bool
    attach_function :blazen_audio_speech_provider_defaults_free, [:pointer], :void

    # -------------------------------------------------------------------
    # AudioMusicProviderDefaults
    # -------------------------------------------------------------------
    attach_function :blazen_audio_music_provider_defaults_new, [], :pointer
    attach_function :blazen_audio_music_provider_defaults_set_base,
                    [:pointer, :pointer], :void
    attach_function :blazen_audio_music_provider_defaults_base,
                    [:pointer], :pointer
    attach_function :blazen_audio_music_provider_defaults_has_before,
                    [:pointer], :bool
    attach_function :blazen_audio_music_provider_defaults_free, [:pointer], :void

    # -------------------------------------------------------------------
    # VoiceCloningProviderDefaults
    # -------------------------------------------------------------------
    attach_function :blazen_voice_cloning_provider_defaults_new, [], :pointer
    attach_function :blazen_voice_cloning_provider_defaults_set_base,
                    [:pointer, :pointer], :void
    attach_function :blazen_voice_cloning_provider_defaults_base,
                    [:pointer], :pointer
    attach_function :blazen_voice_cloning_provider_defaults_has_before,
                    [:pointer], :bool
    attach_function :blazen_voice_cloning_provider_defaults_free, [:pointer], :void

    # -------------------------------------------------------------------
    # ImageGenerationProviderDefaults
    # -------------------------------------------------------------------
    attach_function :blazen_image_generation_provider_defaults_new, [], :pointer
    attach_function :blazen_image_generation_provider_defaults_set_base,
                    [:pointer, :pointer], :void
    attach_function :blazen_image_generation_provider_defaults_base,
                    [:pointer], :pointer
    attach_function :blazen_image_generation_provider_defaults_has_before,
                    [:pointer], :bool
    attach_function :blazen_image_generation_provider_defaults_free, [:pointer], :void

    # -------------------------------------------------------------------
    # ImageUpscaleProviderDefaults
    # -------------------------------------------------------------------
    attach_function :blazen_image_upscale_provider_defaults_new, [], :pointer
    attach_function :blazen_image_upscale_provider_defaults_set_base,
                    [:pointer, :pointer], :void
    attach_function :blazen_image_upscale_provider_defaults_base,
                    [:pointer], :pointer
    attach_function :blazen_image_upscale_provider_defaults_has_before,
                    [:pointer], :bool
    attach_function :blazen_image_upscale_provider_defaults_free, [:pointer], :void

    # -------------------------------------------------------------------
    # VideoProviderDefaults
    # -------------------------------------------------------------------
    attach_function :blazen_video_provider_defaults_new, [], :pointer
    attach_function :blazen_video_provider_defaults_set_base,
                    [:pointer, :pointer], :void
    attach_function :blazen_video_provider_defaults_base,
                    [:pointer], :pointer
    attach_function :blazen_video_provider_defaults_has_before,
                    [:pointer], :bool
    attach_function :blazen_video_provider_defaults_free, [:pointer], :void

    # -------------------------------------------------------------------
    # TranscriptionProviderDefaults (STT)
    # -------------------------------------------------------------------
    attach_function :blazen_transcription_provider_defaults_new, [], :pointer
    attach_function :blazen_transcription_provider_defaults_set_base,
                    [:pointer, :pointer], :void
    attach_function :blazen_transcription_provider_defaults_base,
                    [:pointer], :pointer
    attach_function :blazen_transcription_provider_defaults_has_before,
                    [:pointer], :bool
    attach_function :blazen_transcription_provider_defaults_free, [:pointer], :void

    # -------------------------------------------------------------------
    # ThreeDProviderDefaults
    # -------------------------------------------------------------------
    attach_function :blazen_three_d_provider_defaults_new, [], :pointer
    attach_function :blazen_three_d_provider_defaults_set_base,
                    [:pointer, :pointer], :void
    attach_function :blazen_three_d_provider_defaults_base,
                    [:pointer], :pointer
    attach_function :blazen_three_d_provider_defaults_has_before,
                    [:pointer], :bool
    attach_function :blazen_three_d_provider_defaults_free, [:pointer], :void

    # -------------------------------------------------------------------
    # BackgroundRemovalProviderDefaults
    # -------------------------------------------------------------------
    attach_function :blazen_background_removal_provider_defaults_new, [], :pointer
    attach_function :blazen_background_removal_provider_defaults_set_base,
                    [:pointer, :pointer], :void
    attach_function :blazen_background_removal_provider_defaults_base,
                    [:pointer], :pointer
    attach_function :blazen_background_removal_provider_defaults_has_before,
                    [:pointer], :bool
    attach_function :blazen_background_removal_provider_defaults_free,
                    [:pointer], :void

    # -------------------------------------------------------------------
    # BaseProvider — completion model with instance-level defaults
    # -------------------------------------------------------------------
    attach_function :blazen_base_provider_with_system_prompt,
                    [:pointer, :pointer], :void
    attach_function :blazen_base_provider_with_tools_json,
                    [:pointer, :pointer], :void
    attach_function :blazen_base_provider_with_response_format_json,
                    [:pointer, :pointer], :void
    attach_function :blazen_base_provider_with_defaults,
                    [:pointer, :pointer], :void
    attach_function :blazen_base_provider_defaults,
                    [:pointer], :pointer
    attach_function :blazen_base_provider_model_id,
                    [:pointer], :pointer
    attach_function :blazen_base_provider_provider_id,
                    [:pointer], :pointer
    attach_function :blazen_base_provider_free, [:pointer], :void

    # -------------------------------------------------------------------
    # Streaming
    # -------------------------------------------------------------------
    attach_function :blazen_complete_streaming_blocking,
                    [:pointer, :pointer,
                     BlazenCompletionStreamSinkVTable.by_value, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_complete_streaming,
                    [:pointer, :pointer,
                     BlazenCompletionStreamSinkVTable.by_value],
                    :pointer
    attach_function :blazen_stream_chunk_new,
                    [:pointer, :bool], :pointer
    attach_function :blazen_stream_chunk_tool_calls_push,
                    [:pointer, :pointer], :void
    attach_function :blazen_stream_chunk_content_delta,   [:pointer], :pointer
    attach_function :blazen_stream_chunk_is_final,        [:pointer], :bool
    attach_function :blazen_stream_chunk_tool_calls_count, [:pointer], :size_t
    attach_function :blazen_stream_chunk_tool_calls_get,
                    [:pointer, :size_t], :pointer
    attach_function :blazen_stream_chunk_free, [:pointer], :void

    # -------------------------------------------------------------------
    # Peer — Server / Client
    # -------------------------------------------------------------------
    attach_function :blazen_peer_server_new, [:pointer], :pointer
    attach_function :blazen_peer_server_serve_blocking,
                    [:pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_peer_server_serve,
                    [:pointer, :pointer], :pointer
    attach_function :blazen_peer_server_free, [:pointer], :void

    attach_function :blazen_peer_client_connect,
                    [:pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_peer_client_node_id, [:pointer], :pointer
    attach_function :blazen_peer_client_run_remote_workflow_blocking,
                    [:pointer, :pointer, :pointer, :size_t, :pointer, :int64,
                     :pointer, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_peer_client_run_remote_workflow,
                    [:pointer, :pointer, :pointer, :size_t, :pointer, :int64],
                    :pointer
    attach_function :blazen_peer_client_free, [:pointer], :void

    # -------------------------------------------------------------------
    # Control plane — Client / Worker / records / vtables
    #
    # Only attached when the native lib was built with the +distributed+
    # feature. Symbol probes happen at attach time; missing symbols raise
    # +FFI::NotFoundError+, which we trap by feature-detecting from the
    # idiomatic wrapper in +lib/blazen/controlplane.rb+ (mirrors the peer
    # surface).
    # -------------------------------------------------------------------

    # AssignmentHandler vtable callbacks
    callback :assignment_handler_drop_user_data, [:pointer], :void
    callback :assignment_handler_handle,
             [:pointer, :pointer, :pointer, :pointer, :pointer, :pointer],
             :int32
    callback :assignment_handler_on_cancel, [:pointer, :pointer], :void
    callback :assignment_handler_on_drain, [:pointer, :bool], :void

    # By-value vtable struct (mirrors +BlazenAssignmentHandlerVTable+ in
    # +blazen.h+).
    class BlazenAssignmentHandlerVTable < ::FFI::Struct
      layout :user_data,       :pointer,
             :drop_user_data,  :assignment_handler_drop_user_data,
             :handle,          :assignment_handler_handle,
             :on_cancel,       :assignment_handler_on_cancel,
             :on_drain,        :assignment_handler_on_drain
    end

    # RunEventSink vtable callbacks — used by the control-plane
    # subscription surface to deliver +RunEvent+ frames into Ruby.
    callback :run_event_sink_drop_user_data, [:pointer], :void
    callback :run_event_sink_on_event,
             [:pointer, :pointer, :pointer, :pointer, :uint64],
             :void
    callback :run_event_sink_on_close, [:pointer], :void
    callback :run_event_sink_on_error, [:pointer, :pointer], :void

    # By-value vtable struct (mirrors +BlazenRunEventSinkVTable+ in
    # +blazen.h+).
    class BlazenRunEventSinkVTable < ::FFI::Struct
      layout :user_data,       :pointer,
             :drop_user_data,  :run_event_sink_drop_user_data,
             :on_event,        :run_event_sink_on_event,
             :on_close,        :run_event_sink_on_close,
             :on_error,        :run_event_sink_on_error
    end

    # Client lifecycle
    attach_function :blazen_controlplane_client_connect_blocking,
                    [:pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_controlplane_client_connect,
                    [:pointer], :pointer
    attach_function :blazen_controlplane_client_connect_with_mtls_blocking,
                    [:pointer, :pointer, :pointer, :pointer, :pointer, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_controlplane_client_connect_with_mtls,
                    [:pointer, :pointer, :pointer, :pointer], :pointer
    attach_function :blazen_controlplane_client_free, [:pointer], :void

    # Workflow lifecycle RPCs
    attach_function :blazen_controlplane_client_submit_workflow_blocking,
                    [:pointer, :pointer, :pointer, :pointer, :bool,
                     :pointer, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_controlplane_client_submit_workflow,
                    [:pointer, :pointer, :pointer, :pointer, :bool], :pointer

    attach_function :blazen_controlplane_client_cancel_workflow_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_controlplane_client_cancel_workflow,
                    [:pointer, :pointer], :pointer

    attach_function :blazen_controlplane_client_describe_workflow_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_controlplane_client_describe_workflow,
                    [:pointer, :pointer], :pointer

    attach_function :blazen_controlplane_client_list_workers_blocking,
                    [:pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_controlplane_client_list_workers,
                    [:pointer], :pointer

    attach_function :blazen_controlplane_client_drain_worker_blocking,
                    [:pointer, :pointer, :bool, :pointer], :int32,
                    blocking: true
    attach_function :blazen_controlplane_client_drain_worker,
                    [:pointer, :pointer, :bool], :pointer

    # Worker lifecycle
    attach_function :blazen_controlplane_worker_new_blocking,
                    [:pointer, :pointer, :pointer, :pointer,
                     :uint32, :uint64,
                     BlazenAssignmentHandlerVTable.by_value,
                     :pointer, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_controlplane_worker_new_with_mtls_blocking,
                    [:pointer, :pointer, :pointer, :pointer,
                     :uint32, :uint64,
                     :pointer, :pointer, :pointer,
                     BlazenAssignmentHandlerVTable.by_value,
                     :pointer, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_controlplane_worker_run_blocking,
                    [:pointer, :pointer], :int32, blocking: true
    attach_function :blazen_controlplane_worker_run,
                    [:pointer], :pointer
    attach_function :blazen_controlplane_worker_shutdown, [:pointer], :void
    attach_function :blazen_controlplane_worker_free, [:pointer], :void

    # Subscription lifecycle
    attach_function :blazen_controlplane_client_subscribe_run_events,
                    [:pointer, :pointer,
                     BlazenRunEventSinkVTable.by_value,
                     :pointer, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_controlplane_client_subscribe_all,
                    [:pointer, :pointer,
                     BlazenRunEventSinkVTable.by_value,
                     :pointer, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_controlplane_subscription_cancel,
                    [:pointer], :void
    attach_function :blazen_controlplane_subscription_free,
                    [:pointer], :void

    # ModelClient — lightweight client for the model-serving control plane
    # (the `blazen-controlplane` gRPC `BlazenModelServer`). Only the
    # blocking surface is exposed today; future waves will add the async
    # future-returning variants and richer telemetry RPCs.
    attach_function :blazen_modelclient_connect_blocking,
                    [:pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_modelclient_connect_with_tls_blocking,
                    [:pointer, :pointer, :pointer, :pointer,
                     :pointer, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_modelclient_status_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_modelclient_is_loaded_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_modelclient_load_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_modelclient_unload_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_modelclient_load_from_hf_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_modelclient_load_adapter_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_modelclient_unload_adapter_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_modelclient_list_adapters_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_modelclient_complete_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_modelclient_embed_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_modelclient_generate_image_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_modelclient_text_to_speech_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_modelclient_generate_music_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_modelclient_transcribe_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_modelclient_stream_complete_blocking,
                    [:pointer, :pointer,
                     BlazenCompletionStreamSinkVTable.by_value,
                     :pointer, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_modelclient_upload_blob_blocking,
                    [:pointer, :pointer, :pointer, :pointer, :size_t,
                     :pointer, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_modelclient_fetch_blob_blocking,
                    [:pointer, :pointer, :pointer, :pointer, :pointer],
                    :int32,
                    blocking: true
    attach_function :blazen_modelclient_bytes_free, [:pointer, :size_t], :void
    attach_function :blazen_modelclient_free, [:pointer], :void

    # Typed future takers
    attach_function :blazen_future_take_controlplane_client,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_run_state_snapshot,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_worker_info_list,
                    [:pointer, :pointer, :pointer], :int32

    # RunStateSnapshot accessors
    attach_function :blazen_run_state_snapshot_run_id,        [:pointer], :pointer
    attach_function :blazen_run_state_snapshot_status,        [:pointer], :uint32
    attach_function :blazen_run_state_snapshot_started_at_ms, [:pointer], :uint64
    attach_function :blazen_run_state_snapshot_completed_at_ms,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_run_state_snapshot_assigned_to,   [:pointer], :pointer
    attach_function :blazen_run_state_snapshot_last_event_at_ms,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_run_state_snapshot_output_json,   [:pointer], :pointer
    attach_function :blazen_run_state_snapshot_error,         [:pointer], :pointer
    attach_function :blazen_run_state_snapshot_free,          [:pointer], :void

    # WorkerInfo accessors / list
    attach_function :blazen_worker_info_node_id,           [:pointer], :pointer
    attach_function :blazen_worker_info_capabilities_json, [:pointer], :pointer
    attach_function :blazen_worker_info_tags_json,         [:pointer], :pointer
    attach_function :blazen_worker_info_admission_json,    [:pointer], :pointer
    attach_function :blazen_worker_info_in_flight,         [:pointer], :uint32
    attach_function :blazen_worker_info_connected_at_ms,   [:pointer], :uint64
    attach_function :blazen_worker_info_free,              [:pointer], :void

    attach_function :blazen_worker_info_list_count, [:pointer], :size_t
    attach_function :blazen_worker_info_list_get,
                    [:pointer, :size_t], :pointer
    attach_function :blazen_worker_info_list_free,  [:pointer], :void

    # -------------------------------------------------------------------
    # Persist — CheckpointStore / WorkflowCheckpoint / PersistedEvent
    # -------------------------------------------------------------------
    attach_function :blazen_checkpoint_store_save_blocking,
                    [:pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_checkpoint_store_save,
                    [:pointer, :pointer], :pointer
    attach_function :blazen_checkpoint_store_load_blocking,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_checkpoint_store_load,
                    [:pointer, :pointer], :pointer
    attach_function :blazen_checkpoint_store_delete_blocking,
                    [:pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_checkpoint_store_delete,
                    [:pointer, :pointer], :pointer
    attach_function :blazen_checkpoint_store_list_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_checkpoint_store_list,
                    [:pointer], :pointer
    attach_function :blazen_checkpoint_store_list_run_ids_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_checkpoint_store_list_run_ids,
                    [:pointer], :pointer
    attach_function :blazen_checkpoint_store_free, [:pointer], :void

    attach_function :blazen_workflow_checkpoint_array_free,
                    [:pointer, :size_t], :void

    attach_function :blazen_checkpoint_store_new_redb,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_checkpoint_store_new_valkey,
                    [:pointer, :int64, :pointer, :pointer], :int32

    attach_function :blazen_persisted_event_new,
                    [:pointer, :pointer], :pointer
    attach_function :blazen_persisted_event_event_type, [:pointer], :pointer
    attach_function :blazen_persisted_event_data_json,  [:pointer], :pointer
    attach_function :blazen_persisted_event_free,       [:pointer], :void

    attach_function :blazen_workflow_checkpoint_new,
                    [:pointer, :pointer, :pointer, :pointer, :uint64], :pointer
    attach_function :blazen_workflow_checkpoint_pending_events_push,
                    [:pointer, :pointer], :int32
    attach_function :blazen_workflow_checkpoint_workflow_name,
                    [:pointer], :pointer
    attach_function :blazen_workflow_checkpoint_run_id,        [:pointer], :pointer
    attach_function :blazen_workflow_checkpoint_timestamp_ms,  [:pointer], :uint64
    attach_function :blazen_workflow_checkpoint_state_json,    [:pointer], :pointer
    attach_function :blazen_workflow_checkpoint_metadata_json, [:pointer], :pointer
    attach_function :blazen_workflow_checkpoint_pending_events_count,
                    [:pointer], :size_t
    attach_function :blazen_workflow_checkpoint_pending_events_get,
                    [:pointer, :size_t], :pointer
    attach_function :blazen_workflow_checkpoint_free, [:pointer], :void

    # -------------------------------------------------------------------
    # Pipeline
    # -------------------------------------------------------------------
    attach_function :blazen_pipeline_builder_new, [:pointer], :pointer
    attach_function :blazen_pipeline_builder_add_workflow,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_pipeline_builder_stage,
                    [:pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_pipeline_builder_parallel,
                    [:pointer, :pointer, :pointer, :size_t,
                     :pointer, :size_t, :bool, :pointer],
                    :int32
    attach_function :blazen_pipeline_builder_timeout_per_stage_ms,
                    [:pointer, :uint64, :pointer], :int32
    attach_function :blazen_pipeline_builder_total_timeout_ms,
                    [:pointer, :uint64, :pointer], :int32
    attach_function :blazen_pipeline_builder_build,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_pipeline_builder_free, [:pointer], :void

    attach_function :blazen_pipeline_run_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_pipeline_run,
                    [:pointer, :pointer], :pointer
    attach_function :blazen_pipeline_stage_names_count, [:pointer], :size_t
    attach_function :blazen_pipeline_stage_names_get,
                    [:pointer, :size_t], :pointer
    attach_function :blazen_pipeline_free, [:pointer], :void

    # -------------------------------------------------------------------
    # Telemetry
    # -------------------------------------------------------------------
    attach_function :blazen_init_langfuse,
                    [:pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_init_otlp,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_init_prometheus,
                    [:pointer, :pointer], :int32

    attach_function :blazen_parse_workflow_history,
                    [:pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_workflow_history_entry_array_free,
                    [:pointer, :size_t], :void
    attach_function :blazen_workflow_history_entry_workflow_id,
                    [:pointer], :pointer
    attach_function :blazen_workflow_history_entry_step_name,
                    [:pointer], :pointer
    attach_function :blazen_workflow_history_entry_event_type,
                    [:pointer], :pointer
    attach_function :blazen_workflow_history_entry_event_data_json,
                    [:pointer], :pointer
    attach_function :blazen_workflow_history_entry_timestamp_ms,
                    [:pointer], :uint64
    attach_function :blazen_workflow_history_entry_duration_ms,
                    [:pointer], :int64
    attach_function :blazen_workflow_history_entry_error,
                    [:pointer], :pointer
    attach_function :blazen_workflow_history_entry_free,
                    [:pointer], :void

    # -------------------------------------------------------------------
    # Workflow — Builder / Workflow / WorkflowResult / Event / StepOutput
    # -------------------------------------------------------------------
    attach_function :blazen_workflow_builder_new, [:pointer], :pointer
    attach_function :blazen_workflow_builder_add_step,
                    [:pointer, :pointer, :pointer, :size_t,
                     :pointer, :size_t,
                     BlazenStepHandlerVTable.by_value, :pointer],
                    :int32
    attach_function :blazen_workflow_builder_step_timeout_ms,
                    [:pointer, :uint64, :pointer], :int32
    attach_function :blazen_workflow_builder_timeout_ms,
                    [:pointer, :uint64, :pointer], :int32
    attach_function :blazen_workflow_builder_build,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_workflow_builder_free, [:pointer], :void

    attach_function :blazen_workflow_run_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_workflow_run,
                    [:pointer, :pointer], :pointer
    attach_function :blazen_workflow_step_names_count, [:pointer], :size_t
    attach_function :blazen_workflow_step_names_get,
                    [:pointer, :size_t], :pointer
    attach_function :blazen_workflow_free, [:pointer], :void

    attach_function :blazen_event_new,        [:pointer, :pointer], :pointer
    attach_function :blazen_event_event_type, [:pointer], :pointer
    attach_function :blazen_event_data_json,  [:pointer], :pointer
    attach_function :blazen_event_free,       [:pointer], :void

    attach_function :blazen_workflow_result_event,                [:pointer], :pointer
    attach_function :blazen_workflow_result_total_input_tokens,   [:pointer], :uint64
    attach_function :blazen_workflow_result_total_output_tokens,  [:pointer], :uint64
    attach_function :blazen_workflow_result_total_cost_usd,       [:pointer], :double
    attach_function :blazen_workflow_result_free,                 [:pointer], :void

    attach_function :blazen_step_output_new_none,      [],          :pointer
    attach_function :blazen_step_output_new_single,    [:pointer],  :pointer
    attach_function :blazen_step_output_new_multiple,  [],          :pointer
    attach_function :blazen_step_output_multiple_push, [:pointer, :pointer], :uint32
    attach_function :blazen_step_output_kind,          [:pointer],  :uint32
    attach_function :blazen_step_output_single_event,  [:pointer],  :pointer
    attach_function :blazen_step_output_multiple_count, [:pointer], :size_t
    attach_function :blazen_step_output_multiple_get,
                    [:pointer, :size_t], :pointer
    attach_function :blazen_step_output_free, [:pointer], :void

    # -------------------------------------------------------------------
    # ModelManager — typed future-take entry points (mirror cabi/future.rs)
    # -------------------------------------------------------------------
    attach_function :blazen_future_take_bool,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_model_status_list,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_adapter_handle_info,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_adapter_status_list,
                    [:pointer, :pointer, :pointer], :int32

    # -------------------------------------------------------------------
    # ModelManager — manager construction / verbs (cabi/manager.rs)
    # -------------------------------------------------------------------
    attach_function :blazen_model_manager_new,             [], :pointer
    attach_function :blazen_model_manager_with_budgets_gb, [:double, :double], :pointer
    attach_function :blazen_model_manager_free,            [:pointer], :void

    attach_function :blazen_model_manager_load_blocking,
                    [:pointer, :pointer, :pointer], :int32, blocking: true
    attach_function :blazen_model_manager_load,
                    [:pointer, :pointer], :pointer

    attach_function :blazen_model_manager_unload_blocking,
                    [:pointer, :pointer, :pointer], :int32, blocking: true
    attach_function :blazen_model_manager_unload,
                    [:pointer, :pointer], :pointer

    attach_function :blazen_model_manager_is_loaded_blocking,
                    [:pointer, :pointer, :pointer], :int32, blocking: true
    attach_function :blazen_model_manager_is_loaded,
                    [:pointer, :pointer], :pointer

    attach_function :blazen_model_manager_status_blocking,
                    [:pointer, :pointer], :pointer, blocking: true
    attach_function :blazen_model_manager_status,
                    [:pointer], :pointer

    attach_function :blazen_model_manager_pools,
                    [:pointer], :pointer

    attach_function :blazen_model_manager_load_adapter_blocking,
                    [:pointer, :pointer, :pointer, :pointer, :double, :pointer],
                    :pointer, blocking: true
    attach_function :blazen_model_manager_load_adapter,
                    [:pointer, :pointer, :pointer, :pointer, :double], :pointer

    attach_function :blazen_model_manager_unload_adapter_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32, blocking: true
    attach_function :blazen_model_manager_unload_adapter,
                    [:pointer, :pointer, :pointer], :pointer

    attach_function :blazen_model_manager_list_adapters_blocking,
                    [:pointer, :pointer, :pointer], :pointer, blocking: true
    attach_function :blazen_model_manager_list_adapters,
                    [:pointer, :pointer], :pointer

    BLAZEN_BACKEND_HINT_MISTRALRS = 0
    BLAZEN_BACKEND_HINT_CANDLE    = 1
    BLAZEN_BACKEND_HINT_LLAMACPP  = 2
    BLAZEN_BACKEND_HINT_NONE      = -1

    # Mirror of `BlazenHfLoadOptions` in blazen.h (cabi/manager.rs). All
    # pointer fields are nullable; `memory_estimate_bytes == 0` means unset.
    class BlazenHfLoadOptions < ::FFI::Struct
      layout :backend_hint,           :int32,
             :revision,               :pointer,
             :hf_token,               :pointer,
             :cache_dir,              :pointer,
             :device,                 :pointer,
             :gguf_file,              :pointer,
             :memory_estimate_bytes,  :uint64,
             :pool,                   :pointer
    end

    attach_function :blazen_model_manager_load_from_hf_blocking,
                    [:pointer, :pointer, :pointer, :pointer, :pointer, :pointer],
                    :int32, blocking: true
    attach_function :blazen_model_manager_load_from_hf,
                    [:pointer, :pointer, :pointer, :pointer], :pointer

    # -------------------------------------------------------------------
    # ModelManager — record accessors (cabi/manager_records.rs)
    # -------------------------------------------------------------------
    attach_function :blazen_adapter_status_adapter_id,     [:pointer], :pointer
    attach_function :blazen_adapter_status_source_dir,     [:pointer], :pointer
    attach_function :blazen_adapter_status_scale,          [:pointer], :double
    attach_function :blazen_adapter_status_memory_bytes,   [:pointer], :uint64
    attach_function :blazen_adapter_status_mount_strategy, [:pointer], :uint32
    attach_function :blazen_adapter_status_free,           [:pointer], :void

    attach_function :blazen_adapter_status_list_len,  [:pointer], :size_t
    attach_function :blazen_adapter_status_list_get,  [:pointer, :size_t], :pointer
    attach_function :blazen_adapter_status_list_take, [:pointer, :size_t], :pointer
    attach_function :blazen_adapter_status_list_free, [:pointer], :void

    attach_function :blazen_model_status_id,              [:pointer], :pointer
    attach_function :blazen_model_status_loaded,          [:pointer], :bool
    attach_function :blazen_model_status_memory_bytes,    [:pointer], :uint64
    attach_function :blazen_model_status_pool,            [:pointer], :pointer
    attach_function :blazen_model_status_adapters_count,  [:pointer], :size_t
    attach_function :blazen_model_status_adapters,        [:pointer], :pointer
    attach_function :blazen_model_status_free,            [:pointer], :void

    attach_function :blazen_model_status_list_len,  [:pointer], :size_t
    attach_function :blazen_model_status_list_get,  [:pointer, :size_t], :pointer
    attach_function :blazen_model_status_list_take, [:pointer, :size_t], :pointer
    attach_function :blazen_model_status_list_free, [:pointer], :void

    attach_function :blazen_pool_status_id,             [:pointer], :pointer
    attach_function :blazen_pool_status_budget_bytes,   [:pointer], :uint64
    attach_function :blazen_pool_status_used_bytes,     [:pointer], :uint64
    attach_function :blazen_pool_status_loaded_models,  [:pointer], :size_t
    attach_function :blazen_pool_status_free,           [:pointer], :void

    attach_function :blazen_pool_status_list_len,  [:pointer], :size_t
    attach_function :blazen_pool_status_list_get,  [:pointer, :size_t], :pointer
    attach_function :blazen_pool_status_list_take, [:pointer, :size_t], :pointer
    attach_function :blazen_pool_status_list_free, [:pointer], :void

    attach_function :blazen_adapter_handle_info_adapter_id,     [:pointer], :pointer
    attach_function :blazen_adapter_handle_info_memory_bytes,   [:pointer], :uint64
    attach_function :blazen_adapter_handle_info_mount_strategy, [:pointer], :uint32
    attach_function :blazen_adapter_handle_info_free,           [:pointer], :void

    ADAPTER_MOUNT_STRATEGY_ATTACHED = 1
    ADAPTER_MOUNT_STRATEGY_REBUILT  = 2
    ADAPTER_MOUNT_STRATEGY_MERGED   = 3

    # -------------------------------------------------------------------
    # ModelManager — LoRA training (cabi/manager.rs + training_records.rs)
    # -------------------------------------------------------------------

    BLAZEN_SCHEDULER_CONSTANT = 0
    BLAZEN_SCHEDULER_LINEAR   = 1
    BLAZEN_SCHEDULER_COSINE   = 2

    BLAZEN_MIXED_PRECISION_NONE = 0
    BLAZEN_MIXED_PRECISION_BF16 = 1

    # Mirror of `BlazenLoraConfig` in blazen.h.
    class BlazenLoraConfig < ::FFI::Struct
      layout :rank,               :uint32,
             :alpha,              :float,
             :dropout,            :float,
             :target_modules,     :pointer,
             :target_modules_len, :size_t
    end

    # Mirror of `BlazenOptimConfig` in blazen.h.
    class BlazenOptimConfig < ::FFI::Struct
      layout :learning_rate,     :double,
             :beta1,             :double,
             :beta2,             :double,
             :epsilon,           :double,
             :weight_decay,      :double,
             :has_gradient_clip, :int32,
             :gradient_clip,     :float
    end

    # Mirror of `BlazenSchedulerConfig` in blazen.h.
    class BlazenSchedulerConfig < ::FFI::Struct
      layout :kind,         :int32,
             :warmup_steps, :uint32
    end

    # Mirror of `BlazenTrainConfig` in blazen.h.
    class BlazenTrainConfig < ::FFI::Struct
      layout :base_model_repo,             :pointer,
             :output_dir,                  :pointer,
             :lora,                        BlazenLoraConfig,
             :optim,                       BlazenOptimConfig,
             :scheduler,                   BlazenSchedulerConfig,
             :max_steps,                   :uint32,
             :batch_size,                  :uint32,
             :gradient_accumulation_steps, :uint32,
             :max_seq_len,                 :uint32,
             :has_eval_steps,              :int32,
             :eval_steps,                  :uint32,
             :has_save_steps,              :int32,
             :save_steps,                  :uint32,
             :seed,                        :uint64,
             :mixed_precision,             :int32,
             :device,                      :pointer
    end

    # Mirror of `BlazenTrainedAdapter` in blazen.h. Inner `adapter_dir` is a
    # caller-owned heap C string freed by `blazen_trained_adapter_free`.
    class BlazenTrainedAdapter < ::FFI::Struct
      layout :adapter_dir, :pointer,
             :final_loss, :float,
             :total_steps, :uint64
    end

    attach_function :blazen_jsonl_dataset_from_path,
                    [:pointer, :pointer, :pointer, :uint32, :pointer, :uint32, :pointer],
                    :pointer
    attach_function :blazen_jsonl_dataset_free, [:pointer], :void

    attach_function :blazen_model_manager_train_lora_blocking,
                    [:pointer, :pointer, :pointer, :pointer, :pointer],
                    :int32, blocking: true
    attach_function :blazen_model_manager_train_lora,
                    [:pointer, :pointer, :pointer], :pointer

    attach_function :blazen_trained_adapter_free, [:pointer], :void

    # -------------------------------------------------------------------
    # PR8 — preference / KTO / full-fine-tune configs and entry points
    # -------------------------------------------------------------------

    # Mirror of `BlazenTrainCoreConfig` in blazen.h. Shared across
    # `BlazenDpoConfig`, `BlazenOrpoConfig`, `BlazenSimpoConfig`,
    # `BlazenKtoConfig`, and `BlazenFullFineTuneConfig` as their `core` slot.
    class BlazenTrainCoreConfig < ::FFI::Struct
      layout :base_model_repo,             :pointer,
             :base_model_revision,         :pointer,
             :output_dir,                  :pointer,
             :optim,                       BlazenOptimConfig,
             :scheduler,                   BlazenSchedulerConfig,
             :max_steps,                   :uint32,
             :batch_size,                  :uint32,
             :gradient_accumulation_steps, :uint32,
             :max_seq_len,                 :uint32,
             :has_eval_steps,              :int32,
             :eval_steps,                  :uint32,
             :has_save_steps,              :int32,
             :save_steps,                  :uint32,
             :seed,                        :uint64,
             :mixed_precision,             :int32,
             :device,                      :pointer
    end

    # Mirror of `BlazenDpoConfig` in blazen.h.
    class BlazenDpoConfig < ::FFI::Struct
      layout :core,                       BlazenTrainCoreConfig,
             :lora,                       BlazenLoraConfig,
             :beta,                       :float,
             :label_smoothing,            :float,
             :reference_model_repo,       :pointer,
             :reference_model_revision,   :pointer
    end

    # Mirror of `BlazenOrpoConfig` in blazen.h. Reference-free.
    class BlazenOrpoConfig < ::FFI::Struct
      layout :core,   BlazenTrainCoreConfig,
             :lora,   BlazenLoraConfig,
             :lambda, :float
    end

    # Mirror of `BlazenSimpoConfig` in blazen.h. Reference-free.
    class BlazenSimpoConfig < ::FFI::Struct
      layout :core,  BlazenTrainCoreConfig,
             :lora,  BlazenLoraConfig,
             :beta,  :float,
             :gamma, :float
    end

    # Mirror of `BlazenKtoConfig` in blazen.h.
    class BlazenKtoConfig < ::FFI::Struct
      layout :core,                       BlazenTrainCoreConfig,
             :lora,                       BlazenLoraConfig,
             :beta,                       :float,
             :lambda_d,                   :float,
             :lambda_u,                   :float,
             :reference_model_repo,       :pointer,
             :reference_model_revision,   :pointer
    end

    # Mirror of `BlazenFullFineTuneConfig` in blazen.h.
    class BlazenFullFineTuneConfig < ::FFI::Struct
      layout :core,                   BlazenTrainCoreConfig,
             :gradient_checkpointing, :int32
    end

    # Mirror of `BlazenFullFineTuneResult` in blazen.h. Inner `output_dir`
    # is a caller-owned heap C string freed by
    # `blazen_full_finetune_result_free`.
    class BlazenFullFineTuneResult < ::FFI::Struct
      layout :output_dir,      :pointer,
             :final_loss,      :float,
             :steps_completed, :uint64
    end

    attach_function :blazen_preference_jsonl_dataset_from_path,
                    [:pointer, :pointer, :pointer, :uint32, :pointer, :uint32, :pointer],
                    :pointer
    attach_function :blazen_preference_jsonl_dataset_free, [:pointer], :void

    attach_function :blazen_rated_jsonl_dataset_from_path,
                    [:pointer, :pointer, :pointer, :uint32, :pointer, :uint32, :pointer],
                    :pointer
    attach_function :blazen_rated_jsonl_dataset_free, [:pointer], :void

    attach_function :blazen_model_manager_train_dpo_blocking,
                    [:pointer, :pointer, :pointer, :pointer, :pointer],
                    :int32, blocking: true
    attach_function :blazen_model_manager_train_dpo,
                    [:pointer, :pointer, :pointer], :pointer

    attach_function :blazen_model_manager_train_orpo_blocking,
                    [:pointer, :pointer, :pointer, :pointer, :pointer],
                    :int32, blocking: true
    attach_function :blazen_model_manager_train_orpo,
                    [:pointer, :pointer, :pointer], :pointer

    attach_function :blazen_model_manager_train_simpo_blocking,
                    [:pointer, :pointer, :pointer, :pointer, :pointer],
                    :int32, blocking: true
    attach_function :blazen_model_manager_train_simpo,
                    [:pointer, :pointer, :pointer], :pointer

    attach_function :blazen_model_manager_train_kto_blocking,
                    [:pointer, :pointer, :pointer, :pointer, :pointer],
                    :int32, blocking: true
    attach_function :blazen_model_manager_train_kto,
                    [:pointer, :pointer, :pointer], :pointer

    attach_function :blazen_model_manager_fine_tune_blocking,
                    [:pointer, :pointer, :pointer, :pointer, :pointer],
                    :int32, blocking: true
    attach_function :blazen_model_manager_fine_tune,
                    [:pointer, :pointer, :pointer], :pointer

    attach_function :blazen_future_take_full_finetune_result,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_full_finetune_result_free, [:pointer], :void

    # DistributedConfig — ring-AllReduce config for multi-GPU /
    # multi-node training. Constructed via
    # {blazen_distributed_config_new}, freed via
    # {blazen_distributed_config_free}. Passed to training verbs that
    # accept distributed configuration.
    attach_function :blazen_distributed_config_new,
                    %i[uint64 uint64 pointer pointer uint16],
                    :pointer
    attach_function :blazen_distributed_config_free, [:pointer], :void

    # -------------------------------------------------------------------
    # 3D pipeline (Compat3dProvider — HTTP proxy backend)
    #
    # Built only when libblazen_cabi was compiled with the
    # +threed-compat-proxy+ feature. The Ruby wrapper in
    # +lib/blazen/threed.rb+ probes these symbols and falls back to a
    # +UnsupportedError+ raiser when they're not present.
    # -------------------------------------------------------------------

    # PBR-channel discriminants (mirror BLAZEN_PBR_MAP_* in blazen.h)
    PBR_MAP_ALBEDO    = 0
    PBR_MAP_NORMAL    = 1
    PBR_MAP_ROUGHNESS = 2
    PBR_MAP_METALLIC  = 3

    %i[
      blazen_compat3d_provider_new
      blazen_compat3d_provider_free
      blazen_compat3d_result_free
      blazen_compat3d_result_glb_bytes
      blazen_compat3d_result_mime_type
      blazen_compat3d_result_has_pbr_maps
      blazen_compat3d_result_pbr_map_bytes
      blazen_compat3d_result_bone_names_count
      blazen_compat3d_result_bone_name_get
      blazen_compat3d_result_refine_input_tri_count
      blazen_compat3d_result_refine_output_tri_count
      blazen_compat3d_result_refine_uv_chart_count
      blazen_compat3d_result_animate_duration_seconds
      blazen_compat3d_result_animate_fps
      blazen_compat3d_texturize_blocking
      blazen_compat3d_texturize
      blazen_compat3d_rig_blocking
      blazen_compat3d_rig
      blazen_compat3d_refine_blocking
      blazen_compat3d_refine
      blazen_compat3d_animate_blocking
      blazen_compat3d_animate
      blazen_future_take_compat3d_result
    ].each do |sym|
      begin
        case sym
        when :blazen_compat3d_provider_new
          attach_function sym, %i[pointer pointer uint32], :pointer
        when :blazen_compat3d_provider_free
          attach_function sym, [:pointer], :void
        when :blazen_compat3d_result_free
          attach_function sym, [:pointer], :void
        when :blazen_compat3d_result_glb_bytes
          attach_function sym, %i[pointer pointer pointer], :int32
        when :blazen_compat3d_result_mime_type
          attach_function sym, [:pointer], :pointer
        when :blazen_compat3d_result_has_pbr_maps
          attach_function sym, [:pointer], :int32
        when :blazen_compat3d_result_pbr_map_bytes
          attach_function sym, %i[pointer uint32 pointer pointer], :int32
        when :blazen_compat3d_result_bone_names_count
          attach_function sym, [:pointer], :size_t
        when :blazen_compat3d_result_bone_name_get
          attach_function sym, %i[pointer size_t], :pointer
        when :blazen_compat3d_result_refine_input_tri_count,
             :blazen_compat3d_result_refine_output_tri_count
          attach_function sym, [:pointer], :uint32
        when :blazen_compat3d_result_refine_uv_chart_count
          attach_function sym, [:pointer], :int64
        when :blazen_compat3d_result_animate_duration_seconds
          attach_function sym, [:pointer], :float
        when :blazen_compat3d_result_animate_fps
          attach_function sym, [:pointer], :uint32
        when :blazen_compat3d_texturize_blocking
          attach_function sym,
                          %i[pointer pointer size_t pointer size_t pointer pointer pointer],
                          :int32, blocking: true
        when :blazen_compat3d_texturize
          attach_function sym,
                          %i[pointer pointer size_t pointer size_t pointer],
                          :pointer
        when :blazen_compat3d_rig_blocking
          attach_function sym,
                          %i[pointer pointer size_t pointer pointer pointer],
                          :int32, blocking: true
        when :blazen_compat3d_rig
          attach_function sym,
                          %i[pointer pointer size_t pointer],
                          :pointer
        when :blazen_compat3d_refine_blocking
          attach_function sym,
                          %i[pointer pointer size_t pointer pointer pointer],
                          :int32, blocking: true
        when :blazen_compat3d_refine
          attach_function sym,
                          %i[pointer pointer size_t pointer],
                          :pointer
        when :blazen_compat3d_animate_blocking
          attach_function sym,
                          %i[pointer pointer size_t pointer size_t pointer size_t pointer pointer pointer],
                          :int32, blocking: true
        when :blazen_compat3d_animate
          attach_function sym,
                          %i[pointer pointer size_t pointer size_t pointer size_t pointer],
                          :pointer
        when :blazen_future_take_compat3d_result
          attach_function sym, %i[pointer pointer pointer], :int32
        end
      rescue ::FFI::NotFoundError
        # Library built without the threed-compat-proxy feature; the Ruby
        # wrapper detects via +Blazen::FFI.respond_to?+ and raises
        # +UnsupportedError+ at call-site.
      end
    end

    # Returns +true+ when the native lib was built with the
    # +threed-compat-proxy+ feature (i.e. every Compat3d symbol attached
    # successfully).
    def self.threed_available?
      respond_to?(:blazen_compat3d_provider_new) &&
        respond_to?(:blazen_compat3d_texturize_blocking)
    end

    # -------------------------------------------------------------------
    # Ruby-side helpers
    # -------------------------------------------------------------------

    # Reads a heap-allocated C string returned across the FFI, then frees
    # the C side via {blazen_string_free}. Returns +nil+ on a null pointer.
    #
    # @param ptr [::FFI::Pointer, nil] pointer to a NUL-terminated UTF-8 buffer
    # @return [String, nil] the Ruby string, or +nil+ when +ptr+ was null
    def self.consume_cstring(ptr)
      return nil if ptr.nil? || ptr.null?

      s = ptr.read_string
      blazen_string_free(ptr)
      s.force_encoding(Encoding::UTF_8)
    end

    # Allocates a NUL-terminated UTF-8 buffer for +rb_str+ via
    # {::FFI::MemoryPointer.from_string} and yields it. The buffer is freed
    # automatically when the yielded MemoryPointer is GC'd; the +ensure+
    # arm here keeps a strong reference alive for the duration of the
    # native call, then releases the local before returning so GC can
    # reclaim the buffer promptly.
    #
    # Passing +nil+ for +rb_str+ yields +nil+ to the block — this is the
    # cabi convention for optional/nullable C-string parameters.
    #
    # @param rb_str [String, nil] the input string (or nil for nullable args)
    # @yield [::FFI::MemoryPointer, nil] the C-side pointer
    # @return [Object] whatever the block returns
    def self.with_cstring(rb_str)
      return yield(nil) if rb_str.nil?

      mp = ::FFI::MemoryPointer.from_string(rb_str.to_s)
      begin
        yield mp
      ensure
        # Keep mp alive until after the call returns — Ruby's GC won't
        # reclaim it while we hold a reference here.
        mp = nil # rubocop:disable Lint/UselessAssignment
      end
    end

    # If the writable out-error pointer slot +err_ptr_slot+ holds a non-null
    # {BlazenError} pointer after a fallible cabi call, decode it into a
    # Ruby {Blazen::Error} subclass (via {Blazen.build_error_from_ptr}) and
    # raise. The underlying {BlazenError} is freed by the decoder.
    #
    # @param err_ptr_slot [::FFI::MemoryPointer, nil] a +::FFI::MemoryPointer.new(:pointer)+
    #   that was passed to the cabi call as the +BlazenError **out_err+ slot
    # @raise [Blazen::Error] when an error pointer is present
    # @return [void]
    def self.check_error!(err_ptr_slot)
      return if err_ptr_slot.nil? || err_ptr_slot.null?

      ptr = err_ptr_slot.read_pointer
      return if ptr.nil? || ptr.null?

      raise Blazen.build_error_from_ptr(ptr)
    end

    # Awaits a {BlazenFuture}. Uses {Fiber.scheduler} when one is active
    # (composes with the +async+ gem and other reactor-style schedulers) by
    # registering the future's eventfd-style file descriptor for readability;
    # otherwise falls back to the cabi's thread-blocking
    # {blazen_future_wait}.
    #
    # After the future resolves, yields it to the caller's block (which is
    # expected to invoke the matching +blazen_future_take_<X>+ and return the
    # result). The future is always freed via {blazen_future_free} in an
    # +ensure+ arm — callers must NOT free it themselves.
    #
    # @param fut_ptr [::FFI::Pointer] a live BlazenFuture pointer
    # @yield [::FFI::Pointer] the same +fut_ptr+, now resolved
    # @return [Object] whatever the block returns
    # @raise [ArgumentError] when +fut_ptr+ is null
    def self.await_future(fut_ptr)
      raise ArgumentError, "fut_ptr is null" if fut_ptr.nil? || fut_ptr.null?

      begin
        fd = blazen_future_fd(fut_ptr)
        scheduler = Fiber.respond_to?(:scheduler) ? Fiber.scheduler : nil
        if fd >= 0 && scheduler
          # Yield to the fiber scheduler — composes with the `async` gem.
          # autoclose: false so Ruby doesn't close a Tokio-owned fd on GC.
          io = IO.for_fd(fd.to_i, autoclose: false)
          io.wait_readable
        else
          blazen_future_wait(fut_ptr)
        end
        yield(fut_ptr)
      ensure
        blazen_future_free(fut_ptr)
      end
    end
  end

  # Decodes an opaque {BlazenError} pointer into the appropriate
  # {Blazen::Error} subclass and frees the C side. Phase R7's
  # +lib/blazen/errors.rb+ rewrite will replace this with a richer
  # implementation that uses every error accessor (provider/status/etc.);
  # this default implementation reads kind + message so the FFI boundary
  # remains usable on its own.
  #
  # @param ptr [::FFI::Pointer] a caller-owned BlazenError pointer
  # @return [Blazen::Error] the matching idiomatic Ruby exception
  def self.build_error_from_ptr(ptr)
    kind = FFI.blazen_error_kind(ptr)
    msg  = FFI.consume_cstring(FFI.blazen_error_message(ptr)) || ""
    FFI.blazen_error_free(ptr)

    klass =
      case kind
      when FFI::ERROR_KIND_AUTH           then defined?(AuthError)           ? AuthError           : Error
      when FFI::ERROR_KIND_RATE_LIMIT     then defined?(RateLimitError)      ? RateLimitError      : Error
      when FFI::ERROR_KIND_TIMEOUT        then defined?(TimeoutError)        ? TimeoutError        : Error
      when FFI::ERROR_KIND_VALIDATION     then defined?(ValidationError)     ? ValidationError     : Error
      when FFI::ERROR_KIND_CONTENT_POLICY then defined?(ContentPolicyError)  ? ContentPolicyError  : Error
      when FFI::ERROR_KIND_UNSUPPORTED    then defined?(UnsupportedError)    ? UnsupportedError    : Error
      when FFI::ERROR_KIND_COMPUTE        then defined?(ComputeError)        ? ComputeError        : Error
      when FFI::ERROR_KIND_MEDIA          then defined?(MediaError)          ? MediaError          : Error
      when FFI::ERROR_KIND_PROVIDER       then defined?(ProviderError)       ? ProviderError       : Error
      when FFI::ERROR_KIND_WORKFLOW       then defined?(WorkflowError)       ? WorkflowError       : Error
      when FFI::ERROR_KIND_TOOL           then defined?(ToolError)           ? ToolError           : Error
      when FFI::ERROR_KIND_PEER           then defined?(PeerError)           ? PeerError           : Error
      when FFI::ERROR_KIND_PERSIST        then defined?(PersistError)        ? PersistError        : Error
      when FFI::ERROR_KIND_PROMPT         then defined?(PromptError)         ? PromptError         : Error
      when FFI::ERROR_KIND_MEMORY         then defined?(MemoryError)         ? MemoryError         : Error
      when FFI::ERROR_KIND_CACHE          then defined?(CacheError)          ? CacheError          : Error
      when FFI::ERROR_KIND_CANCELLED      then defined?(CancelledError)      ? CancelledError      : Error
      when FFI::ERROR_KIND_INTERNAL       then defined?(InternalError)       ? InternalError       : Error
      else
        defined?(Error) ? Error : StandardError
      end

    # Subclasses with required keyword args (kind:, etc.) take a single
    # positional message in the default ctor we provide below; the R7
    # rewrite swaps in proper structured constructors with the additional
    # accessor data populated.
    begin
      klass.new(msg)
    rescue ArgumentError
      # Fall back to a plain Blazen::Error if the subclass demands kwargs
      # we don't yet know how to supply (RateLimitError, ProviderError, …).
      (defined?(Error) ? Error : StandardError).new("[kind=#{kind}] #{msg}")
    end
  end
end
