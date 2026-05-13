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

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------
    attach_function :blazen_version,           [],          :pointer
    attach_function :blazen_init,              [],          :int32
    attach_function :blazen_shutdown,          [],          :int32
    attach_function :blazen_shutdown_telemetry, [],         :void
    attach_function :blazen_string_free,       [:pointer],  :void
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
    attach_function :blazen_future_take_completion_response,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_embedding_response,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_workflow_result,
                    [:pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_unit,
                    [:pointer, :pointer], :int32
    attach_function :blazen_future_take_workflow_checkpoint_option,
                    [:pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_workflow_checkpoint_list,
                    [:pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_future_take_string_list,
                    [:pointer, :pointer, :pointer, :pointer], :int32

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
    attach_function :blazen_tts_model_new_piper,
                    [:pointer, :int32, :int32, :pointer, :pointer], :int32
    attach_function :blazen_stt_model_new_whisper,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_image_gen_model_new_diffusion,
                    [:pointer, :pointer, :int32, :int32, :int32, :float,
                     :pointer, :pointer],
                    :int32

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
    # LLM — CompletionModel / EmbeddingModel
    # -------------------------------------------------------------------
    attach_function :blazen_completion_model_model_id, [:pointer], :pointer
    attach_function :blazen_completion_model_complete_blocking,
                    [:pointer, :pointer, :pointer, :pointer], :int32,
                    blocking: true
    attach_function :blazen_completion_model_complete,
                    [:pointer, :pointer], :pointer
    attach_function :blazen_completion_model_free, [:pointer], :void

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
    #               CompletionRequest / CompletionResponse / EmbeddingResponse
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

    attach_function :blazen_completion_request_new, [], :pointer
    attach_function :blazen_completion_request_messages_push,
                    [:pointer, :pointer], :void
    attach_function :blazen_completion_request_tools_push,
                    [:pointer, :pointer], :void
    attach_function :blazen_completion_request_set_temperature,
                    [:pointer, :double], :void
    attach_function :blazen_completion_request_clear_temperature,
                    [:pointer], :void
    attach_function :blazen_completion_request_set_max_tokens,
                    [:pointer, :uint32], :void
    attach_function :blazen_completion_request_clear_max_tokens,
                    [:pointer], :void
    attach_function :blazen_completion_request_set_top_p,
                    [:pointer, :double], :void
    attach_function :blazen_completion_request_clear_top_p,
                    [:pointer], :void
    attach_function :blazen_completion_request_set_model,
                    [:pointer, :pointer], :void
    attach_function :blazen_completion_request_set_response_format_json,
                    [:pointer, :pointer], :void
    attach_function :blazen_completion_request_set_system,
                    [:pointer, :pointer], :void
    attach_function :blazen_completion_request_free, [:pointer], :void

    attach_function :blazen_completion_response_content,          [:pointer], :pointer
    attach_function :blazen_completion_response_finish_reason,    [:pointer], :pointer
    attach_function :blazen_completion_response_model,            [:pointer], :pointer
    attach_function :blazen_completion_response_tool_calls_count, [:pointer], :size_t
    attach_function :blazen_completion_response_tool_calls_get,
                    [:pointer, :size_t], :pointer
    attach_function :blazen_completion_response_usage,            [:pointer], :pointer
    attach_function :blazen_completion_response_free,             [:pointer], :void

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
    # CompletionModel provider factories
    # -------------------------------------------------------------------
    attach_function :blazen_completion_model_new_openai,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_completion_model_new_anthropic,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_completion_model_new_gemini,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_completion_model_new_openrouter,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_completion_model_new_groq,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_completion_model_new_together,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_completion_model_new_mistral,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_completion_model_new_deepseek,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_completion_model_new_fireworks,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_completion_model_new_perplexity,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_completion_model_new_xai,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_completion_model_new_cohere,
                    [:pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_completion_model_new_azure,
                    [:pointer, :pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_completion_model_new_bedrock,
                    [:pointer, :pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_completion_model_new_fal,
                    [:pointer, :pointer, :pointer, :bool, :bool, :pointer,
                     :pointer, :pointer],
                    :int32
    attach_function :blazen_completion_model_new_openai_compat,
                    [:pointer, :pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_completion_model_new_ollama,
                    [:pointer, :uint16, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_completion_model_new_lm_studio,
                    [:pointer, :uint16, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_completion_model_new_custom_with_openai_protocol,
                    [:pointer, :pointer, :pointer, :pointer, :pointer, :pointer], :int32
    attach_function :blazen_completion_model_new_mistralrs,
                    [:pointer, :pointer, :pointer, :int32, :bool, :pointer, :pointer],
                    :int32
    attach_function :blazen_completion_model_new_llamacpp,
                    [:pointer, :pointer, :pointer, :int32, :int32, :pointer, :pointer],
                    :int32
    attach_function :blazen_completion_model_new_candle,
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
