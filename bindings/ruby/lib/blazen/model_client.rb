# frozen_string_literal: true

require "json"

module Blazen
  # Ruby handle for the +blazen-controlplane+ model-serving gRPC client
  # (`BlazenModelServer`). Use this to query the status of, and probe
  # whether the server has loaded, a model.
  #
  # The native surface is exposed by the +blazen-cabi+ crate via the
  # +blazen_modelclient_*+ family of C entry points; this class wraps
  # them with cstring marshalling, ownership management via
  # {::FFI::AutoPointer}, and JSON decoding.
  #
  # Only the blocking surface is exposed in this wave; later waves add
  # the async future-returning variants alongside richer telemetry
  # RPCs (load / unload / metrics).
  #
  # @example
  #   client = Blazen::ModelClient.connect("http://localhost:7446")
  #   client.loaded?("llama-3.1-8b")        # => true
  #   client.status                          # => { "models" => [ ... ] }
  #   client.status(model_id: "llama-3.1-8b")
  #
  # @example mTLS
  #   client = Blazen::ModelClient.connect_with_tls(
  #     "https://models.example.com:7446",
  #     ca_cert: File.read("ca.pem"),
  #     client_cert: File.read("client.pem"),
  #     client_key: File.read("client.key"),
  #   )
  class ModelClient
    # @return [::FFI::AutoPointer] underlying +BlazenModelClient+ pointer
    attr_reader :ptr

    # Opens a connection to a +BlazenModelServer+ on a blocking thread.
    #
    # @param endpoint [String] gRPC endpoint URI such as
    #   +"http://localhost:7446"+ or +"https://models.example.com"+
    # @return [ModelClient]
    # @raise [Blazen::Error] when the endpoint URI is invalid or the
    #   TCP/HTTP-2 handshake fails
    def self.connect(endpoint)
      out_client = ::FFI::MemoryPointer.new(:pointer)
      out_err    = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(endpoint.to_s) do |ep|
        Blazen::FFI.blazen_modelclient_connect_blocking(ep, out_client, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      new(out_client.read_pointer)
    end

    # Opens a TLS / mTLS connection to a +BlazenModelServer+ on a
    # blocking thread. The PEM material is passed inline as Ruby
    # strings (not file paths).
    #
    # @param endpoint [String] gRPC endpoint URI; should use the
    #   +https://+ scheme
    # @param ca_cert [String] PEM-encoded CA bundle to validate the
    #   server's certificate against
    # @param client_cert [String, nil] PEM-encoded client cert
    #   (mTLS only — omit for server-auth-only TLS)
    # @param client_key [String, nil] PEM-encoded client private key
    #   (mTLS only — omit for server-auth-only TLS)
    # @return [ModelClient]
    # @raise [Blazen::Error] when any PEM is unparseable, the TLS
    #   config is rejected, or the handshake fails
    def self.connect_with_tls(endpoint, ca_cert:, client_cert: nil, client_key: nil)
      out_client = ::FFI::MemoryPointer.new(:pointer)
      out_err    = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(endpoint.to_s) do |ep|
        Blazen::FFI.with_cstring(ca_cert.to_s) do |ca|
          Blazen::FFI.with_cstring(client_cert) do |cc|
            Blazen::FFI.with_cstring(client_key) do |ck|
              Blazen::FFI.blazen_modelclient_connect_with_tls_blocking(
                ep, ca, cc, ck, out_client, out_err,
              )
            end
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      new(out_client.read_pointer)
    end

    # @param raw_ptr [::FFI::Pointer] caller-owned +BlazenModelClient*+
    def initialize(raw_ptr)
      raise ArgumentError, "ModelClient: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

      @ptr = ::FFI::AutoPointer.new(
        raw_ptr, Blazen::FFI.method(:blazen_modelclient_free),
      )
    end

    # Fetch the status of one model, or of every model the server
    # knows about. The cabi returns a JSON document; this method
    # parses it into a Ruby +Hash+.
    #
    # @param model_id [String, nil] when +nil+ (default), request the
    #   aggregate status across all models; otherwise restrict to the
    #   named model
    # @return [Hash] parsed JSON status payload
    # @raise [Blazen::Error] on RPC failure (server unreachable,
    #   model not found, etc.)
    def status(model_id: nil)
      out_json = ::FFI::MemoryPointer.new(:pointer)
      out_err  = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(model_id) do |mid|
        Blazen::FFI.blazen_modelclient_status_blocking(@ptr, mid, out_json, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      json = Blazen::FFI.consume_cstring(out_json.read_pointer)
      json.nil? ? {} : JSON.parse(json)
    end

    # Probe whether the server has the named model loaded and ready to
    # serve requests.
    #
    # @param model_id [String] the model identifier (must not be nil)
    # @return [Boolean]
    # @raise [ArgumentError] when +model_id+ is nil
    # @raise [Blazen::Error] on RPC failure
    def loaded?(model_id)
      raise ArgumentError, "loaded?: model_id must not be nil" if model_id.nil?

      out_loaded = ::FFI::MemoryPointer.new(:bool)
      out_err    = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(model_id.to_s) do |mid|
        Blazen::FFI.blazen_modelclient_is_loaded_blocking(@ptr, mid, out_loaded, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      out_loaded.read(:bool)
    end

    # Load a model on the server. The +request+ Hash mirrors the JSON
    # body of the +LoadModel+ RPC; the response (parsed Hash) carries
    # whatever per-model load metadata the server emits.
    #
    # @param request [Hash] load request payload (serialised via JSON)
    # @return [Hash] parsed JSON response payload
    # @raise [Blazen::Error] on RPC failure
    def load(request)
      out_json = ::FFI::MemoryPointer.new(:pointer)
      out_err  = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(JSON.generate(request)) do |req|
        Blazen::FFI.blazen_modelclient_load_blocking(@ptr, req, out_json, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      json = Blazen::FFI.consume_cstring(out_json.read_pointer)
      json.nil? ? {} : JSON.parse(json)
    end

    # Unload a model from the server. The +request+ Hash mirrors the
    # JSON body of the +UnloadModel+ RPC.
    #
    # @param request [Hash] unload request payload (serialised via JSON)
    # @return [Hash] parsed JSON response payload
    # @raise [Blazen::Error] on RPC failure
    def unload(request)
      out_json = ::FFI::MemoryPointer.new(:pointer)
      out_err  = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(JSON.generate(request)) do |req|
        Blazen::FFI.blazen_modelclient_unload_blocking(@ptr, req, out_json, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      json = Blazen::FFI.consume_cstring(out_json.read_pointer)
      json.nil? ? {} : JSON.parse(json)
    end

    # Ask the server to fetch a model from the Hugging Face Hub and
    # load it. The +request+ Hash mirrors the JSON body of the
    # +LoadFromHuggingFace+ RPC (repo id, revision, files, etc.).
    #
    # @param request [Hash] load-from-HF request payload (serialised via JSON)
    # @return [Hash] parsed JSON response payload
    # @raise [Blazen::Error] on RPC failure
    def load_from_hf(request)
      out_json = ::FFI::MemoryPointer.new(:pointer)
      out_err  = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(JSON.generate(request)) do |req|
        Blazen::FFI.blazen_modelclient_load_from_hf_blocking(@ptr, req, out_json, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      json = Blazen::FFI.consume_cstring(out_json.read_pointer)
      json.nil? ? {} : JSON.parse(json)
    end

    # Load a LoRA / PEFT adapter onto a base model on the server. The
    # +request+ Hash mirrors the JSON body of the +LoadAdapter+ RPC.
    #
    # @param request [Hash] load-adapter request payload (serialised via JSON)
    # @return [Hash] parsed JSON response payload
    # @raise [Blazen::Error] on RPC failure
    def load_adapter(request)
      out_json = ::FFI::MemoryPointer.new(:pointer)
      out_err  = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(JSON.generate(request)) do |req|
        Blazen::FFI.blazen_modelclient_load_adapter_blocking(@ptr, req, out_json, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      json = Blazen::FFI.consume_cstring(out_json.read_pointer)
      json.nil? ? {} : JSON.parse(json)
    end

    # Unload a previously loaded adapter. The +request+ Hash mirrors the
    # JSON body of the +UnloadAdapter+ RPC.
    #
    # @param request [Hash] unload-adapter request payload (serialised via JSON)
    # @return [Hash] parsed JSON response payload
    # @raise [Blazen::Error] on RPC failure
    def unload_adapter(request)
      out_json = ::FFI::MemoryPointer.new(:pointer)
      out_err  = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(JSON.generate(request)) do |req|
        Blazen::FFI.blazen_modelclient_unload_adapter_blocking(@ptr, req, out_json, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      json = Blazen::FFI.consume_cstring(out_json.read_pointer)
      json.nil? ? {} : JSON.parse(json)
    end

    # List adapters currently loaded on the server. The +request+ Hash
    # mirrors the JSON body of the +ListAdapters+ RPC (filters, etc.).
    #
    # @param request [Hash] list-adapters request payload (serialised via JSON)
    # @return [Hash] parsed JSON response payload
    # @raise [Blazen::Error] on RPC failure
    def list_adapters(request)
      out_json = ::FFI::MemoryPointer.new(:pointer)
      out_err  = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(JSON.generate(request)) do |req|
        Blazen::FFI.blazen_modelclient_list_adapters_blocking(@ptr, req, out_json, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      json = Blazen::FFI.consume_cstring(out_json.read_pointer)
      json.nil? ? {} : JSON.parse(json)
    end

    # Issue a chat / text-completion request against a loaded model. The
    # +request+ Hash mirrors the JSON body of the +Complete+ RPC.
    #
    # @param request [Hash] completion request payload (serialised via JSON)
    # @return [Hash] parsed JSON response payload
    # @raise [Blazen::Error] on RPC failure
    def complete(request)
      out_json = ::FFI::MemoryPointer.new(:pointer)
      out_err  = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(JSON.generate(request)) do |req|
        Blazen::FFI.blazen_modelclient_complete_blocking(@ptr, req, out_json, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      json = Blazen::FFI.consume_cstring(out_json.read_pointer)
      json.nil? ? {} : JSON.parse(json)
    end

    # Embed one or more inputs against a loaded embedding model. The
    # +request+ Hash mirrors the JSON body of the +Embed+ RPC.
    #
    # @param request [Hash] embed request payload (serialised via JSON)
    # @return [Hash] parsed JSON response payload
    # @raise [Blazen::Error] on RPC failure
    def embed(request)
      out_json = ::FFI::MemoryPointer.new(:pointer)
      out_err  = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(JSON.generate(request)) do |req|
        Blazen::FFI.blazen_modelclient_embed_blocking(@ptr, req, out_json, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      json = Blazen::FFI.consume_cstring(out_json.read_pointer)
      json.nil? ? {} : JSON.parse(json)
    end

    # Generate one or more images from a prompt. The +request+ Hash
    # mirrors the JSON body of the +GenerateImage+ RPC.
    #
    # @param request [Hash] image-generation request payload (serialised via JSON)
    # @return [Hash] parsed JSON response payload
    # @raise [Blazen::Error] on RPC failure
    def generate_image(request)
      out_json = ::FFI::MemoryPointer.new(:pointer)
      out_err  = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(JSON.generate(request)) do |req|
        Blazen::FFI.blazen_modelclient_generate_image_blocking(@ptr, req, out_json, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      json = Blazen::FFI.consume_cstring(out_json.read_pointer)
      json.nil? ? {} : JSON.parse(json)
    end

    # Synthesise speech audio from text. The +request+ Hash mirrors the
    # JSON body of the +TextToSpeech+ RPC.
    #
    # @param request [Hash] TTS request payload (serialised via JSON)
    # @return [Hash] parsed JSON response payload
    # @raise [Blazen::Error] on RPC failure
    def text_to_speech(request)
      out_json = ::FFI::MemoryPointer.new(:pointer)
      out_err  = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(JSON.generate(request)) do |req|
        Blazen::FFI.blazen_modelclient_text_to_speech_blocking(@ptr, req, out_json, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      json = Blazen::FFI.consume_cstring(out_json.read_pointer)
      json.nil? ? {} : JSON.parse(json)
    end

    # Generate a music clip from a prompt. The +request+ Hash mirrors the
    # JSON body of the +GenerateMusic+ RPC.
    #
    # @param request [Hash] music-generation request payload (serialised via JSON)
    # @return [Hash] parsed JSON response payload
    # @raise [Blazen::Error] on RPC failure
    def generate_music(request)
      out_json = ::FFI::MemoryPointer.new(:pointer)
      out_err  = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(JSON.generate(request)) do |req|
        Blazen::FFI.blazen_modelclient_generate_music_blocking(@ptr, req, out_json, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      json = Blazen::FFI.consume_cstring(out_json.read_pointer)
      json.nil? ? {} : JSON.parse(json)
    end

    # Transcribe an audio clip into text. The +request+ Hash mirrors the
    # JSON body of the +Transcribe+ RPC.
    #
    # @param request [Hash] transcription request payload (serialised via JSON)
    # @return [Hash] parsed JSON response payload
    # @raise [Blazen::Error] on RPC failure
    def transcribe(request)
      out_json = ::FFI::MemoryPointer.new(:pointer)
      out_err  = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(JSON.generate(request)) do |req|
        Blazen::FFI.blazen_modelclient_transcribe_blocking(@ptr, req, out_json, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      json = Blazen::FFI.consume_cstring(out_json.read_pointer)
      json.nil? ? {} : JSON.parse(json)
    end

    # Drive a streaming completion against a loaded model. The +request+ Hash
    # mirrors the JSON body of the +StreamComplete+ RPC; the block is
    # invoked once per streamed chunk with the chunk's +content_delta+ as
    # a +String+. The call blocks until the server signals end-of-stream
    # or an error.
    #
    # The streaming sink is plumbed through the same
    # +BlazenCompletionStreamSinkVTable+ trampoline used by
    # {Blazen::Streaming.complete} — see +streaming.rb+ for the sink
    # registry / vtable wiring. This wrapper hides the +on_done+ /
    # +on_error+ events behind the blocking return contract: errors are
    # raised as {Blazen::Error} subclasses (via +check_error!+), and the
    # method simply returns +nil+ when the stream terminates normally.
    #
    # @example
    #   client.stream_complete(request) do |chunk_text|
    #     print chunk_text
    #   end
    #
    # @param request [Hash] streaming-completion request payload
    #   (serialised via JSON)
    # @yield [chunk_text] called once per streamed chunk
    # @yieldparam chunk_text [String] the chunk's incremental content delta
    # @return [nil]
    # @raise [ArgumentError] when no block is supplied
    # @raise [Blazen::Error] on RPC failure or mid-stream error
    def stream_complete(request, &block)
      raise ArgumentError, "stream_complete requires a block" unless block

      handlers = {
        on_chunk: ->(chunk) { block.call(chunk.content_delta.to_s) },
        on_done:  nil,
        on_error: nil,
      }
      sink_id = Blazen::Streaming.register_sink(handlers)
      registered = true

      begin
        vtable = Blazen::FFI::BlazenCompletionStreamSinkVTable.new
        vtable[:user_data]      = ::FFI::Pointer.new(sink_id)
        vtable[:drop_user_data] = Blazen::Streaming::DROP_SINK_USER_DATA_FN
        vtable[:on_chunk]       = Blazen::Streaming::ON_CHUNK_FN
        vtable[:on_done]        = Blazen::Streaming::ON_DONE_FN
        vtable[:on_error]       = Blazen::Streaming::ON_ERROR_FN

        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(JSON.generate(request)) do |req|
          Blazen::FFI.blazen_modelclient_stream_complete_blocking(
            @ptr, req, vtable, ::FFI::Pointer::NULL, out_err,
          )
        end

        # The cabi has now consumed the vtable's +user_data+ and will
        # invoke +drop_user_data+ when the inner sink drops, so the
        # registry entry is no longer ours to release.
        registered = false
        Blazen::FFI.check_error!(out_err)
        nil
      ensure
        Blazen::Streaming.unregister_sink(sink_id) if registered
      end
    end

    # Upload a blob to the server in one shot. The bytes are sent inline
    # as part of the +UploadBlob+ RPC body; for very large payloads the
    # caller is expected to chunk by uploading multiple blobs under a
    # shared content-addressed prefix and stitching them server-side.
    #
    # @param blob_id [String] caller-chosen identifier for the blob
    # @param mime [String] MIME type of the payload (e.g. +"image/png"+)
    # @param data [String, Array<Integer>] raw bytes; either a +String+
    #   (any encoding — only the byte sequence is read) or an +Array+ of
    #   integers in the 0..255 range
    # @return [Hash] parsed JSON ack payload
    # @raise [Blazen::Error] on RPC failure
    def upload_blob(blob_id:, mime:, data:)
      bytes = data.is_a?(String) ? data.b : data.pack("C*")
      length = bytes.bytesize
      buf = ::FFI::MemoryPointer.new(:uint8, length)
      buf.write_bytes(bytes, 0, length) if length.positive?

      out_ack = ::FFI::MemoryPointer.new(:pointer)
      out_err = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(blob_id.to_s) do |bid|
        Blazen::FFI.with_cstring(mime.to_s) do |m|
          Blazen::FFI.blazen_modelclient_upload_blob_blocking(
            @ptr, bid, m, buf, length, out_ack, out_err,
          )
        end
      end
      Blazen::FFI.check_error!(out_err)
      json = Blazen::FFI.consume_cstring(out_ack.read_pointer)
      json.nil? ? {} : JSON.parse(json)
    end

    # Fetch a blob from the server in one shot. The +request+ Hash
    # mirrors the JSON body of the +FetchBlob+ RPC (typically just
    # +{ "blob_id" => "..." }+, possibly with a digest assertion).
    #
    # @param request [Hash] fetch request payload (serialised via JSON)
    # @return [String] raw blob bytes, with +Encoding::BINARY+
    # @raise [Blazen::Error] on RPC failure
    def fetch_blob(request)
      out_data = ::FFI::MemoryPointer.new(:pointer)
      out_len  = ::FFI::MemoryPointer.new(:size_t)
      out_err  = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(JSON.generate(request)) do |req|
        Blazen::FFI.blazen_modelclient_fetch_blob_blocking(
          @ptr, req, out_data, out_len, out_err,
        )
      end
      Blazen::FFI.check_error!(out_err)
      data_ptr = out_data.read_pointer
      length   = out_len.read(:size_t)
      return String.new(encoding: Encoding::BINARY) if data_ptr.nil? || data_ptr.null? || length.zero?

      begin
        data_ptr.read_bytes(length).force_encoding(Encoding::BINARY)
      ensure
        Blazen::FFI.blazen_modelclient_bytes_free(data_ptr, length)
      end
    end
  end
end
