# frozen_string_literal: true

module Blazen
  # Idiomatic Ruby exception hierarchy mirroring Blazen's Rust {BlazenError}
  # enum (see +crates/blazen-uniffi/src/errors.rs+ and
  # +crates/blazen-cabi/src/error.rs+).
  #
  # Under the hand-written {Blazen::FFI} loader, the cabi surface returns an
  # opaque +BlazenError *+ pointer with structured accessors (kind, message,
  # retry-after, elapsed, HTTP status, provider, endpoint, request-id, detail,
  # subkind). {Blazen.build_error_from_ptr} consumes that pointer and returns
  # the matching subclass below — no more string parsing of a flat message.
  #
  # All accessors free the C-string buffers they read via
  # {Blazen::FFI.consume_cstring}, and the error itself is freed in the
  # +ensure+ arm of {Blazen.build_error_from_ptr}.

  # Base class for every error raised by the Blazen gem. Catch this if you
  # want a single rescue clause for all native failures.
  class Error < StandardError; end

  # Authentication or credentials failure (missing API key, invalid token,
  # 401 from a provider).
  class AuthError < Error; end

  # Rate limit hit at the provider. +retry_after_ms+ carries the provider's
  # +Retry-After+ hint when one was returned; otherwise +nil+.
  class RateLimitError < Error
    # @return [Integer, nil] retry-after delay in milliseconds, if hinted
    attr_reader :retry_after_ms

    # @param message [String] human-readable error message
    # @param retry_after_ms [Integer, nil] retry-after delay in milliseconds
    def initialize(message, retry_after_ms: nil)
      super(message)
      @retry_after_ms = retry_after_ms
    end
  end

  # Operation timed out before the provider responded.
  class TimeoutError < Error
    # @return [Integer] elapsed time before timeout, in milliseconds
    attr_reader :elapsed_ms

    # @param message [String] human-readable error message
    # @param elapsed_ms [Integer] elapsed time before timeout, in milliseconds
    def initialize(message, elapsed_ms: 0)
      super(message)
      @elapsed_ms = elapsed_ms
    end
  end

  # Input validation failure (bad schema, missing required field, etc.).
  class ValidationError < Error; end

  # Provider refused due to content-policy / safety filters.
  class ContentPolicyError < Error; end

  # Operation unsupported on the current platform, build, or provider.
  class UnsupportedError < Error; end

  # Compute backend error (CPU / GPU / accelerator failure, OOM).
  class ComputeError < Error; end

  # Media handling error (decode, encode, transcoding).
  class MediaError < Error; end

  # Provider / backend error.
  #
  # +kind+ identifies the specific backend and failure mode (mirrors the
  # Node binding's +[ProviderError]+ sentinel JSON shape). Examples:
  # +"OpenAIHttp"+, +"AnthropicHttp"+, +"LlamaCppModelLoad"+,
  # +"DiffusionGeneration"+, +"CandleEmbedInference"+.
  class ProviderError < Error
    # @return [String, nil] backend / failure-mode discriminator
    attr_reader :kind
    # @return [String, nil] provider name (when known)
    attr_reader :provider
    # @return [Integer, nil] HTTP status code (when applicable)
    attr_reader :status
    # @return [String, nil] endpoint that failed
    attr_reader :endpoint
    # @return [String, nil] provider request ID for log correlation
    attr_reader :request_id
    # @return [String, nil] provider-specific error detail / body excerpt
    attr_reader :detail
    # @return [Integer, nil] retry-after delay in milliseconds, if hinted
    attr_reader :retry_after_ms

    # @param message [String] human-readable error message
    # @param kind [String, nil] backend / failure-mode discriminator
    # @param provider [String, nil] provider name
    # @param status [Integer, nil] HTTP status code
    # @param endpoint [String, nil] endpoint that failed
    # @param request_id [String, nil] provider request ID
    # @param detail [String, nil] provider-specific error detail
    # @param retry_after_ms [Integer, nil] retry-after delay in milliseconds
    def initialize(message, kind: nil, provider: nil, status: nil, endpoint: nil,
                   request_id: nil, detail: nil, retry_after_ms: nil)
      super(message)
      @kind = kind
      @provider = provider
      @status = status
      @endpoint = endpoint
      @request_id = request_id
      @detail = detail
      @retry_after_ms = retry_after_ms
    end
  end

  # Workflow execution error (step panic, deadlock, missing context, etc.).
  class WorkflowError < Error; end

  # Tool / function-call error during agent execution.
  class ToolError < Error; end

  # Distributed peer-to-peer error.
  #
  # +kind+ is one of +"Encode"+, +"Transport"+, +"EnvelopeVersion"+,
  # +"Workflow"+, +"Tls"+, +"UnknownStep"+ (or +nil+ if unset).
  class PeerError < Error
    # @return [String, nil] sub-category discriminator
    attr_reader :kind

    # @param message [String] human-readable error message
    # @param kind [String, nil] sub-category discriminator
    def initialize(message, kind: nil)
      super(message)
      @kind = kind
    end
  end

  # Persistence layer error (redb / valkey checkpoint store).
  class PersistError < Error; end

  # Prompt-template error.
  #
  # +kind+ is one of +"MissingVariable"+, +"NotFound"+, +"VersionNotFound"+,
  # +"Io"+, +"Yaml"+, +"Json"+, +"Validation"+ (or +nil+ if unset).
  class PromptError < Error
    # @return [String, nil] sub-category discriminator
    attr_reader :kind

    # @param message [String] human-readable error message
    # @param kind [String, nil] sub-category discriminator
    def initialize(message, kind: nil)
      super(message)
      @kind = kind
    end
  end

  # Memory subsystem error.
  #
  # +kind+ is one of +"NoEmbedder"+, +"Elid"+, +"Embedding"+, +"NotFound"+,
  # +"Serialization"+, +"Io"+, +"Backend"+ (or +nil+ if unset).
  class MemoryError < Error
    # @return [String, nil] sub-category discriminator
    attr_reader :kind

    # @param message [String] human-readable error message
    # @param kind [String, nil] sub-category discriminator
    def initialize(message, kind: nil)
      super(message)
      @kind = kind
    end
  end

  # Model-cache / download error.
  #
  # +kind+ is one of +"Download"+, +"CacheDir"+, +"Io"+ (or +nil+ if unset).
  class CacheError < Error
    # @return [String, nil] sub-category discriminator
    attr_reader :kind

    # @param message [String] human-readable error message
    # @param kind [String, nil] sub-category discriminator
    def initialize(message, kind: nil)
      super(message)
      @kind = kind
    end
  end

  # Operation was cancelled (e.g. via foreign-language cancellation).
  class CancelledError < Error; end

  # Fallback for errors that don't fit any other variant; should be rare.
  class InternalError < Error; end

  # ---------------------------------------------------------------------------
  # Pointer → Ruby-exception decoder
  # ---------------------------------------------------------------------------

  # Decodes an opaque {BlazenError} pointer into the appropriate
  # {Blazen::Error} subclass and frees the C side.
  #
  # Consumes +ptr+: the underlying allocation is released via
  # {Blazen::FFI.blazen_error_free} before returning. After this call the
  # caller MUST NOT touch +ptr+ again.
  #
  # Optional accessor fields (+retry_after_ms+, +status+, +provider+,
  # +endpoint+, +request_id+, +detail+, +subkind+) are read via the matching
  # +blazen_error_*+ accessors. Sentinel +-1+ values map to +nil+; null
  # C-strings map to +nil+.
  #
  # @param ptr [::FFI::Pointer] a caller-owned BlazenError pointer
  # @return [Blazen::Error] the matching idiomatic Ruby exception
  def self.build_error_from_ptr(ptr)
    return InternalError.new("null error pointer") if ptr.nil? || ptr.null?

    begin
      kind = Blazen::FFI.blazen_error_kind(ptr)
      msg = Blazen::FFI.consume_cstring(Blazen::FFI.blazen_error_message(ptr)) || ""

      case kind
      when Blazen::FFI::ERROR_KIND_AUTH
        AuthError.new(msg)
      when Blazen::FFI::ERROR_KIND_RATE_LIMIT
        retry_ms = Blazen::FFI.blazen_error_retry_after_ms(ptr)
        RateLimitError.new(msg, retry_after_ms: retry_ms.negative? ? nil : retry_ms)
      when Blazen::FFI::ERROR_KIND_TIMEOUT
        TimeoutError.new(msg, elapsed_ms: Blazen::FFI.blazen_error_elapsed_ms(ptr))
      when Blazen::FFI::ERROR_KIND_VALIDATION
        ValidationError.new(msg)
      when Blazen::FFI::ERROR_KIND_CONTENT_POLICY
        ContentPolicyError.new(msg)
      when Blazen::FFI::ERROR_KIND_UNSUPPORTED
        UnsupportedError.new(msg)
      when Blazen::FFI::ERROR_KIND_COMPUTE
        ComputeError.new(msg)
      when Blazen::FFI::ERROR_KIND_MEDIA
        MediaError.new(msg)
      when Blazen::FFI::ERROR_KIND_PROVIDER
        status_raw = Blazen::FFI.blazen_error_status(ptr)
        retry_raw = Blazen::FFI.blazen_error_retry_after_ms(ptr)
        ProviderError.new(
          msg,
          kind: Blazen::FFI.consume_cstring(Blazen::FFI.blazen_error_subkind(ptr)),
          provider: Blazen::FFI.consume_cstring(Blazen::FFI.blazen_error_provider(ptr)),
          endpoint: Blazen::FFI.consume_cstring(Blazen::FFI.blazen_error_endpoint(ptr)),
          request_id: Blazen::FFI.consume_cstring(Blazen::FFI.blazen_error_request_id(ptr)),
          detail: Blazen::FFI.consume_cstring(Blazen::FFI.blazen_error_detail(ptr)),
          status: status_raw.negative? ? nil : status_raw,
          retry_after_ms: retry_raw.negative? ? nil : retry_raw,
        )
      when Blazen::FFI::ERROR_KIND_WORKFLOW
        WorkflowError.new(msg)
      when Blazen::FFI::ERROR_KIND_TOOL
        ToolError.new(msg)
      when Blazen::FFI::ERROR_KIND_PEER
        PeerError.new(msg, kind: Blazen::FFI.consume_cstring(Blazen::FFI.blazen_error_subkind(ptr)))
      when Blazen::FFI::ERROR_KIND_PERSIST
        PersistError.new(msg)
      when Blazen::FFI::ERROR_KIND_PROMPT
        PromptError.new(msg, kind: Blazen::FFI.consume_cstring(Blazen::FFI.blazen_error_subkind(ptr)))
      when Blazen::FFI::ERROR_KIND_MEMORY
        MemoryError.new(msg, kind: Blazen::FFI.consume_cstring(Blazen::FFI.blazen_error_subkind(ptr)))
      when Blazen::FFI::ERROR_KIND_CACHE
        CacheError.new(msg, kind: Blazen::FFI.consume_cstring(Blazen::FFI.blazen_error_subkind(ptr)))
      when Blazen::FFI::ERROR_KIND_CANCELLED
        CancelledError.new(msg)
      else
        InternalError.new(msg)
      end
    ensure
      Blazen::FFI.blazen_error_free(ptr)
    end
  end

  # Backwards-compatibility helper: runs +block+ and returns its value.
  #
  # In the UniFFI era this method re-raised generated +BlazenError::*+
  # exceptions as the idiomatic {Blazen::Error} subclass. The cabi surface
  # now raises typed errors directly from {Blazen::FFI.check_error!}, so
  # there's nothing to translate here — the helper just passes the block
  # value through. It's kept so existing callers in the other helper modules
  # (and downstream user code) don't break.
  #
  # @yield block to wrap
  # @return [Object] whatever +block+ returns
  def self.translate_errors
    yield
  end
end
