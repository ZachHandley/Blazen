# frozen_string_literal: true

module Blazen
  # Idiomatic Ruby exception hierarchy mirroring Blazen's Rust {BlazenError}
  # enum (see +crates/blazen-uniffi/src/errors.rs+).
  #
  # The underlying UniFFI binding uses +#[uniffi(flat_error)]+, which means
  # only the formatted +Display+ string crosses the FFI boundary. Sub-kind
  # discriminators (e.g. +Provider.kind = "OpenAIHttp"+) and structured fields
  # (e.g. +Timeout.elapsed_ms+) are encoded in the message prefix and parsed
  # back out here when constructing the idiomatic exception.
  #
  # The generated UniFFI module defines a +Blazen::InternalError+ that is a
  # plain +StandardError+ subclass used internally by the FFI loader for
  # unexpected scaffolding states (panics, mismatched call status codes).
  # We want our own +InternalError+ rooted in {Blazen::Error}, so undefine
  # the generated one first. The FFI loader still references the constant
  # by name when raising; the redefinition below keeps that path working.
  remove_const(:InternalError) if const_defined?(:InternalError, false)

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
    # @return [String] backend / failure-mode discriminator
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
    # @param kind [String] backend / failure-mode discriminator
    # @param provider [String, nil] provider name
    # @param status [Integer, nil] HTTP status code
    # @param endpoint [String, nil] endpoint that failed
    # @param request_id [String, nil] provider request ID
    # @param detail [String, nil] provider-specific error detail
    # @param retry_after_ms [Integer, nil] retry-after delay in milliseconds
    def initialize(message, kind:, provider: nil, status: nil, endpoint: nil,
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
  # +"Workflow"+, +"Tls"+, +"UnknownStep"+.
  class PeerError < Error
    # @return [String] sub-category discriminator
    attr_reader :kind

    # @param message [String] human-readable error message
    # @param kind [String] sub-category discriminator
    def initialize(message, kind:)
      super(message)
      @kind = kind
    end
  end

  # Persistence layer error (redb / valkey checkpoint store).
  class PersistError < Error; end

  # Prompt-template error.
  #
  # +kind+ is one of +"MissingVariable"+, +"NotFound"+, +"VersionNotFound"+,
  # +"Io"+, +"Yaml"+, +"Json"+, +"Validation"+.
  class PromptError < Error
    # @return [String] sub-category discriminator
    attr_reader :kind

    # @param message [String] human-readable error message
    # @param kind [String] sub-category discriminator
    def initialize(message, kind:)
      super(message)
      @kind = kind
    end
  end

  # Memory subsystem error.
  #
  # +kind+ is one of +"NoEmbedder"+, +"Elid"+, +"Embedding"+, +"NotFound"+,
  # +"Serialization"+, +"Io"+, +"Backend"+.
  class MemoryError < Error
    # @return [String] sub-category discriminator
    attr_reader :kind

    # @param message [String] human-readable error message
    # @param kind [String] sub-category discriminator
    def initialize(message, kind:)
      super(message)
      @kind = kind
    end
  end

  # Model-cache / download error.
  #
  # +kind+ is one of +"Download"+, +"CacheDir"+, +"Io"+.
  class CacheError < Error
    # @return [String] sub-category discriminator
    attr_reader :kind

    # @param message [String] human-readable error message
    # @param kind [String] sub-category discriminator
    def initialize(message, kind:)
      super(message)
      @kind = kind
    end
  end

  # Operation was cancelled (e.g. via foreign-language cancellation).
  class CancelledError < Error; end

  # Fallback for errors that don't fit any other variant; should be rare.
  class InternalError < Error; end

  # ---------------------------------------------------------------------------
  # Internal: convert a raw UniFFI-generated exception into the idiomatic class.
  # ---------------------------------------------------------------------------

  # Wraps a raw UniFFI-generated +Blazen::BlazenError::*+ exception in the
  # idiomatic Ruby class above.
  #
  # Variants that carry sub-discriminators or structured fields encode them
  # in the message prefix (per the Rust +#[error("...")]+ format strings); we
  # parse those out here to populate the Ruby exception attributes.
  #
  # @param err [Exception] raw exception from a UniFFI call
  # @return [Blazen::Error] idiomatic Ruby exception (caller should +raise+)
  def self.wrap_error(err)
    msg = err.message.to_s
    case err
    when BlazenError::Auth          then AuthError.new(msg)
    when BlazenError::RateLimit     then RateLimitError.new(msg)
    when BlazenError::Timeout       then TimeoutError.new(msg, elapsed_ms: _parse_elapsed_ms(msg))
    when BlazenError::Validation    then ValidationError.new(msg)
    when BlazenError::ContentPolicy then ContentPolicyError.new(msg)
    when BlazenError::Unsupported   then UnsupportedError.new(msg)
    when BlazenError::Compute       then ComputeError.new(msg)
    when BlazenError::Media         then MediaError.new(msg)
    when BlazenError::Provider      then ProviderError.new(msg, kind: _parse_kind("provider", msg))
    when BlazenError::Workflow      then WorkflowError.new(msg)
    when BlazenError::Tool          then ToolError.new(msg)
    when BlazenError::Peer          then PeerError.new(msg, kind: _parse_kind("peer", msg))
    when BlazenError::Persist       then PersistError.new(msg)
    when BlazenError::Prompt        then PromptError.new(msg, kind: _parse_kind("prompt", msg))
    when BlazenError::Memory        then MemoryError.new(msg, kind: _parse_kind("memory", msg))
    when BlazenError::Cache         then CacheError.new(msg, kind: _parse_kind("cache", msg))
    when BlazenError::Cancelled     then CancelledError.new(msg)
    when BlazenError::Internal      then InternalError.new(msg)
    else err
    end
  end

  # Runs +block+ and re-raises any UniFFI {BlazenError} as the matching
  # idiomatic {Blazen::Error} subclass.
  #
  # @yield block to wrap
  # @return [Object] whatever +block+ returns
  def self.translate_errors
    yield
  rescue BlazenError::Auth, BlazenError::RateLimit, BlazenError::Timeout,
         BlazenError::Validation, BlazenError::ContentPolicy,
         BlazenError::Unsupported, BlazenError::Compute, BlazenError::Media,
         BlazenError::Provider, BlazenError::Workflow, BlazenError::Tool,
         BlazenError::Peer, BlazenError::Persist, BlazenError::Prompt,
         BlazenError::Memory, BlazenError::Cache, BlazenError::Cancelled,
         BlazenError::Internal => e
    raise wrap_error(e)
  end

  # Parses +"<head> <kind>: ..."+-style messages, returning +kind+.
  #
  # @api private
  def self._parse_kind(head, msg)
    m = msg.match(/\A#{Regexp.escape(head)}\s+([^:\s]+):/)
    m ? m[1] : ""
  end

  # Parses +"timeout: ... (elapsed N ms)"+-style messages, returning +N+.
  #
  # @api private
  def self._parse_elapsed_ms(msg)
    m = msg.match(/\(elapsed (\d+) ms\)\z/)
    m ? m[1].to_i : 0
  end
end
