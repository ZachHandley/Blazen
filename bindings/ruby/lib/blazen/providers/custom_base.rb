# frozen_string_literal: true

require_relative "../ffi"

module Blazen
  # Selects how a +CustomProvider+ talks to its backend for completion calls.
  #
  # Two variants exist today:
  #
  # * {.openai} — the provider speaks the OpenAI chat-completions protocol;
  #   serialisation, transport, retry, and error mapping are handled by the
  #   built-in OpenAI-compat client. The config carries +base_url+,
  #   +api_key+, default model, extra headers, etc.
  # * {.custom} — the provider performs its own HTTP / transport work
  #   inside an overridden +complete+ implementation. Used by host-provided
  #   subclasses that do not want any built-in client behaviour.
  #
  # @example
  #   protocol = Blazen::ApiProtocol.custom
  #   protocol.kind   # => "custom"
  class ApiProtocol
    # Constructs an OpenAI-protocol API selector wrapping the given
    # +BlazenOpenAiCompatConfig+. The config object must respond to
    # +to_ptr+ and return a live native pointer; the Ruby
    # +OpenAiCompatConfig+ wrapper class lives elsewhere in the binding
    # surface.
    #
    # @param config [#to_ptr] a live OpenAiCompatConfig wrapper
    # @return [Blazen::ApiProtocol]
    def self.openai(config)
      raise ArgumentError, "config is required" if config.nil?

      ptr = Blazen::FFI.blazen_api_protocol_openai(config.to_ptr)
      new(ptr)
    end

    # Constructs a "custom" API selector — the provider implements its own
    # transport in an overridden +complete+. No backend client is wired up.
    #
    # @return [Blazen::ApiProtocol]
    def self.custom
      new(Blazen::FFI.blazen_api_protocol_custom)
    end

    # Wraps a raw, already-allocated +BlazenApiProtocol *+ pointer.
    # Most callers should use {.openai} or {.custom} instead.
    #
    # @param ptr [::FFI::Pointer]
    def initialize(ptr)
      raise ArgumentError, "null pointer" if ptr.nil? || ptr.null?

      @ptr = ptr
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    # Builds a finalizer proc that frees the underlying native handle.
    #
    # @param ptr [::FFI::Pointer]
    # @return [Proc]
    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_api_protocol_free(ptr) unless ptr.nil? || ptr.null? }
    end

    # Returns the protocol kind as a Ruby string: +"openai"+ or +"custom"+.
    #
    # @return [String]
    def kind
      Blazen::FFI.consume_cstring(Blazen::FFI.blazen_api_protocol_kind(@ptr))
    end

    # Returns the underlying +OpenAiCompatConfig+ pointer when this is the
    # OpenAI variant; +nil+ for the +"custom"+ variant. The Ruby wrapper
    # class for +OpenAiCompatConfig+ is not part of Phase A — callers that
    # need it can pass the raw pointer to whichever consumer needs it.
    #
    # @return [::FFI::Pointer, nil]
    def config_ptr
      ptr = Blazen::FFI.blazen_api_protocol_config(@ptr)
      return nil if ptr.nil? || ptr.null?

      ptr
    end

    # @return [::FFI::Pointer] the underlying +BlazenApiProtocol *+
    attr_reader :ptr
    alias_method :to_ptr, :ptr
  end

  # Provider-role-agnostic defaults — currently a marker that signals the
  # presence of a host-provided +before_request+ hook. Foreign callbacks
  # for that hook are not wired up in Phase A; the handle exists primarily
  # so role-specific defaults ({CompletionProviderDefaults},
  # {EmbeddingProviderDefaults}, etc.) can hang their +base+ off it.
  class BaseProviderDefaults
    # Allocates a fresh defaults handle, or wraps an existing one.
    #
    # @param ptr [::FFI::Pointer, nil] existing handle to adopt; allocates a new one when nil
    def initialize(ptr = nil)
      ptr ||= Blazen::FFI.blazen_base_provider_defaults_new
      raise ArgumentError, "null pointer" if ptr.nil? || ptr.null?

      @ptr = ptr
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    # @api private
    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_base_provider_defaults_free(ptr) unless ptr.nil? || ptr.null? }
    end

    # @return [Boolean] whether a host +before_request+ hook is attached
    def has_before_request?
      Blazen::FFI.blazen_base_provider_defaults_has_before_request(@ptr)
    end

    # @return [::FFI::Pointer] the underlying +BlazenBaseProviderDefaults *+
    attr_reader :ptr
    alias_method :to_ptr, :ptr
  end

  # Defaults applied to every completion call made through a
  # {BaseProvider}-derived completion model: system prompt, tools, response
  # format, and an optional shared {BaseProviderDefaults}.
  #
  # @example
  #   d = Blazen::CompletionProviderDefaults.new(
  #     system_prompt: "Be terse.",
  #     tools_json: '[{"type":"function","function":{"name":"now"}}]',
  #   )
  class CompletionProviderDefaults
    # @param system_prompt [String, nil]
    # @param tools_json [String, nil] JSON-encoded tools array
    # @param response_format_json [String, nil] JSON-encoded response_format
    # @param base [Blazen::BaseProviderDefaults, nil]
    def initialize(system_prompt: nil, tools_json: nil,
                   response_format_json: nil, base: nil)
      @ptr = Blazen::FFI.blazen_completion_provider_defaults_new
      self.system_prompt = system_prompt unless system_prompt.nil?
      self.tools_json = tools_json unless tools_json.nil?
      self.response_format_json = response_format_json unless response_format_json.nil?
      self.base = base unless base.nil?
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    # @api private
    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_completion_provider_defaults_free(ptr) unless ptr.nil? || ptr.null? }
    end

    # @return [String, nil]
    def system_prompt
      Blazen::FFI.consume_cstring(Blazen::FFI.blazen_completion_provider_defaults_system_prompt(@ptr))
    end

    # @param value [String]
    def system_prompt=(value)
      Blazen::FFI.with_cstring(value.to_s) do |s|
        Blazen::FFI.blazen_completion_provider_defaults_set_system_prompt(@ptr, s)
      end
    end

    # @return [String, nil]
    def tools_json
      Blazen::FFI.consume_cstring(Blazen::FFI.blazen_completion_provider_defaults_tools_json(@ptr))
    end

    # @param value [String]
    def tools_json=(value)
      Blazen::FFI.with_cstring(value.to_s) do |s|
        Blazen::FFI.blazen_completion_provider_defaults_set_tools_json(@ptr, s)
      end
    end

    # @return [String, nil]
    def response_format_json
      Blazen::FFI.consume_cstring(Blazen::FFI.blazen_completion_provider_defaults_response_format_json(@ptr))
    end

    # @param value [String]
    def response_format_json=(value)
      Blazen::FFI.with_cstring(value.to_s) do |s|
        Blazen::FFI.blazen_completion_provider_defaults_set_response_format_json(@ptr, s)
      end
    end

    # Attaches a shared {BaseProviderDefaults} sub-handle (carrying the
    # host +before_request+ hook, when present).
    #
    # @param base [Blazen::BaseProviderDefaults]
    def base=(base)
      Blazen::FFI.blazen_completion_provider_defaults_set_base(@ptr, base.to_ptr)
    end

    # @return [Blazen::BaseProviderDefaults, nil] a fresh wrapper around a
    #   freshly-allocated copy of the inner base defaults, or +nil+ when none
    def base
      ptr = Blazen::FFI.blazen_completion_provider_defaults_base(@ptr)
      return nil if ptr.nil? || ptr.null?

      BaseProviderDefaults.new(ptr)
    end

    # @return [Boolean] whether a host +before_completion+ hook is attached
    def has_before_completion?
      Blazen::FFI.blazen_completion_provider_defaults_has_before_completion(@ptr)
    end

    # @return [::FFI::Pointer] the underlying +BlazenCompletionProviderDefaults *+
    attr_reader :ptr
    alias_method :to_ptr, :ptr
  end

  # Defaults applied to every embedding call. Carries an optional shared
  # {BaseProviderDefaults} only — embedding has no role-specific fields in
  # Phase A.
  class EmbeddingProviderDefaults
    # @param base [Blazen::BaseProviderDefaults, nil]
    def initialize(base: nil)
      @ptr = Blazen::FFI.blazen_embedding_provider_defaults_new
      self.base = base unless base.nil?
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    # @api private
    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_embedding_provider_defaults_free(ptr) unless ptr.nil? || ptr.null? }
    end

    # @param base [Blazen::BaseProviderDefaults]
    def base=(base)
      Blazen::FFI.blazen_embedding_provider_defaults_set_base(@ptr, base.to_ptr)
    end

    # @return [Blazen::BaseProviderDefaults, nil]
    def base
      ptr = Blazen::FFI.blazen_embedding_provider_defaults_base(@ptr)
      return nil if ptr.nil? || ptr.null?

      BaseProviderDefaults.new(ptr)
    end

    # @return [::FFI::Pointer]
    attr_reader :ptr
    alias_method :to_ptr, :ptr
  end

  # Per-role defaults handles. Each one wraps a tiny native struct that
  # carries an optional shared {BaseProviderDefaults} and exposes a
  # +has_before?+ predicate that reflects whether a host-provided
  # per-role hook is attached. Phase B will wire those hooks up to Ruby
  # callbacks; Phase A only ships the lifecycle + getter/setter surface.
  #
  # @!macro [new] role_defaults_doc
  #   @!method initialize(base: nil)
  #     @param base [Blazen::BaseProviderDefaults, nil] shared base defaults
  #   @!method base
  #     @return [Blazen::BaseProviderDefaults, nil]
  #   @!method base=(base)
  #     @param base [Blazen::BaseProviderDefaults]
  #   @!method has_before?
  #     @return [Boolean] whether a host per-role +before+ hook is attached
  #   @!attribute [r] ptr
  #     @return [::FFI::Pointer]

  # Defaults handle for the +audio.speech+ (TTS) role.
  # @!macro role_defaults_doc
  class AudioSpeechProviderDefaults
    def initialize(base: nil)
      @ptr = Blazen::FFI.blazen_audio_speech_provider_defaults_new
      self.base = base unless base.nil?
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_audio_speech_provider_defaults_free(ptr) unless ptr.nil? || ptr.null? }
    end

    def base=(base)
      Blazen::FFI.blazen_audio_speech_provider_defaults_set_base(@ptr, base.to_ptr)
    end

    def base
      ptr = Blazen::FFI.blazen_audio_speech_provider_defaults_base(@ptr)
      return nil if ptr.nil? || ptr.null?

      BaseProviderDefaults.new(ptr)
    end

    def has_before?
      Blazen::FFI.blazen_audio_speech_provider_defaults_has_before(@ptr)
    end

    attr_reader :ptr
    alias_method :to_ptr, :ptr
  end

  # Defaults handle for the +audio.music+ role.
  # @!macro role_defaults_doc
  class AudioMusicProviderDefaults
    def initialize(base: nil)
      @ptr = Blazen::FFI.blazen_audio_music_provider_defaults_new
      self.base = base unless base.nil?
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_audio_music_provider_defaults_free(ptr) unless ptr.nil? || ptr.null? }
    end

    def base=(base)
      Blazen::FFI.blazen_audio_music_provider_defaults_set_base(@ptr, base.to_ptr)
    end

    def base
      ptr = Blazen::FFI.blazen_audio_music_provider_defaults_base(@ptr)
      return nil if ptr.nil? || ptr.null?

      BaseProviderDefaults.new(ptr)
    end

    def has_before?
      Blazen::FFI.blazen_audio_music_provider_defaults_has_before(@ptr)
    end

    attr_reader :ptr
    alias_method :to_ptr, :ptr
  end

  # Defaults handle for the +voice_cloning+ role.
  # @!macro role_defaults_doc
  class VoiceCloningProviderDefaults
    def initialize(base: nil)
      @ptr = Blazen::FFI.blazen_voice_cloning_provider_defaults_new
      self.base = base unless base.nil?
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_voice_cloning_provider_defaults_free(ptr) unless ptr.nil? || ptr.null? }
    end

    def base=(base)
      Blazen::FFI.blazen_voice_cloning_provider_defaults_set_base(@ptr, base.to_ptr)
    end

    def base
      ptr = Blazen::FFI.blazen_voice_cloning_provider_defaults_base(@ptr)
      return nil if ptr.nil? || ptr.null?

      BaseProviderDefaults.new(ptr)
    end

    def has_before?
      Blazen::FFI.blazen_voice_cloning_provider_defaults_has_before(@ptr)
    end

    attr_reader :ptr
    alias_method :to_ptr, :ptr
  end

  # Defaults handle for the +image.generation+ role.
  # @!macro role_defaults_doc
  class ImageGenerationProviderDefaults
    def initialize(base: nil)
      @ptr = Blazen::FFI.blazen_image_generation_provider_defaults_new
      self.base = base unless base.nil?
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_image_generation_provider_defaults_free(ptr) unless ptr.nil? || ptr.null? }
    end

    def base=(base)
      Blazen::FFI.blazen_image_generation_provider_defaults_set_base(@ptr, base.to_ptr)
    end

    def base
      ptr = Blazen::FFI.blazen_image_generation_provider_defaults_base(@ptr)
      return nil if ptr.nil? || ptr.null?

      BaseProviderDefaults.new(ptr)
    end

    def has_before?
      Blazen::FFI.blazen_image_generation_provider_defaults_has_before(@ptr)
    end

    attr_reader :ptr
    alias_method :to_ptr, :ptr
  end

  # Defaults handle for the +image.upscale+ role.
  # @!macro role_defaults_doc
  class ImageUpscaleProviderDefaults
    def initialize(base: nil)
      @ptr = Blazen::FFI.blazen_image_upscale_provider_defaults_new
      self.base = base unless base.nil?
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_image_upscale_provider_defaults_free(ptr) unless ptr.nil? || ptr.null? }
    end

    def base=(base)
      Blazen::FFI.blazen_image_upscale_provider_defaults_set_base(@ptr, base.to_ptr)
    end

    def base
      ptr = Blazen::FFI.blazen_image_upscale_provider_defaults_base(@ptr)
      return nil if ptr.nil? || ptr.null?

      BaseProviderDefaults.new(ptr)
    end

    def has_before?
      Blazen::FFI.blazen_image_upscale_provider_defaults_has_before(@ptr)
    end

    attr_reader :ptr
    alias_method :to_ptr, :ptr
  end

  # Defaults handle for the +video+ role.
  # @!macro role_defaults_doc
  class VideoProviderDefaults
    def initialize(base: nil)
      @ptr = Blazen::FFI.blazen_video_provider_defaults_new
      self.base = base unless base.nil?
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_video_provider_defaults_free(ptr) unless ptr.nil? || ptr.null? }
    end

    def base=(base)
      Blazen::FFI.blazen_video_provider_defaults_set_base(@ptr, base.to_ptr)
    end

    def base
      ptr = Blazen::FFI.blazen_video_provider_defaults_base(@ptr)
      return nil if ptr.nil? || ptr.null?

      BaseProviderDefaults.new(ptr)
    end

    def has_before?
      Blazen::FFI.blazen_video_provider_defaults_has_before(@ptr)
    end

    attr_reader :ptr
    alias_method :to_ptr, :ptr
  end

  # Defaults handle for the +transcription+ (STT) role.
  # @!macro role_defaults_doc
  class TranscriptionProviderDefaults
    def initialize(base: nil)
      @ptr = Blazen::FFI.blazen_transcription_provider_defaults_new
      self.base = base unless base.nil?
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_transcription_provider_defaults_free(ptr) unless ptr.nil? || ptr.null? }
    end

    def base=(base)
      Blazen::FFI.blazen_transcription_provider_defaults_set_base(@ptr, base.to_ptr)
    end

    def base
      ptr = Blazen::FFI.blazen_transcription_provider_defaults_base(@ptr)
      return nil if ptr.nil? || ptr.null?

      BaseProviderDefaults.new(ptr)
    end

    def has_before?
      Blazen::FFI.blazen_transcription_provider_defaults_has_before(@ptr)
    end

    attr_reader :ptr
    alias_method :to_ptr, :ptr
  end

  # Defaults handle for the +three_d+ (3D model generation) role.
  # @!macro role_defaults_doc
  class ThreeDProviderDefaults
    def initialize(base: nil)
      @ptr = Blazen::FFI.blazen_three_d_provider_defaults_new
      self.base = base unless base.nil?
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_three_d_provider_defaults_free(ptr) unless ptr.nil? || ptr.null? }
    end

    def base=(base)
      Blazen::FFI.blazen_three_d_provider_defaults_set_base(@ptr, base.to_ptr)
    end

    def base
      ptr = Blazen::FFI.blazen_three_d_provider_defaults_base(@ptr)
      return nil if ptr.nil? || ptr.null?

      BaseProviderDefaults.new(ptr)
    end

    def has_before?
      Blazen::FFI.blazen_three_d_provider_defaults_has_before(@ptr)
    end

    attr_reader :ptr
    alias_method :to_ptr, :ptr
  end

  # Defaults handle for the +background_removal+ role.
  # @!macro role_defaults_doc
  class BackgroundRemovalProviderDefaults
    def initialize(base: nil)
      @ptr = Blazen::FFI.blazen_background_removal_provider_defaults_new
      self.base = base unless base.nil?
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_background_removal_provider_defaults_free(ptr) unless ptr.nil? || ptr.null? }
    end

    def base=(base)
      Blazen::FFI.blazen_background_removal_provider_defaults_set_base(@ptr, base.to_ptr)
    end

    def base
      ptr = Blazen::FFI.blazen_background_removal_provider_defaults_base(@ptr)
      return nil if ptr.nil? || ptr.null?

      BaseProviderDefaults.new(ptr)
    end

    def has_before?
      Blazen::FFI.blazen_background_removal_provider_defaults_has_before(@ptr)
    end

    attr_reader :ptr
    alias_method :to_ptr, :ptr
  end

  # Wraps a +BlazenBaseProvider *+ — a completion model that has been
  # configured with instance-level defaults (system prompt, tools,
  # response_format, plus an optional shared {CompletionProviderDefaults}
  # handle). In V1, instances are not constructible from Ruby directly;
  # they are returned from {Blazen::CustomProvider} factories (Phase B).
  class BaseProvider
    # @param ptr [::FFI::Pointer] a live +BlazenBaseProvider *+ — typically
    #   returned by the cabi
    def initialize(ptr)
      if ptr.nil? || ptr.null?
        raise ArgumentError,
              "BaseProvider cannot be constructed directly in V1 — pass a " \
              "live BlazenBaseProvider * (e.g. from a CustomProvider factory)"
      end

      @ptr = ptr
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    # @api private
    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_base_provider_free(ptr) unless ptr.nil? || ptr.null? }
    end

    # Overrides the per-call system prompt that the provider injects ahead
    # of every completion. Returns +self+ for chaining.
    #
    # @param value [String]
    # @return [self]
    def with_system_prompt(value)
      Blazen::FFI.with_cstring(value.to_s) do |s|
        Blazen::FFI.blazen_base_provider_with_system_prompt(@ptr, s)
      end
      self
    end

    # Overrides the per-call tools list (JSON-encoded array). Returns +self+.
    #
    # @param json [String]
    # @return [self]
    def with_tools_json(json)
      Blazen::FFI.with_cstring(json.to_s) do |s|
        Blazen::FFI.blazen_base_provider_with_tools_json(@ptr, s)
      end
      self
    end

    # Overrides the per-call response_format (JSON-encoded). Returns +self+.
    #
    # @param json [String]
    # @return [self]
    def with_response_format_json(json)
      Blazen::FFI.with_cstring(json.to_s) do |s|
        Blazen::FFI.blazen_base_provider_with_response_format_json(@ptr, s)
      end
      self
    end

    # Replaces the shared {CompletionProviderDefaults} attached to this
    # provider. Returns +self+ for chaining.
    #
    # @param defaults [Blazen::CompletionProviderDefaults]
    # @return [self]
    def with_defaults(defaults)
      Blazen::FFI.blazen_base_provider_with_defaults(@ptr, defaults.to_ptr)
      self
    end

    # Returns the currently-attached {CompletionProviderDefaults} as a
    # fresh wrapper, or +nil+ if none.
    #
    # @return [Blazen::CompletionProviderDefaults, nil]
    def defaults
      ptr = Blazen::FFI.blazen_base_provider_defaults(@ptr)
      return nil if ptr.nil? || ptr.null?

      cpd = CompletionProviderDefaults.allocate
      cpd.instance_variable_set(:@ptr, ptr)
      ObjectSpace.define_finalizer(cpd, CompletionProviderDefaults.finalizer(ptr))
      cpd
    end

    # @return [String]
    def model_id
      Blazen::FFI.consume_cstring(Blazen::FFI.blazen_base_provider_model_id(@ptr))
    end

    # @return [String, nil]
    def provider_id
      Blazen::FFI.consume_cstring(Blazen::FFI.blazen_base_provider_provider_id(@ptr))
    end

    # @return [::FFI::Pointer] the underlying +BlazenBaseProvider *+
    attr_reader :ptr
    alias_method :to_ptr, :ptr
  end

  # ---------------------------------------------------------------------
  # OpenAiCompatConfig — Ruby wrapper around +BlazenOpenAiCompatConfig+
  # ---------------------------------------------------------------------

  # Lightweight Ruby wrapper around the cabi
  # +BlazenOpenAiCompatConfig+ struct. Built ad-hoc by callers of
  # {Blazen.openai_compat} (and similar) so the underlying
  # +blazen_custom_provider_openai_compat+ factory has a config handle to
  # clone its state from.
  #
  # Mirrors the same +to_ptr+ contract as the other wrappers in this file:
  # the underlying allocation is freed by a finalizer when the Ruby wrapper
  # is GC'd.
  #
  # @example
  #   cfg = Blazen::OpenAiCompatConfig.new(
  #     provider_name: "groq",
  #     base_url:      "https://api.groq.com/openai/v1",
  #     api_key:       ENV.fetch("GROQ_API_KEY"),
  #     default_model: "llama-3.1-70b-versatile",
  #   )
  class OpenAiCompatConfig
    # Auth-code enum mirrors +blazen_llm::providers::openai_compat::AuthMethod+:
    # +0+ = Bearer (default), +1+ = ApiKeyHeader, +2+ = AzureApiKey,
    # +3+ = KeyPrefix.
    AUTH_BEARER         = 0
    AUTH_API_KEY_HEADER = 1
    AUTH_AZURE_API_KEY  = 2
    AUTH_KEY_PREFIX     = 3

    # @param provider_name [String] short identifier, e.g. +"groq"+
    # @param base_url      [String] absolute URL incl. the +/v1+ path
    # @param api_key       [String] bearer / header / prefix credential
    # @param default_model [String] model id to ship in completion requests
    # @param auth_code     [Integer] one of +AUTH_*+ constants (default +Bearer+)
    # @param auth_header_name [String, nil] only used for +AUTH_API_KEY_HEADER+
    # @param supports_model_listing [Boolean] does the server implement +GET /v1/models+
    # @param extra_headers [Hash{String=>String}, nil] additional request headers
    # @param query_params  [Hash{String=>String}, nil] additional query params
    def initialize(provider_name:, base_url:, api_key:, default_model:,
                   auth_code: AUTH_BEARER, auth_header_name: nil,
                   supports_model_listing: false,
                   extra_headers: nil, query_params: nil)
      ptr = Blazen::FFI.with_cstring(provider_name) do |p|
        Blazen::FFI.with_cstring(base_url) do |b|
          Blazen::FFI.with_cstring(api_key) do |k|
            Blazen::FFI.with_cstring(default_model) do |m|
              Blazen::FFI.with_cstring(auth_header_name) do |h|
                Blazen::FFI.blazen_openai_compat_config_new(
                  p, b, k, m, auth_code.to_i, h, supports_model_listing ? true : false
                )
              end
            end
          end
        end
      end
      if ptr.nil? || ptr.null?
        raise ArgumentError,
              "blazen_openai_compat_config_new returned null — one of " \
              "provider_name/base_url/api_key/default_model was null or non-UTF-8"
      end

      @ptr = ptr
      (extra_headers || {}).each { |k, v| push_extra_header(k, v) }
      (query_params  || {}).each { |k, v| push_query_param(k, v)  }
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    # @api private
    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_openai_compat_config_free(ptr) unless ptr.nil? || ptr.null? }
    end

    # Appends an +extra_headers+ entry.
    # @param name [String]
    # @param value [String]
    # @return [self]
    def push_extra_header(name, value)
      Blazen::FFI.with_cstring(name) do |n|
        Blazen::FFI.with_cstring(value) do |v|
          Blazen::FFI.blazen_openai_compat_config_push_extra_header(@ptr, n, v)
        end
      end
      self
    end

    # Appends a +query_params+ entry.
    # @param name [String]
    # @param value [String]
    # @return [self]
    def push_query_param(name, value)
      Blazen::FFI.with_cstring(name) do |n|
        Blazen::FFI.with_cstring(value) do |v|
          Blazen::FFI.blazen_openai_compat_config_push_query_param(@ptr, n, v)
        end
      end
      self
    end

    # @return [String, nil]
    def provider_name
      Blazen::FFI.consume_cstring(Blazen::FFI.blazen_openai_compat_config_provider_name(@ptr))
    end

    # @return [String, nil]
    def base_url
      Blazen::FFI.consume_cstring(Blazen::FFI.blazen_openai_compat_config_base_url(@ptr))
    end

    # @return [String, nil]
    def api_key
      Blazen::FFI.consume_cstring(Blazen::FFI.blazen_openai_compat_config_api_key(@ptr))
    end

    # @return [String, nil]
    def default_model
      Blazen::FFI.consume_cstring(Blazen::FFI.blazen_openai_compat_config_default_model(@ptr))
    end

    # @return [Integer] one of the +AUTH_*+ constants
    def auth_code
      Blazen::FFI.blazen_openai_compat_config_auth_code(@ptr)
    end

    # @return [::FFI::Pointer] the underlying +BlazenOpenAiCompatConfig *+
    attr_reader :ptr
    alias_method :to_ptr, :ptr
  end

  # ---------------------------------------------------------------------
  # CustomProvider — typed-method base class users subclass
  # ---------------------------------------------------------------------

  # User-facing base class for foreign-implemented providers. Subclass and
  # override only the methods you implement; the other 15 typed methods will
  # raise {Blazen::UnsupportedError} by default so the rest of the Blazen
  # runtime can fall through to its documented "no such capability" semantics.
  #
  # ## Wire-up
  #
  # When a {CustomProvider} subclass instance is handed to
  # {Blazen::CustomProvider.from_subclass} (or, equivalently, when {.new} is
  # called on a subclass with no overrides at all), the binding builds a
  # cabi {Blazen::FFI::BlazenCustomProviderVTable} whose 16 fn-pointers all
  # dispatch back into the Ruby instance via {Registry}. The vtable is then
  # consumed by +blazen_custom_provider_from_vtable+, producing a
  # +BlazenCustomProvider *+ that the rest of Blazen treats as a regular
  # provider.
  #
  # ## Ruby/FFI subclass limitations (V1)
  #
  # The cabi does NOT (yet) expose Ruby-callable constructors for the
  # typed result records (+BlazenAudioResult+, +BlazenImageResult+,
  # +BlazenVoiceHandle+, +BlazenError+, etc.), so a Ruby override can't
  # currently synthesise a success value to hand back across the FFI.
  # Overrides therefore have two practical options:
  #
  # 1. Raise +Blazen::UnsupportedError+ (or any +StandardError+) — the
  #    binding catches it, the callback returns +-1+ to the cabi, and
  #    callers see a +Blazen::InternalError+ wrapping the original message.
  # 2. Leave the method un-overridden — the default implementation raises
  #    +Blazen::UnsupportedError+ for you.
  #
  # Full Ruby-side typed-result construction is tracked for a follow-on
  # wave; this V1 surface still lets Ruby users register their own provider
  # identity / metadata and reuse the cabi presets (ollama, lm_studio,
  # openai_compat) without any callback synthesis.
  #
  # @example Subclass with a custom +text_to_speech+
  #   class MyTts < Blazen::CustomProvider
  #     def text_to_speech(_req)
  #       # No Ruby-side result constructors yet — see "Ruby/FFI subclass
  #       # limitations" above; raise so the caller sees a structured error.
  #       raise Blazen::UnsupportedError, "MyTts: result synthesis not wired"
  #     end
  #   end
  #
  #   handle = Blazen::CustomProvider.from_subclass(MyTts.new)
  #   handle.provider_id  # => "my_tts" (derived from class name)
  class CustomProvider
    # ---------------------------------------------------------------------
    # Class-level callback registry
    #
    # The cabi vtable's +user_data+ field is an opaque +void*+. We stuff a
    # monotonically-increasing integer ID into it and use it as the key into
    # a class-level hash holding the Ruby instance + the per-vtable string
    # buffers. The hash entry is what keeps the Ruby instance + buffers
    # reachable to GC for as long as the cabi adapter holds the vtable.
    # ---------------------------------------------------------------------
    @registry = {}
    @registry_mutex = Mutex.new
    @next_registry_id = 0

    # @api private
    # @param instance [CustomProvider]
    # @param keepalive [Array] strong-ref jail for vtable string buffers
    # @return [Integer] the registry ID (stuffed into +user_data+)
    def self.register(instance, keepalive)
      @registry_mutex.synchronize do
        @next_registry_id += 1
        id = @next_registry_id
        @registry[id] = { instance: instance, keepalive: keepalive }
        id
      end
    end

    # @api private
    # @param id [Integer]
    # @return [CustomProvider, nil]
    def self.lookup(id)
      @registry_mutex.synchronize { @registry.dig(id, :instance) }
    end

    # @api private
    # @param id [Integer]
    def self.unregister(id)
      @registry_mutex.synchronize { @registry.delete(id) }
    end

    # ---------------------------------------------------------------------
    # Trampoline thunks — one per vtable slot
    #
    # Every typed thunk follows the same pattern:
    #   1. Look up the Ruby instance from +user_data+.
    #   2. Free / discard the incoming request pointer (we have nothing
    #      idiomatic to deserialize it into yet; see the class docs).
    #   3. Call the Ruby method, capturing any +StandardError+.
    #   4. Return +-1+. We never write +out_err+ because the cabi doesn't
    #      expose +BlazenError+ constructors to foreign callers — the Rust
    #      adapter synthesises a +InternalError+ when it sees status=-1
    #      with a null +out_err+, which carries the Ruby method's message
    #      as the "vtable returned -1" detail.
    # ---------------------------------------------------------------------

    # Free function map: maps request pointer types to the matching cabi
    # +blazen_<x>_free+ symbol so the trampolines can release the
    # caller-owned request without leaking on every dispatch.
    #
    # @api private
    REQUEST_FREE_FNS = {
      complete:          :blazen_completion_request_free,
      text_to_speech:    :blazen_speech_request_free,
      generate_music:    :blazen_music_request_free,
      generate_sfx:      :blazen_music_request_free,
      clone_voice:       :blazen_voice_clone_request_free,
      delete_voice:      :blazen_voice_handle_free,
      generate_image:    :blazen_image_request_free,
      upscale_image:     :blazen_upscale_request_free,
      text_to_video:     :blazen_video_request_free,
      image_to_video:    :blazen_video_request_free,
      transcribe:        :blazen_transcription_request_free,
      generate_3d:       :blazen_three_d_request_free,
      remove_background: :blazen_background_removal_request_free,
    }.freeze

    # @api private
    # Shared core for every typed-method trampoline.
    def self.dispatch_typed(user_data_ptr, method_name, req_ptr, free_sym)
      id = user_data_ptr.address
      instance = lookup(id)
      begin
        if instance.nil?
          warn "[blazen custom_provider] no instance registered for id=#{id}"
        else
          # Hand the raw request pointer in — V1 callers either ignore it or
          # consume it through the cabi accessors. We free it via the
          # +free_sym+ below regardless, so callers MUST NOT free it
          # themselves.
          instance.public_send(method_name, req_ptr)
        end
      rescue StandardError => e
        warn "[blazen custom_provider #{method_name}] #{e.class}: #{e.message}"
        warn e.backtrace.first(4).join("\n") if e.backtrace
      ensure
        if free_sym && req_ptr && !req_ptr.null?
          Blazen::FFI.send(free_sym, req_ptr)
        end
      end
      -1
    end

    # +drop_user_data+ — invoked exactly once when the inner cabi adapter
    # drops. Tears down our registry entry, releasing the strong reference
    # to the Ruby instance + string buffers.
    DROP_USER_DATA_FN = ::FFI::Function.new(
      :void,
      [:pointer],
      proc { |user_data_ptr| unregister(user_data_ptr.address) },
    )

    # +complete+ — special-cased because the success path (writing a
    # +BlazenCompletionResponse *+) is also not Ruby-constructible in V1.
    COMPLETE_FN = ::FFI::Function.new(
      :int32,
      %i[pointer pointer pointer pointer],
      proc do |user_data_ptr, req_ptr, _out_response, _out_err|
        dispatch_typed(user_data_ptr, :complete, req_ptr, :blazen_completion_request_free)
      end,
      blocking: true,
    )

    # +stream+ — similarly stubbed; Ruby streaming through a subclass is a
    # follow-on wave (the cabi stream-pusher surface needs a Ruby-side
    # wrapper before this becomes useful).
    STREAM_FN = ::FFI::Function.new(
      :int32,
      %i[pointer pointer pointer pointer],
      proc do |user_data_ptr, req_ptr, _pusher, _out_err|
        # Note: +req_ptr+ here is a +BlazenCompletionRequest *+; same as
        # complete. We don't own +pusher+ (the cabi keeps it).
        dispatch_typed(user_data_ptr, :stream, req_ptr, :blazen_completion_request_free)
      end,
      blocking: true,
    )

    # +embed+ — the texts array is borrowed-for-duration-of-the-call;
    # the trampoline doesn't free it.
    EMBED_FN = ::FFI::Function.new(
      :int32,
      %i[pointer pointer size_t pointer pointer],
      proc do |user_data_ptr, _texts, _count, _out_response, _out_err|
        id = user_data_ptr.address
        instance = lookup(id)
        begin
          instance&.public_send(:embed, _texts, _count)
        rescue StandardError => e
          warn "[blazen custom_provider embed] #{e.class}: #{e.message}"
        end
        -1
      end,
      blocking: true,
    )

    # +list_voices+ — no request, just out-array / out-count / out-err.
    LIST_VOICES_FN = ::FFI::Function.new(
      :int32,
      %i[pointer pointer pointer pointer],
      proc do |user_data_ptr, _out_array, _out_count, _out_err|
        id = user_data_ptr.address
        instance = lookup(id)
        begin
          instance&.public_send(:list_voices)
        rescue StandardError => e
          warn "[blazen custom_provider list_voices] #{e.class}: #{e.message}"
        end
        -1
      end,
      blocking: true,
    )

    # +delete_voice+ — request pointer is a +BlazenVoiceHandle *+ that the
    # callback must free.
    DELETE_VOICE_FN = ::FFI::Function.new(
      :int32,
      %i[pointer pointer pointer],
      proc do |user_data_ptr, voice_ptr, _out_err|
        dispatch_typed(user_data_ptr, :delete_voice, voice_ptr, :blazen_voice_handle_free)
      end,
      blocking: true,
    )

    # Per-method trampolines. Each one wraps {dispatch_typed} with the
    # appropriate request-free symbol baked in.
    TEXT_TO_SPEECH_FN = ::FFI::Function.new(
      :int32, %i[pointer pointer pointer pointer],
      proc { |u, r, *| dispatch_typed(u, :text_to_speech, r, :blazen_speech_request_free) },
      blocking: true,
    )
    GENERATE_MUSIC_FN = ::FFI::Function.new(
      :int32, %i[pointer pointer pointer pointer],
      proc { |u, r, *| dispatch_typed(u, :generate_music, r, :blazen_music_request_free) },
      blocking: true,
    )
    GENERATE_SFX_FN = ::FFI::Function.new(
      :int32, %i[pointer pointer pointer pointer],
      proc { |u, r, *| dispatch_typed(u, :generate_sfx, r, :blazen_music_request_free) },
      blocking: true,
    )
    CLONE_VOICE_FN = ::FFI::Function.new(
      :int32, %i[pointer pointer pointer pointer],
      proc { |u, r, *| dispatch_typed(u, :clone_voice, r, :blazen_voice_clone_request_free) },
      blocking: true,
    )
    GENERATE_IMAGE_FN = ::FFI::Function.new(
      :int32, %i[pointer pointer pointer pointer],
      proc { |u, r, *| dispatch_typed(u, :generate_image, r, :blazen_image_request_free) },
      blocking: true,
    )
    UPSCALE_IMAGE_FN = ::FFI::Function.new(
      :int32, %i[pointer pointer pointer pointer],
      proc { |u, r, *| dispatch_typed(u, :upscale_image, r, :blazen_upscale_request_free) },
      blocking: true,
    )
    TEXT_TO_VIDEO_FN = ::FFI::Function.new(
      :int32, %i[pointer pointer pointer pointer],
      proc { |u, r, *| dispatch_typed(u, :text_to_video, r, :blazen_video_request_free) },
      blocking: true,
    )
    IMAGE_TO_VIDEO_FN = ::FFI::Function.new(
      :int32, %i[pointer pointer pointer pointer],
      proc { |u, r, *| dispatch_typed(u, :image_to_video, r, :blazen_video_request_free) },
      blocking: true,
    )
    TRANSCRIBE_FN = ::FFI::Function.new(
      :int32, %i[pointer pointer pointer pointer],
      proc { |u, r, *| dispatch_typed(u, :transcribe, r, :blazen_transcription_request_free) },
      blocking: true,
    )
    GENERATE_3D_FN = ::FFI::Function.new(
      :int32, %i[pointer pointer pointer pointer],
      proc { |u, r, *| dispatch_typed(u, :generate_3d, r, :blazen_three_d_request_free) },
      blocking: true,
    )
    REMOVE_BACKGROUND_FN = ::FFI::Function.new(
      :int32, %i[pointer pointer pointer pointer],
      proc { |u, r, *| dispatch_typed(u, :remove_background, r, :blazen_background_removal_request_free) },
      blocking: true,
    )

    # ---------------------------------------------------------------------
    # Public class API
    # ---------------------------------------------------------------------

    # Builds a {CustomProviderHandle} that wraps +instance+ in a cabi vtable.
    # The returned handle is a regular Blazen provider — it can be turned
    # into a base provider via {CustomProviderHandle#as_base_provider} and
    # plugged into the rest of Blazen.
    #
    # @param instance [CustomProvider] a user subclass instance
    # @return [CustomProviderHandle]
    def self.from_subclass(instance)
      unless instance.is_a?(CustomProvider)
        raise ArgumentError, "from_subclass: instance must be a Blazen::CustomProvider"
      end

      keepalive = []
      id = register(instance, keepalive)

      provider_id_ptr = ::FFI::MemoryPointer.from_string(instance.provider_id.to_s)
      model_id_ptr    = ::FFI::MemoryPointer.from_string(instance.model_id.to_s)
      keepalive << provider_id_ptr
      keepalive << model_id_ptr

      vtable = Blazen::FFI::BlazenCustomProviderVTable.new
      vtable[:user_data]         = ::FFI::Pointer.new(:void, id)
      vtable[:drop_user_data]    = DROP_USER_DATA_FN
      vtable[:provider_id]       = provider_id_ptr
      vtable[:model_id]          = model_id_ptr
      vtable[:complete]          = COMPLETE_FN
      vtable[:stream]            = STREAM_FN
      vtable[:embed]             = EMBED_FN
      vtable[:text_to_speech]    = TEXT_TO_SPEECH_FN
      vtable[:generate_music]    = GENERATE_MUSIC_FN
      vtable[:generate_sfx]      = GENERATE_SFX_FN
      vtable[:clone_voice]       = CLONE_VOICE_FN
      vtable[:list_voices]       = LIST_VOICES_FN
      vtable[:delete_voice]      = DELETE_VOICE_FN
      vtable[:generate_image]    = GENERATE_IMAGE_FN
      vtable[:upscale_image]     = UPSCALE_IMAGE_FN
      vtable[:text_to_video]     = TEXT_TO_VIDEO_FN
      vtable[:image_to_video]    = IMAGE_TO_VIDEO_FN
      vtable[:transcribe]        = TRANSCRIBE_FN
      vtable[:generate_3d]       = GENERATE_3D_FN
      vtable[:remove_background] = REMOVE_BACKGROUND_FN

      ptr = Blazen::FFI.blazen_custom_provider_from_vtable(vtable)
      if ptr.nil? || ptr.null?
        unregister(id)
        raise Blazen::InternalError, "blazen_custom_provider_from_vtable returned null"
      end

      CustomProviderHandle.new(ptr)
    end

    # ---------------------------------------------------------------------
    # Default identity + 16 typed-method defaults
    # ---------------------------------------------------------------------

    # Derive a provider_id from the class name: +My::FunkyTts+ → +"funky_tts"+
    # (final segment, CamelCase → snake_case, downcased).
    def provider_id
      name = self.class.name || "custom_provider"
      seg  = name.split("::").last
      seg.gsub(/([A-Z]+)([A-Z][a-z])/, '\1_\2')
         .gsub(/([a-z\d])([A-Z])/, '\1_\2')
         .downcase
    end

    # Subclasses typically override +model_id+ to expose their model name;
    # the default reuses +provider_id+ so the metadata pair always renders.
    def model_id
      provider_id
    end

    # @raise [Blazen::UnsupportedError] always (override to implement)
    def complete(_request)
      raise Blazen::UnsupportedError, "#{self.class.name}: #complete not implemented"
    end

    # @raise [Blazen::UnsupportedError] always (override to implement)
    def stream(_request)
      raise Blazen::UnsupportedError, "#{self.class.name}: #stream not implemented"
    end

    # @raise [Blazen::UnsupportedError] always (override to implement)
    def embed(_texts, _count = nil)
      raise Blazen::UnsupportedError, "#{self.class.name}: #embed not implemented"
    end

    # @raise [Blazen::UnsupportedError] always (override to implement)
    def text_to_speech(_request)
      raise Blazen::UnsupportedError, "#{self.class.name}: #text_to_speech not implemented"
    end

    # @raise [Blazen::UnsupportedError] always (override to implement)
    def generate_music(_request)
      raise Blazen::UnsupportedError, "#{self.class.name}: #generate_music not implemented"
    end

    # @raise [Blazen::UnsupportedError] always (override to implement)
    def generate_sfx(_request)
      raise Blazen::UnsupportedError, "#{self.class.name}: #generate_sfx not implemented"
    end

    # @raise [Blazen::UnsupportedError] always (override to implement)
    def clone_voice(_request)
      raise Blazen::UnsupportedError, "#{self.class.name}: #clone_voice not implemented"
    end

    # @raise [Blazen::UnsupportedError] always (override to implement)
    def list_voices
      raise Blazen::UnsupportedError, "#{self.class.name}: #list_voices not implemented"
    end

    # @raise [Blazen::UnsupportedError] always (override to implement)
    def delete_voice(_voice)
      raise Blazen::UnsupportedError, "#{self.class.name}: #delete_voice not implemented"
    end

    # @raise [Blazen::UnsupportedError] always (override to implement)
    def generate_image(_request)
      raise Blazen::UnsupportedError, "#{self.class.name}: #generate_image not implemented"
    end

    # @raise [Blazen::UnsupportedError] always (override to implement)
    def upscale_image(_request)
      raise Blazen::UnsupportedError, "#{self.class.name}: #upscale_image not implemented"
    end

    # @raise [Blazen::UnsupportedError] always (override to implement)
    def text_to_video(_request)
      raise Blazen::UnsupportedError, "#{self.class.name}: #text_to_video not implemented"
    end

    # @raise [Blazen::UnsupportedError] always (override to implement)
    def image_to_video(_request)
      raise Blazen::UnsupportedError, "#{self.class.name}: #image_to_video not implemented"
    end

    # @raise [Blazen::UnsupportedError] always (override to implement)
    def transcribe(_request)
      raise Blazen::UnsupportedError, "#{self.class.name}: #transcribe not implemented"
    end

    # @raise [Blazen::UnsupportedError] always (override to implement)
    def generate_3d(_request)
      raise Blazen::UnsupportedError, "#{self.class.name}: #generate_3d not implemented"
    end

    # @raise [Blazen::UnsupportedError] always (override to implement)
    def remove_background(_request)
      raise Blazen::UnsupportedError, "#{self.class.name}: #remove_background not implemented"
    end
  end

  # ---------------------------------------------------------------------
  # CustomProviderHandle — wraps a live +BlazenCustomProvider *+
  # ---------------------------------------------------------------------

  # Wraps a caller-owned +BlazenCustomProvider *+, exposing the cabi's
  # provider_id / model_id accessors plus the +as_base_provider+ adapter
  # that lets the rest of Blazen plug the handle in wherever a
  # {BaseProvider} is expected.
  #
  # Construct one of these via {Blazen::CustomProvider.from_subclass} (for
  # Ruby-implemented providers) or via {Blazen.ollama} / {Blazen.lm_studio}
  # / {Blazen.openai_compat} (for the cabi's built-in OpenAI-protocol
  # presets).
  class CustomProviderHandle
    # @param ptr [::FFI::Pointer] live +BlazenCustomProvider *+
    def initialize(ptr)
      if ptr.nil? || ptr.null?
        raise ArgumentError, "CustomProviderHandle: ptr must be non-null"
      end

      @ptr = ptr
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    # @api private
    def self.finalizer(ptr)
      proc { Blazen::FFI.blazen_custom_provider_free(ptr) unless ptr.nil? || ptr.null? }
    end

    # @return [String, nil]
    def provider_id
      Blazen::FFI.consume_cstring(Blazen::FFI.blazen_custom_provider_provider_id(@ptr))
    end

    # @return [String, nil]
    def model_id
      Blazen::FFI.consume_cstring(Blazen::FFI.blazen_custom_provider_model_id(@ptr))
    end

    # Wraps the underlying handle as a {BaseProvider} — gives callers
    # access to the system-prompt / tools / response-format / defaults
    # mutators. The returned {BaseProvider} owns its own pointer; this
    # handle remains usable.
    #
    # @return [BaseProvider]
    def as_base_provider
      bp_ptr = Blazen::FFI.blazen_custom_provider_as_base_provider(@ptr)
      if bp_ptr.nil? || bp_ptr.null?
        raise Blazen::InternalError, "blazen_custom_provider_as_base_provider returned null"
      end

      BaseProvider.new(bp_ptr)
    end

    # @return [::FFI::Pointer]
    attr_reader :ptr
    alias_method :to_ptr, :ptr
  end

  # ---------------------------------------------------------------------
  # Module-level factories
  # ---------------------------------------------------------------------

  # CustomProvider preset for an Ollama server.
  #
  # Builds the canonical +http://<host>:<port>/v1+ URL on the cabi side and
  # constructs an OpenAI-protocol +CustomProviderHandle+ with
  # +provider_id = "ollama"+. Pass +host: nil+ to use +"localhost"+;
  # +port: 0+ uses +11434+.
  #
  # @param model [String] required, e.g. +"llama3.1"+
  # @param host  [String, nil] default +"localhost"+
  # @param port  [Integer] default +11434+
  # @return [CustomProviderHandle]
  def self.ollama(model:, host: nil, port: 11_434)
    ptr = Blazen::FFI.with_cstring(model) do |m|
      Blazen::FFI.with_cstring(host) do |h|
        Blazen::FFI.blazen_custom_provider_ollama(m, h, port.to_i)
      end
    end
    if ptr.nil? || ptr.null?
      raise ArgumentError, "blazen_custom_provider_ollama returned null (check model / host / port)"
    end

    CustomProviderHandle.new(ptr)
  end

  # CustomProvider preset for an LM Studio server.
  #
  # @param model [String]
  # @param host  [String, nil] default +"localhost"+
  # @param port  [Integer] default +1234+
  # @return [CustomProviderHandle]
  def self.lm_studio(model:, host: nil, port: 1234)
    ptr = Blazen::FFI.with_cstring(model) do |m|
      Blazen::FFI.with_cstring(host) do |h|
        Blazen::FFI.blazen_custom_provider_lm_studio(m, h, port.to_i)
      end
    end
    if ptr.nil? || ptr.null?
      raise ArgumentError, "blazen_custom_provider_lm_studio returned null (check model / host / port)"
    end

    CustomProviderHandle.new(ptr)
  end

  # Generic OpenAI-protocol CustomProvider.
  #
  # @param provider_id [String] stable identifier (e.g. +"groq"+, +"vllm"+)
  # @param config [OpenAiCompatConfig] config bundle
  # @return [CustomProviderHandle]
  def self.openai_compat(provider_id:, config:)
    unless config.is_a?(OpenAiCompatConfig)
      raise ArgumentError, "config must be a Blazen::OpenAiCompatConfig"
    end

    ptr = Blazen::FFI.with_cstring(provider_id) do |p|
      Blazen::FFI.blazen_custom_provider_openai_compat(p, config.to_ptr)
    end
    if ptr.nil? || ptr.null?
      raise ArgumentError,
            "blazen_custom_provider_openai_compat returned null (check provider_id / config)"
    end

    CustomProviderHandle.new(ptr)
  end
end
