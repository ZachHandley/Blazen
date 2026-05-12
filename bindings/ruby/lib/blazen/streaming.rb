# frozen_string_literal: true

module Blazen
  # Streaming-completion support.
  #
  # @note Streaming sinks in Ruby
  #   The underlying UniFFI binding declares {Blazen::CompletionStreamSink}
  #   as a +with_foreign+ trait so that Go/Swift/Kotlin can supply a sink
  #   implemented in their host language. The upstream +uniffi-bindgen+ Ruby
  #   template does **not** currently generate foreign-callback scaffolding,
  #   so Ruby can only consume a {Blazen::CompletionStreamSink} handle that
  #   was produced on the Rust side — it cannot construct one from a Ruby
  #   object. As a result, {Blazen.complete_streaming_blocking} cannot yet
  #   be called from this gem.
  #
  #   Until upstream Ruby bindgen grows callback support, callers wanting
  #   token-by-token streaming from Ruby should either:
  #
  #   * fall back to non-streaming {Blazen::CompletionModel#complete_blocking}
  #     (gets the full response in one call), or
  #   * use the Node or Python bindings, which have first-class streaming.
  module Streaming
    # Raised when a caller tries to use streaming from Ruby on a build of
    # this gem that doesn't have foreign-callback scaffolding wired up.
    class Unsupported < Blazen::UnsupportedError; end

    module_function

    # Placeholder: streams a completion. Currently raises {Unsupported}
    # because the upstream UniFFI Ruby bindgen does not generate foreign
    # callback scaffolding (see module docs).
    #
    # @param _model [Blazen::CompletionModel]
    # @param _request [Blazen::CompletionRequest]
    # @yieldparam chunk [Blazen::StreamChunk]
    # @raise [Unsupported]
    def complete(_model, _request, &_block)
      raise Unsupported,
            "streaming completion sinks are not yet supported from Ruby; the " \
            "upstream uniffi-bindgen ruby template does not generate " \
            "foreign-callback scaffolding. Use #complete_blocking on the " \
            "CompletionModel for the full response, or switch to the Node " \
            "or Python bindings for token-by-token streaming."
    end
  end
end
