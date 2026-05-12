# frozen_string_literal: true

module Blazen
  # Telemetry / observability helpers (Langfuse, OTLP, Prometheus).
  #
  # All three exporters are best-effort initializers: they configure global
  # state inside the native library. Call them once at process start, and
  # be sure to invoke {Blazen.shutdown_telemetry} (alias {Blazen.shutdown})
  # at shutdown so buffered spans / metrics are flushed.
  module Telemetry
    module_function

    # Initializes the Langfuse exporter.
    #
    # Requires the native library to have been built with the +langfuse+
    # feature; otherwise raises {Blazen::UnsupportedError}.
    #
    # @param public_key [String] Langfuse project public key
    # @param secret_key [String] Langfuse project secret key
    # @param host [String, nil] Langfuse host (defaults to the SaaS endpoint)
    # @return [void]
    def init_langfuse(public_key:, secret_key:, host: nil)
      Blazen.translate_errors { Blazen.init_langfuse(public_key, secret_key, host) }
    end

    # Initializes the OTLP (OpenTelemetry) exporter.
    #
    # Requires the native library to have been built with the +otlp+
    # feature; otherwise raises {Blazen::UnsupportedError}.
    #
    # @param endpoint [String] OTLP collector endpoint
    # @param service_name [String, nil] service name to tag spans with
    # @return [void]
    def init_otlp(endpoint:, service_name: nil)
      Blazen.translate_errors { Blazen.init_otlp(endpoint, service_name) }
    end

    # Initializes the Prometheus metrics exporter.
    #
    # Requires the native library to have been built with the +prometheus+
    # feature; otherwise raises {Blazen::UnsupportedError}.
    #
    # @param listen_address [String] +"host:port"+ to bind the metrics
    #   endpoint on (e.g. +"0.0.0.0:9090"+)
    # @return [void]
    def init_prometheus(listen_address)
      Blazen.translate_errors { Blazen.init_prometheus(listen_address) }
    end

    # Parses a JSON-encoded workflow-history dump into
    # {Blazen::WorkflowHistoryEntry} records.
    #
    # @param history_json [String] JSON string emitted by the native lib
    # @return [Array<Blazen::WorkflowHistoryEntry>]
    def parse_workflow_history(history_json)
      Blazen.translate_errors { Blazen.parse_workflow_history(history_json) }
    end
  end
end
