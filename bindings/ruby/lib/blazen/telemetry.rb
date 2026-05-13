# frozen_string_literal: true

require "json"

module Blazen
  # Telemetry / observability helpers (Langfuse, OTLP, Prometheus) plus the
  # JSON workflow-history decoder.
  #
  # All three exporter initializers are best-effort: they configure global
  # state inside the native library. Call them once at process start, and be
  # sure to invoke {Blazen.shutdown_telemetry} (alias {Blazen.shutdown}) at
  # process shutdown so buffered spans / metrics are flushed.
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
      out_err = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(public_key.to_s) do |pk|
        Blazen::FFI.with_cstring(secret_key.to_s) do |sk|
          Blazen::FFI.with_cstring(host) do |h|
            Blazen::FFI.blazen_init_langfuse(pk, sk, h, out_err)
          end
        end
      end
      Blazen::FFI.check_error!(out_err)
      nil
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
      out_err = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(endpoint.to_s) do |ep|
        Blazen::FFI.with_cstring(service_name) do |sn|
          Blazen::FFI.blazen_init_otlp(ep, sn, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      nil
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
      out_err = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(listen_address.to_s) do |addr|
        Blazen::FFI.blazen_init_prometheus(addr, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      nil
    end

    # Parses a JSON-encoded +blazen_telemetry::WorkflowHistory+ dump into a
    # flat list of {Blazen::Telemetry::WorkflowHistoryEntry} records.
    #
    # The entire backing array (slice + each element's C handle) is freed
    # before this method returns — every returned
    # {WorkflowHistoryEntry} holds a fully-decoded snapshot of the
    # underlying record, so the native side has no lingering allocations.
    #
    # @param history_json [String] JSON string produced by
    #   +serde_json::to_string(&history)+ on the Rust side
    # @return [Array<Blazen::Telemetry::WorkflowHistoryEntry>]
    def parse_workflow_history(history_json)
      out_array = ::FFI::MemoryPointer.new(:pointer)
      out_count = ::FFI::MemoryPointer.new(:size_t)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(history_json.to_s) do |h|
        Blazen::FFI.blazen_parse_workflow_history(h, out_array, out_count, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      count = out_count.read(:size_t)
      base  = out_array.read_pointer
      return [] if base.nil? || base.null? || count.zero?

      begin
        slot_stride = ::FFI::Pointer.size
        Array.new(count) do |i|
          entry_raw = base.get_pointer(i * slot_stride)
          WorkflowHistoryEntry.from_native(entry_raw)
        end
      ensure
        # Free both the backing slice AND every element handle in one call
        # — accessor results above are already eager Ruby strings/ints.
        Blazen::FFI.blazen_workflow_history_entry_array_free(base, count)
      end
    end

    # Snapshot of a single workflow-history record.
    #
    # All fields are decoded eagerly at construction time and the
    # underlying +BlazenWorkflowHistoryEntry+ handle is released as part of
    # the parent array's batch free, so instances of this class carry no
    # foreign-memory ownership of their own.
    class WorkflowHistoryEntry
      # @return [String] the enclosing run's UUID
      attr_reader :workflow_id

      # @return [String] step name (empty string for workflow-level events)
      attr_reader :step_name

      # @return [String] event variant tag (e.g. +"StepCompleted"+)
      attr_reader :event_type

      # @return [String] full event payload as serde-encoded JSON
      attr_reader :event_data_json

      # @return [Integer] Unix-epoch milliseconds the event was recorded at
      attr_reader :timestamp_ms

      # @return [Integer, nil] event duration in ms, or +nil+ when the
      #   underlying record carried +duration_ms: None+ (e.g.
      #   +WorkflowStarted+ / +StepDispatched+ / +LlmCallStarted+)
      attr_reader :duration_ms

      # @return [String, nil] error message attached to failure events, or
      #   +nil+ for success variants
      attr_reader :error

      # Builds a snapshot directly — primarily useful in tests. Production
      # callers should use {.from_native}.
      #
      # @param workflow_id [String]
      # @param step_name [String]
      # @param event_type [String]
      # @param event_data_json [String]
      # @param timestamp_ms [Integer]
      # @param duration_ms [Integer, nil]
      # @param error [String, nil]
      def initialize(workflow_id:, step_name:, event_type:, event_data_json:,
                     timestamp_ms:, duration_ms: nil, error: nil)
        @workflow_id     = workflow_id
        @step_name       = step_name
        @event_type      = event_type
        @event_data_json = event_data_json
        @timestamp_ms    = timestamp_ms
        @duration_ms     = duration_ms
        @error           = error
      end

      # Decodes a live +BlazenWorkflowHistoryEntry+ handle into a Ruby
      # snapshot. The handle itself is NOT freed here — the surrounding
      # +parse_workflow_history+ batch-frees the whole array afterwards.
      #
      # @param raw_ptr [::FFI::Pointer] non-null entry handle
      # @return [WorkflowHistoryEntry]
      def self.from_native(raw_ptr)
        duration = Blazen::FFI.blazen_workflow_history_entry_duration_ms(raw_ptr)
        err_raw  = Blazen::FFI.blazen_workflow_history_entry_error(raw_ptr)
        new(
          workflow_id:     Blazen::FFI.consume_cstring(
            Blazen::FFI.blazen_workflow_history_entry_workflow_id(raw_ptr),
          ),
          step_name:       Blazen::FFI.consume_cstring(
            Blazen::FFI.blazen_workflow_history_entry_step_name(raw_ptr),
          ),
          event_type:      Blazen::FFI.consume_cstring(
            Blazen::FFI.blazen_workflow_history_entry_event_type(raw_ptr),
          ),
          event_data_json: Blazen::FFI.consume_cstring(
            Blazen::FFI.blazen_workflow_history_entry_event_data_json(raw_ptr),
          ),
          timestamp_ms:    Blazen::FFI.blazen_workflow_history_entry_timestamp_ms(raw_ptr),
          duration_ms:     duration.negative? ? nil : duration,
          error:           err_raw.nil? || err_raw.null? ? nil : Blazen::FFI.consume_cstring(err_raw),
        )
      end

      # @return [Object] decoded JSON value of {#event_data_json}, or +nil+
      #   when the payload is empty
      def event_data
        return nil if event_data_json.nil? || event_data_json.empty?

        JSON.parse(event_data_json)
      end
    end
  end

  # Best-effort flush + shutdown of any initialised telemetry exporters.
  # Safe to call even if no exporter was initialised.
  #
  # @return [void]
  def self.shutdown_telemetry
    Blazen::FFI.blazen_shutdown_telemetry
    nil
  end
end
