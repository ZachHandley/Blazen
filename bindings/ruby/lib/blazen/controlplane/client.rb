# frozen_string_literal: true

require "json"

module Blazen
  module ControlPlane
    # Orchestrator-side handle for the Blazen control plane.
    #
    # @example
    #   client = Blazen::ControlPlane::Client.connect("http://cp.example.com:7445")
    #   snap = client.submit_workflow(
    #     workflow_name: "summarize",
    #     input: { text: "..." },
    #     required_tags: ["region=us-west"],
    #   )
    #   puts snap[:status]   # => :pending or :running
    class Client
      # @return [::FFI::AutoPointer] underlying +BlazenControlPlaneClient+ pointer
      attr_reader :ptr

      # Opens a connection to the control plane asynchronously, integrating
      # with {Fiber.scheduler} when one is active.
      #
      # @param endpoint [String] gRPC endpoint URI such as
      #   +"http://cp.example.com:7445"+ or +"https://cp.example.com"+
      # @return [Client]
      # @raise [Blazen::Error] when the endpoint URI is invalid or the
      #   TCP/HTTP-2 handshake fails
      # @raise [Blazen::UnsupportedError] when the +distributed+ feature
      #   is missing from the loaded native library
      def self.connect(endpoint)
        ControlPlane.ensure_available!
        fut = Blazen::FFI.with_cstring(endpoint.to_s) do |ep|
          Blazen::FFI.blazen_controlplane_client_connect(ep)
        end
        if fut.nil? || fut.null?
          raise Blazen::ValidationError, "blazen_controlplane_client_connect returned a null future"
        end

        out_client = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_controlplane_client(f, out_client, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        new(out_client.read_pointer)
      end

      # Blocking-thread variant of {.connect}.
      #
      # @param endpoint [String]
      # @return [Client]
      def self.connect_blocking(endpoint)
        ControlPlane.ensure_available!
        out_client = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(endpoint.to_s) do |ep|
          Blazen::FFI.blazen_controlplane_client_connect_blocking(ep, out_client, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        new(out_client.read_pointer)
      end

      # Opens an mTLS connection to the control plane asynchronously,
      # loading the client identity + CA bundle from PEM files on disk.
      #
      # @param endpoint [String] gRPC endpoint URI such as
      #   +"https://cp.example.com:7445"+
      # @param cert_path [String] path to the PEM-encoded client cert
      # @param key_path [String] path to the PEM-encoded client private key
      # @param ca_path [String] path to the PEM-encoded CA bundle
      # @return [Client]
      # @raise [Blazen::Error] when any PEM file is missing / unparseable,
      #   the TLS config is rejected, or the handshake fails
      def self.connect_with_mtls(endpoint, cert_path:, key_path:, ca_path:)
        ControlPlane.ensure_available!
        fut = Blazen::FFI.with_cstring(endpoint.to_s) do |ep|
          Blazen::FFI.with_cstring(cert_path.to_s) do |cp|
            Blazen::FFI.with_cstring(key_path.to_s) do |kp|
              Blazen::FFI.with_cstring(ca_path.to_s) do |ca|
                Blazen::FFI.blazen_controlplane_client_connect_with_mtls(ep, cp, kp, ca)
              end
            end
          end
        end
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_controlplane_client_connect_with_mtls returned a null future"
        end

        out_client = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_controlplane_client(f, out_client, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        new(out_client.read_pointer)
      end

      # Blocking-thread variant of {.connect_with_mtls}.
      #
      # @param endpoint [String]
      # @param cert_path [String]
      # @param key_path [String]
      # @param ca_path [String]
      # @return [Client]
      def self.connect_with_mtls_blocking(endpoint, cert_path:, key_path:, ca_path:)
        ControlPlane.ensure_available!
        out_client = ::FFI::MemoryPointer.new(:pointer)
        out_err    = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(endpoint.to_s) do |ep|
          Blazen::FFI.with_cstring(cert_path.to_s) do |cp|
            Blazen::FFI.with_cstring(key_path.to_s) do |kp|
              Blazen::FFI.with_cstring(ca_path.to_s) do |ca|
                Blazen::FFI.blazen_controlplane_client_connect_with_mtls_blocking(
                  ep, cp, kp, ca, out_client, out_err,
                )
              end
            end
          end
        end
        Blazen::FFI.check_error!(out_err)
        new(out_client.read_pointer)
      end

      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "Client: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(
          raw_ptr, Blazen::FFI.method(:blazen_controlplane_client_free),
        )
      end

      # Submit a new workflow run.
      #
      # @param workflow_name [String]
      # @param input [Object, nil] JSON-serialisable payload for the
      #   workflow's first step
      # @param required_tags [Array<String>] +"key=value"+ tag
      #   predicates a matching worker must advertise
      # @param wait_for_worker [Boolean] if +true+ (default), the
      #   control plane queues the run when no worker matches at
      #   submit time; if +false+ it rejects with +FailedPrecondition+
      # @return [Hash] the initial {RunStateSnapshot} as a hash
      def submit_workflow(workflow_name:, input: nil, required_tags: [], wait_for_worker: true)
        input_json = JSON.generate(input.nil? ? nil : input)
        tags_json  = JSON.generate(Array(required_tags))
        fut = Blazen::FFI.with_cstring(workflow_name.to_s) do |wn|
          Blazen::FFI.with_cstring(input_json) do |ij|
            Blazen::FFI.with_cstring(tags_json) do |tj|
              Blazen::FFI.blazen_controlplane_client_submit_workflow(
                @ptr, wn, ij, tj, wait_for_worker,
              )
            end
          end
        end
        await_snapshot(fut)
      end

      # Cancel an in-flight run.
      #
      # @param run_id [String] hyphenated UUID
      # @return [Hash] the post-cancel {RunStateSnapshot} as a hash
      def cancel_workflow(run_id)
        fut = Blazen::FFI.with_cstring(run_id.to_s) do |id|
          Blazen::FFI.blazen_controlplane_client_cancel_workflow(@ptr, id)
        end
        await_snapshot(fut)
      end

      # Look up the current state of a run.
      #
      # @param run_id [String] hyphenated UUID
      # @return [Hash] {RunStateSnapshot} as a hash
      def describe_workflow(run_id)
        fut = Blazen::FFI.with_cstring(run_id.to_s) do |id|
          Blazen::FFI.blazen_controlplane_client_describe_workflow(@ptr, id)
        end
        await_snapshot(fut)
      end

      # List currently-connected workers.
      #
      # @return [Array<Hash>] one {WorkerInfo} hash per worker
      def list_workers
        fut = Blazen::FFI.blazen_controlplane_client_list_workers(@ptr)
        if fut.nil? || fut.null?
          raise Blazen::ValidationError, "blazen_controlplane_client_list_workers returned a null future"
        end

        out_list = ::FFI::MemoryPointer.new(:pointer)
        out_err  = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_worker_info_list(f, out_list, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        ControlPlane.consume_worker_info_list(out_list.read_pointer)
      end

      # Subscribe to the event stream for a single run. Yields each
      # event as a Hash to the supplied block:
      #
      #   { run_id: "<uuid>", event_type: "<str>",
      #     data: <decoded JSON>, timestamp_ms: <Integer> }
      #
      # The block runs on a worker thread the cabi schedules
      # callbacks on (so any shared state must be thread-safe).
      # Returns a {Subscription} handle whose +#cancel+ stops the
      # stream and whose +#close+ (or finalizer) frees the underlying
      # cabi resources.
      #
      # @param run_id [String] hyphenated UUID
      # @yield [Hash] each decoded event
      # @return [Subscription]
      # @raise [ArgumentError] when no block is given
      # @raise [Blazen::Error] when the subscribe RPC fails
      def subscribe_run_events(run_id, &block)
        raise ArgumentError, "subscribe_run_events requires a block" if block.nil?

        sink_state = build_run_event_sink(block)
        out_sub = ::FFI::MemoryPointer.new(:pointer)
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(run_id.to_s) do |id|
          Blazen::FFI.blazen_controlplane_client_subscribe_run_events(
            @ptr, id, sink_state[:vtable], out_sub, out_err,
          )
        end
        Blazen::FFI.check_error!(out_err)
        Subscription.new(out_sub.read_pointer, sink_state)
      end

      # Subscribe to the fan-out event stream across all runs,
      # optionally filtered by tag predicates. Same callback / return
      # contract as {#subscribe_run_events}.
      #
      # @param required_tags [Array<String>] +"key=value"+ tag
      #   predicates a matching run must advertise (empty = no filter)
      # @yield [Hash] each decoded event
      # @return [Subscription]
      # @raise [ArgumentError] when no block is given
      # @raise [Blazen::Error] when the subscribe RPC fails
      def subscribe_all(required_tags: [], &block)
        raise ArgumentError, "subscribe_all requires a block" if block.nil?

        tags_json = JSON.generate(Array(required_tags).map(&:to_s))
        sink_state = build_run_event_sink(block)
        out_sub = ::FFI::MemoryPointer.new(:pointer)
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(tags_json) do |tj|
          Blazen::FFI.blazen_controlplane_client_subscribe_all(
            @ptr, tj, sink_state[:vtable], out_sub, out_err,
          )
        end
        Blazen::FFI.check_error!(out_err)
        Subscription.new(out_sub.read_pointer, sink_state)
      end

      # Send a drain instruction to the named worker.
      #
      # @param node_id [String]
      # @param immediate [Boolean] +true+ to abort in-flight assignments;
      #   +false+ to wait for them to complete
      # @return [void]
      def drain_worker(node_id, immediate: false)
        fut = Blazen::FFI.with_cstring(node_id.to_s) do |nid|
          Blazen::FFI.blazen_controlplane_client_drain_worker(@ptr, nid, immediate)
        end
        if fut.nil? || fut.null?
          raise Blazen::ValidationError, "blazen_controlplane_client_drain_worker returned a null future"
        end

        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_unit(f, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        nil
      end

      private

      # Drives a snapshot-returning future to completion and returns the
      # parsed snapshot hash.
      def await_snapshot(fut)
        if fut.nil? || fut.null?
          raise Blazen::ValidationError, "control-plane RPC returned a null future"
        end

        out_snap = ::FFI::MemoryPointer.new(:pointer)
        out_err  = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_run_state_snapshot(f, out_snap, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        ControlPlane.consume_snapshot(out_snap.read_pointer)
      end

      # Build the +BlazenRunEventSinkVTable+ + persistent
      # +FFI::Function+ wrappers for a single subscription. Returns a
      # state Hash that the caller MUST hand off to the
      # {Subscription} so the callbacks aren't GC'd while the C side
      # still references them.
      def build_run_event_sink(block)
        # The callbacks share state via these procs. We hold strong
        # references to each FFI::Function until the subscription is
        # freed (mirrors the worker vtable pattern).
        drop_user_data = ::FFI::Function.new(:void, [:pointer]) { |_| }

        on_event = ::FFI::Function.new(
          :void,
          [:pointer, :pointer, :pointer, :pointer, :uint64],
        ) do |_user_data, run_id_ptr, event_type_ptr, data_json_ptr, timestamp_ms|
          dispatch_run_event(block, run_id_ptr, event_type_ptr, data_json_ptr, timestamp_ms)
        end

        on_close = ::FFI::Function.new(:void, [:pointer]) { |_| nil }
        on_error = ::FFI::Function.new(:void, [:pointer, :pointer]) { |_, _| nil }

        vtable = Blazen::FFI::BlazenRunEventSinkVTable.new
        vtable[:user_data]      = ::FFI::Pointer::NULL
        vtable[:drop_user_data] = drop_user_data
        vtable[:on_event]       = on_event
        vtable[:on_close]       = on_close
        vtable[:on_error]       = on_error

        {
          vtable:         vtable,
          drop_user_data: drop_user_data,
          on_event:       on_event,
          on_close:       on_close,
          on_error:       on_error,
        }
      end

      # Decode a single +on_event+ callback and yield to the
      # user-supplied block.
      def dispatch_run_event(block, run_id_ptr, event_type_ptr, data_json_ptr, timestamp_ms)
        run_id     = run_id_ptr.null? ? nil : run_id_ptr.read_string.force_encoding(Encoding::UTF_8)
        event_type = event_type_ptr.null? ? nil : event_type_ptr.read_string.force_encoding(Encoding::UTF_8)
        data_raw   = data_json_ptr.null? ? "null" : data_json_ptr.read_string.force_encoding(Encoding::UTF_8)
        data       = JSON.parse(data_raw)
        block.call(
          run_id:       run_id,
          event_type:   event_type,
          data:         data,
          timestamp_ms: timestamp_ms,
        )
      rescue StandardError
        # Swallow user-block errors so a bad consumer doesn't poison
        # the entire stream. Foreign callbacks must not unwind across
        # the FFI boundary.
        nil
      end
    end

    # Handle for an open control-plane subscription. Keeps the
    # +FFI::Function+ callbacks alive on the Ruby side so they aren't
    # GC'd while the cabi pump task still holds references to them,
    # and exposes +#cancel+ + +#close+ for explicit teardown.
    #
    # Construct via {Client#subscribe_run_events} or
    # {Client#subscribe_all}; subscriptions auto-free on Ruby GC.
    class Subscription
      # @param raw_ptr [::FFI::Pointer] cabi-owned
      #   +BlazenControlPlaneSubscription*+
      # @param sink_state [Hash] callback-keepalive state from
      #   +Client#build_run_event_sink+
      def initialize(raw_ptr, sink_state)
        if raw_ptr.nil? || raw_ptr.null?
          raise ArgumentError, "Subscription: pointer must be non-null"
        end

        @sink_state = sink_state
        @ptr = ::FFI::AutoPointer.new(
          raw_ptr, Blazen::FFI.method(:blazen_controlplane_subscription_free),
        )
      end

      # Cancel the subscription. The cabi pump task observes the
      # cancellation on its next stream poll, exits without firing
      # terminal callbacks, and releases the vtable's +user_data+.
      # Idempotent.
      #
      # @return [void]
      def cancel
        ptr = @ptr
        return if ptr.nil? || ptr.null?

        Blazen::FFI.blazen_controlplane_subscription_cancel(ptr)
        nil
      end

      # Explicitly tear down the subscription. Equivalent to calling
      # {#cancel} followed by releasing the underlying handle.
      # Subsequent calls are no-ops.
      #
      # @return [void]
      def close
        cancel
        @ptr&.free
        @ptr = nil
        @sink_state = nil
      end
    end
  end
end
