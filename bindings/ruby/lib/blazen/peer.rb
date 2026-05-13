# frozen_string_literal: true

module Blazen
  # Distributed peer-to-peer helpers (workflow remoting between Blazen
  # nodes over gRPC).
  #
  # Requires the native library to have been built with the +distributed+
  # feature — if that feature was omitted, the cabi entry points used here
  # are not present in the loaded library and the wrappers raise
  # {Blazen::UnsupportedError} on construction.
  module Peer
    module_function

    # @return [Boolean] whether the distributed peer surface is available
    def self.available?
      Blazen::FFI.respond_to?(:blazen_peer_server_new) &&
        Blazen::FFI.respond_to?(:blazen_peer_client_connect)
    end

    # Connects to a remote Blazen peer at +address+.
    #
    # @param address [String] gRPC endpoint URI such as
    #   +"http://node-a.local:7443"+
    # @param client_node_id [String] this client's node id
    # @return [Blazen::Peer::PeerClient]
    # @raise [Blazen::UnsupportedError] when the +distributed+ feature is
    #   missing from the loaded native library
    def self.connect(address:, client_node_id:)
      ensure_available!
      out_client = ::FFI::MemoryPointer.new(:pointer)
      out_err    = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(address.to_s) do |addr|
        Blazen::FFI.with_cstring(client_node_id.to_s) do |nid|
          Blazen::FFI.blazen_peer_client_connect(addr, nid, out_client, out_err)
        end
      end
      Blazen::FFI.check_error!(out_err)
      PeerClient.new(out_client.read_pointer)
    end

    # Creates a new {PeerServer} ready to host workflows for remote
    # clients.
    #
    # @param node_id [String] this server's node id
    # @return [Blazen::Peer::PeerServer]
    # @raise [Blazen::UnsupportedError] when the +distributed+ feature is
    #   missing
    def self.server(node_id:)
      ensure_available!
      raw = Blazen::FFI.with_cstring(node_id.to_s) do |nid|
        Blazen::FFI.blazen_peer_server_new(nid)
      end
      if raw.nil? || raw.null?
        raise Blazen::ValidationError, "blazen_peer_server_new rejected node_id=#{node_id.inspect}"
      end

      PeerServer.new(raw)
    end

    def self.ensure_available!
      return if available?

      raise Blazen::UnsupportedError,
            "blazen was built without the 'distributed' feature — peer surface is unavailable"
    end
    private_class_method :ensure_available!

    # Idiomatic Ruby wrapper around a +BlazenPeerClient+ handle.
    class PeerClient
      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "PeerClient: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_peer_client_free))
      end

      # @return [String] the client's node id
      def node_id
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_peer_client_node_id(@ptr))
      end

      # Invokes a workflow on the connected peer asynchronously
      # (integrates with {Fiber.scheduler}).
      #
      # @param workflow_name [String] symbolic workflow name known to the
      #   remote's step registry
      # @param step_ids [Array<String>] step ids to execute; empty array
      #   means "use the workflow's declared steps"
      # @param input_json [String] JSON-encoded entry-step payload
      # @param timeout_secs [Integer, nil] wall-clock bound; +nil+ defers
      #   to the server default
      # @return [Blazen::Workflow::WorkflowResult]
      def run_remote_workflow(workflow_name, step_ids: [], input_json:, timeout_secs: nil)
        ids = Array(step_ids)
        Peer.send(:with_string_array, ids) do |arr_mp, count|
          out_result = ::FFI::MemoryPointer.new(:pointer)
          out_err    = ::FFI::MemoryPointer.new(:pointer)
          timeout    = timeout_secs.nil? ? -1 : Integer(timeout_secs)

          fut = Blazen::FFI.with_cstring(workflow_name.to_s) do |wn|
            Blazen::FFI.with_cstring(input_json.to_s) do |ij|
              Blazen::FFI.blazen_peer_client_run_remote_workflow(
                @ptr, wn, arr_mp, count, ij, timeout,
              )
            end
          end
          if fut.nil? || fut.null?
            raise Blazen::ValidationError, "blazen_peer_client_run_remote_workflow returned a null future"
          end

          Blazen::FFI.await_future(fut) do |f|
            Blazen::FFI.blazen_future_take_workflow_result(f, out_result, out_err)
          end
          Blazen::FFI.check_error!(out_err)
          Peer.send(:wrap_workflow_result, out_result.read_pointer)
        end
      end

      # Blocking-thread variant of {#run_remote_workflow}.
      #
      # @param workflow_name [String]
      # @param step_ids [Array<String>]
      # @param input_json [String]
      # @param timeout_secs [Integer, nil]
      # @return [Blazen::Workflow::WorkflowResult]
      def run_remote_workflow_blocking(workflow_name, step_ids: [], input_json:, timeout_secs: nil)
        ids = Array(step_ids)
        Peer.send(:with_string_array, ids) do |arr_mp, count|
          out_result = ::FFI::MemoryPointer.new(:pointer)
          out_err    = ::FFI::MemoryPointer.new(:pointer)
          timeout    = timeout_secs.nil? ? -1 : Integer(timeout_secs)

          Blazen::FFI.with_cstring(workflow_name.to_s) do |wn|
            Blazen::FFI.with_cstring(input_json.to_s) do |ij|
              Blazen::FFI.blazen_peer_client_run_remote_workflow_blocking(
                @ptr, wn, arr_mp, count, ij, timeout, out_result, out_err,
              )
            end
          end
          Blazen::FFI.check_error!(out_err)
          Peer.send(:wrap_workflow_result, out_result.read_pointer)
        end
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenPeerServer+ handle.
    class PeerServer
      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @param raw_ptr [::FFI::Pointer]
      def initialize(raw_ptr)
        raise ArgumentError, "PeerServer: pointer must be non-null" if raw_ptr.nil? || raw_ptr.null?

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_peer_server_free))
      end

      # Asynchronously binds the gRPC server to +listen_address+ and serves
      # until the listener errors or the future is dropped (integrates
      # with {Fiber.scheduler}).
      #
      # @param listen_address [String] a +"host:port"+ socket addr
      # @return [void]
      def serve(listen_address)
        out_err = ::FFI::MemoryPointer.new(:pointer)
        fut = Blazen::FFI.with_cstring(listen_address.to_s) do |addr|
          Blazen::FFI.blazen_peer_server_serve(@ptr, addr)
        end
        if fut.nil? || fut.null?
          raise Blazen::ValidationError, "blazen_peer_server_serve returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_unit(f, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        nil
      end

      # Blocking-thread variant of {#serve}.
      #
      # @param listen_address [String]
      # @return [void]
      def serve_blocking(listen_address)
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(listen_address.to_s) do |addr|
          Blazen::FFI.blazen_peer_server_serve_blocking(@ptr, addr, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        nil
      end
    end

    # Packs an Array<String> into a +const char *[]+ pointer array and
    # yields +(array_mp, count)+. The yielded pointers and the inner C
    # strings remain valid until the block returns.
    #
    # @api private
    def self.with_string_array(strings)
      if strings.empty?
        return yield(::FFI::Pointer::NULL, 0)
      end

      cstrings = strings.map { |s| ::FFI::MemoryPointer.from_string(s.to_s) }
      array_mp = ::FFI::MemoryPointer.new(:pointer, cstrings.length)
      cstrings.each_with_index do |mp, i|
        array_mp.put_pointer(i * ::FFI::Pointer.size, mp)
      end
      begin
        yield(array_mp, cstrings.length)
      ensure
        # Keep cstrings alive across the call — Ruby's GC won't reclaim
        # them while we hold a reference here.
        cstrings = nil # rubocop:disable Lint/UselessAssignment
      end
    end
    private_class_method :with_string_array

    # Wraps a +BlazenWorkflowResult*+ pointer in the canonical
    # {Blazen::Workflow::WorkflowResult} class when available (Phase R7
    # Agent C), or falls back to an {::FFI::AutoPointer} drop hook so the
    # handle is always released.
    #
    # @api private
    def self.wrap_workflow_result(raw_ptr)
      return nil if raw_ptr.nil? || raw_ptr.null?

      if defined?(Blazen::Workflow) && defined?(Blazen::Workflow::WorkflowResult)
        Blazen::Workflow::WorkflowResult.new(raw_ptr)
      elsif defined?(Blazen::WorkflowResult) &&
            Blazen::WorkflowResult.instance_method(:initialize).arity != 0
        Blazen::WorkflowResult.new(raw_ptr)
      else
        ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_workflow_result_free))
      end
    end
    private_class_method :wrap_workflow_result
  end
end
