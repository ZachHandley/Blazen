# frozen_string_literal: true

require_relative "controlplane/client"
require_relative "controlplane/worker"

module Blazen
  # Distributed control-plane bindings (central-server topology for
  # orchestrating workflow runs across a fleet of workers).
  #
  # Unlike {Blazen::Peer}, which models a flat mesh of peer processes
  # that dial each other directly, this module talks to a central
  # +BlazenControlPlane+ server. Orchestrators construct a
  # {Blazen::ControlPlane::Client} and submit / cancel / observe
  # workflow runs; workers construct a {Blazen::ControlPlane::Worker}
  # and run assignments dispatched by the server.
  #
  # Requires the native library to have been built with the
  # +distributed+ feature — if that feature was omitted, the cabi entry
  # points used here are not present in the loaded library and the
  # wrappers raise {Blazen::UnsupportedError} on construction.
  module ControlPlane
    module_function

    # Run-status integer codes returned by
    # {Blazen::ControlPlane.snapshot_to_hash}. Mirror the
    # +BLAZEN_RUN_STATUS_*+ constants in +blazen.h+.
    PENDING   = 0
    RUNNING   = 1
    COMPLETED = 2
    FAILED    = 3
    CANCELLED = 4

    # @return [Boolean] whether the distributed control-plane surface
    #   is available in the loaded native library.
    def self.available?
      Blazen::FFI.respond_to?(:blazen_controlplane_client_connect) &&
        Blazen::FFI.respond_to?(:blazen_controlplane_worker_new_blocking)
    end

    def self.ensure_available!
      return if available?

      raise Blazen::UnsupportedError,
            "blazen was built without the 'distributed' feature — control plane is unavailable"
    end

    # Convert a status integer into a symbol used by the snapshot/event
    # serialisation helpers below.
    #
    # @param status [Integer]
    # @return [Symbol]
    def self.status_symbol(status)
      case status
      when PENDING   then :pending
      when RUNNING   then :running
      when COMPLETED then :completed
      when FAILED    then :failed
      when CANCELLED then :cancelled
      else                :unknown
      end
    end

    # Read every accessor on a {BlazenRunStateSnapshot} pointer and
    # return a Ruby hash. Frees the underlying handle.
    #
    # @param ptr [::FFI::Pointer] caller-owned snapshot pointer
    # @return [Hash]
    def self.consume_snapshot(ptr)
      return nil if ptr.nil? || ptr.null?

      run_id        = Blazen::FFI.consume_cstring(Blazen::FFI.blazen_run_state_snapshot_run_id(ptr))
      status_int    = Blazen::FFI.blazen_run_state_snapshot_status(ptr)
      started_at_ms = Blazen::FFI.blazen_run_state_snapshot_started_at_ms(ptr)
      assigned_to   = Blazen::FFI.consume_cstring(Blazen::FFI.blazen_run_state_snapshot_assigned_to(ptr))
      output_json   = Blazen::FFI.consume_cstring(Blazen::FFI.blazen_run_state_snapshot_output_json(ptr))
      error_msg     = Blazen::FFI.consume_cstring(Blazen::FFI.blazen_run_state_snapshot_error(ptr))

      completed_at_ms = read_optional_u64(
        ptr, :blazen_run_state_snapshot_completed_at_ms
      )
      last_event_at_ms = read_optional_u64(
        ptr, :blazen_run_state_snapshot_last_event_at_ms
      )

      output = output_json && JSON.parse(output_json)

      Blazen::FFI.blazen_run_state_snapshot_free(ptr)

      {
        run_id:           run_id,
        status:           status_symbol(status_int),
        started_at_ms:    started_at_ms,
        completed_at_ms:  completed_at_ms,
        assigned_to:      assigned_to,
        last_event_at_ms: last_event_at_ms,
        output:           output,
        error:            error_msg,
      }
    end

    # Read every accessor on a {BlazenWorkerInfo} pointer and return a
    # Ruby hash. Frees the underlying handle.
    #
    # @param ptr [::FFI::Pointer] caller-owned worker-info pointer
    # @return [Hash]
    def self.consume_worker_info(ptr)
      return nil if ptr.nil? || ptr.null?

      node_id           = Blazen::FFI.consume_cstring(Blazen::FFI.blazen_worker_info_node_id(ptr))
      capabilities_json = Blazen::FFI.consume_cstring(Blazen::FFI.blazen_worker_info_capabilities_json(ptr))
      tags_json         = Blazen::FFI.consume_cstring(Blazen::FFI.blazen_worker_info_tags_json(ptr))
      admission_json    = Blazen::FFI.consume_cstring(Blazen::FFI.blazen_worker_info_admission_json(ptr))
      in_flight         = Blazen::FFI.blazen_worker_info_in_flight(ptr)
      connected_at_ms   = Blazen::FFI.blazen_worker_info_connected_at_ms(ptr)

      Blazen::FFI.blazen_worker_info_free(ptr)

      {
        node_id:         node_id,
        capabilities:    capabilities_json ? JSON.parse(capabilities_json) : [],
        tags:            tags_json ? JSON.parse(tags_json) : {},
        admission:       admission_json ? JSON.parse(admission_json) : nil,
        in_flight:       in_flight,
        connected_at_ms: connected_at_ms,
      }
    end

    # Drain a {BlazenWorkerInfoList} into an array of hashes, freeing
    # both the per-entry handles and the list.
    #
    # @api private
    # @param list_ptr [::FFI::Pointer]
    # @return [Array<Hash>]
    def self.consume_worker_info_list(list_ptr)
      return [] if list_ptr.nil? || list_ptr.null?

      count = Blazen::FFI.blazen_worker_info_list_count(list_ptr)
      out = Array.new(count) do |i|
        entry = Blazen::FFI.blazen_worker_info_list_get(list_ptr, i)
        consume_worker_info(entry)
      end
      Blazen::FFI.blazen_worker_info_list_free(list_ptr)
      out
    end

    # Read an optional u64 out-param pair (+ms+, +has_value+) and
    # return either the value or +nil+.
    #
    # @api private
    def self.read_optional_u64(ptr, method_name)
      ms_slot   = ::FFI::MemoryPointer.new(:uint64)
      flag_slot = ::FFI::MemoryPointer.new(:int32)
      rc = Blazen::FFI.public_send(method_name, ptr, ms_slot, flag_slot)
      return nil if rc != 0

      flag_slot.read_int32 == 1 ? ms_slot.read_uint64 : nil
    end
  end
end

require "json"
