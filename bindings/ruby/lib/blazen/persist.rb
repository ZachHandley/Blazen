# frozen_string_literal: true

module Blazen
  # Persistence (workflow checkpointing) helpers.
  #
  # Provides two factory constructors for the built-in backends —
  # {redb} (embedded file-backed K/V store) and {valkey}
  # (Valkey/Redis-compatible) — plus rich Ruby wrappers for the
  # {CheckpointStore}, {WorkflowCheckpoint}, and {PersistedEvent} types.
  #
  # Every async store method (save / load / delete / list / list_run_ids) is
  # exposed in two flavours:
  #
  # - the default name (e.g. {CheckpointStore#save}) integrates with
  #   {Fiber.scheduler} (works with the +async+ gem), and
  # - the +_blocking+ suffix variant (e.g. {CheckpointStore#save_blocking})
  #   blocks the calling thread on the cabi tokio runtime.
  module Persist
    module_function

    # Opens (or creates) a redb-backed checkpoint store at +path+.
    #
    # @param path [String] filesystem path to the redb database file
    # @return [Blazen::Persist::CheckpointStore]
    def redb(path)
      out_store = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(path.to_s) do |p|
        Blazen::FFI.blazen_checkpoint_store_new_redb(p, out_store, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      CheckpointStore.new(out_store.read_pointer)
    end

    # Connects to a Valkey/Redis server for checkpoint storage.
    #
    # @param url [String] Valkey/Redis connection URL
    #   (e.g. +"redis://127.0.0.1:6379"+)
    # @param ttl_seconds [Integer, nil] TTL in seconds for stored
    #   checkpoints — pass +nil+ for no TTL
    # @return [Blazen::Persist::CheckpointStore]
    def valkey(url:, ttl_seconds: nil)
      ttl = ttl_seconds.nil? ? -1 : Integer(ttl_seconds)
      out_store = ::FFI::MemoryPointer.new(:pointer)
      out_err   = ::FFI::MemoryPointer.new(:pointer)
      Blazen::FFI.with_cstring(url.to_s) do |u|
        Blazen::FFI.blazen_checkpoint_store_new_valkey(u, ttl, out_store, out_err)
      end
      Blazen::FFI.check_error!(out_err)
      CheckpointStore.new(out_store.read_pointer)
    end

    # Idiomatic Ruby wrapper around a +BlazenCheckpointStore+ handle.
    #
    # Each instance owns the underlying foreign handle via
    # {::FFI::AutoPointer} — GC releases the C-side allocation
    # automatically. All async methods are available in two variants:
    # the default name integrates with {Fiber.scheduler}, and the
    # +_blocking+ suffix variant calls the matching +_blocking+ cabi entry
    # point directly (always thread-blocking).
    class CheckpointStore
      # @return [::FFI::AutoPointer] wrapped handle
      attr_reader :ptr

      # @param raw_ptr [::FFI::Pointer] live store handle (caller-owned)
      def initialize(raw_ptr)
        @ptr = ::FFI::AutoPointer.new(
          raw_ptr,
          Blazen::FFI.method(:blazen_checkpoint_store_free),
        )
      end

      # Persists +checkpoint+ — the checkpoint handle is **consumed** and
      # MUST NOT be referenced again by the caller.
      #
      # @param checkpoint [Blazen::Persist::WorkflowCheckpoint]
      # @return [void]
      def save(checkpoint)
        raw = checkpoint.consume!
        out_err = ::FFI::MemoryPointer.new(:pointer)
        fut = Blazen::FFI.blazen_checkpoint_store_save(@ptr, raw)
        if fut.nil? || fut.null?
          raise Blazen::PersistError, "blazen_checkpoint_store_save returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_unit(f, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        nil
      end

      # Blocking-thread variant of {#save}.
      #
      # @param checkpoint [Blazen::Persist::WorkflowCheckpoint]
      # @return [void]
      def save_blocking(checkpoint)
        raw = checkpoint.consume!
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_checkpoint_store_save_blocking(@ptr, raw, out_err)
        Blazen::FFI.check_error!(out_err)
        nil
      end

      # Loads a checkpoint by +run_id+.
      #
      # @param run_id [String] UUID of the run to fetch
      # @return [Blazen::Persist::WorkflowCheckpoint, nil] +nil+ when no
      #   checkpoint exists for +run_id+
      def load(run_id)
        out_ckpt  = ::FFI::MemoryPointer.new(:pointer)
        out_found = ::FFI::MemoryPointer.new(:int32)
        out_err   = ::FFI::MemoryPointer.new(:pointer)
        fut = Blazen::FFI.with_cstring(run_id.to_s) do |r|
          Blazen::FFI.blazen_checkpoint_store_load(@ptr, r)
        end
        if fut.nil? || fut.null?
          raise Blazen::PersistError, "blazen_checkpoint_store_load returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_workflow_checkpoint_option(f, out_ckpt, out_found, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        return nil if out_found.read(:int32).zero?

        WorkflowCheckpoint.new(out_ckpt.read_pointer)
      end

      # Blocking-thread variant of {#load}.
      #
      # @param run_id [String]
      # @return [Blazen::Persist::WorkflowCheckpoint, nil]
      def load_blocking(run_id)
        out_ckpt  = ::FFI::MemoryPointer.new(:pointer)
        out_found = ::FFI::MemoryPointer.new(:int32)
        out_err   = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(run_id.to_s) do |r|
          Blazen::FFI.blazen_checkpoint_store_load_blocking(@ptr, r, out_ckpt, out_found, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        return nil if out_found.read(:int32).zero?

        WorkflowCheckpoint.new(out_ckpt.read_pointer)
      end

      # Deletes the checkpoint for +run_id+. Delete-of-missing is a no-op
      # on every backend, so this never raises +PersistError+ for a missing
      # key.
      #
      # @param run_id [String]
      # @return [void]
      def delete(run_id)
        out_err = ::FFI::MemoryPointer.new(:pointer)
        fut = Blazen::FFI.with_cstring(run_id.to_s) do |r|
          Blazen::FFI.blazen_checkpoint_store_delete(@ptr, r)
        end
        if fut.nil? || fut.null?
          raise Blazen::PersistError, "blazen_checkpoint_store_delete returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_unit(f, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        nil
      end

      # Blocking-thread variant of {#delete}.
      #
      # @param run_id [String]
      # @return [void]
      def delete_blocking(run_id)
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(run_id.to_s) do |r|
          Blazen::FFI.blazen_checkpoint_store_delete_blocking(@ptr, r, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        nil
      end

      # Lists every stored checkpoint, ordered by timestamp descending.
      #
      # @return [Array<Blazen::Persist::WorkflowCheckpoint>]
      def list
        out_array = ::FFI::MemoryPointer.new(:pointer)
        out_count = ::FFI::MemoryPointer.new(:size_t)
        out_err   = ::FFI::MemoryPointer.new(:pointer)
        fut = Blazen::FFI.blazen_checkpoint_store_list(@ptr)
        if fut.nil? || fut.null?
          raise Blazen::PersistError, "blazen_checkpoint_store_list returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_workflow_checkpoint_list(f, out_array, out_count, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        Persist.send(:consume_checkpoint_array, out_array.read_pointer, out_count.read(:size_t))
      end

      # Blocking-thread variant of {#list}.
      #
      # @return [Array<Blazen::Persist::WorkflowCheckpoint>]
      def list_blocking
        out_array = ::FFI::MemoryPointer.new(:pointer)
        out_count = ::FFI::MemoryPointer.new(:size_t)
        out_err   = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_checkpoint_store_list_blocking(@ptr, out_array, out_count, out_err)
        Blazen::FFI.check_error!(out_err)
        Persist.send(:consume_checkpoint_array, out_array.read_pointer, out_count.read(:size_t))
      end

      # Lists every stored run id, ordered by timestamp descending.
      #
      # @return [Array<String>] run-id UUIDs
      def list_run_ids
        out_array = ::FFI::MemoryPointer.new(:pointer)
        out_count = ::FFI::MemoryPointer.new(:size_t)
        out_err   = ::FFI::MemoryPointer.new(:pointer)
        fut = Blazen::FFI.blazen_checkpoint_store_list_run_ids(@ptr)
        if fut.nil? || fut.null?
          raise Blazen::PersistError, "blazen_checkpoint_store_list_run_ids returned a null future"
        end

        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_string_list(f, out_array, out_count, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        Persist.send(:consume_string_array, out_array.read_pointer, out_count.read(:size_t))
      end

      # Blocking-thread variant of {#list_run_ids}.
      #
      # @return [Array<String>]
      def list_run_ids_blocking
        out_array = ::FFI::MemoryPointer.new(:pointer)
        out_count = ::FFI::MemoryPointer.new(:size_t)
        out_err   = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_checkpoint_store_list_run_ids_blocking(@ptr, out_array, out_count, out_err)
        Blazen::FFI.check_error!(out_err)
        Persist.send(:consume_string_array, out_array.read_pointer, out_count.read(:size_t))
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenWorkflowCheckpoint+ handle.
    #
    # Mutation helpers (+pending_events_push+) follow the cabi's
    # consume-on-push semantics; once the inner handle is passed to
    # {CheckpointStore#save} via {#consume!} the wrapper is empty and
    # any further access raises +Blazen::Error+.
    class WorkflowCheckpoint
      # Constructs a new checkpoint. All four string fields default to the
      # empty string for ergonomic Ruby callers.
      #
      # When called with a single +raw_ptr+ positional argument the wrapper
      # adopts that pre-existing handle instead of allocating a fresh one
      # — used by accessors that return cabi-allocated handles.
      #
      # @overload initialize(raw_ptr)
      #   @param raw_ptr [::FFI::Pointer] live BlazenWorkflowCheckpoint
      # @overload initialize(workflow_name:, run_id:, state_json:, metadata_json:, timestamp_ms:)
      #   @param workflow_name [String]
      #   @param run_id [String] UUID; empty string lets the persist layer
      #     pick one
      #   @param state_json [String] JSON-encoded state (empty for "no state")
      #   @param metadata_json [String] JSON-encoded metadata (empty for "no
      #     metadata")
      #   @param timestamp_ms [Integer] Unix-epoch milliseconds
      def initialize(*args, **kwargs)
        if !args.empty?
          raw = args.first
          raise ArgumentError, "WorkflowCheckpoint.new(raw_ptr): pointer must be non-null" if raw.nil? || raw.null?

          @ptr = ::FFI::AutoPointer.new(raw, Blazen::FFI.method(:blazen_workflow_checkpoint_free))
          return
        end

        workflow_name = (kwargs[:workflow_name] || "").to_s
        run_id        = (kwargs[:run_id] || "").to_s
        state_json    = (kwargs[:state_json] || "").to_s
        metadata_json = (kwargs[:metadata_json] || "").to_s
        timestamp_ms  = Integer(kwargs.fetch(:timestamp_ms, 0))

        raw = Blazen::FFI.with_cstring(workflow_name) do |wn|
          Blazen::FFI.with_cstring(run_id) do |rid|
            Blazen::FFI.with_cstring(state_json) do |st|
              Blazen::FFI.with_cstring(metadata_json) do |md|
                Blazen::FFI.blazen_workflow_checkpoint_new(wn, rid, st, md, timestamp_ms)
              end
            end
          end
        end
        raise Blazen::ValidationError, "blazen_workflow_checkpoint_new rejected its inputs" if raw.nil? || raw.null?

        @ptr = ::FFI::AutoPointer.new(raw, Blazen::FFI.method(:blazen_workflow_checkpoint_free))
      end

      # @return [::FFI::AutoPointer, nil] the live underlying pointer, or
      #   +nil+ after {#consume!}
      def ptr
        ensure_live!
        @ptr
      end

      # @return [String]
      def workflow_name
        ensure_live!
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_workflow_checkpoint_workflow_name(@ptr))
      end

      # @return [String]
      def run_id
        ensure_live!
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_workflow_checkpoint_run_id(@ptr))
      end

      # @return [Integer]
      def timestamp_ms
        ensure_live!
        Blazen::FFI.blazen_workflow_checkpoint_timestamp_ms(@ptr)
      end

      # @return [String]
      def state_json
        ensure_live!
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_workflow_checkpoint_state_json(@ptr))
      end

      # @return [String]
      def metadata_json
        ensure_live!
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_workflow_checkpoint_metadata_json(@ptr))
      end

      # @return [Integer] number of pending events
      def pending_events_count
        ensure_live!
        Blazen::FFI.blazen_workflow_checkpoint_pending_events_count(@ptr)
      end

      # Returns a copy of the pending event at +idx+.
      #
      # The cabi returns a freshly-cloned event handle, so the returned
      # wrapper is independent of the checkpoint's lifetime.
      #
      # @param idx [Integer]
      # @return [Blazen::Persist::PersistedEvent, nil]
      def pending_event_at(idx)
        ensure_live!
        raw = Blazen::FFI.blazen_workflow_checkpoint_pending_events_get(@ptr, Integer(idx))
        return nil if raw.nil? || raw.null?

        PersistedEvent.new(raw)
      end

      # @return [Array<Blazen::Persist::PersistedEvent>] every pending event
      def pending_events
        Array.new(pending_events_count) { |i| pending_event_at(i) }
      end

      # Appends +event+ to the checkpoint's pending-events list. The event
      # handle is **consumed** — callers must not reference it again after
      # a successful push.
      #
      # @param event [Blazen::Persist::PersistedEvent]
      # @return [self]
      def pending_events_push(event)
        ensure_live!
        raw_event = event.consume!
        status = Blazen::FFI.blazen_workflow_checkpoint_pending_events_push(@ptr, raw_event)
        if status.negative?
          raise Blazen::PersistError, "blazen_workflow_checkpoint_pending_events_push rejected its inputs"
        end

        self
      end

      # Transfers ownership of the underlying handle to the C side
      # (typically {CheckpointStore#save}) and disables this wrapper's
      # auto-free hook. After this call any other method raises
      # +Blazen::Error+.
      #
      # @return [::FFI::Pointer] the raw, now-orphaned handle
      def consume!
        ensure_live!
        raw_address = @ptr.address
        @ptr.autorelease = false
        @ptr = nil
        ::FFI::Pointer.new(raw_address)
      end

      # @return [Boolean] true once {#consume!} has been called
      def consumed?
        @ptr.nil?
      end

      private

      def ensure_live!
        return unless @ptr.nil?

        raise Blazen::Error, "WorkflowCheckpoint handle was already consumed"
      end
    end

    # Idiomatic Ruby wrapper around a +BlazenPersistedEvent+ handle.
    #
    # Lifecycle mirrors {WorkflowCheckpoint}: GC releases the C handle via
    # {::FFI::AutoPointer}, and {#consume!} transfers ownership when the
    # event is pushed onto a checkpoint.
    class PersistedEvent
      # @overload initialize(raw_ptr)
      #   @param raw_ptr [::FFI::Pointer]
      # @overload initialize(event_type:, data_json:)
      #   @param event_type [String]
      #   @param data_json [String]
      def initialize(*args, **kwargs)
        if !args.empty?
          raw = args.first
          raise ArgumentError, "PersistedEvent.new(raw_ptr): pointer must be non-null" if raw.nil? || raw.null?

          @ptr = ::FFI::AutoPointer.new(raw, Blazen::FFI.method(:blazen_persisted_event_free))
          return
        end

        event_type = (kwargs[:event_type] || "").to_s
        data_json  = (kwargs[:data_json] || "").to_s
        raw = Blazen::FFI.with_cstring(event_type) do |et|
          Blazen::FFI.with_cstring(data_json) do |dj|
            Blazen::FFI.blazen_persisted_event_new(et, dj)
          end
        end
        raise Blazen::ValidationError, "blazen_persisted_event_new rejected its inputs" if raw.nil? || raw.null?

        @ptr = ::FFI::AutoPointer.new(raw, Blazen::FFI.method(:blazen_persisted_event_free))
      end

      # @return [::FFI::AutoPointer] live handle (raises after {#consume!})
      def ptr
        ensure_live!
        @ptr
      end

      # @return [String]
      def event_type
        ensure_live!
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_persisted_event_event_type(@ptr))
      end

      # @return [String]
      def data_json
        ensure_live!
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_persisted_event_data_json(@ptr))
      end

      # Transfers ownership of the underlying handle. After this call any
      # other method raises +Blazen::Error+.
      #
      # @return [::FFI::Pointer]
      def consume!
        ensure_live!
        raw_address = @ptr.address
        @ptr.autorelease = false
        @ptr = nil
        ::FFI::Pointer.new(raw_address)
      end

      # @return [Boolean]
      def consumed?
        @ptr.nil?
      end

      private

      def ensure_live!
        return unless @ptr.nil?

        raise Blazen::Error, "PersistedEvent handle was already consumed"
      end
    end

    # --------------------------------------------------------------------
    # Internal array consumers — extracts each element into a Ruby wrapper
    # that owns the per-element handle, then releases only the backing
    # slice via the matching +*_array_free+ helper. Each element pointer
    # in the slice is nulled out before the free call so the cabi side
    # skips its own per-element drop.
    # --------------------------------------------------------------------

    def self.consume_checkpoint_array(base, count)
      return [] if base.nil? || base.null? || count.zero?

      slot_stride = ::FFI::Pointer.size
      begin
        Array.new(count) do |i|
          raw = base.get_pointer(i * slot_stride)
          base.put_pointer(i * slot_stride, ::FFI::Pointer::NULL)
          WorkflowCheckpoint.new(raw)
        end
      ensure
        Blazen::FFI.blazen_workflow_checkpoint_array_free(base, count)
      end
    end
    private_class_method :consume_checkpoint_array

    def self.consume_string_array(base, count)
      return [] if base.nil? || base.null? || count.zero?

      slot_stride = ::FFI::Pointer.size
      strings = Array.new(count) do |i|
        raw = base.get_pointer(i * slot_stride)
        raw.null? ? "" : raw.read_string.dup.force_encoding(Encoding::UTF_8)
      end
      # The cabi's +blazen_string_array_free+ frees both backing slice and
      # every per-element C string in one call — we don't null slots here
      # because Ruby has already copied each string's bytes out.
      Blazen::FFI.blazen_string_array_free(base, count)
      strings
    end
    private_class_method :consume_string_array
  end
end
