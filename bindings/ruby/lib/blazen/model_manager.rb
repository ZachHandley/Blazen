# frozen_string_literal: true

module Blazen
  # Knobs controlling how a `LoRA` adapter is mounted onto a base model.
  #
  # Today only the `scale` (delta-weight multiplier) is exposed across the
  # cabi boundary; richer fields (target-module overrides, dropout, etc.)
  # remain provider-internal until the cabi grows a structured adapter
  # options record.
  class AdapterOptions
    # @return [Float]
    attr_reader :scale

    # @param scale [Float] delta-weight multiplier; `1.0` is full PEFT
    #   strength
    def initialize(scale: 1.0)
      @scale = scale.to_f
    end
  end

  # Snapshot of one registered model: id, residency, estimated footprint,
  # owning pool, and the list of mounted adapters.
  ModelStatus = Struct.new(
    :id, :loaded?, :memory_estimate_bytes, :pool, :adapters,
    keyword_init: true,
  )

  # Snapshot of one memory pool: label, budget, currently-used bytes, and
  # the count of loaded models charged to it.
  PoolStatus = Struct.new(
    :pool, :budget_bytes, :used_bytes, :loaded_models,
    keyword_init: true,
  )

  # Snapshot of one mounted adapter on a model. `mount_strategy` is the
  # backend-reported strategy tag (`"attached"` / `"rebuilt"` / `"merged"`).
  AdapterStatus = Struct.new(
    :id, :scale, :on_disk_bytes, :mount_strategy,
    keyword_init: true,
  )

  # Handle returned by {ModelManager#load_adapter}. Field-compatible with
  # {AdapterStatus} but represents the live mount, not a snapshot.
  AdapterHandle = Struct.new(
    :id, :scale, :on_disk_bytes, :mount_strategy,
    keyword_init: true,
  )

  # Memory-budget-aware loader with per-pool LRU eviction and `LoRA` adapter
  # orchestration.
  #
  # Wraps the cabi `BlazenModelManager*` opaque handle. The async verbs
  # (`load`, `unload`, `loaded?`, `status`, `load_adapter`,
  # `unload_adapter`, `list_adapters`) compose with `Fiber.scheduler` when
  # one is active (via {Blazen::FFI.await_future}) and fall back to the
  # cabi's thread-blocking flavor otherwise.
  #
  # @example No-enforcement manager (u64::MAX budgets)
  #   mgr = Blazen::ModelManager.new
  #   mgr.pools.map(&:pool)   # => ["cpu", "gpu:0"]
  #
  # @example Explicit budgets
  #   mgr = Blazen::ModelManager.with_budgets_gb(cpu_ram_gb: 8.0, gpu_vram_gb: 4.0)
  class ModelManager
    # @return [::FFI::AutoPointer] underlying `BlazenModelManager *`
    attr_reader :ptr

    # Constructs a manager with `u64::MAX` budgets on `Pool::Cpu` and
    # `Pool::Gpu(0)` (matching the Python binding's no-args sentinel).
    def initialize
      raw = Blazen::FFI.blazen_model_manager_new
      install_handle!(raw)
    end

    # Constructs a manager with explicit CPU RAM and GPU VRAM budgets, both
    # in gigabytes. Pass `0.0` to disable a pool.
    #
    # @param cpu_ram_gb [Float]
    # @param gpu_vram_gb [Float]
    # @return [ModelManager]
    def self.with_budgets_gb(cpu_ram_gb:, gpu_vram_gb:)
      mgr = allocate
      raw = Blazen::FFI.blazen_model_manager_with_budgets_gb(
        cpu_ram_gb.to_f, gpu_vram_gb.to_f,
      )
      mgr.send(:install_handle!, raw)
      mgr
    end

    # Registers a Ruby-side {LocalModel}-like object with the manager.
    #
    # @raise [Blazen::UnsupportedError] always — the cabi intentionally does
    #   not expose this verb (see `crates/blazen-cabi/src/manager.rs`'s
    #   "Deferred surface" doc). Use provider factories to register native
    #   `LocalModel` impls.
    def register_local(_model)
      raise Blazen::UnsupportedError,
            "ModelManager#register_local requires a foreign-callback trampoline " \
            "that is not yet wired through the C-ABI. Use provider factories to " \
            "register native LocalModel impls."
    end

    # Loads (or evicts-then-loads) the model registered as `model_id`.
    #
    # @param model_id [String]
    # @return [void]
    def load(model_id)
      Blazen::FFI.with_cstring(model_id) do |mid|
        fut = Blazen::FFI.blazen_model_manager_load(@ptr, mid)
        Blazen::FFI.await_future(fut) do |f|
          out_err = ::FFI::MemoryPointer.new(:pointer)
          Blazen::FFI.blazen_future_take_unit(f, out_err)
          Blazen::FFI.check_error!(out_err)
        end
      end
      nil
    end

    # Synchronously loads the model. Use when no `Fiber.scheduler` is
    # available and you explicitly want a blocking call.
    #
    # @param model_id [String]
    # @return [void]
    def load_blocking(model_id)
      Blazen::FFI.with_cstring(model_id) do |mid|
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_model_manager_load_blocking(@ptr, mid, out_err)
        Blazen::FFI.check_error!(out_err)
      end
      nil
    end

    # Unloads the model registered as `model_id`.
    #
    # @param model_id [String]
    # @return [void]
    def unload(model_id)
      Blazen::FFI.with_cstring(model_id) do |mid|
        fut = Blazen::FFI.blazen_model_manager_unload(@ptr, mid)
        Blazen::FFI.await_future(fut) do |f|
          out_err = ::FFI::MemoryPointer.new(:pointer)
          Blazen::FFI.blazen_future_take_unit(f, out_err)
          Blazen::FFI.check_error!(out_err)
        end
      end
      nil
    end

    # Returns `true` if `model_id` is currently loaded.
    #
    # @param model_id [String]
    # @return [Boolean]
    def loaded?(model_id)
      Blazen::FFI.with_cstring(model_id) do |mid|
        fut = Blazen::FFI.blazen_model_manager_is_loaded(@ptr, mid)
        Blazen::FFI.await_future(fut) do |f|
          out = ::FFI::MemoryPointer.new(:bool)
          out_err = ::FFI::MemoryPointer.new(:pointer)
          Blazen::FFI.blazen_future_take_bool(f, out, out_err)
          Blazen::FFI.check_error!(out_err)
          out.read(:bool)
        end
      end
    end

    # Snapshots the status of every registered model.
    #
    # @return [Array<Blazen::ModelStatus>]
    def status
      fut = Blazen::FFI.blazen_model_manager_status(@ptr)
      Blazen::FFI.await_future(fut) do |f|
        out = ::FFI::MemoryPointer.new(:pointer)
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_future_take_model_status_list(f, out, out_err)
        Blazen::FFI.check_error!(out_err)
        list_ptr = out.read_pointer
        decode_model_status_list(list_ptr)
      end
    end

    # Snapshots configured pools together with their live `used_bytes` and
    # loaded-model counts. Synchronous on the cabi side (`pools` does not
    # spawn onto the runtime).
    #
    # @return [Array<Blazen::PoolStatus>]
    def pools
      list_ptr = Blazen::FFI.blazen_model_manager_pools(@ptr)
      decode_pool_status_list(list_ptr)
    end

    # Returns bytes currently used by loaded models in `pool` (label,
    # e.g. `"cpu"` / `"gpu:0"`). Raises {Blazen::ValidationError} if no
    # such pool exists.
    #
    # Why: the cabi exposes `used_bytes` only on each `BlazenPoolStatus`
    # entry, so we look it up via {#pools} rather than via a dedicated
    # manager verb.
    #
    # @param pool [String]
    # @return [Integer]
    def used_bytes(pool)
      entry = pools.find { |p| p.pool == pool }
      raise Blazen::ValidationError, "unknown pool: #{pool.inspect}" if entry.nil?

      entry.used_bytes
    end

    # Returns the remaining headroom (`budget - used`) in bytes for `pool`.
    # Saturates at `0` to avoid underflow when the budget has been
    # exceeded (which is possible during the brief window between an
    # eviction failure and the next snapshot).
    #
    # @param pool [String]
    # @return [Integer]
    def available_bytes(pool)
      entry = pools.find { |p| p.pool == pool }
      raise Blazen::ValidationError, "unknown pool: #{pool.inspect}" if entry.nil?

      diff = entry.budget_bytes - entry.used_bytes
      diff.negative? ? 0 : diff
    end

    # Mounts a PEFT-format `LoRA` adapter. The base model is loaded
    # automatically if not already in residence.
    #
    # @param model_id [String]
    # @param adapter_dir [String] filesystem path containing the adapter
    #   weights + tokenizer-config
    # @param options [Blazen::AdapterOptions]
    # @param adapter_id [String, nil] caller-chosen handle; defaults to
    #   the basename of `adapter_dir`
    # @return [Blazen::AdapterHandle]
    def load_adapter(model_id, adapter_dir, options = AdapterOptions.new, adapter_id: nil)
      handle_id = adapter_id || File.basename(adapter_dir.to_s)
      Blazen::FFI.with_cstring(model_id) do |mid|
        Blazen::FFI.with_cstring(adapter_dir) do |dir|
          Blazen::FFI.with_cstring(handle_id) do |aid|
            fut = Blazen::FFI.blazen_model_manager_load_adapter(
              @ptr, mid, dir, aid, options.scale.to_f,
            )
            Blazen::FFI.await_future(fut) do |f|
              out = ::FFI::MemoryPointer.new(:pointer)
              out_err = ::FFI::MemoryPointer.new(:pointer)
              Blazen::FFI.blazen_future_take_adapter_handle_info(f, out, out_err)
              Blazen::FFI.check_error!(out_err)
              info_ptr = out.read_pointer
              decode_adapter_handle_info(info_ptr, options.scale.to_f)
            end
          end
        end
      end
    end

    # Unmounts a previously-loaded adapter.
    #
    # @param model_id [String]
    # @param adapter_id [String]
    # @return [void]
    def unload_adapter(model_id, adapter_id)
      Blazen::FFI.with_cstring(model_id) do |mid|
        Blazen::FFI.with_cstring(adapter_id) do |aid|
          fut = Blazen::FFI.blazen_model_manager_unload_adapter(@ptr, mid, aid)
          Blazen::FFI.await_future(fut) do |f|
            out_err = ::FFI::MemoryPointer.new(:pointer)
            Blazen::FFI.blazen_future_take_unit(f, out_err)
            Blazen::FFI.check_error!(out_err)
          end
        end
      end
      nil
    end

    # Lists adapters mounted on `model_id`.
    #
    # @param model_id [String]
    # @return [Array<Blazen::AdapterStatus>]
    def list_adapters(model_id)
      Blazen::FFI.with_cstring(model_id) do |mid|
        fut = Blazen::FFI.blazen_model_manager_list_adapters(@ptr, mid)
        Blazen::FFI.await_future(fut) do |f|
          out = ::FFI::MemoryPointer.new(:pointer)
          out_err = ::FFI::MemoryPointer.new(:pointer)
          Blazen::FFI.blazen_future_take_adapter_status_list(f, out, out_err)
          Blazen::FFI.check_error!(out_err)
          list_ptr = out.read_pointer
          decode_adapter_status_list(list_ptr)
        end
      end
    end

    private

    def install_handle!(raw_ptr)
      if raw_ptr.nil? || raw_ptr.null?
        raise Blazen::InternalError, "ModelManager: native constructor returned null"
      end

      @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_model_manager_free))
    end

    def decode_model_status_list(list_ptr)
      return [] if list_ptr.nil? || list_ptr.null?

      begin
        len = Blazen::FFI.blazen_model_status_list_len(list_ptr)
        Array.new(len) do |i|
          entry = Blazen::FFI.blazen_model_status_list_get(list_ptr, i)
          decode_model_status(entry)
        end
      ensure
        Blazen::FFI.blazen_model_status_list_free(list_ptr)
      end
    end

    def decode_model_status(entry)
      return nil if entry.nil? || entry.null?

      adapters_ptr = Blazen::FFI.blazen_model_status_adapters(entry)
      ModelStatus.new(
        id: Blazen::FFI.consume_cstring(Blazen::FFI.blazen_model_status_id(entry)),
        loaded?: Blazen::FFI.blazen_model_status_loaded(entry),
        memory_estimate_bytes: Blazen::FFI.blazen_model_status_memory_bytes(entry),
        pool: Blazen::FFI.consume_cstring(Blazen::FFI.blazen_model_status_pool(entry)),
        adapters: decode_adapter_status_list(adapters_ptr),
      )
    end

    def decode_pool_status_list(list_ptr)
      return [] if list_ptr.nil? || list_ptr.null?

      begin
        len = Blazen::FFI.blazen_pool_status_list_len(list_ptr)
        Array.new(len) do |i|
          entry = Blazen::FFI.blazen_pool_status_list_get(list_ptr, i)
          PoolStatus.new(
            pool: Blazen::FFI.consume_cstring(Blazen::FFI.blazen_pool_status_id(entry)),
            budget_bytes: Blazen::FFI.blazen_pool_status_budget_bytes(entry),
            used_bytes: Blazen::FFI.blazen_pool_status_used_bytes(entry),
            loaded_models: Blazen::FFI.blazen_pool_status_loaded_models(entry),
          )
        end
      ensure
        Blazen::FFI.blazen_pool_status_list_free(list_ptr)
      end
    end

    def decode_adapter_status_list(list_ptr)
      return [] if list_ptr.nil? || list_ptr.null?

      begin
        len = Blazen::FFI.blazen_adapter_status_list_len(list_ptr)
        Array.new(len) do |i|
          entry = Blazen::FFI.blazen_adapter_status_list_get(list_ptr, i)
          AdapterStatus.new(
            id: Blazen::FFI.consume_cstring(Blazen::FFI.blazen_adapter_status_adapter_id(entry)),
            scale: Blazen::FFI.blazen_adapter_status_scale(entry),
            on_disk_bytes: Blazen::FFI.blazen_adapter_status_memory_bytes(entry),
            mount_strategy: decode_mount_strategy(
              Blazen::FFI.blazen_adapter_status_mount_strategy(entry),
            ),
          )
        end
      ensure
        Blazen::FFI.blazen_adapter_status_list_free(list_ptr)
      end
    end

    def decode_adapter_handle_info(info_ptr, scale)
      return nil if info_ptr.nil? || info_ptr.null?

      begin
        AdapterHandle.new(
          id: Blazen::FFI.consume_cstring(Blazen::FFI.blazen_adapter_handle_info_adapter_id(info_ptr)),
          scale: scale,
          on_disk_bytes: Blazen::FFI.blazen_adapter_handle_info_memory_bytes(info_ptr),
          mount_strategy: decode_mount_strategy(
            Blazen::FFI.blazen_adapter_handle_info_mount_strategy(info_ptr),
          ),
        )
      ensure
        Blazen::FFI.blazen_adapter_handle_info_free(info_ptr)
      end
    end

    def decode_mount_strategy(tag)
      case tag
      when Blazen::FFI::ADAPTER_MOUNT_STRATEGY_ATTACHED then "attached"
      when Blazen::FFI::ADAPTER_MOUNT_STRATEGY_REBUILT  then "rebuilt"
      when Blazen::FFI::ADAPTER_MOUNT_STRATEGY_MERGED   then "merged"
      else "unknown"
      end
    end
  end
end
