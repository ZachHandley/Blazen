# frozen_string_literal: true

module Blazen
  # Stable string labels for the backend hints accepted by
  # {ModelManager#load_from_hf}. Pass one of these (or +nil+ for auto-detect)
  # as `HfLoadOptions#backend_hint`.
  module BackendHint
    MISTRALRS = "mistralrs"
    CANDLE    = "candle"
    LLAMACPP  = "llamacpp"

    # @api private
    # Maps the public string label to the cabi's int32 sentinel. Returns the
    # `BLAZEN_BACKEND_HINT_NONE` sentinel (-1) for +nil+ / unknown labels so
    # the loader falls back to auto-detect.
    def self.to_cabi(label)
      case label
      when MISTRALRS then Blazen::FFI::BLAZEN_BACKEND_HINT_MISTRALRS
      when CANDLE    then Blazen::FFI::BLAZEN_BACKEND_HINT_CANDLE
      when LLAMACPP  then Blazen::FFI::BLAZEN_BACKEND_HINT_LLAMACPP
      else Blazen::FFI::BLAZEN_BACKEND_HINT_NONE
      end
    end
  end

  # Options bag for {ModelManager#load_from_hf}. All fields are optional; the
  # loader auto-detects backend, uses `$HF_TOKEN`/anonymous auth, the default
  # `hf-hub` cache dir, and `Pool::Cpu` when their respective fields are nil.
  class HfLoadOptions
    # @return [String, nil] one of {BackendHint::MISTRALRS} /
    #   {BackendHint::CANDLE} / {BackendHint::LLAMACPP} or +nil+ for
    #   auto-detect.
    attr_reader :backend_hint
    # @return [String, nil] git revision (branch, tag, or commit sha)
    attr_reader :revision
    # @return [String, nil] Hugging Face access token; falls back to `$HF_TOKEN`
    attr_reader :hf_token
    # @return [String, nil] override for the on-disk `hf-hub` cache dir
    attr_reader :cache_dir
    # @return [String, nil] device specifier (`"cpu"`, `"cuda:0"`, `"metal"`)
    attr_reader :device
    # @return [String, nil] explicit GGUF filename for multi-quant repos
    attr_reader :gguf_file
    # @return [Integer, nil] override the manager's memory-budget estimate, in
    #   bytes. nil = let the manager sum the chosen backend's weight files.
    attr_reader :memory_estimate_bytes
    # @return [String, nil] target pool label (`"cpu"` / `"gpu"` / `"gpu:N"`)
    attr_reader :pool

    def initialize(backend_hint: nil, revision: nil, hf_token: nil, cache_dir: nil,
                   device: nil, gguf_file: nil, memory_estimate_bytes: nil, pool: nil)
      @backend_hint = backend_hint
      @revision = revision
      @hf_token = hf_token
      @cache_dir = cache_dir
      @device = device
      @gguf_file = gguf_file
      @memory_estimate_bytes = memory_estimate_bytes
      @pool = pool
    end
  end

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

  # Stable string labels for the LR scheduler kinds accepted by
  # {TrainConfig#scheduler}. Pass one of these as `SchedulerConfig#kind`.
  module SchedulerKind
    CONSTANT = "constant"
    LINEAR   = "linear"
    COSINE   = "cosine"

    # @api private
    def self.to_cabi(label)
      case label
      when CONSTANT then Blazen::FFI::BLAZEN_SCHEDULER_CONSTANT
      when LINEAR   then Blazen::FFI::BLAZEN_SCHEDULER_LINEAR
      when COSINE   then Blazen::FFI::BLAZEN_SCHEDULER_COSINE
      else
        raise Blazen::ValidationError,
              "unknown SchedulerKind: #{label.inspect} " \
              "(expected one of #{[CONSTANT, LINEAR, COSINE].inspect})"
      end
    end
  end

  # Stable string labels for the mixed-precision modes accepted by
  # {TrainConfig#mixed_precision}.
  module MixedPrecision
    NONE = "none"
    BF16 = "bf16"

    # @api private
    def self.to_cabi(label)
      case label
      when NONE then Blazen::FFI::BLAZEN_MIXED_PRECISION_NONE
      when BF16 then Blazen::FFI::BLAZEN_MIXED_PRECISION_BF16
      else
        raise Blazen::ValidationError,
              "unknown MixedPrecision: #{label.inspect} " \
              "(expected one of #{[NONE, BF16].inspect})"
      end
    end
  end

  # `LoRA` hyperparameters (rank, alpha, dropout, and the list of attention
  # projection names to adapt).
  class LoraConfig
    attr_reader :rank, :alpha, :dropout, :target_modules

    def initialize(rank: 16, alpha: 32.0, dropout: 0.0,
                   target_modules: %w[q_proj k_proj v_proj o_proj])
      @rank = Integer(rank)
      @alpha = Float(alpha)
      @dropout = Float(dropout)
      @target_modules = Array(target_modules).map(&:to_s)
    end
  end

  # `AdamW` optimizer hyperparameters. `gradient_clip` is nullable — pass
  # `nil` to disable global L2 clipping.
  class OptimConfig
    attr_reader :learning_rate, :beta1, :beta2, :epsilon, :weight_decay, :gradient_clip

    def initialize(learning_rate: 2e-4, beta1: 0.9, beta2: 0.999,
                   epsilon: 1e-8, weight_decay: 0.0, gradient_clip: 1.0)
      @learning_rate = Float(learning_rate)
      @beta1 = Float(beta1)
      @beta2 = Float(beta2)
      @epsilon = Float(epsilon)
      @weight_decay = Float(weight_decay)
      @gradient_clip = gradient_clip.nil? ? nil : Float(gradient_clip)
    end
  end

  # LR-schedule configuration. `kind` is one of {SchedulerKind::CONSTANT} /
  # {SchedulerKind::LINEAR} / {SchedulerKind::COSINE}.
  class SchedulerConfig
    attr_reader :kind, :warmup_steps

    def initialize(kind: SchedulerKind::COSINE, warmup_steps: 0)
      @kind = kind.to_s
      @warmup_steps = Integer(warmup_steps)
    end
  end

  # End-to-end `LoRA` training configuration consumed by
  # {ModelManager#train_lora}.
  class TrainConfig
    attr_reader :base_model_repo, :output_dir, :lora, :optim, :scheduler,
                :max_steps, :batch_size, :gradient_accumulation_steps,
                :max_seq_len, :eval_steps, :save_steps, :seed,
                :mixed_precision, :device

    def initialize(base_model_repo:, output_dir:,
                   lora: LoraConfig.new, optim: OptimConfig.new,
                   scheduler: SchedulerConfig.new, max_steps: 100,
                   batch_size: 1, gradient_accumulation_steps: 1,
                   max_seq_len: 2048, eval_steps: nil, save_steps: nil,
                   seed: 42, mixed_precision: MixedPrecision::NONE,
                   device: nil)
      @base_model_repo = base_model_repo.to_s
      @output_dir = output_dir.to_s
      @lora = lora
      @optim = optim
      @scheduler = scheduler
      @max_steps = Integer(max_steps)
      @batch_size = Integer(batch_size)
      @gradient_accumulation_steps = Integer(gradient_accumulation_steps)
      @max_seq_len = Integer(max_seq_len)
      @eval_steps = eval_steps.nil? ? nil : Integer(eval_steps)
      @save_steps = save_steps.nil? ? nil : Integer(save_steps)
      @seed = Integer(seed)
      @mixed_precision = mixed_precision.to_s
      @device = device.nil? ? nil : device.to_s
    end
  end

  # Result returned by {ModelManager#train_lora}.
  TrainedAdapter = Struct.new(:adapter_dir, :final_loss, :total_steps, keyword_init: true)

  # Shared training hyperparameters consumed by the preference-optimization
  # configs ({DpoConfig}, {OrpoConfig}, {SimpoConfig}, {KtoConfig}) and the
  # full-fine-tune config ({FullFineTuneConfig}) as their `core` slot.
  #
  # Mirrors `BlazenTrainCoreConfig`. The legacy SFT-only {TrainConfig} keeps
  # a flat layout for PR7 backward compatibility; new training surfaces in
  # PR8 nest a `TrainCoreConfig` plus per-algorithm hyperparameters.
  class TrainCoreConfig
    attr_reader :base_model_repo, :base_model_revision, :output_dir,
                :optim, :scheduler, :max_steps, :batch_size,
                :gradient_accumulation_steps, :max_seq_len,
                :eval_steps, :save_steps, :seed,
                :mixed_precision, :device

    def initialize(base_model_repo:, output_dir:,
                   base_model_revision: nil,
                   optim: OptimConfig.new, scheduler: SchedulerConfig.new,
                   max_steps: 100, batch_size: 1,
                   gradient_accumulation_steps: 1, max_seq_len: 2048,
                   eval_steps: nil, save_steps: nil, seed: 42,
                   mixed_precision: MixedPrecision::NONE, device: nil)
      @base_model_repo = base_model_repo.to_s
      @base_model_revision = base_model_revision.nil? ? nil : base_model_revision.to_s
      @output_dir = output_dir.to_s
      @optim = optim
      @scheduler = scheduler
      @max_steps = Integer(max_steps)
      @batch_size = Integer(batch_size)
      @gradient_accumulation_steps = Integer(gradient_accumulation_steps)
      @max_seq_len = Integer(max_seq_len)
      @eval_steps = eval_steps.nil? ? nil : Integer(eval_steps)
      @save_steps = save_steps.nil? ? nil : Integer(save_steps)
      @seed = Integer(seed)
      @mixed_precision = mixed_precision.to_s
      @device = device.nil? ? nil : device.to_s
    end
  end

  # DPO (Direct Preference Optimization) training configuration. Wraps
  # {TrainCoreConfig} with DPO-specific knobs (`beta`, `label_smoothing`)
  # plus an optional separate reference model repo (defaults to
  # `core.base_model_repo` when nil).
  class DpoConfig
    attr_reader :core, :lora, :beta, :label_smoothing,
                :reference_model_repo, :reference_model_revision

    def initialize(core:, lora: LoraConfig.new, beta: 0.1,
                   label_smoothing: 0.0,
                   reference_model_repo: nil,
                   reference_model_revision: nil)
      @core = core
      @lora = lora
      @beta = Float(beta)
      @label_smoothing = Float(label_smoothing)
      @reference_model_repo = reference_model_repo.nil? ? nil : reference_model_repo.to_s
      @reference_model_revision = reference_model_revision.nil? ? nil : reference_model_revision.to_s
    end
  end

  # ORPO (Odds-Ratio Preference Optimization) training configuration.
  # Reference-free: combines the SFT loss with an odds-ratio penalty term
  # whose weight is `lambda`.
  class OrpoConfig
    attr_reader :core, :lora, :lambda

    def initialize(core:, lora: LoraConfig.new, lambda: 0.1)
      @core = core
      @lora = lora
      @lambda = Float(lambda)
    end
  end

  # SimPO (Simple Preference Optimization) training configuration.
  # Reference-free: length-normalized preference margin with logit scaling
  # `beta` and target margin `gamma`.
  class SimpoConfig
    attr_reader :core, :lora, :beta, :gamma

    def initialize(core:, lora: LoraConfig.new, beta: 2.0, gamma: 1.0)
      @core = core
      @lora = lora
      @beta = Float(beta)
      @gamma = Float(gamma)
    end
  end

  # KTO (Kahneman-Tversky Optimization) training configuration. Consumes a
  # {RatedJsonlDataset} (single-completion-plus-rating, not preference
  # pairs). Requires a reference model — pass `nil` for
  # `reference_model_repo` to reuse `core.base_model_repo`.
  class KtoConfig
    attr_reader :core, :lora, :beta, :lambda_d, :lambda_u,
                :reference_model_repo, :reference_model_revision

    def initialize(core:, lora: LoraConfig.new, beta: 0.1,
                   lambda_d: 1.0, lambda_u: 1.0,
                   reference_model_repo: nil,
                   reference_model_revision: nil)
      @core = core
      @lora = lora
      @beta = Float(beta)
      @lambda_d = Float(lambda_d)
      @lambda_u = Float(lambda_u)
      @reference_model_repo = reference_model_repo.nil? ? nil : reference_model_repo.to_s
      @reference_model_revision = reference_model_revision.nil? ? nil : reference_model_revision.to_s
    end
  end

  # Full-fine-tune configuration (every parameter trainable, no `LoRA`
  # wrapping). `gradient_checkpointing` is exposed for forward compatibility
  # but the trainer currently rejects `true` because candle 0.10.2 has no
  # checkpointing primitive.
  class FullFineTuneConfig
    attr_reader :core, :gradient_checkpointing

    def initialize(core:, gradient_checkpointing: false)
      @core = core
      @gradient_checkpointing = gradient_checkpointing ? true : false
    end
  end

  # Result returned by {ModelManager#fine_tune}. Mirrors
  # `BlazenFullFineTuneResult`.
  FullFineTuneResult = Struct.new(
    :output_dir, :final_loss, :steps_completed,
    keyword_init: true,
  )

  # Tokenized preference-pair JSONL dataset handle consumed by
  # {ModelManager#train_dpo} / `train_orpo` / `train_simpo`. Wraps
  # `BlazenPreferenceJsonlDataset *`; the inner pointer is freed
  # automatically when the Ruby object is garbage-collected (via
  # `FFI::AutoPointer`).
  class PreferenceJsonlDataset
    # @return [::FFI::AutoPointer] underlying `BlazenPreferenceJsonlDataset *`
    attr_reader :ptr

    # Loads a preference JSONL training file with the given tokenizer.
    # The file format is one JSON object per line with `prompt`, `chosen`,
    # and `rejected` fields (see `blazen_train::dataset`).
    #
    # @param path [String]
    # @param tokenizer_path [String]
    # @param chat_template [String, nil]
    # @param max_seq_len [Integer]
    # @param device [String, nil]
    # @param pad_token_id [Integer]
    # @return [PreferenceJsonlDataset]
    # @raise [Blazen::Error] on validation, tokenizer-load, or dataset-parse errors
    def self.from_path(path, tokenizer_path:, chat_template: nil,
                       max_seq_len: 2048, device: nil, pad_token_id: 0)
      raw = nil
      Blazen::FFI.with_cstring(path.to_s) do |path_ptr|
        Blazen::FFI.with_cstring(tokenizer_path.to_s) do |tok_ptr|
          Blazen::FFI.with_cstring(chat_template) do |tmpl_ptr|
            Blazen::FFI.with_cstring(device) do |dev_ptr|
              out_err = ::FFI::MemoryPointer.new(:pointer)
              raw = Blazen::FFI.blazen_preference_jsonl_dataset_from_path(
                path_ptr, tok_ptr, tmpl_ptr,
                Integer(max_seq_len), dev_ptr, Integer(pad_token_id),
                out_err,
              )
              Blazen::FFI.check_error!(out_err)
            end
          end
        end
      end
      if raw.nil? || raw.null?
        raise Blazen::InternalError,
              "blazen_preference_jsonl_dataset_from_path returned null without an error"
      end

      new(raw)
    end

    # @api private
    def initialize(raw_ptr)
      @ptr = ::FFI::AutoPointer.new(
        raw_ptr, Blazen::FFI.method(:blazen_preference_jsonl_dataset_free),
      )
    end
  end

  # Tokenized rated (KTO) JSONL dataset handle consumed by
  # {ModelManager#train_kto}. Each row carries a single
  # prompt/completion pair plus a boolean `desirable` rating (see
  # `blazen_train::dataset`).
  class RatedJsonlDataset
    # @return [::FFI::AutoPointer] underlying `BlazenRatedJsonlDataset *`
    attr_reader :ptr

    # Loads a rated JSONL training file with the given tokenizer.
    #
    # @param path [String]
    # @param tokenizer_path [String]
    # @param chat_template [String, nil]
    # @param max_seq_len [Integer]
    # @param device [String, nil]
    # @param pad_token_id [Integer]
    # @return [RatedJsonlDataset]
    # @raise [Blazen::Error] on validation, tokenizer-load, or dataset-parse errors
    def self.from_path(path, tokenizer_path:, chat_template: nil,
                       max_seq_len: 2048, device: nil, pad_token_id: 0)
      raw = nil
      Blazen::FFI.with_cstring(path.to_s) do |path_ptr|
        Blazen::FFI.with_cstring(tokenizer_path.to_s) do |tok_ptr|
          Blazen::FFI.with_cstring(chat_template) do |tmpl_ptr|
            Blazen::FFI.with_cstring(device) do |dev_ptr|
              out_err = ::FFI::MemoryPointer.new(:pointer)
              raw = Blazen::FFI.blazen_rated_jsonl_dataset_from_path(
                path_ptr, tok_ptr, tmpl_ptr,
                Integer(max_seq_len), dev_ptr, Integer(pad_token_id),
                out_err,
              )
              Blazen::FFI.check_error!(out_err)
            end
          end
        end
      end
      if raw.nil? || raw.null?
        raise Blazen::InternalError,
              "blazen_rated_jsonl_dataset_from_path returned null without an error"
      end

      new(raw)
    end

    # @api private
    def initialize(raw_ptr)
      @ptr = ::FFI::AutoPointer.new(
        raw_ptr, Blazen::FFI.method(:blazen_rated_jsonl_dataset_free),
      )
    end
  end

  # Tokenized JSONL dataset handle consumed by {ModelManager#train_lora}.
  # Wraps `BlazenJsonlDataset *`; the inner pointer is freed automatically
  # when the Ruby object is garbage-collected (via `FFI::AutoPointer`).
  class JsonlDataset
    # @return [::FFI::AutoPointer] underlying `BlazenJsonlDataset *`
    attr_reader :ptr

    # Loads a JSONL training file with the given tokenizer.
    #
    # @param path [String] JSONL dataset path on disk
    # @param tokenizer_path [String] path to the tokenizer `.json`
    # @param chat_template [String, nil] optional chat template override
    # @param max_seq_len [Integer] max tokenized sequence length per example
    # @param device [String, nil] device specifier (`"cpu"`, `"cuda:0"`, …)
    # @param pad_token_id [Integer] pad token id used for collation
    # @return [JsonlDataset]
    # @raise [Blazen::Error] on validation, tokenizer-load, or dataset-parse errors
    def self.from_path(path, tokenizer_path:, chat_template: nil,
                       max_seq_len: 2048, device: nil, pad_token_id: 0)
      raw = nil
      Blazen::FFI.with_cstring(path.to_s) do |path_ptr|
        Blazen::FFI.with_cstring(tokenizer_path.to_s) do |tok_ptr|
          Blazen::FFI.with_cstring(chat_template) do |tmpl_ptr|
            Blazen::FFI.with_cstring(device) do |dev_ptr|
              out_err = ::FFI::MemoryPointer.new(:pointer)
              raw = Blazen::FFI.blazen_jsonl_dataset_from_path(
                path_ptr, tok_ptr, tmpl_ptr,
                Integer(max_seq_len), dev_ptr, Integer(pad_token_id),
                out_err,
              )
              Blazen::FFI.check_error!(out_err)
            end
          end
        end
      end
      if raw.nil? || raw.null?
        raise Blazen::InternalError,
              "blazen_jsonl_dataset_from_path returned null without an error"
      end

      new(raw)
    end

    # @api private
    def initialize(raw_ptr)
      @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_jsonl_dataset_free))
    end
  end

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

    # Registers a remote LLM provider under `id` so it can be dispatched by
    # name with {#complete} / {#complete_blocking}.
    #
    # `provider` is any per-engine provider object (e.g.
    # +Blazen::Providers.openai(...)+, +Blazen::Providers.anthropic(...)+) or a
    # polymorphic {Blazen::LlmProvider}. The manager files it for by-name
    # dispatch; remote providers own no local weights, so they never count
    # against a memory budget (pass `memory_estimate_bytes: 0`, the default).
    #
    # @param id [String] dispatch handle to register the provider under
    # @param provider [Blazen::LlmProvider, #as_llm_provider] a provider that
    #   responds to +#as_llm_provider+ (every +Blazen::Providers.<name>+
    #   factory result does)
    # @param memory_estimate_bytes [Integer] bookkeeping only for remote
    #   providers; defaults to 0
    # @return [void]
    # @raise [ArgumentError] if `provider` does not respond to
    #   +#as_llm_provider+
    def register(id, provider, memory_estimate_bytes: 0)
      unless provider.respond_to?(:as_llm_provider)
        raise ArgumentError,
              "provider must respond to #as_llm_provider " \
              "(e.g. a Blazen::Providers.<name> result or a Blazen::LlmProvider); " \
              "got #{provider.class}"
      end

      llm = provider.as_llm_provider
      # Keep the polymorphic handle alive for the duration of the call. The
      # cabi clones the inner Arc, so the registration outlives `llm`, but we
      # must not let GC free `llm` before register_remote reads its pointer.
      Blazen::FFI.with_cstring(id.to_s) do |id_ptr|
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_model_manager_register_remote(
          @ptr, id_ptr, llm.handle, Integer(memory_estimate_bytes), out_err,
        )
        Blazen::FFI.check_error!(out_err)
      end
      nil
    end

    # Runs a chat completion against the provider registered under `id`.
    #
    # Composes with `Fiber.scheduler` when one is active (via
    # {Blazen::FFI.await_future}); otherwise blocks the calling thread on the
    # cabi-side wait. The `request` is consumed by the call.
    #
    # @param id [String]
    # @param request [Blazen::Llm::ModelRequest]
    # @return [Blazen::Llm::ModelResponse]
    def complete(id, request)
      unless request.is_a?(Blazen::Llm::ModelRequest)
        raise ArgumentError, "request must be a Blazen::Llm::ModelRequest"
      end
      unless Blazen::FFI.respond_to?(:blazen_future_take_model_response)
        raise Blazen::UnsupportedError,
              "blazen cabi does not export blazen_future_take_model_response; " \
              "use #complete_blocking instead"
      end

      Blazen::FFI.with_cstring(id.to_s) do |id_ptr|
        req_ptr = request.consume!
        fut = Blazen::FFI.blazen_model_manager_complete(@ptr, id_ptr, req_ptr)
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_model_manager_complete rejected the call " \
                "(null manager, null/non-UTF-8 id, or invalid request)"
        end
        out_resp = ::FFI::MemoryPointer.new(:pointer)
        out_err  = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.await_future(fut) do |f|
          Blazen::FFI.blazen_future_take_model_response(f, out_resp, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        Blazen::Llm::ModelResponse.new(out_resp.read_pointer)
      end
    end

    # Synchronous variant of {#complete}. Use when no `Fiber.scheduler` is
    # available and you explicitly want a blocking call. The `request` is
    # consumed by the call.
    #
    # @param id [String]
    # @param request [Blazen::Llm::ModelRequest]
    # @return [Blazen::Llm::ModelResponse]
    def complete_blocking(id, request)
      unless request.is_a?(Blazen::Llm::ModelRequest)
        raise ArgumentError, "request must be a Blazen::Llm::ModelRequest"
      end

      Blazen::FFI.with_cstring(id.to_s) do |id_ptr|
        req_ptr  = request.consume!
        out_resp = ::FFI::MemoryPointer.new(:pointer)
        out_err  = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_model_manager_complete_blocking(
          @ptr, id_ptr, req_ptr, out_resp, out_err,
        )
        Blazen::FFI.check_error!(out_err)
        Blazen::Llm::ModelResponse.new(out_resp.read_pointer)
      end
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

    # Downloads (or hits the `hf-hub` cache for) a model from Hugging Face,
    # picks an appropriate local backend (mistral.rs / candle / llama.cpp)
    # via `choose_backend`, registers it with the manager under `id`, and
    # eagerly loads it.
    #
    # Composes with `Fiber.scheduler` when one is active; otherwise blocks
    # the calling thread on the cabi-side wait.
    #
    # @param id [String] the manager-side handle to register the model under
    # @param repo [String] HF repo id (e.g. `"meta-llama/Llama-3.2-1B"`)
    # @param options [Blazen::HfLoadOptions]
    # @return [String] the stable label of the chosen backend
    #   (one of {BackendHint::MISTRALRS} / {BackendHint::CANDLE} /
    #   {BackendHint::LLAMACPP})
    def load_from_hf(id, repo, options = HfLoadOptions.new)
      with_hf_load_options(options) do |opts_struct|
        Blazen::FFI.with_cstring(id) do |id_ptr|
          Blazen::FFI.with_cstring(repo) do |repo_ptr|
            opts_ptr = opts_struct ? opts_struct.pointer : nil
            fut = Blazen::FFI.blazen_model_manager_load_from_hf(
              @ptr, id_ptr, repo_ptr, opts_ptr,
            )
            Blazen::FFI.await_future(fut) do |f|
              out = ::FFI::MemoryPointer.new(:pointer)
              out_err = ::FFI::MemoryPointer.new(:pointer)
              Blazen::FFI.blazen_future_take_string(f, out, out_err)
              Blazen::FFI.check_error!(out_err)
              Blazen::FFI.consume_cstring(out.read_pointer)
            end
          end
        end
      end
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

    # Trains a `LoRA` adapter end-to-end against `dataset` and returns the
    # resulting {TrainedAdapter} (adapter directory + final loss + total
    # steps).
    #
    # Composes with `Fiber.scheduler` when one is active; otherwise blocks
    # the calling thread on the cabi-side wait. The `dataset` handle remains
    # valid after the call — the cabi clones the inner `Arc<JsonlDataset>`
    # before spawning the training run, so the Ruby-side `AutoPointer` can
    # free the handle whenever.
    #
    # No progress-callback hook is exposed in v1 — the cabi does not yet
    # surface a `BlazenTrainProgressVTable`. Progress callbacks will be
    # added once that vtable lands (deferred upstream).
    #
    # @param config [Blazen::TrainConfig]
    # @param dataset [Blazen::JsonlDataset]
    # @return [Blazen::TrainedAdapter]
    def train_lora(config, dataset)
      with_train_config(config) do |cfg_struct|
        fut = Blazen::FFI.blazen_model_manager_train_lora(
          @ptr, cfg_struct.pointer, dataset.ptr,
        )
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_model_manager_train_lora rejected the call " \
                "(null manager, null dataset, or invalid config)"
        end
        Blazen::FFI.await_future(fut) do |f|
          out_adapter = Blazen::FFI::BlazenTrainedAdapter.new
          out_err = ::FFI::MemoryPointer.new(:pointer)
          Blazen::FFI.blazen_future_take_trained_adapter(f, out_adapter.pointer, out_err)
          Blazen::FFI.check_error!(out_err)
          decode_trained_adapter(out_adapter)
        end
      end
    end

    # Trains a `LoRA` adapter end-to-end against `dataset` using Direct
    # Preference Optimization (DPO) and returns the resulting
    # {TrainedAdapter}.
    #
    # Composes with `Fiber.scheduler` when one is active (via
    # {Blazen::FFI.await_future}); otherwise blocks the calling thread on
    # the cabi-side wait. The `dataset` handle remains valid after the
    # call — the cabi clones the inner `Arc<PreferenceJsonlDataset>` before
    # spawning the training run.
    #
    # @param config [Blazen::DpoConfig]
    # @param dataset [Blazen::PreferenceJsonlDataset]
    # @return [Blazen::TrainedAdapter]
    def train_dpo(config, dataset)
      validate_dpo_config!(config)
      with_dpo_config(config) do |cfg_struct|
        fut = Blazen::FFI.blazen_model_manager_train_dpo(
          @ptr, cfg_struct.pointer, dataset.ptr,
        )
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_model_manager_train_dpo rejected the call " \
                "(null manager, null dataset, or invalid config)"
        end
        take_trained_adapter_future(fut)
      end
    end

    # Trains a `LoRA` adapter end-to-end against `dataset` using ORPO
    # (Odds-Ratio Preference Optimization, reference-free).
    #
    # @param config [Blazen::OrpoConfig]
    # @param dataset [Blazen::PreferenceJsonlDataset]
    # @return [Blazen::TrainedAdapter]
    def train_orpo(config, dataset)
      validate_orpo_config!(config)
      with_orpo_config(config) do |cfg_struct|
        fut = Blazen::FFI.blazen_model_manager_train_orpo(
          @ptr, cfg_struct.pointer, dataset.ptr,
        )
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_model_manager_train_orpo rejected the call " \
                "(null manager, null dataset, or invalid config)"
        end
        take_trained_adapter_future(fut)
      end
    end

    # Trains a `LoRA` adapter end-to-end against `dataset` using SimPO
    # (Simple Preference Optimization, reference-free, length-normalized).
    #
    # @param config [Blazen::SimpoConfig]
    # @param dataset [Blazen::PreferenceJsonlDataset]
    # @return [Blazen::TrainedAdapter]
    def train_simpo(config, dataset)
      validate_simpo_config!(config)
      with_simpo_config(config) do |cfg_struct|
        fut = Blazen::FFI.blazen_model_manager_train_simpo(
          @ptr, cfg_struct.pointer, dataset.ptr,
        )
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_model_manager_train_simpo rejected the call " \
                "(null manager, null dataset, or invalid config)"
        end
        take_trained_adapter_future(fut)
      end
    end

    # Trains a `LoRA` adapter end-to-end against `dataset` using KTO
    # (Kahneman-Tversky Optimization). `dataset` must be a
    # {RatedJsonlDataset}, not a {PreferenceJsonlDataset}.
    #
    # @param config [Blazen::KtoConfig]
    # @param dataset [Blazen::RatedJsonlDataset]
    # @return [Blazen::TrainedAdapter]
    def train_kto(config, dataset)
      validate_kto_config!(config)
      with_kto_config(config) do |cfg_struct|
        fut = Blazen::FFI.blazen_model_manager_train_kto(
          @ptr, cfg_struct.pointer, dataset.ptr,
        )
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_model_manager_train_kto rejected the call " \
                "(null manager, null dataset, or invalid config)"
        end
        take_trained_adapter_future(fut)
      end
    end

    # Runs a full (every-parameter) fine-tune against `dataset` and returns
    # the resulting {FullFineTuneResult} (output directory + final loss +
    # steps completed). Uses the same SFT-style JSONL format as
    # {ModelManager#train_lora}, so {JsonlDataset} is the expected dataset
    # type.
    #
    # @param config [Blazen::FullFineTuneConfig]
    # @param dataset [Blazen::JsonlDataset]
    # @return [Blazen::FullFineTuneResult]
    def fine_tune(config, dataset)
      validate_full_finetune_config!(config)
      with_full_finetune_config(config) do |cfg_struct|
        fut = Blazen::FFI.blazen_model_manager_fine_tune(
          @ptr, cfg_struct.pointer, dataset.ptr,
        )
        if fut.nil? || fut.null?
          raise Blazen::ValidationError,
                "blazen_model_manager_fine_tune rejected the call " \
                "(null manager, null dataset, or invalid config)"
        end
        Blazen::FFI.await_future(fut) do |f|
          out_result = Blazen::FFI::BlazenFullFineTuneResult.new
          out_err = ::FFI::MemoryPointer.new(:pointer)
          Blazen::FFI.blazen_future_take_full_finetune_result(f, out_result.pointer, out_err)
          Blazen::FFI.check_error!(out_err)
          decode_full_finetune_result(out_result)
        end
      end
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

    # Why: every string + array buffer referenced by `BlazenTrainConfig`
    # (plus the per-element strings inside `target_modules`) is borrowed
    # for the duration of the cabi call. We collect strong refs in
    # `keepalive` so GC can't reclaim them mid-call, then release the
    # whole batch in the `ensure` arm.
    def with_train_config(config)
      raise Blazen::ValidationError, "config must be a Blazen::TrainConfig" \
        unless config.is_a?(TrainConfig)
      raise Blazen::ValidationError, "TrainConfig#lora must be a Blazen::LoraConfig" \
        unless config.lora.is_a?(LoraConfig)
      raise Blazen::ValidationError, "TrainConfig#optim must be a Blazen::OptimConfig" \
        unless config.optim.is_a?(OptimConfig)
      raise Blazen::ValidationError, "TrainConfig#scheduler must be a Blazen::SchedulerConfig" \
        unless config.scheduler.is_a?(SchedulerConfig)

      keepalive = []
      to_cstr = lambda do |s|
        next nil if s.nil?

        mp = ::FFI::MemoryPointer.from_string(s.to_s)
        keepalive << mp
        mp
      end

      target_mod_ptrs = config.lora.target_modules.map { |m| to_cstr.call(m) }
      target_mod_array = ::FFI::MemoryPointer.new(:pointer, target_mod_ptrs.length)
      target_mod_array.write_array_of_pointer(target_mod_ptrs)
      keepalive << target_mod_array

      struct = Blazen::FFI::BlazenTrainConfig.new
      struct[:base_model_repo] = to_cstr.call(config.base_model_repo)
      struct[:output_dir]      = to_cstr.call(config.output_dir)

      lora = struct[:lora]
      lora[:rank]               = config.lora.rank
      lora[:alpha]              = config.lora.alpha
      lora[:dropout]            = config.lora.dropout
      lora[:target_modules]     = target_mod_array
      lora[:target_modules_len] = target_mod_ptrs.length

      optim = struct[:optim]
      optim[:learning_rate] = config.optim.learning_rate
      optim[:beta1]         = config.optim.beta1
      optim[:beta2]         = config.optim.beta2
      optim[:epsilon]       = config.optim.epsilon
      optim[:weight_decay]  = config.optim.weight_decay
      if config.optim.gradient_clip.nil?
        optim[:has_gradient_clip] = 0
        optim[:gradient_clip]     = 0.0
      else
        optim[:has_gradient_clip] = 1
        optim[:gradient_clip]     = config.optim.gradient_clip
      end

      sched = struct[:scheduler]
      sched[:kind]         = SchedulerKind.to_cabi(config.scheduler.kind)
      sched[:warmup_steps] = config.scheduler.warmup_steps

      struct[:max_steps]                   = config.max_steps
      struct[:batch_size]                  = config.batch_size
      struct[:gradient_accumulation_steps] = config.gradient_accumulation_steps
      struct[:max_seq_len]                 = config.max_seq_len
      if config.eval_steps.nil?
        struct[:has_eval_steps] = 0
        struct[:eval_steps]     = 0
      else
        struct[:has_eval_steps] = 1
        struct[:eval_steps]     = config.eval_steps
      end
      if config.save_steps.nil?
        struct[:has_save_steps] = 0
        struct[:save_steps]     = 0
      else
        struct[:has_save_steps] = 1
        struct[:save_steps]     = config.save_steps
      end
      struct[:seed]            = config.seed
      struct[:mixed_precision] = MixedPrecision.to_cabi(config.mixed_precision)
      struct[:device]          = to_cstr.call(config.device)

      begin
        yield struct
      ensure
        keepalive.clear
      end
    end

    # Shared helper: drives a `BlazenFuture *` that resolves to a
    # `BlazenTrainedAdapter` (the result type for all four LoRA-based
    # training surfaces — DPO/ORPO/SimPO/KTO).
    def take_trained_adapter_future(fut)
      Blazen::FFI.await_future(fut) do |f|
        out_adapter = Blazen::FFI::BlazenTrainedAdapter.new
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.blazen_future_take_trained_adapter(f, out_adapter.pointer, out_err)
        Blazen::FFI.check_error!(out_err)
        decode_trained_adapter(out_adapter)
      end
    end

    def validate_dpo_config!(config)
      raise Blazen::ValidationError, "config must be a Blazen::DpoConfig" \
        unless config.is_a?(DpoConfig)
      validate_core_config!(config.core)
      raise Blazen::ValidationError, "DpoConfig#lora must be a Blazen::LoraConfig" \
        unless config.lora.is_a?(LoraConfig)
    end

    def validate_orpo_config!(config)
      raise Blazen::ValidationError, "config must be a Blazen::OrpoConfig" \
        unless config.is_a?(OrpoConfig)
      validate_core_config!(config.core)
      raise Blazen::ValidationError, "OrpoConfig#lora must be a Blazen::LoraConfig" \
        unless config.lora.is_a?(LoraConfig)
    end

    def validate_simpo_config!(config)
      raise Blazen::ValidationError, "config must be a Blazen::SimpoConfig" \
        unless config.is_a?(SimpoConfig)
      validate_core_config!(config.core)
      raise Blazen::ValidationError, "SimpoConfig#lora must be a Blazen::LoraConfig" \
        unless config.lora.is_a?(LoraConfig)
    end

    def validate_kto_config!(config)
      raise Blazen::ValidationError, "config must be a Blazen::KtoConfig" \
        unless config.is_a?(KtoConfig)
      validate_core_config!(config.core)
      raise Blazen::ValidationError, "KtoConfig#lora must be a Blazen::LoraConfig" \
        unless config.lora.is_a?(LoraConfig)
    end

    def validate_full_finetune_config!(config)
      raise Blazen::ValidationError, "config must be a Blazen::FullFineTuneConfig" \
        unless config.is_a?(FullFineTuneConfig)
      validate_core_config!(config.core)
    end

    def validate_core_config!(core)
      raise Blazen::ValidationError, "core must be a Blazen::TrainCoreConfig" \
        unless core.is_a?(TrainCoreConfig)
      raise Blazen::ValidationError, "TrainCoreConfig#optim must be a Blazen::OptimConfig" \
        unless core.optim.is_a?(OptimConfig)
      raise Blazen::ValidationError, "TrainCoreConfig#scheduler must be a Blazen::SchedulerConfig" \
        unless core.scheduler.is_a?(SchedulerConfig)
    end

    # Why: every C-string + array buffer referenced by the nested
    # `BlazenTrainCoreConfig` and `BlazenLoraConfig` is borrowed by the
    # cabi only for the duration of the call. The `keepalive` array holds
    # strong refs to every `MemoryPointer.from_string` and per-array
    # `MemoryPointer.new(:pointer, …)` allocation so GC can't reclaim them
    # mid-call; we release the batch in the `ensure` arm after the FFI
    # function returns.
    def populate_core_struct(core_struct, core, keepalive)
      to_cstr = lambda do |s|
        next nil if s.nil?

        mp = ::FFI::MemoryPointer.from_string(s.to_s)
        keepalive << mp
        mp
      end
      core_struct[:base_model_repo]     = to_cstr.call(core.base_model_repo)
      core_struct[:base_model_revision] = to_cstr.call(core.base_model_revision)
      core_struct[:output_dir]          = to_cstr.call(core.output_dir)

      optim = core_struct[:optim]
      optim[:learning_rate] = core.optim.learning_rate
      optim[:beta1]         = core.optim.beta1
      optim[:beta2]         = core.optim.beta2
      optim[:epsilon]       = core.optim.epsilon
      optim[:weight_decay]  = core.optim.weight_decay
      if core.optim.gradient_clip.nil?
        optim[:has_gradient_clip] = 0
        optim[:gradient_clip]     = 0.0
      else
        optim[:has_gradient_clip] = 1
        optim[:gradient_clip]     = core.optim.gradient_clip
      end

      sched = core_struct[:scheduler]
      sched[:kind]         = SchedulerKind.to_cabi(core.scheduler.kind)
      sched[:warmup_steps] = core.scheduler.warmup_steps

      core_struct[:max_steps]                   = core.max_steps
      core_struct[:batch_size]                  = core.batch_size
      core_struct[:gradient_accumulation_steps] = core.gradient_accumulation_steps
      core_struct[:max_seq_len]                 = core.max_seq_len
      if core.eval_steps.nil?
        core_struct[:has_eval_steps] = 0
        core_struct[:eval_steps]     = 0
      else
        core_struct[:has_eval_steps] = 1
        core_struct[:eval_steps]     = core.eval_steps
      end
      if core.save_steps.nil?
        core_struct[:has_save_steps] = 0
        core_struct[:save_steps]     = 0
      else
        core_struct[:has_save_steps] = 1
        core_struct[:save_steps]     = core.save_steps
      end
      core_struct[:seed]            = core.seed
      core_struct[:mixed_precision] = MixedPrecision.to_cabi(core.mixed_precision)
      core_struct[:device]          = to_cstr.call(core.device)
    end

    # Populates a `BlazenLoraConfig` slot. Allocates the `target_modules`
    # pointer array on the caller's `keepalive`.
    def populate_lora_struct(lora_struct, lora, keepalive)
      target_mod_ptrs = lora.target_modules.map do |m|
        mp = ::FFI::MemoryPointer.from_string(m.to_s)
        keepalive << mp
        mp
      end
      target_mod_array = ::FFI::MemoryPointer.new(:pointer, target_mod_ptrs.length)
      target_mod_array.write_array_of_pointer(target_mod_ptrs)
      keepalive << target_mod_array

      lora_struct[:rank]               = lora.rank
      lora_struct[:alpha]              = lora.alpha
      lora_struct[:dropout]            = lora.dropout
      lora_struct[:target_modules]     = target_mod_array
      lora_struct[:target_modules_len] = target_mod_ptrs.length
    end

    def with_dpo_config(config)
      keepalive = []
      to_cstr = lambda do |s|
        next nil if s.nil?

        mp = ::FFI::MemoryPointer.from_string(s.to_s)
        keepalive << mp
        mp
      end

      struct = Blazen::FFI::BlazenDpoConfig.new
      populate_core_struct(struct[:core], config.core, keepalive)
      populate_lora_struct(struct[:lora], config.lora, keepalive)
      struct[:beta]                     = config.beta
      struct[:label_smoothing]          = config.label_smoothing
      struct[:reference_model_repo]     = to_cstr.call(config.reference_model_repo)
      struct[:reference_model_revision] = to_cstr.call(config.reference_model_revision)

      begin
        yield struct
      ensure
        keepalive.clear
      end
    end

    def with_orpo_config(config)
      keepalive = []
      struct = Blazen::FFI::BlazenOrpoConfig.new
      populate_core_struct(struct[:core], config.core, keepalive)
      populate_lora_struct(struct[:lora], config.lora, keepalive)
      struct[:lambda] = config.lambda

      begin
        yield struct
      ensure
        keepalive.clear
      end
    end

    def with_simpo_config(config)
      keepalive = []
      struct = Blazen::FFI::BlazenSimpoConfig.new
      populate_core_struct(struct[:core], config.core, keepalive)
      populate_lora_struct(struct[:lora], config.lora, keepalive)
      struct[:beta]  = config.beta
      struct[:gamma] = config.gamma

      begin
        yield struct
      ensure
        keepalive.clear
      end
    end

    def with_kto_config(config)
      keepalive = []
      to_cstr = lambda do |s|
        next nil if s.nil?

        mp = ::FFI::MemoryPointer.from_string(s.to_s)
        keepalive << mp
        mp
      end

      struct = Blazen::FFI::BlazenKtoConfig.new
      populate_core_struct(struct[:core], config.core, keepalive)
      populate_lora_struct(struct[:lora], config.lora, keepalive)
      struct[:beta]                     = config.beta
      struct[:lambda_d]                 = config.lambda_d
      struct[:lambda_u]                 = config.lambda_u
      struct[:reference_model_repo]     = to_cstr.call(config.reference_model_repo)
      struct[:reference_model_revision] = to_cstr.call(config.reference_model_revision)

      begin
        yield struct
      ensure
        keepalive.clear
      end
    end

    def with_full_finetune_config(config)
      keepalive = []
      struct = Blazen::FFI::BlazenFullFineTuneConfig.new
      populate_core_struct(struct[:core], config.core, keepalive)
      struct[:gradient_checkpointing] = config.gradient_checkpointing ? 1 : 0

      begin
        yield struct
      ensure
        keepalive.clear
      end
    end

    def decode_full_finetune_result(out_result)
      dir_ptr = out_result[:output_dir]
      output_dir =
        if dir_ptr.nil? || dir_ptr.null?
          nil
        else
          dir_ptr.read_string.force_encoding(Encoding::UTF_8)
        end
      final_loss      = out_result[:final_loss]
      steps_completed = out_result[:steps_completed]
      Blazen::FFI.blazen_full_finetune_result_free(out_result.pointer)
      FullFineTuneResult.new(
        output_dir: output_dir,
        final_loss: final_loss,
        steps_completed: steps_completed,
      )
    end

    def decode_trained_adapter(out_adapter)
      dir_ptr = out_adapter[:adapter_dir]
      adapter_dir =
        if dir_ptr.nil? || dir_ptr.null?
          nil
        else
          dir_ptr.read_string.force_encoding(Encoding::UTF_8)
        end
      final_loss  = out_adapter[:final_loss]
      total_steps = out_adapter[:total_steps]
      Blazen::FFI.blazen_trained_adapter_free(out_adapter.pointer)
      TrainedAdapter.new(
        adapter_dir: adapter_dir,
        final_loss: final_loss,
        total_steps: total_steps,
      )
    end

    # Why: the cabi reads C-string fields of `BlazenHfLoadOptions` during the
    # call but copies them before returning, so the MemoryPointer buffers
    # only need to outlive the yield. We hold strong refs in a local array
    # so GC can't reclaim them mid-call.
    def with_hf_load_options(options)
      return yield(nil) if options.nil?

      keepalive = []
      to_cstr = lambda do |s|
        next nil if s.nil?

        mp = ::FFI::MemoryPointer.from_string(s.to_s)
        keepalive << mp
        mp
      end

      struct = Blazen::FFI::BlazenHfLoadOptions.new
      struct[:backend_hint] = Blazen::BackendHint.to_cabi(options.backend_hint)
      struct[:revision]  = to_cstr.call(options.revision)
      struct[:hf_token]  = to_cstr.call(options.hf_token)
      struct[:cache_dir] = to_cstr.call(options.cache_dir)
      struct[:device]    = to_cstr.call(options.device)
      struct[:gguf_file] = to_cstr.call(options.gguf_file)
      struct[:memory_estimate_bytes] = (options.memory_estimate_bytes || 0).to_i
      struct[:pool]      = to_cstr.call(options.pool)

      begin
        yield struct
      ensure
        keepalive.clear
      end
    end

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
