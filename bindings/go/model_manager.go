package blazen

import (
	"context"
	"runtime"
	"sync"

	uniffiblazen "github.com/zachhandley/Blazen/bindings/go/internal/uniffi/blazen"
)

// AdapterOptions controls how an adapter is mounted by
// [ModelManager.LoadAdapter] (and by foreign [LocalModel] implementations).
//
// AdapterID is the empty string when the caller wants the backend to assign
// an id; otherwise it pins a stable id for cross-call references. Scale is
// the adapter scaling factor (1.0 means the trained weights are applied at
// full strength).
type AdapterOptions struct {
	AdapterID string
	Scale     float32
}

func (o AdapterOptions) toFFI() uniffiblazen.AdapterOptionsRecord {
	return uniffiblazen.AdapterOptionsRecord{
		AdapterId: o.AdapterID,
		Scale:     o.Scale,
	}
}

// AdapterHandle is the result returned by [LocalModel.LoadAdapter]. It
// mirrors the upstream `blazen_llm::AdapterHandle` and is what the
// [ModelManager] feeds back into [LocalModel.UnloadAdapter] later.
//
// MountStrategy is one of "attached", "rebuilt", "merged" — kept as a
// string discriminator so adding a new strategy upstream does not break
// the binding.
type AdapterHandle struct {
	AdapterID     string
	MemoryBytes   uint64
	MountStrategy string
}

func (h AdapterHandle) toFFI() uniffiblazen.AdapterHandleRecord {
	return uniffiblazen.AdapterHandleRecord{
		AdapterId:     h.AdapterID,
		MemoryBytes:   h.MemoryBytes,
		MountStrategy: h.MountStrategy,
	}
}

func adapterHandleFromFFI(h uniffiblazen.AdapterHandleRecord) AdapterHandle {
	return AdapterHandle{
		AdapterID:     h.AdapterId,
		MemoryBytes:   h.MemoryBytes,
		MountStrategy: h.MountStrategy,
	}
}

// AdapterStatus is the snapshot returned by [ModelManager.ListAdapters] and
// is also embedded in [ModelStatus.Adapters].
type AdapterStatus struct {
	AdapterID   string
	Scale       float32
	SourceDir   string
	MemoryBytes uint64
}

func adapterStatusFromFFI(s uniffiblazen.AdapterStatusRecord) AdapterStatus {
	return AdapterStatus{
		AdapterID:   s.AdapterId,
		Scale:       s.Scale,
		SourceDir:   s.SourceDir,
		MemoryBytes: s.MemoryBytes,
	}
}

func adapterStatusToFFI(s AdapterStatus) uniffiblazen.AdapterStatusRecord {
	return uniffiblazen.AdapterStatusRecord{
		AdapterId:   s.AdapterID,
		Scale:       s.Scale,
		SourceDir:   s.SourceDir,
		MemoryBytes: s.MemoryBytes,
	}
}

func adapterStatusSliceFromFFI(in []uniffiblazen.AdapterStatusRecord) []AdapterStatus {
	if in == nil {
		return nil
	}
	out := make([]AdapterStatus, len(in))
	for i, s := range in {
		out[i] = adapterStatusFromFFI(s)
	}
	return out
}

func adapterStatusSliceToFFI(in []AdapterStatus) []uniffiblazen.AdapterStatusRecord {
	if in == nil {
		return nil
	}
	out := make([]uniffiblazen.AdapterStatusRecord, len(in))
	for i, s := range in {
		out[i] = adapterStatusToFFI(s)
	}
	return out
}

// ModelStatus is the per-model snapshot returned by [ModelManager.Status].
//
// Pool is one of "cpu" or "gpu:N" where N is the GPU index.
type ModelStatus struct {
	ID                  string
	Loaded              bool
	MemoryEstimateBytes uint64
	Pool                string
	Adapters            []AdapterStatus
}

func modelStatusFromFFI(s uniffiblazen.ModelStatusRecord) ModelStatus {
	return ModelStatus{
		ID:                  s.Id,
		Loaded:              s.Loaded,
		MemoryEstimateBytes: s.MemoryEstimateBytes,
		Pool:                s.Pool,
		Adapters:            adapterStatusSliceFromFFI(s.Adapters),
	}
}

func modelStatusSliceFromFFI(in []uniffiblazen.ModelStatusRecord) []ModelStatus {
	if in == nil {
		return nil
	}
	out := make([]ModelStatus, len(in))
	for i, s := range in {
		out[i] = modelStatusFromFFI(s)
	}
	return out
}

// PoolStatus describes a configured memory pool and its byte budget.
type PoolStatus struct {
	Pool        string
	BudgetBytes uint64
}

func poolStatusFromFFI(p uniffiblazen.PoolStatusRecord) PoolStatus {
	return PoolStatus{
		Pool:        p.Pool,
		BudgetBytes: p.BudgetBytes,
	}
}

func poolStatusSliceFromFFI(in []uniffiblazen.PoolStatusRecord) []PoolStatus {
	if in == nil {
		return nil
	}
	out := make([]PoolStatus, len(in))
	for i, p := range in {
		out[i] = poolStatusFromFFI(p)
	}
	return out
}

// LocalModel is the Go-side trait foreign callers implement to plug a
// custom on-device model into [ModelManager].
//
// All methods are called from a goroutine driven by the underlying
// async runtime, so implementations must be safe for concurrent use.
//
// Device returns one of "cpu", "cuda:N", "metal", "rocm:N", "vulkan",
// or "xla". MemoryBytes returns a pointer for parity with the upstream
// optional — return nil to mean "unknown". The adapter verbs may return
// a [*UnsupportedError] when the backend does not implement adapters.
type LocalModel interface {
	Load(ctx context.Context) error
	Unload(ctx context.Context) error
	IsLoaded(ctx context.Context) bool
	Device(ctx context.Context) string
	MemoryBytes(ctx context.Context) *uint64
	LoadAdapter(ctx context.Context, adapterDir string, options AdapterOptions) (AdapterHandle, error)
	UnloadAdapter(ctx context.Context, handle AdapterHandle) error
	ListAdapters(ctx context.Context) []AdapterStatus
}

// localModelAdapter satisfies the generated [uniffiblazen.ForeignLocalModel]
// interface and forwards each FFI call into the user-supplied [LocalModel].
type localModelAdapter struct {
	inner LocalModel
}

func (a *localModelAdapter) Load() error {
	if err := a.inner.Load(context.Background()); err != nil {
		return unwrapToValidation(err)
	}
	return nil
}

func (a *localModelAdapter) Unload() error {
	if err := a.inner.Unload(context.Background()); err != nil {
		return unwrapToValidation(err)
	}
	return nil
}

func (a *localModelAdapter) IsLoaded() bool {
	return a.inner.IsLoaded(context.Background())
}

func (a *localModelAdapter) Device() string {
	return a.inner.Device(context.Background())
}

func (a *localModelAdapter) MemoryBytes() *uint64 {
	return a.inner.MemoryBytes(context.Background())
}

func (a *localModelAdapter) LoadAdapter(adapterDir string, options uniffiblazen.AdapterOptionsRecord) (uniffiblazen.AdapterHandleRecord, error) {
	opts := AdapterOptions{AdapterID: options.AdapterId, Scale: options.Scale}
	h, err := a.inner.LoadAdapter(context.Background(), adapterDir, opts)
	if err != nil {
		return uniffiblazen.AdapterHandleRecord{}, unwrapToValidation(err)
	}
	return h.toFFI(), nil
}

func (a *localModelAdapter) UnloadAdapter(handle uniffiblazen.AdapterHandleRecord) error {
	h := adapterHandleFromFFI(handle)
	if err := a.inner.UnloadAdapter(context.Background(), h); err != nil {
		return unwrapToValidation(err)
	}
	return nil
}

func (a *localModelAdapter) ListAdapters() []uniffiblazen.AdapterStatusRecord {
	return adapterStatusSliceToFFI(a.inner.ListAdapters(context.Background()))
}

// ModelManager is a memory-budget-aware registry of local
// (on-device) models with per-pool LRU eviction.
//
// Construct via [NewModelManager], [NewModelManagerWithBudgetsGB], or
// [NewModelManagerWithPoolBudgets]. Register foreign-side models with
// [ModelManager.RegisterLocal], then drive loads / unloads / adapter
// lifecycle from any goroutine.
//
// ModelManager owns a native handle; call [ModelManager.Close] when
// finished. A finalizer is attached as a safety net.
type ModelManager struct {
	inner *uniffiblazen.UniffiModelManager
	once  sync.Once
	// Why: keep adapter values alive for the registered foreign models so
	// the GC does not collect them while the Rust side still holds the
	// callback-interface handle.
	pinned sync.Map
}

func newModelManager(inner *uniffiblazen.UniffiModelManager) *ModelManager {
	m := &ModelManager{inner: inner}
	runtime.SetFinalizer(m, func(mm *ModelManager) { mm.Close() })
	return m
}

// NewModelManager constructs a manager with no budget enforcement (both
// "cpu" and "gpu:0" pools seeded with the maximum u64 budget). Suitable
// for tests and small scripts; production deployments should prefer one
// of the budgeted constructors.
func NewModelManager() *ModelManager {
	ensureInit()
	return newModelManager(uniffiblazen.NewUniffiModelManager())
}

// NewModelManagerWithBudgetsGB constructs a manager with one CPU-pool
// budget and one GPU-pool ("gpu:0") budget, both expressed in
// gigabytes.
func NewModelManagerWithBudgetsGB(cpuRAMGB, gpuVRAMGB float64) *ModelManager {
	ensureInit()
	return newModelManager(uniffiblazen.UniffiModelManagerWithBudgetsGb(cpuRAMGB, gpuVRAMGB))
}

// NewModelManagerWithPoolBudgets constructs a manager with explicit
// per-pool budgets. Keys are pool labels ("cpu", "gpu", "gpu:0",
// "gpu:1", ...); values are budgets in gigabytes.
func NewModelManagerWithPoolBudgets(budgetsGB map[string]float64) (*ModelManager, error) {
	ensureInit()
	inner, err := uniffiblazen.UniffiModelManagerWithPoolBudgets(budgetsGB)
	if err != nil {
		return nil, wrapErr(err)
	}
	return newModelManager(inner), nil
}

// RegisterLocal registers a foreign-implemented [LocalModel] under
// modelID. memoryEstimateBytes is the model's estimated footprint and
// is charged against the pool derived from the model's Device() when
// it loads.
func (m *ModelManager) RegisterLocal(ctx context.Context, modelID string, model LocalModel, memoryEstimateBytes uint64) error {
	if m.inner == nil {
		return &ValidationError{Message: "model manager has been closed"}
	}
	if model == nil {
		return &ValidationError{Message: "model must not be nil"}
	}
	adapter := &localModelAdapter{inner: model}
	m.pinned.Store(modelID, adapter)

	done := make(chan error, 1)
	go func() {
		done <- m.inner.RegisterLocal(modelID, adapter, memoryEstimateBytes)
	}()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-done:
		if err != nil {
			m.pinned.Delete(modelID)
			return wrapErr(err)
		}
		return nil
	}
}

// Load drives the model identified by modelID through its load path,
// reserving budget in the appropriate pool and evicting LRU peers as
// needed.
func (m *ModelManager) Load(ctx context.Context, modelID string) error {
	if m.inner == nil {
		return &ValidationError{Message: "model manager has been closed"}
	}
	done := make(chan error, 1)
	go func() {
		done <- m.inner.Load(modelID)
	}()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-done:
		return wrapErr(err)
	}
}

// Unload releases the model identified by modelID and returns its
// budget to the owning pool.
func (m *ModelManager) Unload(ctx context.Context, modelID string) error {
	if m.inner == nil {
		return &ValidationError{Message: "model manager has been closed"}
	}
	done := make(chan error, 1)
	go func() {
		done <- m.inner.Unload(modelID)
	}()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-done:
		return wrapErr(err)
	}
}

// IsLoaded reports whether the model identified by modelID is currently
// resident. Unregistered ids return (false, nil).
func (m *ModelManager) IsLoaded(ctx context.Context, modelID string) (bool, error) {
	if m.inner == nil {
		return false, &ValidationError{Message: "model manager has been closed"}
	}
	done := make(chan bool, 1)
	go func() {
		done <- m.inner.IsLoaded(modelID)
	}()
	select {
	case <-ctx.Done():
		return false, ctx.Err()
	case v := <-done:
		return v, nil
	}
}

// EnsureLoaded loads the model if it is not already resident. A no-op
// when the model is already loaded.
func (m *ModelManager) EnsureLoaded(ctx context.Context, modelID string) error {
	if m.inner == nil {
		return &ValidationError{Message: "model manager has been closed"}
	}
	done := make(chan error, 1)
	go func() {
		done <- m.inner.EnsureLoaded(modelID)
	}()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-done:
		return wrapErr(err)
	}
}

// Status returns a snapshot of every registered model with its current
// load state, pool assignment, and mounted adapters.
func (m *ModelManager) Status(ctx context.Context) ([]ModelStatus, error) {
	if m.inner == nil {
		return nil, &ValidationError{Message: "model manager has been closed"}
	}
	done := make(chan []uniffiblazen.ModelStatusRecord, 1)
	go func() {
		done <- m.inner.Status()
	}()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case res := <-done:
		return modelStatusSliceFromFFI(res), nil
	}
}

// Pools returns the configured memory pools and their byte budgets.
// This is a synchronous accessor on the Rust side and does not honour
// context cancellation.
func (m *ModelManager) Pools() []PoolStatus {
	if m.inner == nil {
		return nil
	}
	return poolStatusSliceFromFFI(m.inner.Pools())
}

// UsedBytes returns the bytes currently checked out from the named
// pool ("cpu", "gpu", "gpu:N").
func (m *ModelManager) UsedBytes(ctx context.Context, pool string) (uint64, error) {
	if m.inner == nil {
		return 0, &ValidationError{Message: "model manager has been closed"}
	}
	type result struct {
		v   uint64
		err error
	}
	done := make(chan result, 1)
	go func() {
		v, err := m.inner.UsedBytes(pool)
		done <- result{v: v, err: err}
	}()
	select {
	case <-ctx.Done():
		return 0, ctx.Err()
	case r := <-done:
		if r.err != nil {
			return 0, wrapErr(r.err)
		}
		return r.v, nil
	}
}

// AvailableBytes returns the bytes remaining in the named pool.
func (m *ModelManager) AvailableBytes(ctx context.Context, pool string) (uint64, error) {
	if m.inner == nil {
		return 0, &ValidationError{Message: "model manager has been closed"}
	}
	type result struct {
		v   uint64
		err error
	}
	done := make(chan result, 1)
	go func() {
		v, err := m.inner.AvailableBytes(pool)
		done <- result{v: v, err: err}
	}()
	select {
	case <-ctx.Done():
		return 0, ctx.Err()
	case r := <-done:
		if r.err != nil {
			return 0, wrapErr(r.err)
		}
		return r.v, nil
	}
}

// BackendHint identifies a local-inference backend. It is returned by
// [ModelManager.LoadFromHf] and may be passed via [HfLoadOptions.BackendHint]
// to force a specific provider instead of letting the loader infer one.
type BackendHint string

const (
	// BackendHintMistralrs selects `mistral.rs` — broad architecture
	// coverage, safetensors + GGUF, vision/multimodal.
	BackendHintMistralrs BackendHint = "mistralrs"
	// BackendHintCandle selects pure-Rust `candle` — safetensors + GGUF
	// for the architectures candle ships.
	BackendHintCandle BackendHint = "candle"
	// BackendHintLlamacpp selects `llama.cpp` — GGUF only, best CPU
	// performance and lowest memory.
	BackendHintLlamacpp BackendHint = "llamacpp"
)

func (h BackendHint) toFFI() uniffiblazen.BackendHintEnum {
	switch h {
	case BackendHintCandle:
		return uniffiblazen.BackendHintEnumCandle
	case BackendHintLlamacpp:
		return uniffiblazen.BackendHintEnumLlamacpp
	default:
		return uniffiblazen.BackendHintEnumMistralrs
	}
}

// HfLoadOptions controls how [ModelManager.LoadFromHf] probes a Hugging
// Face repo and builds the local-inference provider. Every field is
// optional; a zero-value struct accepts loader defaults.
//
// Pool is a label ("cpu", "gpu", "gpu:N"); it defaults to "cpu" when nil.
type HfLoadOptions struct {
	BackendHint         *BackendHint
	Revision            *string
	HfToken             *string
	CacheDir            *string
	Device              *string
	GgufFile            *string
	MemoryEstimateBytes *uint64
	Pool                *string
}

func (o HfLoadOptions) toFFI() uniffiblazen.HfLoadOptionsRecord {
	rec := uniffiblazen.HfLoadOptionsRecord{
		Revision:            o.Revision,
		HfToken:             o.HfToken,
		CacheDir:            o.CacheDir,
		Device:              o.Device,
		GgufFile:            o.GgufFile,
		MemoryEstimateBytes: o.MemoryEstimateBytes,
		Pool:                o.Pool,
	}
	if o.BackendHint != nil {
		hint := o.BackendHint.toFFI()
		rec.BackendHint = &hint
	}
	return rec
}

// LoadFromHf probes a Hugging Face repo, picks a local-inference backend,
// builds the provider, and registers it under id. The returned
// [BackendHint] identifies which backend was selected (or the forced
// override when opts.BackendHint was non-nil).
//
// The model starts unloaded — call [ModelManager.Load] or
// [ModelManager.EnsureLoaded] to materialize it.
//
// Errors on empty repo id, gated/missing repo, PEFT-adapter-only repo
// (use [ModelManager.LoadAdapter] instead), missing backend feature, or
// any provider construction failure.
func (m *ModelManager) LoadFromHf(ctx context.Context, id, repo string, opts HfLoadOptions) (BackendHint, error) {
	if m.inner == nil {
		return "", &ValidationError{Message: "model manager has been closed"}
	}
	type result struct {
		backend string
		err     error
	}
	done := make(chan result, 1)
	go func() {
		b, err := m.inner.LoadFromHf(id, repo, opts.toFFI())
		done <- result{backend: b, err: err}
	}()
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case r := <-done:
		if r.err != nil {
			return "", wrapErr(r.err)
		}
		return BackendHint(r.backend), nil
	}
}

// LoadAdapter mounts a PEFT-format LoRA adapter on the model and
// returns the adapter id assigned by the backend.
func (m *ModelManager) LoadAdapter(ctx context.Context, modelID, adapterDir string, opts AdapterOptions) (string, error) {
	if m.inner == nil {
		return "", &ValidationError{Message: "model manager has been closed"}
	}
	type result struct {
		id  string
		err error
	}
	done := make(chan result, 1)
	go func() {
		id, err := m.inner.LoadAdapter(modelID, adapterDir, opts.toFFI())
		done <- result{id: id, err: err}
	}()
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case r := <-done:
		if r.err != nil {
			return "", wrapErr(r.err)
		}
		return r.id, nil
	}
}

// UnloadAdapter detaches the adapter identified by adapterID from the
// model.
func (m *ModelManager) UnloadAdapter(ctx context.Context, modelID, adapterID string) error {
	if m.inner == nil {
		return &ValidationError{Message: "model manager has been closed"}
	}
	done := make(chan error, 1)
	go func() {
		done <- m.inner.UnloadAdapter(modelID, adapterID)
	}()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-done:
		return wrapErr(err)
	}
}

// ListAdapters returns the adapters currently mounted on the model.
func (m *ModelManager) ListAdapters(ctx context.Context, modelID string) ([]AdapterStatus, error) {
	if m.inner == nil {
		return nil, &ValidationError{Message: "model manager has been closed"}
	}
	type result struct {
		v   []uniffiblazen.AdapterStatusRecord
		err error
	}
	done := make(chan result, 1)
	go func() {
		v, err := m.inner.ListAdapters(modelID)
		done <- result{v: v, err: err}
	}()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case r := <-done:
		if r.err != nil {
			return nil, wrapErr(r.err)
		}
		return adapterStatusSliceFromFFI(r.v), nil
	}
}

// Close releases the underlying native handle. It is safe to call
// multiple times and from multiple goroutines; subsequent calls are
// no-ops.
func (m *ModelManager) Close() {
	m.once.Do(func() {
		if m.inner != nil {
			m.inner.Destroy()
			m.inner = nil
		}
		m.pinned.Range(func(k, _ any) bool {
			m.pinned.Delete(k)
			return true
		})
	})
}

// Why: catch generator drift early. If UniFFI renames or retypes the
// ForeignLocalModel trait, this declaration fails to build.
var _ uniffiblazen.ForeignLocalModel = (*localModelAdapter)(nil)

// SchedulerKind selects the learning-rate scheduler shape used by
// [TrainConfig.Scheduler].
type SchedulerKind string

const (
	// SchedulerKindConstant keeps the learning rate flat for the entire run.
	SchedulerKindConstant SchedulerKind = "constant"
	// SchedulerKindLinear linearly decays the learning rate from peak to zero.
	SchedulerKindLinear SchedulerKind = "linear"
	// SchedulerKindCosine follows a half-cosine decay from peak to zero.
	SchedulerKindCosine SchedulerKind = "cosine"
)

func (s SchedulerKind) toFFI() uniffiblazen.SchedulerKindEnum {
	switch s {
	case SchedulerKindLinear:
		return uniffiblazen.SchedulerKindEnumLinear
	case SchedulerKindCosine:
		return uniffiblazen.SchedulerKindEnumCosine
	default:
		return uniffiblazen.SchedulerKindEnumConstant
	}
}

// MixedPrecision selects the numerical precision used by the training
// loop.
type MixedPrecision string

const (
	// MixedPrecisionNone runs the loop in full f32.
	MixedPrecisionNone MixedPrecision = "none"
	// MixedPrecisionBf16 runs the loop with bfloat16 activations and
	// gradients while keeping a f32 master copy of the optimizer state.
	MixedPrecisionBf16 MixedPrecision = "bf16"
)

func (m MixedPrecision) toFFI() uniffiblazen.MixedPrecisionEnum {
	switch m {
	case MixedPrecisionBf16:
		return uniffiblazen.MixedPrecisionEnumBf16
	default:
		return uniffiblazen.MixedPrecisionEnumNone
	}
}

// LoraConfig describes the LoRA adapter shape that
// [ModelManager.TrainLora] will fit.
type LoraConfig struct {
	Rank          uint32
	Alpha         float32
	Dropout       float32
	TargetModules []string
}

func (l LoraConfig) toFFI() uniffiblazen.LoraConfigRecord {
	return uniffiblazen.LoraConfigRecord{
		Rank:          l.Rank,
		Alpha:         l.Alpha,
		Dropout:       l.Dropout,
		TargetModules: l.TargetModules,
	}
}

// OptimConfig holds AdamW optimizer hyperparameters. GradientClip is
// nil-able — leave it nil to disable gradient clipping.
type OptimConfig struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64
	WeightDecay  float64
	GradientClip *float32
}

func (o OptimConfig) toFFI() uniffiblazen.OptimConfigRecord {
	return uniffiblazen.OptimConfigRecord{
		LearningRate: o.LearningRate,
		Beta1:        o.Beta1,
		Beta2:        o.Beta2,
		Epsilon:      o.Epsilon,
		WeightDecay:  o.WeightDecay,
		GradientClip: o.GradientClip,
	}
}

// SchedulerConfig configures the learning-rate scheduler.
type SchedulerConfig struct {
	Kind        SchedulerKind
	WarmupSteps uint32
}

func (s SchedulerConfig) toFFI() uniffiblazen.SchedulerConfigRecord {
	return uniffiblazen.SchedulerConfigRecord{
		Kind:        s.Kind.toFFI(),
		WarmupSteps: s.WarmupSteps,
	}
}

// TrainConfig is the full configuration for one [ModelManager.TrainLora]
// run.
//
// BaseModelRepo is a HuggingFace repo id ("owner/name") that will be
// downloaded (cached) before training. OutputDir is the directory the
// resulting PEFT adapter is written to. EvalSteps and SaveSteps are
// nil-able — leave them nil to skip the corresponding cadence. Device
// is "cpu", "cuda:N", "metal", etc.; nil lets the trainer pick.
type TrainConfig struct {
	BaseModelRepo             string
	OutputDir                 string
	Lora                      LoraConfig
	Optim                     OptimConfig
	Scheduler                 SchedulerConfig
	MaxSteps                  uint32
	BatchSize                 uint32
	GradientAccumulationSteps uint32
	MaxSeqLen                 uint32
	EvalSteps                 *uint32
	SaveSteps                 *uint32
	Seed                      uint64
	MixedPrecision            MixedPrecision
	Device                    *string
}

func (c TrainConfig) toFFI() uniffiblazen.TrainConfigRecord {
	return uniffiblazen.TrainConfigRecord{
		BaseModelRepo:             c.BaseModelRepo,
		OutputDir:                 c.OutputDir,
		Lora:                      c.Lora.toFFI(),
		Optim:                     c.Optim.toFFI(),
		Scheduler:                 c.Scheduler.toFFI(),
		MaxSteps:                  c.MaxSteps,
		BatchSize:                 c.BatchSize,
		GradientAccumulationSteps: c.GradientAccumulationSteps,
		MaxSeqLen:                 c.MaxSeqLen,
		EvalSteps:                 c.EvalSteps,
		SaveSteps:                 c.SaveSteps,
		Seed:                      c.Seed,
		MixedPrecision:            c.MixedPrecision.toFFI(),
		Device:                    c.Device,
	}
}

// TrainedAdapter is the on-disk descriptor returned by
// [ModelManager.TrainLora]. AdapterDir is immediately mountable via
// [ModelManager.LoadAdapter] on a compatible backend.
type TrainedAdapter struct {
	AdapterDir string
	FinalLoss  float32
	TotalSteps uint64
}

func trainedAdapterFromFFI(r uniffiblazen.TrainedAdapterRecord) TrainedAdapter {
	return TrainedAdapter{
		AdapterDir: r.AdapterDir,
		FinalLoss:  r.FinalLoss,
		TotalSteps: r.TotalSteps,
	}
}

// TrainingEvent is the flattened, host-friendly projection of every
// variant of UniFFI's tagged-union `TrainingEventEnum`.
//
// Kind selects the variant; the optional pointers carry per-variant
// payload fields. Recognised Kind values are "started", "step_completed",
// "evaluating", "eval_completed", "checkpoint_saved", and "finished".
type TrainingEvent struct {
	Kind           string
	Step           *uint64
	Loss           *float32
	LearningRate   *float64
	ElapsedMs      *uint64
	TotalSteps     *uint64
	EvalLoss       *float32
	CheckpointPath *string
	AdapterDir     *string
	FinalLoss      *float32
}

func trainingEventFromFFI(ev uniffiblazen.TrainingEventEnum) TrainingEvent {
	switch v := ev.(type) {
	case uniffiblazen.TrainingEventEnumStarted:
		total := v.TotalSteps
		return TrainingEvent{Kind: "started", TotalSteps: &total}
	case uniffiblazen.TrainingEventEnumStepCompleted:
		step, loss, lr, elapsed := v.Step, v.Loss, v.LearningRate, v.ElapsedMs
		return TrainingEvent{
			Kind:         "step_completed",
			Step:         &step,
			Loss:         &loss,
			LearningRate: &lr,
			ElapsedMs:    &elapsed,
		}
	case uniffiblazen.TrainingEventEnumEvaluating:
		step := v.Step
		return TrainingEvent{Kind: "evaluating", Step: &step}
	case uniffiblazen.TrainingEventEnumEvalCompleted:
		step, loss := v.Step, v.EvalLoss
		return TrainingEvent{Kind: "eval_completed", Step: &step, EvalLoss: &loss}
	case uniffiblazen.TrainingEventEnumCheckpointSaved:
		step, path := v.Step, v.Path
		return TrainingEvent{Kind: "checkpoint_saved", Step: &step, CheckpointPath: &path}
	case uniffiblazen.TrainingEventEnumFinished:
		loss, total, dir := v.FinalLoss, v.TotalSteps, v.AdapterDir
		return TrainingEvent{
			Kind:       "finished",
			FinalLoss:  &loss,
			TotalSteps: &total,
			AdapterDir: &dir,
		}
	default:
		return TrainingEvent{Kind: "unknown"}
	}
}

// TrainingProgress is the callback shape passed to
// [ModelManager.TrainLora]. Returning a non-nil error cancels the run
// with a Cancelled error.
type TrainingProgress func(event TrainingEvent) error

// trainingProgressAdapter satisfies the generated
// [uniffiblazen.ForeignTrainingProgress] interface and forwards each FFI
// event into the user-supplied [TrainingProgress].
type trainingProgressAdapter struct {
	inner TrainingProgress
}

func (a *trainingProgressAdapter) OnEvent(event uniffiblazen.TrainingEventEnum) error {
	if a.inner == nil {
		return nil
	}
	if err := a.inner(trainingEventFromFFI(event)); err != nil {
		return unwrapToValidation(err)
	}
	return nil
}

// JsonlDataset is a handle to a tokenized JSONL fine-tuning corpus
// loaded from disk. Pass it to [ModelManager.TrainLora].
//
// The dataset owns a native handle; UniFFI attaches its own finalizer
// so callers do not need to release it explicitly.
type JsonlDataset struct {
	inner *uniffiblazen.UniffiJsonlDataset
}

// NewJsonlDatasetFromPath loads and tokenizes a JSONL fine-tuning
// corpus from path. tokenizerPath points at a HuggingFace tokenizer.json
// (or equivalent) compatible with the base model. chatTemplate is nil-able;
// when nil the trainer falls back to the tokenizer's built-in template.
// device pins the tensor device ("cpu", "cuda:N", ...); nil uses cpu.
func NewJsonlDatasetFromPath(path, tokenizerPath string, chatTemplate *string, maxSeqLen uint32, device *string, padTokenId uint32) (*JsonlDataset, error) {
	ensureInit()
	inner, err := uniffiblazen.UniffiJsonlDatasetFromPath(path, tokenizerPath, chatTemplate, maxSeqLen, device, padTokenId)
	if err != nil {
		return nil, wrapErr(err)
	}
	return &JsonlDataset{inner: inner}, nil
}

// IsEmpty reports whether the dataset contains zero training examples.
func (d *JsonlDataset) IsEmpty() bool {
	if d == nil || d.inner == nil {
		return true
	}
	return d.inner.IsEmpty()
}

// Len returns the number of training examples in the dataset.
func (d *JsonlDataset) Len() uint64 {
	if d == nil || d.inner == nil {
		return 0
	}
	return d.inner.Len()
}

// TrainLora runs a LoRA fine-tune end-to-end on the model identified
// by config.BaseModelRepo. dataset must be non-nil. progress may be nil
// to skip per-step callbacks.
//
// Errors on closed manager, nil dataset, invalid config (e.g.
// MaxSteps=0), missing base model, training failure, or any callback
// that returns a non-nil error (surfaced as a Cancelled error).
func (m *ModelManager) TrainLora(ctx context.Context, config TrainConfig, dataset *JsonlDataset, progress TrainingProgress) (TrainedAdapter, error) {
	if m.inner == nil {
		return TrainedAdapter{}, &ValidationError{Message: "model manager has been closed"}
	}
	if dataset == nil || dataset.inner == nil {
		return TrainedAdapter{}, &ValidationError{Message: "dataset must not be nil"}
	}

	var progressArg *uniffiblazen.ForeignTrainingProgress
	if progress != nil {
		// Why: keep the adapter rooted on the goroutine stack for the
		// lifetime of the call so the UniFFI handle table cannot drop
		// it while Rust still holds the callback handle.
		var iface uniffiblazen.ForeignTrainingProgress = &trainingProgressAdapter{inner: progress}
		progressArg = &iface
	}

	type result struct {
		v   uniffiblazen.TrainedAdapterRecord
		err error
	}
	done := make(chan result, 1)
	go func() {
		v, err := m.inner.TrainLora(config.toFFI(), dataset.inner, progressArg)
		done <- result{v: v, err: err}
	}()
	select {
	case <-ctx.Done():
		return TrainedAdapter{}, ctx.Err()
	case r := <-done:
		if r.err != nil {
			return TrainedAdapter{}, wrapErr(r.err)
		}
		return trainedAdapterFromFFI(r.v), nil
	}
}

// Why: catch generator drift early. If UniFFI renames or retypes the
// ForeignTrainingProgress trait, this declaration fails to build.
var _ uniffiblazen.ForeignTrainingProgress = (*trainingProgressAdapter)(nil)
