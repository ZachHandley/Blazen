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
