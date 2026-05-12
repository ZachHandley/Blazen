package blazen

import (
	"context"
	"runtime"
	"sync"

	uniffiblazen "github.com/zachhandley/Blazen/bindings/go/internal/uniffi/blazen"
)

// PersistedEvent is a serialised representation of a queued event captured
// in a [WorkflowCheckpoint].
//
// EventType is the event type identifier (e.g. `"blazen::StartEvent"`).
// DataJSON is the JSON-encoded payload of the original event — the
// upstream Rust type stores a `serde_json::Value`, which is not a
// UniFFI-supported wire type, so a JSON string crosses the FFI boundary
// instead. Decode it with `encoding/json` on the Go side.
type PersistedEvent struct {
	EventType string
	DataJSON  string
}

// toFFI converts the Go-facing PersistedEvent to the generated FFI record.
//
// The generated bindgen named the second field `DataJson` (PascalCase
// Json, not JSON); this converter is the single place that knows about
// that naming quirk.
func (e PersistedEvent) toFFI() uniffiblazen.PersistedEvent {
	return uniffiblazen.PersistedEvent{
		EventType: e.EventType,
		DataJson:  e.DataJSON,
	}
}

// persistedEventFromFFI converts a generated FFI PersistedEvent into the
// package-public form. See [PersistedEvent.toFFI] for the naming
// rationale.
func persistedEventFromFFI(e uniffiblazen.PersistedEvent) PersistedEvent {
	return PersistedEvent{
		EventType: e.EventType,
		DataJSON:  e.DataJson,
	}
}

// WorkflowCheckpoint is a snapshot of a workflow's state at a point in
// time, persisted via a [CheckpointStore] so a run can be resumed later.
//
// StateJSON and MetadataJSON are JSON-encoded strings — decode them with
// `encoding/json`. PendingEvents are the events still queued at the
// moment the checkpoint was taken. TimestampMs is Unix-epoch
// milliseconds. RunID is a UUID-formatted string identifying the run.
type WorkflowCheckpoint struct {
	WorkflowName  string
	RunID         string
	TimestampMs   uint64
	StateJSON     string
	PendingEvents []PersistedEvent
	MetadataJSON  string
}

// toFFI converts the Go-facing WorkflowCheckpoint into the generated FFI
// record. Field-name acronym casing (RunID, StateJSON, MetadataJSON) is
// remapped to the bindgen's `RunId` / `StateJson` / `MetadataJson` here.
func (c WorkflowCheckpoint) toFFI() uniffiblazen.WorkflowCheckpoint {
	ffiEvents := make([]uniffiblazen.PersistedEvent, len(c.PendingEvents))
	for i, ev := range c.PendingEvents {
		ffiEvents[i] = ev.toFFI()
	}
	return uniffiblazen.WorkflowCheckpoint{
		WorkflowName:  c.WorkflowName,
		RunId:         c.RunID,
		TimestampMs:   c.TimestampMs,
		StateJson:     c.StateJSON,
		PendingEvents: ffiEvents,
		MetadataJson:  c.MetadataJSON,
	}
}

// workflowCheckpointFromFFI converts the generated FFI record into the
// package-public form. See [WorkflowCheckpoint.toFFI] for the naming
// rationale.
func workflowCheckpointFromFFI(c uniffiblazen.WorkflowCheckpoint) WorkflowCheckpoint {
	events := make([]PersistedEvent, len(c.PendingEvents))
	for i, ev := range c.PendingEvents {
		events[i] = persistedEventFromFFI(ev)
	}
	return WorkflowCheckpoint{
		WorkflowName:  c.WorkflowName,
		RunID:         c.RunId,
		TimestampMs:   c.TimestampMs,
		StateJSON:     c.StateJson,
		PendingEvents: events,
		MetadataJSON:  c.MetadataJson,
	}
}

// CheckpointStore is an opaque handle to a durable workflow-checkpoint
// backend. Two factories are provided: [NewRedbCheckpointStore] (embedded
// file-backed via redb) and [NewValkeyCheckpointStore] (Redis/Valkey for
// distributed setups).
//
// Methods come in two flavours: an async form taking a [context.Context]
// which honours cancellation by detaching from the wait (the Rust-side
// operation continues in the background; cancellation propagation into
// the runtime is a known gap pending an upstream UniFFI feature), and a
// `…Blocking` form that runs synchronously on the calling goroutine.
//
// Call [CheckpointStore.Close] when finished to release the underlying
// native handle. A finalizer is attached as a safety net but explicit
// Close is preferred for predictable resource release.
type CheckpointStore struct {
	inner *uniffiblazen.CheckpointStore
	once  sync.Once
}

// Save persists the checkpoint, overwriting any existing entry with the
// same RunID.
//
// The Rust side of the call is blocking from Go's perspective. To honour
// ctx cancellation Save launches the FFI call on a background goroutine
// and waits on ctx.Done(); when ctx fires first the function returns
// ctx.Err() immediately. The Rust-side write may still complete in the
// background.
func (s *CheckpointStore) Save(ctx context.Context, cp WorkflowCheckpoint) error {
	if s == nil || s.inner == nil {
		return &ValidationError{Message: "checkpoint store has been closed"}
	}
	ffi := cp.toFFI()
	done := make(chan error, 1)
	go func() {
		done <- s.inner.Save(ffi)
	}()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-done:
		if err != nil {
			return wrapErr(err)
		}
		return nil
	}
}

// SaveBlocking is the synchronous form of [CheckpointStore.Save]. It
// blocks the calling goroutine until the Rust-side write completes.
func (s *CheckpointStore) SaveBlocking(cp WorkflowCheckpoint) error {
	if s == nil || s.inner == nil {
		return &ValidationError{Message: "checkpoint store has been closed"}
	}
	if err := s.inner.SaveBlocking(cp.toFFI()); err != nil {
		return wrapErr(err)
	}
	return nil
}

// Load fetches the checkpoint for runID, returning (nil, nil) when no
// checkpoint exists for that id.
//
// ctx cancellation semantics match [CheckpointStore.Save].
func (s *CheckpointStore) Load(ctx context.Context, runID string) (*WorkflowCheckpoint, error) {
	if s == nil || s.inner == nil {
		return nil, &ValidationError{Message: "checkpoint store has been closed"}
	}
	type loadResult struct {
		cp  *uniffiblazen.WorkflowCheckpoint
		err error
	}
	done := make(chan loadResult, 1)
	go func() {
		cp, err := s.inner.Load(runID)
		done <- loadResult{cp: cp, err: err}
	}()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case r := <-done:
		if r.err != nil {
			return nil, wrapErr(r.err)
		}
		if r.cp == nil {
			return nil, nil
		}
		out := workflowCheckpointFromFFI(*r.cp)
		return &out, nil
	}
}

// LoadBlocking is the synchronous form of [CheckpointStore.Load]. It
// returns (nil, nil) when no checkpoint exists for runID.
func (s *CheckpointStore) LoadBlocking(runID string) (*WorkflowCheckpoint, error) {
	if s == nil || s.inner == nil {
		return nil, &ValidationError{Message: "checkpoint store has been closed"}
	}
	cp, err := s.inner.LoadBlocking(runID)
	if err != nil {
		return nil, wrapErr(err)
	}
	if cp == nil {
		return nil, nil
	}
	out := workflowCheckpointFromFFI(*cp)
	return &out, nil
}

// Delete removes the checkpoint for runID. Deleting a runID with no
// stored checkpoint is a no-op (no error returned).
//
// ctx cancellation semantics match [CheckpointStore.Save].
func (s *CheckpointStore) Delete(ctx context.Context, runID string) error {
	if s == nil || s.inner == nil {
		return &ValidationError{Message: "checkpoint store has been closed"}
	}
	done := make(chan error, 1)
	go func() {
		done <- s.inner.Delete(runID)
	}()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-done:
		if err != nil {
			return wrapErr(err)
		}
		return nil
	}
}

// DeleteBlocking is the synchronous form of [CheckpointStore.Delete]. A
// runID with no stored checkpoint is a no-op.
func (s *CheckpointStore) DeleteBlocking(runID string) error {
	if s == nil || s.inner == nil {
		return &ValidationError{Message: "checkpoint store has been closed"}
	}
	if err := s.inner.DeleteBlocking(runID); err != nil {
		return wrapErr(err)
	}
	return nil
}

// List returns every checkpoint stored in this backend. The order is
// backend-defined and should not be relied upon.
//
// ctx cancellation semantics match [CheckpointStore.Save].
func (s *CheckpointStore) List(ctx context.Context) ([]WorkflowCheckpoint, error) {
	if s == nil || s.inner == nil {
		return nil, &ValidationError{Message: "checkpoint store has been closed"}
	}
	type listResult struct {
		cps []uniffiblazen.WorkflowCheckpoint
		err error
	}
	done := make(chan listResult, 1)
	go func() {
		cps, err := s.inner.List()
		done <- listResult{cps: cps, err: err}
	}()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case r := <-done:
		if r.err != nil {
			return nil, wrapErr(r.err)
		}
		return convertCheckpointSlice(r.cps), nil
	}
}

// ListBlocking is the synchronous form of [CheckpointStore.List].
func (s *CheckpointStore) ListBlocking() ([]WorkflowCheckpoint, error) {
	if s == nil || s.inner == nil {
		return nil, &ValidationError{Message: "checkpoint store has been closed"}
	}
	cps, err := s.inner.ListBlocking()
	if err != nil {
		return nil, wrapErr(err)
	}
	return convertCheckpointSlice(cps), nil
}

// ListRunIDs returns just the run-id strings for every stored
// checkpoint. Useful for indexing UIs where the full state payload is
// not needed.
//
// ctx cancellation semantics match [CheckpointStore.Save].
func (s *CheckpointStore) ListRunIDs(ctx context.Context) ([]string, error) {
	if s == nil || s.inner == nil {
		return nil, &ValidationError{Message: "checkpoint store has been closed"}
	}
	type idsResult struct {
		ids []string
		err error
	}
	done := make(chan idsResult, 1)
	go func() {
		ids, err := s.inner.ListRunIds()
		done <- idsResult{ids: ids, err: err}
	}()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case r := <-done:
		if r.err != nil {
			return nil, wrapErr(r.err)
		}
		return r.ids, nil
	}
}

// ListRunIDsBlocking is the synchronous form of
// [CheckpointStore.ListRunIDs].
func (s *CheckpointStore) ListRunIDsBlocking() ([]string, error) {
	if s == nil || s.inner == nil {
		return nil, &ValidationError{Message: "checkpoint store has been closed"}
	}
	ids, err := s.inner.ListRunIdsBlocking()
	if err != nil {
		return nil, wrapErr(err)
	}
	return ids, nil
}

// Close releases the underlying native handle. It is safe to call Close
// multiple times and from multiple goroutines; subsequent calls are
// no-ops.
func (s *CheckpointStore) Close() {
	s.once.Do(func() {
		if s.inner != nil {
			s.inner.Destroy()
			s.inner = nil
		}
	})
}

// convertCheckpointSlice maps a slice of FFI checkpoint records into the
// Go-facing form. Factored out so the async and blocking List paths
// share one conversion implementation.
func convertCheckpointSlice(cps []uniffiblazen.WorkflowCheckpoint) []WorkflowCheckpoint {
	out := make([]WorkflowCheckpoint, len(cps))
	for i, cp := range cps {
		out[i] = workflowCheckpointFromFFI(cp)
	}
	return out
}

// NewRedbCheckpointStore opens (or creates) a redb-backed checkpoint
// store at path. The database file is created if it does not already
// exist; re-opening an existing file is safe and preserves prior
// checkpoints. Suitable for single-process workflows where embedded
// durable storage is sufficient.
//
// Call [CheckpointStore.Close] when finished to release the underlying
// native handle.
func NewRedbCheckpointStore(path string) (*CheckpointStore, error) {
	ensureInit()
	inner, err := uniffiblazen.NewRedbCheckpointStore(path)
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCheckpointStore(inner), nil
}

// NewValkeyCheckpointStore opens a Valkey-/Redis-backed checkpoint store
// connected to url (`redis://host:port/db` or `rediss://` for TLS).
//
// Pass ttlSeconds > 0 to make every saved checkpoint auto-expire after
// that many seconds — useful for transient workflows where stale
// checkpoints should not accumulate. Pass 0 to keep checkpoints
// indefinitely.
//
// Call [CheckpointStore.Close] when finished to release the underlying
// native handle.
func NewValkeyCheckpointStore(url string, ttlSeconds uint64) (*CheckpointStore, error) {
	ensureInit()
	var ttl *uint64
	if ttlSeconds > 0 {
		t := ttlSeconds
		ttl = &t
	}
	inner, err := uniffiblazen.NewValkeyCheckpointStore(url, ttl)
	if err != nil {
		return nil, wrapErr(err)
	}
	return newCheckpointStore(inner), nil
}

// newCheckpointStore wraps an FFI handle in the public opaque type and
// attaches a finalizer as a safety net. Callers should still invoke
// [CheckpointStore.Close] explicitly for predictable release.
func newCheckpointStore(inner *uniffiblazen.CheckpointStore) *CheckpointStore {
	s := &CheckpointStore{inner: inner}
	runtime.SetFinalizer(s, func(st *CheckpointStore) { st.Close() })
	return s
}
