package blazen

import (
	"context"

	uniffiblazen "github.com/zachhandley/Blazen/bindings/go/internal/uniffi/blazen"
)

// TrainCoreConfig is the shared training-loop core configuration used by
// every preference-optimization verb ([ModelManager.TrainDpo],
// [ModelManager.TrainOrpo], [ModelManager.TrainSimpo],
// [ModelManager.TrainKto]) and by [ModelManager.FineTune].
//
// BaseModelRepo is a HuggingFace repo id ("owner/name") that will be
// downloaded (cached) before training. BaseModelRevision pins a git
// revision (branch, tag, or commit sha); nil uses the repo's default.
// OutputDir is the directory the resulting PEFT adapter or full-fine-tuned
// weights are written to. EvalSteps and SaveSteps are nil-able — leave
// them nil to skip the corresponding cadence. Device is "cpu", "cuda:N",
// "metal", etc.; nil lets the trainer pick.
type TrainCoreConfig struct {
	BaseModelRepo             string
	BaseModelRevision         *string
	OutputDir                 string
	MaxSteps                  uint32
	BatchSize                 uint32
	GradientAccumulationSteps uint32
	MaxSeqLen                 uint32
	EvalSteps                 *uint32
	SaveSteps                 *uint32
	Seed                      uint64
	MixedPrecision            MixedPrecision
	Device                    *string
	Optim                     OptimConfig
	Scheduler                 SchedulerConfig
}

func (c TrainCoreConfig) toFFI() uniffiblazen.TrainCoreConfigRecord {
	return uniffiblazen.TrainCoreConfigRecord{
		BaseModelRepo:             c.BaseModelRepo,
		BaseModelRevision:         c.BaseModelRevision,
		OutputDir:                 c.OutputDir,
		MaxSteps:                  c.MaxSteps,
		BatchSize:                 c.BatchSize,
		GradientAccumulationSteps: c.GradientAccumulationSteps,
		MaxSeqLen:                 c.MaxSeqLen,
		EvalSteps:                 c.EvalSteps,
		SaveSteps:                 c.SaveSteps,
		Seed:                      c.Seed,
		MixedPrecision:            c.MixedPrecision.toFFI(),
		Device:                    c.Device,
		Optim:                     c.Optim.toFFI(),
		Scheduler:                 c.Scheduler.toFFI(),
	}
}

// DpoConfig is the Direct Preference Optimization configuration.
//
// DPO requires a frozen reference model. If ReferenceModelRepo is nil,
// the trainer reuses Core.BaseModelRepo (with ReferenceModelRevision as
// the optional pin). Beta controls the KL-regularization strength
// (TRL default: 0.1). LabelSmoothing applies cDPO-style label smoothing
// (0.0 disables it).
type DpoConfig struct {
	Core                   TrainCoreConfig
	Lora                   LoraConfig
	Beta                   float32
	LabelSmoothing         float32
	ReferenceModelRepo     *string
	ReferenceModelRevision *string
}

// DefaultDpoConfig returns a DpoConfig with the documented TRL defaults
// applied (Beta=0.1, LabelSmoothing=0.0, no reference-model override).
// Core, Lora, Optim, and Scheduler are left at their Go zero values —
// callers must populate BaseModelRepo, OutputDir, MaxSteps, etc.
func DefaultDpoConfig() DpoConfig {
	return DpoConfig{
		Beta:           0.1,
		LabelSmoothing: 0.0,
	}
}

func (c DpoConfig) toFFI() uniffiblazen.DpoConfigRecord {
	return uniffiblazen.DpoConfigRecord{
		Core:                   c.Core.toFFI(),
		Lora:                   c.Lora.toFFI(),
		Beta:                   c.Beta,
		LabelSmoothing:         c.LabelSmoothing,
		ReferenceModelRepo:     c.ReferenceModelRepo,
		ReferenceModelRevision: c.ReferenceModelRevision,
	}
}

// OrpoConfig is the Odds Ratio Preference Optimization configuration.
//
// ORPO is reference-free — it combines an SFT loss on chosen responses
// with an odds-ratio loss term weighted by Lambda (TRL default: 0.1).
type OrpoConfig struct {
	Core   TrainCoreConfig
	Lora   LoraConfig
	Lambda float32
}

// DefaultOrpoConfig returns an OrpoConfig with the documented TRL
// default applied (Lambda=0.1). Core and Lora are left at their Go zero
// values — callers must populate BaseModelRepo, OutputDir, MaxSteps, etc.
func DefaultOrpoConfig() OrpoConfig {
	return OrpoConfig{
		Lambda: 0.1,
	}
}

func (c OrpoConfig) toFFI() uniffiblazen.OrpoConfigRecord {
	return uniffiblazen.OrpoConfigRecord{
		Core:   c.Core.toFFI(),
		Lora:   c.Lora.toFFI(),
		Lambda: c.Lambda,
	}
}

// SimpoConfig is the Simple Preference Optimization configuration.
//
// SimPO is reference-free and length-normalized. Defaults track TRL
// `main` (Beta=2.0, Gamma=1.0).
type SimpoConfig struct {
	Core  TrainCoreConfig
	Lora  LoraConfig
	Beta  float32
	Gamma float32
}

// DefaultSimpoConfig returns a SimpoConfig with the TRL-main defaults
// applied (Beta=2.0, Gamma=1.0). Core and Lora are left at their Go
// zero values — callers must populate BaseModelRepo, OutputDir,
// MaxSteps, etc.
func DefaultSimpoConfig() SimpoConfig {
	return SimpoConfig{
		Beta:  2.0,
		Gamma: 1.0,
	}
}

func (c SimpoConfig) toFFI() uniffiblazen.SimpoConfigRecord {
	return uniffiblazen.SimpoConfigRecord{
		Core:  c.Core.toFFI(),
		Lora:  c.Lora.toFFI(),
		Beta:  c.Beta,
		Gamma: c.Gamma,
	}
}

// KtoConfig is the Kahneman-Tversky Optimization configuration.
//
// Like DPO, KTO requires a frozen reference model (defaults to
// Core.BaseModelRepo when ReferenceModelRepo is nil) — but the dataset
// schema differs: each row is a (prompt, completion, desirable) triple
// ([RatedJsonlDataset]) rather than a chosen/rejected pair.
//
// Beta controls KL strength (default 0.1). LambdaD and LambdaU weight
// the desirable- and undesirable-completion losses respectively
// (defaults: 1.0 each).
type KtoConfig struct {
	Core                   TrainCoreConfig
	Lora                   LoraConfig
	Beta                   float32
	LambdaD                float32
	LambdaU                float32
	ReferenceModelRepo     *string
	ReferenceModelRevision *string
}

// DefaultKtoConfig returns a KtoConfig with the documented defaults
// applied (Beta=0.1, LambdaD=1.0, LambdaU=1.0). Core and Lora are left
// at their Go zero values — callers must populate BaseModelRepo,
// OutputDir, MaxSteps, etc.
func DefaultKtoConfig() KtoConfig {
	return KtoConfig{
		Beta:    0.1,
		LambdaD: 1.0,
		LambdaU: 1.0,
	}
}

func (c KtoConfig) toFFI() uniffiblazen.KtoConfigRecord {
	return uniffiblazen.KtoConfigRecord{
		Core:                   c.Core.toFFI(),
		Lora:                   c.Lora.toFFI(),
		Beta:                   c.Beta,
		LambdaD:                c.LambdaD,
		LambdaU:                c.LambdaU,
		ReferenceModelRepo:     c.ReferenceModelRepo,
		ReferenceModelRevision: c.ReferenceModelRevision,
	}
}

// FullFineTuneConfig is the configuration for [ModelManager.FineTune]
// (full-parameter fine-tune — no LoRA; every model parameter is updated).
//
// GradientCheckpointing is accepted for forward compatibility but the
// trainer currently rejects `true` at init time with a typed validation
// error — candle 0.10.2 has no activation-checkpointing primitive.
type FullFineTuneConfig struct {
	Core                  TrainCoreConfig
	GradientCheckpointing bool
}

// DefaultFullFineTuneConfig returns a FullFineTuneConfig with the
// documented default applied (GradientCheckpointing=false). Core is left
// at its Go zero value — callers must populate BaseModelRepo,
// OutputDir, MaxSteps, etc.
func DefaultFullFineTuneConfig() FullFineTuneConfig {
	return FullFineTuneConfig{
		GradientCheckpointing: false,
	}
}

func (c FullFineTuneConfig) toFFI() uniffiblazen.FullFineTuneConfigRecord {
	return uniffiblazen.FullFineTuneConfigRecord{
		Core:                  c.Core.toFFI(),
		GradientCheckpointing: c.GradientCheckpointing,
	}
}

// FullFineTuneResult is the on-disk descriptor returned by
// [ModelManager.FineTune]. Unlike [TrainedAdapter], no PEFT adapter is
// written — the entire model's weights are saved to OutputDir directly.
type FullFineTuneResult struct {
	OutputDir      string
	FinalLoss      float32
	StepsCompleted uint64
}

func fullFineTuneResultFromFFI(r uniffiblazen.FullFineTuneResultRecord) FullFineTuneResult {
	return FullFineTuneResult{
		OutputDir:      r.OutputDir,
		FinalLoss:      r.FinalLoss,
		StepsCompleted: r.StepsCompleted,
	}
}

// PreferenceJsonlDataset is a handle to a tokenized preference-pair
// JSONL corpus loaded from disk. Pass it to [ModelManager.TrainDpo],
// [ModelManager.TrainOrpo], or [ModelManager.TrainSimpo].
//
// Each line of the input file must deserialize to either
// `{"prompt": "...", "chosen": "...", "rejected": "..."}` or
// `{"messages": [...], "chosen": "...", "rejected": "..."}` (the
// latter requires a chatTemplate).
//
// The dataset owns a native handle; UniFFI attaches its own finalizer
// so callers do not need to release it explicitly.
type PreferenceJsonlDataset struct {
	inner *uniffiblazen.UniffiPreferenceJsonlDataset
}

// NewPreferenceJsonlDataset loads and tokenizes a preference-pair JSONL
// corpus from path. tokenizerPath points at a HuggingFace tokenizer.json
// (or equivalent) compatible with the base model. chatTemplate is
// nil-able; when nil the trainer falls back to the tokenizer's built-in
// template. device pins the tensor device ("cpu", "cuda:N", ...); nil
// uses cpu.
func NewPreferenceJsonlDataset(path, tokenizerPath string, chatTemplate *string, maxSeqLen uint32, device *string, padTokenId uint32) (*PreferenceJsonlDataset, error) {
	ensureInit()
	inner, err := uniffiblazen.UniffiPreferenceJsonlDatasetFromPath(path, tokenizerPath, chatTemplate, maxSeqLen, device, padTokenId)
	if err != nil {
		return nil, wrapErr(err)
	}
	return &PreferenceJsonlDataset{inner: inner}, nil
}

// IsEmpty reports whether the dataset contains zero preference pairs.
func (d *PreferenceJsonlDataset) IsEmpty() bool {
	if d == nil || d.inner == nil {
		return true
	}
	return d.inner.IsEmpty()
}

// Len returns the number of preference pairs in the dataset.
func (d *PreferenceJsonlDataset) Len() uint64 {
	if d == nil || d.inner == nil {
		return 0
	}
	return d.inner.Len()
}

// RatedJsonlDataset is a handle to a tokenized rated-completion JSONL
// corpus loaded from disk. Pass it to [ModelManager.TrainKto].
//
// Each line of the input file must deserialize to either
// `{"prompt": "...", "completion": "...", "desirable": true|false}` or
// `{"messages": [...], "completion": "...", "desirable": true|false}`
// (the latter requires a chatTemplate).
//
// The dataset owns a native handle; UniFFI attaches its own finalizer
// so callers do not need to release it explicitly.
type RatedJsonlDataset struct {
	inner *uniffiblazen.UniffiRatedJsonlDataset
}

// NewRatedJsonlDataset loads and tokenizes a rated-completion JSONL
// corpus from path. tokenizerPath points at a HuggingFace tokenizer.json
// (or equivalent) compatible with the base model. chatTemplate is
// nil-able; when nil the trainer falls back to the tokenizer's built-in
// template. device pins the tensor device ("cpu", "cuda:N", ...); nil
// uses cpu.
func NewRatedJsonlDataset(path, tokenizerPath string, chatTemplate *string, maxSeqLen uint32, device *string, padTokenId uint32) (*RatedJsonlDataset, error) {
	ensureInit()
	inner, err := uniffiblazen.UniffiRatedJsonlDatasetFromPath(path, tokenizerPath, chatTemplate, maxSeqLen, device, padTokenId)
	if err != nil {
		return nil, wrapErr(err)
	}
	return &RatedJsonlDataset{inner: inner}, nil
}

// IsEmpty reports whether the dataset contains zero rated completions.
func (d *RatedJsonlDataset) IsEmpty() bool {
	if d == nil || d.inner == nil {
		return true
	}
	return d.inner.IsEmpty()
}

// Len returns the number of rated completions in the dataset.
func (d *RatedJsonlDataset) Len() uint64 {
	if d == nil || d.inner == nil {
		return 0
	}
	return d.inner.Len()
}

// progressFFIArg wraps the optional [TrainingProgress] callback in the
// interface-pointer shape UniFFI expects. Returns nil when progress is
// nil so the FFI layer sees `Option::None`. The returned interface value
// is rooted on the caller's stack via the returned local variable, so
// the UniFFI handle table cannot drop it while Rust still holds the
// callback handle.
func progressFFIArg(progress TrainingProgress) *uniffiblazen.ForeignTrainingProgress {
	if progress == nil {
		return nil
	}
	var iface uniffiblazen.ForeignTrainingProgress = &trainingProgressAdapter{inner: progress}
	return &iface
}

// TrainDpo runs a Direct Preference Optimization fine-tune end-to-end
// on the model identified by config.Core.BaseModelRepo. dataset must be
// non-nil. progress may be nil to skip per-step callbacks.
//
// Errors on closed manager, nil dataset, invalid config (e.g.
// MaxSteps=0), missing base or reference model, training failure, or
// any callback that returns a non-nil error (surfaced as a Cancelled
// error).
func (m *ModelManager) TrainDpo(ctx context.Context, config DpoConfig, dataset *PreferenceJsonlDataset, progress TrainingProgress) (TrainedAdapter, error) {
	if m.inner == nil {
		return TrainedAdapter{}, &ValidationError{Message: "model manager has been closed"}
	}
	if dataset == nil || dataset.inner == nil {
		return TrainedAdapter{}, &ValidationError{Message: "dataset must not be nil"}
	}

	progressArg := progressFFIArg(progress)

	type result struct {
		v   uniffiblazen.TrainedAdapterRecord
		err error
	}
	done := make(chan result, 1)
	go func() {
		v, err := m.inner.TrainDpo(config.toFFI(), dataset.inner, progressArg)
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

// TrainOrpo runs an Odds Ratio Preference Optimization fine-tune
// end-to-end on the model identified by config.Core.BaseModelRepo.
// dataset must be non-nil. progress may be nil to skip per-step
// callbacks. Error contract matches [ModelManager.TrainDpo] (sans the
// missing-reference-model variant — ORPO is reference-free).
func (m *ModelManager) TrainOrpo(ctx context.Context, config OrpoConfig, dataset *PreferenceJsonlDataset, progress TrainingProgress) (TrainedAdapter, error) {
	if m.inner == nil {
		return TrainedAdapter{}, &ValidationError{Message: "model manager has been closed"}
	}
	if dataset == nil || dataset.inner == nil {
		return TrainedAdapter{}, &ValidationError{Message: "dataset must not be nil"}
	}

	progressArg := progressFFIArg(progress)

	type result struct {
		v   uniffiblazen.TrainedAdapterRecord
		err error
	}
	done := make(chan result, 1)
	go func() {
		v, err := m.inner.TrainOrpo(config.toFFI(), dataset.inner, progressArg)
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

// TrainSimpo runs a Simple Preference Optimization fine-tune end-to-end
// on the model identified by config.Core.BaseModelRepo. dataset must
// be non-nil. progress may be nil to skip per-step callbacks. SimPO is
// reference-free and length-normalized.
func (m *ModelManager) TrainSimpo(ctx context.Context, config SimpoConfig, dataset *PreferenceJsonlDataset, progress TrainingProgress) (TrainedAdapter, error) {
	if m.inner == nil {
		return TrainedAdapter{}, &ValidationError{Message: "model manager has been closed"}
	}
	if dataset == nil || dataset.inner == nil {
		return TrainedAdapter{}, &ValidationError{Message: "dataset must not be nil"}
	}

	progressArg := progressFFIArg(progress)

	type result struct {
		v   uniffiblazen.TrainedAdapterRecord
		err error
	}
	done := make(chan result, 1)
	go func() {
		v, err := m.inner.TrainSimpo(config.toFFI(), dataset.inner, progressArg)
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

// TrainKto runs a Kahneman-Tversky Optimization fine-tune end-to-end on
// the model identified by config.Core.BaseModelRepo. dataset must be a
// non-nil [RatedJsonlDataset] (KTO's schema differs from DPO/ORPO/SimPO).
// progress may be nil to skip per-step callbacks.
func (m *ModelManager) TrainKto(ctx context.Context, config KtoConfig, dataset *RatedJsonlDataset, progress TrainingProgress) (TrainedAdapter, error) {
	if m.inner == nil {
		return TrainedAdapter{}, &ValidationError{Message: "model manager has been closed"}
	}
	if dataset == nil || dataset.inner == nil {
		return TrainedAdapter{}, &ValidationError{Message: "dataset must not be nil"}
	}

	progressArg := progressFFIArg(progress)

	type result struct {
		v   uniffiblazen.TrainedAdapterRecord
		err error
	}
	done := make(chan result, 1)
	go func() {
		v, err := m.inner.TrainKto(config.toFFI(), dataset.inner, progressArg)
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

// FineTune runs a full-parameter fine-tune end-to-end on the model
// identified by config.Core.BaseModelRepo. Unlike the LoRA / PO verbs,
// every model parameter is updated and the entire weight set (not a
// PEFT adapter) is written to config.Core.OutputDir.
//
// dataset is the same SFT-style [JsonlDataset] used by
// [ModelManager.TrainLora]. progress may be nil to skip per-step
// callbacks.
func (m *ModelManager) FineTune(ctx context.Context, config FullFineTuneConfig, dataset *JsonlDataset, progress TrainingProgress) (FullFineTuneResult, error) {
	if m.inner == nil {
		return FullFineTuneResult{}, &ValidationError{Message: "model manager has been closed"}
	}
	if dataset == nil || dataset.inner == nil {
		return FullFineTuneResult{}, &ValidationError{Message: "dataset must not be nil"}
	}

	progressArg := progressFFIArg(progress)

	type result struct {
		v   uniffiblazen.FullFineTuneResultRecord
		err error
	}
	done := make(chan result, 1)
	go func() {
		v, err := m.inner.FineTune(config.toFFI(), dataset.inner, progressArg)
		done <- result{v: v, err: err}
	}()
	select {
	case <-ctx.Done():
		return FullFineTuneResult{}, ctx.Err()
	case r := <-done:
		if r.err != nil {
			return FullFineTuneResult{}, wrapErr(r.err)
		}
		return fullFineTuneResultFromFFI(r.v), nil
	}
}
