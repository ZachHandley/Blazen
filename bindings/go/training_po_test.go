package blazen

import (
	"context"
	"errors"
	"testing"
)

// newTrainCoreConfig builds a minimal [TrainCoreConfig] suitable for the
// compile-only / validation-only smoke tests below. It points at a
// guaranteed-nonexistent base repo so any test that accidentally
// progresses past the dataset guard fails fast at fetch time rather
// than launching a real download.
func newTrainCoreConfig(maxSteps uint32) TrainCoreConfig {
	return TrainCoreConfig{
		BaseModelRepo: "nonexistent/repo-abc123-xyz",
		OutputDir:     "blazen-train-po-out-test",
		Optim: OptimConfig{
			LearningRate: 2e-4,
			Beta1:        0.9,
			Beta2:        0.999,
			Epsilon:      1e-8,
			WeightDecay:  0.0,
		},
		Scheduler: SchedulerConfig{
			Kind:        SchedulerKindCosine,
			WarmupSteps: 0,
		},
		MaxSteps:                  maxSteps,
		BatchSize:                 1,
		GradientAccumulationSteps: 1,
		MaxSeqLen:                 64,
		Seed:                      42,
		MixedPrecision:            MixedPrecisionNone,
	}
}

func newLoraConfigForPO() LoraConfig {
	return LoraConfig{
		Rank:          8,
		Alpha:         16,
		Dropout:       0.05,
		TargetModules: []string{"q_proj", "v_proj"},
	}
}

func TestDpoConfigDefaults(t *testing.T) {
	c := DefaultDpoConfig()
	if c.Beta != 0.1 {
		t.Errorf("DefaultDpoConfig().Beta = %v, want 0.1", c.Beta)
	}
	if c.LabelSmoothing != 0.0 {
		t.Errorf("DefaultDpoConfig().LabelSmoothing = %v, want 0.0", c.LabelSmoothing)
	}
	if c.ReferenceModelRepo != nil {
		t.Errorf("DefaultDpoConfig().ReferenceModelRepo = %v, want nil", c.ReferenceModelRepo)
	}
	if c.ReferenceModelRevision != nil {
		t.Errorf("DefaultDpoConfig().ReferenceModelRevision = %v, want nil", c.ReferenceModelRevision)
	}
}

func TestOrpoConfigDefaults(t *testing.T) {
	c := DefaultOrpoConfig()
	if c.Lambda != 0.1 {
		t.Errorf("DefaultOrpoConfig().Lambda = %v, want 0.1", c.Lambda)
	}
}

func TestSimpoConfigDefaults(t *testing.T) {
	c := DefaultSimpoConfig()
	if c.Beta != 2.0 {
		t.Errorf("DefaultSimpoConfig().Beta = %v, want 2.0", c.Beta)
	}
	if c.Gamma != 1.0 {
		t.Errorf("DefaultSimpoConfig().Gamma = %v, want 1.0", c.Gamma)
	}
}

func TestKtoConfigDefaults(t *testing.T) {
	c := DefaultKtoConfig()
	if c.Beta != 0.1 {
		t.Errorf("DefaultKtoConfig().Beta = %v, want 0.1", c.Beta)
	}
	if c.LambdaD != 1.0 {
		t.Errorf("DefaultKtoConfig().LambdaD = %v, want 1.0", c.LambdaD)
	}
	if c.LambdaU != 1.0 {
		t.Errorf("DefaultKtoConfig().LambdaU = %v, want 1.0", c.LambdaU)
	}
}

func TestFullFineTuneConfigDefaults(t *testing.T) {
	c := DefaultFullFineTuneConfig()
	if c.GradientCheckpointing {
		t.Error("DefaultFullFineTuneConfig().GradientCheckpointing = true, want false")
	}
}

func TestPreferenceJsonlDatasetInvalidPath(t *testing.T) {
	ds, err := NewPreferenceJsonlDataset("/nonexistent/pref.jsonl", "/nonexistent/tok.json", nil, 64, nil, 0)
	if err == nil {
		t.Fatal("expected NewPreferenceJsonlDataset to error on missing file")
	}
	if ds != nil {
		t.Fatalf("expected nil dataset on error, got %+v", ds)
	}
	var be Error
	if !errors.As(err, &be) {
		t.Fatalf("expected typed blazen.Error, got %T: %v", err, err)
	}
}

func TestRatedJsonlDatasetInvalidPath(t *testing.T) {
	ds, err := NewRatedJsonlDataset("/nonexistent/rated.jsonl", "/nonexistent/tok.json", nil, 64, nil, 0)
	if err == nil {
		t.Fatal("expected NewRatedJsonlDataset to error on missing file")
	}
	if ds != nil {
		t.Fatalf("expected nil dataset on error, got %+v", ds)
	}
	var be Error
	if !errors.As(err, &be) {
		t.Fatalf("expected typed blazen.Error, got %T: %v", err, err)
	}
}

// TestTrainDpoSignatureCompiles is a "method exists" smoke test —
// it constructs the argument shape and references each PO/FFT method to
// guarantee the public signatures are present and accept the documented
// types. Each call is expected to fail at the nil-dataset guard
// without launching a training run; the contract under test is that
// the call shape itself compiles and routes through the guard.
func TestTrainDpoSignatureCompiles(t *testing.T) {
	mm, ctx, cancel := newTestManagerCtx(t)
	defer cancel()

	dpo := DefaultDpoConfig()
	dpo.Core = newTrainCoreConfig(1)
	dpo.Lora = newLoraConfigForPO()
	if _, err := mm.TrainDpo(ctx, dpo, nil, nil); err == nil {
		t.Error("expected TrainDpo to reject nil dataset")
	}

	orpo := DefaultOrpoConfig()
	orpo.Core = newTrainCoreConfig(1)
	orpo.Lora = newLoraConfigForPO()
	if _, err := mm.TrainOrpo(ctx, orpo, nil, nil); err == nil {
		t.Error("expected TrainOrpo to reject nil dataset")
	}

	simpo := DefaultSimpoConfig()
	simpo.Core = newTrainCoreConfig(1)
	simpo.Lora = newLoraConfigForPO()
	if _, err := mm.TrainSimpo(ctx, simpo, nil, nil); err == nil {
		t.Error("expected TrainSimpo to reject nil dataset")
	}

	kto := DefaultKtoConfig()
	kto.Core = newTrainCoreConfig(1)
	kto.Lora = newLoraConfigForPO()
	if _, err := mm.TrainKto(ctx, kto, nil, nil); err == nil {
		t.Error("expected TrainKto to reject nil dataset")
	}

	fft := DefaultFullFineTuneConfig()
	fft.Core = newTrainCoreConfig(1)
	if _, err := mm.FineTune(ctx, fft, nil, nil); err == nil {
		t.Error("expected FineTune to reject nil dataset")
	}
}

func TestTrainPONilProgressDoesntPanic(t *testing.T) {
	mm, ctx, cancel := newTestManagerCtx(t)
	defer cancel()

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("PO verb panicked with nil progress: %v", r)
		}
	}()

	// Reach the dataset guard with progress=nil. The guard rejects with
	// a typed error; the contract under test is "no panic".
	dpo := DefaultDpoConfig()
	dpo.Core = newTrainCoreConfig(1)
	dpo.Lora = newLoraConfigForPO()
	_, _ = mm.TrainDpo(ctx, dpo, nil, nil)

	fft := DefaultFullFineTuneConfig()
	fft.Core = newTrainCoreConfig(1)
	_, _ = mm.FineTune(ctx, fft, nil, nil)
	_ = context.Canceled // keep context import used through linter pruning
}
