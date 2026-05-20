package blazen

import (
	"context"
	"errors"
	"testing"
	"time"
)

func newTestManagerCtx(t *testing.T) (*ModelManager, context.Context, context.CancelFunc) {
	t.Helper()
	mm := NewModelManager()
	t.Cleanup(mm.Close)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	return mm, ctx, cancel
}

func TestModelManagerConstructors(t *testing.T) {
	t.Run("default", func(t *testing.T) {
		mm := NewModelManager()
		if mm == nil {
			t.Fatal("NewModelManager returned nil")
		}
		mm.Close()
	})

	t.Run("budgets_gb", func(t *testing.T) {
		mm := NewModelManagerWithBudgetsGB(8, 16)
		if mm == nil {
			t.Fatal("NewModelManagerWithBudgetsGB returned nil")
		}
		mm.Close()
	})

	t.Run("pool_budgets", func(t *testing.T) {
		mm, err := NewModelManagerWithPoolBudgets(map[string]float64{
			"cpu":   4,
			"gpu:0": 12,
		})
		if err != nil {
			t.Fatalf("NewModelManagerWithPoolBudgets: %v", err)
		}
		if mm == nil {
			t.Fatal("NewModelManagerWithPoolBudgets returned nil")
		}
		mm.Close()
	})

	t.Run("pool_budgets_invalid_label", func(t *testing.T) {
		_, err := NewModelManagerWithPoolBudgets(map[string]float64{
			"not-a-real-pool": 1,
		})
		if err == nil {
			t.Fatal("expected error for invalid pool label")
		}
	})
}

func TestModelManagerStatusEmpty(t *testing.T) {
	mm, ctx, cancel := newTestManagerCtx(t)
	defer cancel()

	status, err := mm.Status(ctx)
	if err != nil {
		t.Fatalf("Status: %v", err)
	}
	if len(status) != 0 {
		t.Fatalf("expected empty status, got %d entries: %+v", len(status), status)
	}
}

func TestModelManagerPoolsDefault(t *testing.T) {
	mm := NewModelManager()
	t.Cleanup(mm.Close)

	pools := mm.Pools()
	if len(pools) != 2 {
		t.Fatalf("expected 2 default pools (cpu + gpu:0), got %d: %+v", len(pools), pools)
	}
	seen := map[string]bool{}
	for _, p := range pools {
		seen[p.Pool] = true
		if p.BudgetBytes == 0 {
			t.Errorf("pool %q has zero budget; default constructor should seed u64::MAX", p.Pool)
		}
	}
	if !seen["cpu"] {
		t.Errorf("default pools missing %q: %+v", "cpu", pools)
	}
	if !seen["gpu:0"] {
		t.Errorf("default pools missing %q: %+v", "gpu:0", pools)
	}
}

func TestModelManagerLoadNonexistent(t *testing.T) {
	mm, ctx, cancel := newTestManagerCtx(t)
	defer cancel()

	err := mm.Load(ctx, "definitely-not-registered")
	if err == nil {
		t.Fatal("expected error loading unregistered model")
	}
	var be Error
	if !errors.As(err, &be) {
		t.Fatalf("expected typed blazen.Error, got %T: %v", err, err)
	}
}

func TestModelManagerIsLoadedNonexistent(t *testing.T) {
	mm, ctx, cancel := newTestManagerCtx(t)
	defer cancel()

	loaded, err := mm.IsLoaded(ctx, "definitely-not-registered")
	if err != nil {
		t.Fatalf("IsLoaded: %v", err)
	}
	if loaded {
		t.Fatal("IsLoaded for unregistered model should be false")
	}
}

func TestModelManagerLoadAdapterNonexistent(t *testing.T) {
	mm, ctx, cancel := newTestManagerCtx(t)
	defer cancel()

	_, err := mm.LoadAdapter(ctx, "definitely-not-registered", "/nonexistent/adapter/dir", AdapterOptions{Scale: 1.0})
	if err == nil {
		t.Fatal("expected error loading adapter for unregistered model")
	}
	var be Error
	if !errors.As(err, &be) {
		t.Fatalf("expected typed blazen.Error, got %T: %v", err, err)
	}
}

func TestModelManagerListAdaptersNonexistent(t *testing.T) {
	mm, ctx, cancel := newTestManagerCtx(t)
	defer cancel()

	_, err := mm.ListAdapters(ctx, "definitely-not-registered")
	if err == nil {
		t.Fatal("expected error listing adapters for unregistered model")
	}
}

func TestModelManagerPoolBytes(t *testing.T) {
	mm, ctx, cancel := newTestManagerCtx(t)
	defer cancel()

	used, err := mm.UsedBytes(ctx, "cpu")
	if err != nil {
		t.Fatalf("UsedBytes(cpu): %v", err)
	}
	if used != 0 {
		t.Errorf("expected UsedBytes(cpu)=0 on empty manager, got %d", used)
	}

	if _, err := mm.AvailableBytes(ctx, "cpu"); err != nil {
		t.Fatalf("AvailableBytes(cpu): %v", err)
	}

	if _, err := mm.UsedBytes(ctx, "not-a-pool"); err == nil {
		t.Fatal("expected error from UsedBytes with invalid pool label")
	}
}

func TestModelManager_LoadFromHf_NoNetwork(t *testing.T) {
	mm, ctx, cancel := newTestManagerCtx(t)
	defer cancel()

	// Why: an obviously-invalid repo slug must fail with a typed Blazen
	// error rather than panic. Either the loader rejects validation
	// up-front or the HF probe surfaces a network/404 error — both
	// outcomes are acceptable as long as the call returns cleanly.
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("LoadFromHf panicked on invalid repo: %v", r)
		}
	}()

	backend, err := mm.LoadFromHf(ctx, "test-id", "nonexistent/repo-abc123-xyz", HfLoadOptions{})
	if err == nil {
		t.Fatalf("expected error loading nonexistent repo, got backend=%q", backend)
	}
	var be Error
	if !errors.As(err, &be) {
		t.Fatalf("expected typed blazen.Error, got %T: %v", err, err)
	}
}

func TestModelManager_LoadFromHf_NilOptions_DoesntPanic(t *testing.T) {
	mm, ctx, cancel := newTestManagerCtx(t)
	defer cancel()

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("LoadFromHf panicked on zero-value HfLoadOptions: %v", r)
		}
	}()

	_, err := mm.LoadFromHf(ctx, "test-id-empty-opts", "nonexistent/repo-abc123-xyz", HfLoadOptions{})
	// Why: success would be unexpected for a fake repo, but we accept it
	// — the contract is "no panic". Any error path is also fine.
	_ = err
}

func TestModelManager_LoadFromHf_AllOptionsPlumbed(t *testing.T) {
	mm, ctx, cancel := newTestManagerCtx(t)
	defer cancel()

	hint := BackendHintLlamacpp
	rev := "main"
	token := "hf_dummy"
	cache := "/tmp/blazen-hf-cache-test"
	device := "cpu"
	gguf := "model-q4.gguf"
	var mem uint64 = 1 << 30
	pool := "cpu"
	opts := HfLoadOptions{
		BackendHint:         &hint,
		Revision:            &rev,
		HfToken:             &token,
		CacheDir:            &cache,
		Device:              &device,
		GgufFile:            &gguf,
		MemoryEstimateBytes: &mem,
		Pool:                &pool,
	}

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("LoadFromHf panicked with all options set: %v", r)
		}
	}()
	if _, err := mm.LoadFromHf(ctx, "test-id-all-opts", "nonexistent/repo-abc123-xyz", opts); err == nil {
		t.Fatal("expected error loading nonexistent repo")
	}
}

func TestModelManager_LoadFromHf_AfterClose(t *testing.T) {
	mm := NewModelManager()
	mm.Close()

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	_, err := mm.LoadFromHf(ctx, "id", "owner/repo", HfLoadOptions{})
	if err == nil {
		t.Fatal("expected error from LoadFromHf after Close")
	}
	var ve *ValidationError
	if !errors.As(err, &ve) {
		t.Fatalf("expected *ValidationError after Close, got %T: %v", err, err)
	}
}

func newTrainConfig(maxSteps uint32) TrainConfig {
	return TrainConfig{
		BaseModelRepo: "nonexistent/repo-abc123-xyz",
		OutputDir:     "/tmp/blazen-train-out-test",
		Lora: LoraConfig{
			Rank:          8,
			Alpha:         16,
			Dropout:       0.05,
			TargetModules: []string{"q_proj", "v_proj"},
		},
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

func TestTrainLora_RejectsInvalidConfig(t *testing.T) {
	mm, ctx, cancel := newTestManagerCtx(t)
	defer cancel()

	ds, err := NewJsonlDatasetFromPath("/nonexistent/data.jsonl", "/nonexistent/tok.json", nil, 64, nil, 0)
	// Why: the dataset constructor must fail because the path does not
	// exist; we still cover the TrainLora invalid-config path below by
	// passing a nil dataset, which TrainLora must reject up-front.
	if err == nil {
		t.Fatal("expected NewJsonlDatasetFromPath to error on bogus path")
	}
	if ds != nil {
		t.Fatalf("expected nil dataset on error, got %+v", ds)
	}

	_, err = mm.TrainLora(ctx, newTrainConfig(0), nil, nil)
	if err == nil {
		t.Fatal("expected TrainLora to reject MaxSteps=0 / nil dataset")
	}
	var be Error
	if !errors.As(err, &be) {
		t.Fatalf("expected typed blazen.Error, got %T: %v", err, err)
	}
}

func TestTrainLora_NilProgress_DoesntPanic(t *testing.T) {
	mm, ctx, cancel := newTestManagerCtx(t)
	defer cancel()

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("TrainLora panicked with nil progress: %v", r)
		}
	}()

	// Why: we have no real dataset to construct, so we expect the call
	// to fail at the dataset-validation guard rather than launch a
	// training run. The contract under test is "no panic".
	_, err := mm.TrainLora(ctx, newTrainConfig(1), nil, nil)
	if err == nil {
		t.Fatal("expected error from TrainLora with nil dataset")
	}
}

func TestJsonlDataset_RequiresValidPath(t *testing.T) {
	_, err := NewJsonlDatasetFromPath("/nonexistent/file.jsonl", "/nonexistent/tok.json", nil, 64, nil, 0)
	if err == nil {
		t.Fatal("expected NewJsonlDatasetFromPath to error on missing file")
	}
	var be Error
	if !errors.As(err, &be) {
		t.Fatalf("expected typed blazen.Error, got %T: %v", err, err)
	}
}

func TestModelManagerCloseIsIdempotent(t *testing.T) {
	mm := NewModelManager()
	mm.Close()
	mm.Close()
	mm.Close()
}

func TestModelManagerOperationsAfterClose(t *testing.T) {
	mm := NewModelManager()
	mm.Close()

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	if err := mm.Load(ctx, "anything"); err == nil {
		t.Error("expected error from Load after Close")
	}
	if _, err := mm.IsLoaded(ctx, "anything"); err == nil {
		t.Error("expected error from IsLoaded after Close")
	}
	if _, err := mm.Status(ctx); err == nil {
		t.Error("expected error from Status after Close")
	}
	if pools := mm.Pools(); pools != nil {
		t.Errorf("expected nil Pools after Close, got %+v", pools)
	}
}
