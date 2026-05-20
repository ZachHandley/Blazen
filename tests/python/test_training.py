"""Smoke tests for the Python training surface exposed by `blazen`.

These tests exercise the type surface only — they construct config objects,
exercise validation, and confirm the supporting pyclasses are discoverable.
A live training run requires HF + a base model download + compute and is
not part of this suite; the live path is exercised via the `pytest -m live`
opt-in once that marker is wired up.
"""

import pytest

import blazen

pytestmark = pytest.mark.skipif(
    not hasattr(blazen, "TrainConfig"),
    reason="blazen built without the `training` feature",
)


def test_train_config_defaults():
    cfg = blazen.TrainConfig(
        base_model_repo="Qwen/Qwen2.5-0.5B",
        output_dir="/tmp/blazen-test-out",
    )
    assert cfg.base_model_repo == "Qwen/Qwen2.5-0.5B"
    assert cfg.output_dir == "/tmp/blazen-test-out"
    assert cfg.max_steps == 1000
    assert cfg.batch_size == 4
    assert cfg.gradient_accumulation_steps == 1
    assert cfg.max_seq_len == 2048
    assert cfg.seed == 42
    assert cfg.lora.rank == 16
    assert cfg.lora.alpha == 32.0
    assert cfg.lora.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert cfg.optim.learning_rate == pytest.approx(2e-4)
    assert cfg.optim.gradient_clip == pytest.approx(1.0)
    assert cfg.scheduler.kind == blazen.SchedulerKind.COSINE
    assert cfg.mixed_precision == blazen.MixedPrecision.BF16


def test_train_config_validation():
    with pytest.raises(ValueError):
        blazen.TrainConfig(
            base_model_repo="x",
            output_dir="/tmp/out",
            max_steps=0,
        )
    with pytest.raises(ValueError):
        blazen.TrainConfig(
            base_model_repo="x",
            output_dir="/tmp/out",
            batch_size=0,
        )
    with pytest.raises(ValueError):
        blazen.TrainConfig(
            base_model_repo="",
            output_dir="/tmp/out",
        )
    with pytest.raises(ValueError):
        blazen.LoraConfig(rank=0)
    with pytest.raises(ValueError):
        blazen.OptimConfig(learning_rate=0.0)


def test_training_surface_is_registered():
    for name in (
        "TrainConfig",
        "LoraConfig",
        "OptimConfig",
        "SchedulerConfig",
        "SchedulerKind",
        "MixedPrecision",
        "TrainedAdapter",
        "TrainingEvent",
        "JsonlDataset",
    ):
        assert hasattr(blazen, name), f"blazen.{name} missing"

    assert blazen.ModelManager is not None
    assert hasattr(blazen.ModelManager, "train_lora")
