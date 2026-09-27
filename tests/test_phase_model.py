from pathlib import Path
import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from respi_net.breath_phases import (  # noqa: E402
    CLASS_NAMES,
    EXHALE,
    HOLD_AFTER_INHALE,
    IGNORE,
    INHALE,
    NOISE,
    NUM_CLASSES,
)
from respi_net.phase_model import (  # noqa: E402
    OnlinePhaseSmoother,
    StreamingPhaseDetector,
    build_model,
    class_weights,
    load_checkpoint,
    masked_cross_entropy,
    save_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import train_breath_phase_model as trainer  # noqa: E402

FS = 20.0
IN_CHANNELS = 3
MODEL_NAMES = ("tcn", "gru")


def _model(name: str, **kwargs: object) -> torch.nn.Module:
    torch.manual_seed(0)
    return build_model(name, IN_CHANNELS, **kwargs).eval()


def _confident(label: int, frames: int, p: float = 0.9) -> np.ndarray:
    probs = np.full((frames, NUM_CLASSES), (1.0 - p) / (NUM_CLASSES - 1))
    probs[:, label] = p
    return probs


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_give_class_logits_for_every_frame(name: str) -> None:
    assert _model(name)(torch.randn(2, IN_CHANNELS, 57)).shape == (2, NUM_CLASSES, 57)
    assert build_model(name, 1, num_classes=3)(torch.randn(1, 1, 10)).shape == (1, 3, 10)


def test_build_model_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown phase model"):
        build_model("transformer", IN_CHANNELS)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_future_frames_do_not_change_earlier_outputs(name: str) -> None:
    model = _model(name)
    x = torch.randn(1, IN_CHANNELS, 300)
    t = 150
    changed = x.clone()
    changed[..., t + 1 :] = 5.0 * torch.randn(1, IN_CHANNELS, 300 - t - 1)

    with torch.no_grad():
        before, after = model(x), model(changed)

    torch.testing.assert_close(before[..., : t + 1], after[..., : t + 1], rtol=0.0, atol=1e-5)
    assert not torch.allclose(before[..., t + 1 :], after[..., t + 1 :])


def test_default_tcn_sees_about_25_seconds_back() -> None:
    model = _model("tcn")
    field = model.receptive_field_samples()

    assert field == 1 + 4 * (1 + 2 + 4 + 8 + 16 + 32 + 64) == 509
    assert 20.0 <= field / FS <= 30.0

    # The last output depends on exactly the last `field` input frames.
    x = torch.randn(1, IN_CHANNELS, field + 40, requires_grad=True)
    model(x)[0, :, -1].sum().backward()
    reach = torch.nonzero(x.grad.abs().sum(dim=1)[0]).flatten()
    assert reach.min().item() == x.shape[-1] - field
    assert reach.max().item() == x.shape[-1] - 1


@pytest.mark.parametrize(
    ("name", "history_s"),
    [("tcn", None), ("tcn", 508 / FS), ("gru", None)],
    ids=["tcn-state", "tcn-history", "gru-state"],
)
def test_streaming_equals_one_pass_over_the_whole_sequence(name: str, history_s: float | None) -> None:
    model = _model(name)
    frames = torch.randn(IN_CHANNELS, 700)  # longer than the TCN's receptive field
    with torch.no_grad():
        expected = torch.softmax(model(frames[None]), dim=1)[0].T.numpy()
    detector = StreamingPhaseDetector(model, FS, history_s=history_s)

    rng = np.random.default_rng(0)
    parts, start = [], 0
    while start < frames.shape[1]:
        size = int(rng.choice([0, 1, 1, 3, 7, 20, 64]))
        chunk = frames[:, start : start + size].numpy()
        probs = detector.push(chunk)
        assert probs.shape == (chunk.shape[1], NUM_CLASSES)
        parts.append(probs)
        start += size

    np.testing.assert_allclose(np.concatenate(parts), expected, atol=1e-5)
    assert detector.frames_seen == 700 and detector.time_s == pytest.approx(35.0)

    detector.reset()
    np.testing.assert_allclose(detector.push(frames[:, 0].numpy()), expected[:1], atol=1e-5)


def test_smoother_blocks_transitions_the_breath_cannot_make() -> None:
    smoother = OnlinePhaseSmoother(FS)  # 3 frames to confirm, phases of at least 0.3 s = 6 frames

    out = smoother.update(
        np.concatenate([_confident(EXHALE, 20), _confident(HOLD_AFTER_INHALE, 20), _confident(INHALE, 10)])
    )

    assert list(out[:2]) == [IGNORE, IGNORE]  # nothing confirmed yet
    assert np.all(out[2:42] == EXHALE)  # exhale -> hold after inhale is impossible
    assert np.all(out[42:] == INHALE)  # inhale takes over on its third confident frame


def test_smoother_ignores_flicker_unsure_frames_and_too_short_phases() -> None:
    smoother = OnlinePhaseSmoother(FS)

    flicker = smoother.update(
        np.concatenate([_confident(INHALE, 10), _confident(EXHALE, 2), _confident(EXHALE, 10, p=0.4), _confident(INHALE, 3)])
    )
    assert np.all(flicker[2:] == INHALE)

    smoother.reset()
    out = smoother.update(np.concatenate([_confident(INHALE, 10), _confident(EXHALE, 3), _confident(INHALE, 10)]))
    # The exhale is confirmed at frame 12; the inhale right after it has to wait
    # until the exhale has lasted 0.3 s (6 frames).
    assert list(out[10:20]) == [INHALE, INHALE] + [EXHALE] * 6 + [INHALE, INHALE]


def test_noise_interrupts_any_phase() -> None:
    smoother = OnlinePhaseSmoother(FS, allowed={HOLD_AFTER_INHALE: frozenset()})

    out = smoother.update(np.concatenate([_confident(HOLD_AFTER_INHALE, 10), _confident(NOISE, 5)]))

    assert out[9] == HOLD_AFTER_INHALE and out[-1] == NOISE


def test_masked_loss_skips_ignored_frames() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, NUM_CLASSES, 8, requires_grad=True)
    targets = torch.randint(0, NUM_CLASSES, (2, 8))
    masked = targets.clone()
    masked[:, :3] = IGNORE

    expected = torch.nn.functional.cross_entropy(logits[..., 3:], targets[:, 3:])
    torch.testing.assert_close(masked_cross_entropy(logits, masked), expected)

    nothing = masked_cross_entropy(logits, torch.full_like(targets, IGNORE))
    assert nothing.item() == 0.0
    nothing.backward()  # a batch without labels still gives a usable (zero) gradient


def test_class_weights_are_inverse_frequency_over_labelled_samples() -> None:
    labels = np.array([INHALE] * 6 + [EXHALE] * 2 + [IGNORE] * 5)

    weights = class_weights(labels)

    assert weights[INHALE] == pytest.approx(8 / (2 * 6))
    assert weights[EXHALE] == pytest.approx(8 / (2 * 2))
    assert weights[NOISE] == 0.0
    assert weights[labels[labels != IGNORE]].mean() == pytest.approx(1.0)
    np.testing.assert_allclose(class_weights(labels, power=0.0)[[INHALE, EXHALE]], [1.0, 1.0])


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_one_optimiser_step_lowers_the_loss(name: str) -> None:
    data = trainer.toy_windows(n_windows=6, seed=1)
    x, y = torch.from_numpy(data.X), torch.from_numpy(data.y)
    weight = torch.as_tensor(class_weights(data.y), dtype=torch.float32)
    model = _model(name, dropout=0.0).train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss = masked_cross_entropy(model(x), y, weight)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    with torch.no_grad():
        after = masked_cross_entropy(model(x), y, weight)

    assert after.item() < loss.item()


def test_augmentation_moves_signal_and_labels_together() -> None:
    x = torch.arange(10, dtype=torch.float32).repeat(4, IN_CHANNELS, 1)
    y = torch.arange(10).repeat(4, 1)

    shifted_x, shifted_y = trainer.augment_batch(
        x, y, torch.Generator().manual_seed(0), amplitude_channels=[0, 1], scale_range=(2.0, 2.0), max_shift=3
    )

    for window in range(4):
        labelled = shifted_y[window] != IGNORE
        assert 7 <= int(labelled.sum()) <= 10
        # Every labelled frame still carries the signal value it was labelled with.
        torch.testing.assert_close(shifted_x[window, 2, labelled], shifted_y[window, labelled].float())
        torch.testing.assert_close(shifted_x[window, :2, labelled], 2.0 * shifted_x[window, 2:, labelled].expand(2, -1))


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = _model("gru", hidden=16, layers=1)
    save_checkpoint(tmp_path / "model.pt", model, channels=["chest", "velocity", "echo_db"], fs=FS, note="test")

    loaded, info = load_checkpoint(tmp_path / "model.pt")

    x = torch.randn(1, IN_CHANNELS, 40)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x))
    assert (info["model_name"], info["fs"], info["note"]) == ("gru", FS, "test")
    assert info["channels"] == ["chest", "velocity", "echo_db"]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_training_script_smoke_run(name: str, tmp_path: Path) -> None:
    assert trainer.main(["--smoke", "--model", name, "--out-dir", str(tmp_path)]) == 0

    run = tmp_path / f"smoke_{name}"
    assert {"model.pt", "metrics.json", "history.csv"} <= {path.name for path in run.iterdir()}
    metrics = json.loads((run / "metrics.json").read_text())
    assert set(metrics["splits"]) == {"train", "val", "test"}
    val = metrics["splits"]["val"]
    assert {"accuracy", "macro_f1", "per_class", "confusion", "boundary"} <= set(val)
    assert np.array(val["confusion"]).shape == (NUM_CLASSES, NUM_CLASSES)
    assert set(val["per_class"]) == set(CLASS_NAMES)
    assert (run / "history.csv").read_text().splitlines()[0].startswith("epoch,steps,train_loss")


def _write_tiny_dataset(path: Path, fs: float = FS, width: int = 120) -> None:
    """Four windows in the nn_dataset .npz layout, written with numpy alone."""

    rng = np.random.default_rng(0)
    y = np.full((4, width), IGNORE, dtype=np.int8)
    y[:, 10:50] = INHALE
    y[:, 50:60] = HOLD_AFTER_INHALE
    y[:, 62:110] = EXHALE
    meta = {
        "fs": fs,
        "window_s": width / fs,
        "stride_s": width / fs,
        "channels": ["chest", "velocity", "echo_db"],
        "class_names": list(CLASS_NAMES),
    }
    np.savez(
        path,
        X=rng.normal(size=(4, 3, width)).astype(np.float32),
        y=y,
        split=np.array(["train", "val", "test", "test"]),
        run_id=np.array(["a", "b", "c", "c"]),
        start_s=np.array([0.0, 0.0, 0.0, 6.0], dtype=np.float32),
        meta=np.array(json.dumps(meta)),
    )


def test_evaluate_scores_a_saved_model_on_a_dataset_file(tmp_path: Path) -> None:
    dataset = tmp_path / "tiny.npz"
    _write_tiny_dataset(dataset)
    # A model reading two of the channels, in another order than the file has them.
    torch.manual_seed(0)
    model = build_model("tcn", 2, channels=8, dilations=(1, 2))
    save_checkpoint(tmp_path / "model.pt", model, channels=["velocity", "chest"], fs=FS)
    out = tmp_path / "eval.json"

    assert trainer.main(["--evaluate", str(tmp_path / "model.pt"), str(dataset), "--out", str(out)]) == 0

    report = json.loads(out.read_text())
    assert report["channels"] == ["velocity", "chest"]
    assert set(report["splits"]) == {"train", "val", "test"}
    test = report["splits"]["test"]
    assert test["windows"] == 2
    assert test["labelled_samples"] == 2 * (40 + 10 + 48)
    assert np.array(test["confusion"]).sum() == test["labelled_samples"]
    assert test["boundary"]["reference_boundaries"] == 2 * 2


def test_evaluate_refuses_a_dataset_at_another_sample_rate(tmp_path: Path) -> None:
    dataset = tmp_path / "tiny.npz"
    _write_tiny_dataset(dataset, fs=10.0)
    save_checkpoint(tmp_path / "model.pt", _model("gru", hidden=8, layers=1), channels=["chest", "velocity", "echo_db"], fs=FS)

    with pytest.raises(SystemExit, match="20 Hz"):
        trainer.main(["--evaluate", str(tmp_path / "model.pt"), str(dataset), "--out", str(tmp_path / "eval.json")])
