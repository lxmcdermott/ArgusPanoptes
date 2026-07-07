"""Tests for the Day-3 deep-learning models (:mod:`models.dl_models`).

Skipped entirely when the ``dl`` extra (torch) is not installed, so the core /
ml test runs are unaffected. Covers forward-pass shapes, the ``get_model``
factory, multi-task training + evaluation on a tiny synthetic loader, ONNX
export round-trip parity, and integration with ``SignalProcessor``'s new DL
convenience methods.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.dl_models import (  # noqa: E402
    DLModelConfig,
    Vibration1DCNN,
    evaluate_dl_model,
    example_inputs_for,
    export_to_onnx,
    get_model,
    set_seed,
    train_dl_model,
)

_SPECS = [("1dcnn", "waveform"), ("spectrogram", "spectrogram"), ("fusion", "fusion")]


def _fake_loader(kind, *, n=16, bs=8, L=4096, n_freq=64, n_time=16, n_reg=3, n_classes=4, thermal_dim=5):
    """A minimal list-of-dict-batches 'loader' for the given input kind."""
    g = torch.Generator().manual_seed(0)
    batches = []
    for start in range(0, n, bs):
        b = min(bs, n - start)
        batch = {
            "y_reg": torch.rand(b, n_reg, generator=g),
            "y_clf": torch.randint(0, n_classes, (b,), generator=g),
        }
        if kind in ("waveform", "fusion"):
            batch["waveform"] = torch.randn(b, 1, L, generator=g)
        if kind == "spectrogram":
            batch["spectrogram"] = torch.randn(b, 1, n_freq, n_time, generator=g)
        if kind == "fusion":
            batch["thermal"] = torch.randn(b, thermal_dim, generator=g)
        batches.append(batch)
    return batches


# --------------------------------------------------------------------------- #
# Construction / forward
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,kind", _SPECS)
def test_forward_shapes(name, kind):
    model = get_model(name).eval()
    assert model.input_kind == kind
    ex = example_inputs_for(model, batch=2, seq_len=4096, n_freq=64, n_time=16)
    with torch.no_grad():
        reg, logits = model(*ex)
    assert reg.shape == (2, 3)
    assert logits.shape == (2, 4)
    assert torch.all(torch.isfinite(reg)) and torch.all(torch.isfinite(logits))


def test_get_model_config_and_errors():
    m = get_model("1dcnn", {"n_classes": 6, "cnn1d_channels": (8, 16)})
    assert isinstance(m, Vibration1DCNN)
    assert m.cfg.n_classes == 6
    ex = example_inputs_for(m, batch=1, seq_len=2048)
    with torch.no_grad():
        _, logits = m(*ex)
    assert logits.shape == (1, 6)
    with pytest.raises(ValueError):
        get_model("does-not-exist")


def test_waveform_accepts_2d_input():
    m = get_model("1dcnn").eval()
    with torch.no_grad():
        reg, _ = m(torch.randn(3, 4096))  # (B, L) auto-unsqueezed
    assert reg.shape == (3, 3)


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def test_train_and_evaluate_smoke():
    set_seed(0)
    cfg = DLModelConfig(cnn1d_channels=(8, 16, 32), head_hidden=16)
    model = get_model("1dcnn", cfg)
    train = _fake_loader("waveform", L=4096)
    val = _fake_loader("waveform", n=8, L=4096)
    history = train_dl_model(model, train, val, epochs=2, patience=5, seed=0, verbose=False)
    assert len(history["train_loss"]) >= 1
    assert history["epochs_run"] >= 1
    metrics = evaluate_dl_model(model, val, class_names=["a", "b", "c", "d"])
    assert set(metrics["regression"]) == {"wear_level", "cycle_time_factor", "quality_score"}
    for m in metrics["regression"].values():
        assert set(m) == {"mae", "rmse", "r2"}
    assert 0.0 <= metrics["classification"]["accuracy"] <= 1.0
    assert 0.0 <= metrics["classification"]["macro_f1"] <= 1.0


# --------------------------------------------------------------------------- #
# ONNX export round-trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,kind", _SPECS)
def test_onnx_export_roundtrip(name, kind, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    model = get_model(name).eval()
    ex = example_inputs_for(model, batch=2, seq_len=4096, n_freq=64, n_time=16)
    with torch.no_grad():
        reg, logits = model(*ex)
    onnx_path = tmp_path / f"{name}.onnx"
    export_to_onnx(model, onnx_path, example_input=ex)
    assert onnx_path.is_file()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {i.name: e.numpy() for i, e in zip(sess.get_inputs(), ex)}
    out = sess.run(None, feeds)
    assert np.max(np.abs(out[0] - reg.numpy())) < 1e-4
    assert np.max(np.abs(out[1] - logits.numpy())) < 1e-4


def test_onnx_dynamic_batch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    model = get_model("1dcnn").eval()
    ex = example_inputs_for(model, batch=1, seq_len=4096)
    onnx_path = tmp_path / "dyn.onnx"
    export_to_onnx(model, onnx_path, example_input=ex)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # A different batch size and sequence length must still run (dynamic axes).
    out = sess.run(None, {"waveform": np.random.randn(5, 1, 6000).astype(np.float32)})
    assert out[0].shape == (5, 3)
    assert out[1].shape == (5, 4)


# --------------------------------------------------------------------------- #
# Integration with SignalProcessor DL methods -> models
# --------------------------------------------------------------------------- #
def test_signalprocessor_to_onnx_perceptor(tmp_path, proc, vib):
    """End-to-end: normalized waveform -> 1D-CNN -> ONNX -> ONNXPerceptor."""
    pytest.importorskip("onnxruntime")
    from models.onnx_inference import ONNXPerceptor

    _, accel, meta = vib.generate(duration_s=1.0, wear=0.5, seed=3)
    w = proc.get_normalized_waveform(accel, fs=meta["fs_hz"])
    model = get_model("1dcnn").eval()
    onnx_path = tmp_path / "e2e.onnx"
    export_to_onnx(model, onnx_path, example_input=example_inputs_for(model, seq_len=w.size))
    perc = ONNXPerceptor(onnx_path)
    out = perc.predict(w[None, None, :])
    assert out["regression"].shape == (1, 3)
    assert out["health_probs"].shape == (1, 4)
    assert np.allclose(out["health_probs"].sum(axis=1), 1.0, atol=1e-5)
