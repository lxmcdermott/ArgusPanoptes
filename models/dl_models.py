"""PyTorch deep-learning models for Argus Panoptes (Day 3).

Three multi-task architectures that predict blade condition **directly from the
signals** (rather than the hand-crafted DSP scalars used by the XGBoost
baseline), plus training / evaluation / ONNX-export utilities:

* :class:`Vibration1DCNN` - 1-D conv backbone on the normalized raw waveform
  (``SignalProcessor.get_normalized_waveform``). Learns the *shape* of the
  tooth-strike transients that sharpen with wear.
* :class:`SpectrogramCNN` - 2-D conv backbone on the log-power STFT spectrogram
  (``SignalProcessor.compute_spectrogram``). Learns time-frequency wear
  signatures (tooth-pass harmonics + broadband rise).
* :class:`FusionModel` - late-fusion of a 1-D vibration branch and a thermal
  scalar-feature MLP branch, mirroring the "sensor fusion wins" result of the
  tabular ablation.

Every model shares a small trunk feeding **two heads** so a single network is
trained multi-task and evaluated with the *same* metrics as
:mod:`models.baseline`:

* a **regression** head -> ``(wear_level, cycle_time_factor, quality_score)``
  (MAE / RMSE / R2), and
* a **classification** head -> ``health_state`` (Accuracy / macro-F1).

``forward`` returns a ``(regression, health_logits)`` tuple (tuple, not dict, so
``torch.onnx.export`` stays simple and the output names are stable).

This module requires the optional ``dl`` extra (``torch``, ``onnx``,
``onnxruntime``) and is imported lazily so ``import models`` never needs torch.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DLModelConfig",
    "Vibration1DCNN",
    "SpectrogramCNN",
    "FusionModel",
    "get_model",
    "set_seed",
    "train_dl_model",
    "evaluate_dl_model",
    "export_to_onnx",
    "example_inputs_for",
    "default_dynamic_axes",
    "REGRESSION_TARGETS",
    "DEFAULT_CLASS_NAMES",
    "__version__",
]

__version__ = "0.1.0"

#: Regression head output order (matches models.baseline REGRESSION_TARGETS).
REGRESSION_TARGETS: tuple[str, ...] = ("wear_level", "cycle_time_factor", "quality_score")
#: Default health-state class order (alphabetical, as LabelEncoder produces).
DEFAULT_CLASS_NAMES: tuple[str, ...] = ("critical", "healthy", "monitor", "warning")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class DLModelConfig:
    """Architecture hyper-parameters shared by the three DL models.

    Kept intentionally small / edge-friendly (a few hundred k params) - the goal
    is a fast, exportable v1, not a leaderboard model.
    """

    # --- shared head ---
    n_regression: int = 3
    n_classes: int = 4
    dropout: float = 0.2
    head_hidden: int = 64

    # --- 1-D vibration CNN ---
    cnn1d_channels: tuple[int, ...] = (16, 32, 64, 128)
    cnn1d_kernel: int = 7
    cnn1d_pool: int = 4

    # --- 2-D spectrogram CNN ---
    cnn2d_channels: tuple[int, ...] = (16, 32, 64)
    cnn2d_kernel: int = 3

    # --- fusion: thermal scalar branch ---
    thermal_dim: int = 5
    thermal_hidden: tuple[int, ...] = (32, 32)
    fusion_hidden: int = 128


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class _MultiTaskHead(nn.Module):
    """Two linear heads (regression + classification) on a shared embedding."""

    def __init__(self, in_dim: int, cfg: DLModelConfig) -> None:
        super().__init__()
        self.reg = nn.Linear(in_dim, cfg.n_regression)
        self.clf = nn.Linear(in_dim, cfg.n_classes)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.reg(z), self.clf(z)


def _conv1d_stack(channels: Sequence[int], kernel: int, pool: int) -> tuple[nn.Sequential, int]:
    """Conv1d -> BN -> ReLU -> MaxPool blocks; returns (module, out_channels)."""
    layers: list[nn.Module] = []
    in_ch = 1
    for out_ch in channels:
        layers += [
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
        ]
        in_ch = out_ch
    return nn.Sequential(*layers), in_ch


def _mlp(in_dim: int, hidden: Sequence[int], dropout: float) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        d = h
    return nn.Sequential(*layers), d


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class Vibration1DCNN(nn.Module):
    """1-D CNN on the normalized vibration waveform.

    Input: ``(B, 1, L)`` (or ``(B, L)``, auto-unsqueezed). A stack of
    Conv1d/BN/ReLU/MaxPool blocks followed by **global average pooling**, so the
    model accepts *variable-length* chunks (handy for streaming and ONNX dynamic
    sequence length). Output: ``(regression (B, 3), health_logits (B, C))``.
    """

    input_kind = "waveform"

    def __init__(self, cfg: DLModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DLModelConfig()
        self.features, feat_dim = _conv1d_stack(
            self.cfg.cnn1d_channels, self.cfg.cnn1d_kernel, self.cfg.cnn1d_pool
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, self.cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
        )
        self.head = _MultiTaskHead(self.cfg.head_hidden, self.cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z = self.features(x)
        z = self.pool(z).flatten(1)
        z = self.trunk(z)
        return self.head(z)


class SpectrogramCNN(nn.Module):
    """2-D CNN on the log-power STFT spectrogram.

    Input: ``(B, 1, n_freq, n_time)`` (or ``(B, n_freq, n_time)``,
    auto-unsqueezed). Conv2d/BN/ReLU/MaxPool blocks + **global average pooling**,
    so it tolerates variable time (and frequency) extents. Output:
    ``(regression (B, 3), health_logits (B, C))``.
    """

    input_kind = "spectrogram"

    def __init__(self, cfg: DLModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DLModelConfig()
        layers: list[nn.Module] = []
        in_ch = 1
        k = self.cfg.cnn2d_kernel
        for out_ch in self.cfg.cnn2d_channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.trunk = nn.Sequential(
            nn.Linear(in_ch, self.cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
        )
        self.head = _MultiTaskHead(self.cfg.head_hidden, self.cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.features(x)
        z = self.pool(z).flatten(1)
        z = self.trunk(z)
        return self.head(z)


class FusionModel(nn.Module):
    """Late-fusion of a 1-D vibration branch and a thermal scalar-feature MLP.

    Inputs: ``waveform (B, 1, L)`` (or ``(B, L)``) and ``thermal (B, thermal_dim)``
    (observable thermal scalars, e.g. ``max_temp_c, temp_rise_c, therm_std_c,
    therm_slope_c_per_s, mean_temp_c``). The two branch embeddings are
    concatenated and passed through a shared fusion FC before the multi-task
    head. Output: ``(regression (B, 3), health_logits (B, C))``.
    """

    input_kind = "fusion"

    def __init__(self, cfg: DLModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DLModelConfig()
        self.vib_features, vib_dim = _conv1d_stack(
            self.cfg.cnn1d_channels, self.cfg.cnn1d_kernel, self.cfg.cnn1d_pool
        )
        self.vib_pool = nn.AdaptiveAvgPool1d(1)
        self.thermal_mlp, therm_dim = _mlp(
            self.cfg.thermal_dim, self.cfg.thermal_hidden, self.cfg.dropout
        )
        self.fusion = nn.Sequential(
            nn.Linear(vib_dim + therm_dim, self.cfg.fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
        )
        self.head = _MultiTaskHead(self.cfg.fusion_hidden, self.cfg)

    def forward(
        self, waveform: torch.Tensor, thermal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        v = self.vib_pool(self.vib_features(waveform)).flatten(1)
        t = self.thermal_mlp(thermal)
        z = self.fusion(torch.cat([v, t], dim=1))
        return self.head(z)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_MODEL_ALIASES: dict[str, type[nn.Module]] = {
    "1dcnn": Vibration1DCNN,
    "cnn1d": Vibration1DCNN,
    "vibration1dcnn": Vibration1DCNN,
    "spectrogram": SpectrogramCNN,
    "spectrogramcnn": SpectrogramCNN,
    "cnn2d": SpectrogramCNN,
    "fusion": FusionModel,
    "fusionmodel": FusionModel,
}


def get_model(name: str, config: DLModelConfig | dict[str, Any] | None = None) -> nn.Module:
    """Construct a DL model by name.

    Parameters
    ----------
    name:
        One of ``1dcnn`` / ``spectrogram`` / ``fusion`` (aliases accepted).
    config:
        A :class:`DLModelConfig`, a plain ``dict`` of overrides, or ``None`` for
        defaults.
    """
    key = name.strip().lower()
    if key not in _MODEL_ALIASES:
        raise ValueError(
            f"Unknown model {name!r}. Choose from: 1dcnn, spectrogram, fusion."
        )
    if isinstance(config, dict):
        cfg = DLModelConfig(**config)
    elif isinstance(config, DLModelConfig) or config is None:
        cfg = config or DLModelConfig()
    else:  # pragma: no cover - defensive
        raise TypeError("config must be DLModelConfig, dict, or None.")
    return _MODEL_ALIASES[key](cfg)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed Python / NumPy / Torch RNGs for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CPU CI
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Metrics (numpy; no sklearn dependency so the dl extra stays lean)
# --------------------------------------------------------------------------- #
def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, class_names: Sequence[str] | None
) -> dict[str, Any]:
    acc = float(np.mean(y_true == y_pred)) if y_true.size else 0.0
    per_class_f1: list[float] = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class_f1.append(f1)
    names = list(class_names) if class_names is not None else [str(c) for c in range(n_classes)]
    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(per_class_f1)) if per_class_f1 else 0.0,
        "per_class_f1": dict(zip(names, per_class_f1)),
    }


# --------------------------------------------------------------------------- #
# Batch plumbing (unifies the three input kinds)
# --------------------------------------------------------------------------- #
def _model_forward(model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device):
    kind = getattr(model, "input_kind", "waveform")
    if kind == "waveform":
        return model(batch["waveform"].to(device))
    if kind == "spectrogram":
        return model(batch["spectrogram"].to(device))
    if kind == "fusion":
        return model(batch["waveform"].to(device), batch["thermal"].to(device))
    raise ValueError(f"Unknown model.input_kind: {kind!r}")  # pragma: no cover


def _multitask_loss(
    out: tuple[torch.Tensor, torch.Tensor],
    batch: dict[str, torch.Tensor],
    device: torch.device,
    reg_weight: float,
    clf_weight: float,
) -> torch.Tensor:
    reg, logits = out
    y_reg = batch["y_reg"].to(device)
    y_clf = batch["y_clf"].to(device)
    loss = reg_weight * F.mse_loss(reg, y_reg) + clf_weight * F.cross_entropy(logits, y_clf)
    return loss


@torch.no_grad()
def _eval_loss(model, loader, device, reg_weight, clf_weight) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        out = _model_forward(model, batch, device)
        bs = int(batch["y_reg"].shape[0])
        total += float(_multitask_loss(out, batch, device, reg_weight, clf_weight)) * bs
        n += bs
    return total / max(1, n)


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def train_dl_model(
    model: nn.Module,
    train_loader: Iterable[dict[str, torch.Tensor]],
    val_loader: Iterable[dict[str, torch.Tensor]] | None = None,
    *,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | torch.device = "cpu",
    reg_weight: float = 1.0,
    clf_weight: float = 0.5,
    patience: int = 8,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train a multi-task DL model with Adam + early stopping on val loss.

    Restores the best-val checkpoint into ``model`` in place. Returns a history
    dict (``train_loss``, ``val_loss`` per epoch, ``best_val_loss``,
    ``epochs_run``). ``clf_weight`` defaults to 0.5 because the 3-target
    regression MSE and the 4-class cross-entropy live on different scales.
    """
    set_seed(seed)
    device = torch.device(device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_ctr = 0
    history: dict[str, Any] = {"train_loss": [], "val_loss": []}
    epochs_run = 0

    for epoch in range(epochs):
        epochs_run = epoch + 1
        model.train()
        run, n = 0.0, 0
        for batch in train_loader:
            opt.zero_grad()
            out = _model_forward(model, batch, device)
            loss = _multitask_loss(out, batch, device, reg_weight, clf_weight)
            loss.backward()
            opt.step()
            bs = int(batch["y_reg"].shape[0])
            run += loss.item() * bs
            n += bs
        train_loss = run / max(1, n)
        val_loss = (
            _eval_loss(model, val_loader, device, reg_weight, clf_weight)
            if val_loader is not None
            else train_loss
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if verbose:
            print(f"  epoch {epoch + 1:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch + 1} (best val={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    history["epochs_run"] = epochs_run
    return history


@torch.no_grad()
def evaluate_dl_model(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    *,
    device: str | torch.device = "cpu",
    regression_targets: Sequence[str] = REGRESSION_TARGETS,
    class_names: Sequence[str] | None = DEFAULT_CLASS_NAMES,
) -> dict[str, Any]:
    """Compute baseline-parity metrics: per-target MAE/RMSE/R2 + Acc/macro-F1."""
    device = torch.device(device)
    model.to(device)
    model.eval()
    reg_p, reg_y, clf_p, clf_y = [], [], [], []
    n_classes = model.cfg.n_classes if hasattr(model, "cfg") else len(class_names or [])
    for batch in loader:
        reg, logits = _model_forward(model, batch, device)
        reg_p.append(reg.cpu().numpy())
        reg_y.append(batch["y_reg"].cpu().numpy())
        clf_p.append(logits.argmax(dim=1).cpu().numpy())
        clf_y.append(batch["y_clf"].cpu().numpy())

    reg_p = np.concatenate(reg_p) if reg_p else np.empty((0, len(regression_targets)))
    reg_y = np.concatenate(reg_y) if reg_y else np.empty((0, len(regression_targets)))
    clf_p = np.concatenate(clf_p) if clf_p else np.empty((0,), dtype=int)
    clf_y = np.concatenate(clf_y) if clf_y else np.empty((0,), dtype=int)

    metrics: dict[str, Any] = {"regression": {}, "classification": {}}
    for i, name in enumerate(regression_targets):
        metrics["regression"][name] = _regression_metrics(reg_y[:, i], reg_p[:, i])
    metrics["classification"] = _classification_metrics(clf_y, clf_p, n_classes, class_names)
    return metrics


# --------------------------------------------------------------------------- #
# ONNX export
# --------------------------------------------------------------------------- #
def example_inputs_for(
    model: nn.Module, *, batch: int = 1, seq_len: int = 40960, n_freq: int = 513, n_time: int = 161
) -> tuple[torch.Tensor, ...]:
    """Build a representative example input tuple for ONNX export / tracing."""
    kind = getattr(model, "input_kind", "waveform")
    if kind == "waveform":
        return (torch.randn(batch, 1, seq_len),)
    if kind == "spectrogram":
        return (torch.randn(batch, 1, n_freq, n_time),)
    if kind == "fusion":
        thermal_dim = model.cfg.thermal_dim if hasattr(model, "cfg") else 5
        return (torch.randn(batch, 1, seq_len), torch.randn(batch, thermal_dim))
    raise ValueError(f"Unknown model.input_kind: {kind!r}")  # pragma: no cover


def default_dynamic_axes(model: nn.Module) -> dict[str, dict[int, str]]:
    """Dynamic-axis spec: batch (all) + sequence/time length where meaningful."""
    kind = getattr(model, "input_kind", "waveform")
    axes: dict[str, dict[int, str]] = {
        "regression": {0: "batch"},
        "health_logits": {0: "batch"},
    }
    if kind == "waveform":
        axes["waveform"] = {0: "batch", 2: "length"}
    elif kind == "spectrogram":
        axes["spectrogram"] = {0: "batch", 3: "time"}
    elif kind == "fusion":
        axes["waveform"] = {0: "batch", 2: "length"}
        axes["thermal"] = {0: "batch"}
    return axes


def export_to_onnx(
    model: nn.Module,
    onnx_path: str | Path,
    example_input: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
    *,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset: int = 17,
) -> Path:
    """Export a DL model to ONNX with stable IO names and dynamic axes.

    Input names follow the model's ``input_kind`` (``waveform`` / ``spectrogram``
    / [``waveform``, ``thermal``] for fusion); outputs are ``regression`` and
    ``health_logits``. Batch (and sequence/time length for the streaming models)
    are exported as dynamic axes so a single artifact serves any chunk size.
    """
    model.eval()
    kind = getattr(model, "input_kind", "waveform")
    if example_input is None:
        example_input = example_inputs_for(model)
    if isinstance(example_input, torch.Tensor):
        example_input = (example_input,)
    if kind == "waveform":
        input_names = ["waveform"]
    elif kind == "spectrogram":
        input_names = ["spectrogram"]
    else:
        input_names = ["waveform", "thermal"]
    output_names = ["regression", "health_logits"]
    if dynamic_axes is None:
        dynamic_axes = default_dynamic_axes(model)

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # Use the legacy TorchScript exporter (dynamo=False): it consumes the
    # ``dynamic_axes`` mapping directly and avoids the onnxscript dependency the
    # newer dynamo path pulls in - keeping the ``dl`` extra lean. (The dynamo
    # exporter uses ``dynamic_shapes`` instead; we deliberately keep dynamic_axes.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        torch.onnx.export(
            model,
            example_input,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    return onnx_path
