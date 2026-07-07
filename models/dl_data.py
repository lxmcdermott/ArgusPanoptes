"""Data loading for the Argus Panoptes DL models (Day 3).

Bridges the Parquet dataset produced by ``scripts/generate_dataset.py`` to the
PyTorch models in :mod:`models.dl_models`:

* Raw ``float32`` vibration waveforms are read from the partitioned ``records/``
  dataset (no extra storage - the 1D-CNN reuses the existing waveform columns).
* The normalized waveform and (optionally) the log-power spectrogram are
  produced by :class:`dsp.SignalProcessor` (``get_normalized_waveform`` /
  ``compute_spectrogram``) so the DL front-end is *identical* to the DSP path -
  and are **precomputed once** into arrays to keep per-epoch cost low.
* Labels / thermal scalars come from ``manifest.parquet``.

Crucially, the train/val/test split reproduces :mod:`models.baseline` exactly
(one stratified 80/20 split on ``wear_bin`` with the same seed, then a further
80/20 carve of train for early-stopping validation), so a DL model and the
XGBoost baseline are always compared on the **same held-out test rows**.

Requires the ``dl`` extra (torch) and ``scikit-learn`` (for the stratified
split, shared with the ``ml`` extra).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

#: Observable thermal scalars for the fusion branch (leakage-aware; identical to
#: models.baseline.THERMAL_FEATURES). Requires the dataset to be generated with
#: ``--extract-features`` (adds therm_std_c / therm_slope_c_per_s).
THERMAL_FEATURES: tuple[str, ...] = (
    "mean_temp_c",
    "max_temp_c",
    "temp_rise_c",
    "therm_std_c",
    "therm_slope_c_per_s",
)
REGRESSION_LABEL_COLS: tuple[str, ...] = (
    "label_wear_level",
    "label_cycle_time_factor",
    "label_quality_score",
)
CLASSIFICATION_LABEL_COL = "label_health_state"


# --------------------------------------------------------------------------- #
# Parquet readers
# --------------------------------------------------------------------------- #
def load_waveforms(records_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``(sample_id, vibration_waveform)`` from the partitioned records.

    Returns ``(sample_ids, waveforms_object_array)`` sorted by ``sample_id`` so
    the row order matches ``manifest.parquet`` (and therefore the baseline split).
    Waveforms may be variable length, so they are returned as an object array of
    1-D ``float32`` arrays.
    """
    import pyarrow as pa
    import pyarrow.dataset as ds

    part = ds.partitioning(
        schema=pa.schema([("alloy", pa.string()), ("wear_bin", pa.string())]),
        flavor="hive",
    )
    dataset = ds.dataset(str(records_dir), partitioning=part, format="parquet")
    table = dataset.to_table(columns=["sample_id", "vibration_waveform"])
    sample_ids = table.column("sample_id").to_numpy()
    wf_list = table.column("vibration_waveform").to_pylist()
    order = np.argsort(sample_ids, kind="stable")
    sample_ids = sample_ids[order]
    waveforms = np.empty(len(wf_list), dtype=object)
    for new_i, old_i in enumerate(order):
        waveforms[new_i] = np.asarray(wf_list[old_i], dtype=np.float32)
    return sample_ids, waveforms


# --------------------------------------------------------------------------- #
# Fixed-length helper
# --------------------------------------------------------------------------- #
def _crop_or_pad(x: np.ndarray, target_len: int | None) -> np.ndarray:
    """Center-crop or zero-pad a 1-D signal to ``target_len`` (no-op if None)."""
    if target_len is None or x.size == target_len:
        return x
    if x.size > target_len:
        start = (x.size - target_len) // 2
        return x[start : start + target_len]
    out = np.zeros(target_len, dtype=x.dtype)
    out[: x.size] = x
    return out


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ArgusDLDataset(Dataset):
    """In-memory dataset of precomputed DL inputs + multi-task targets.

    Parameters
    ----------
    waveforms_norm:
        ``(N, L)`` float32 normalized waveforms (already fixed-length).
    spectrograms:
        ``(N, F, T)`` float32 log-power spectrograms, or ``None`` if unused.
    thermal:
        ``(N, thermal_dim)`` float32 thermal scalars, or ``None`` if unused.
    y_reg:
        ``(N, 3)`` float32 regression targets.
    y_clf:
        ``(N,)`` int64 encoded health-state labels.
    mode:
        ``"waveform"`` | ``"spectrogram"`` | ``"fusion"`` - selects which tensors
        the batch dict carries.
    """

    def __init__(
        self,
        waveforms_norm: np.ndarray | None,
        spectrograms: np.ndarray | None,
        thermal: np.ndarray | None,
        y_reg: np.ndarray,
        y_clf: np.ndarray,
        mode: str,
    ) -> None:
        self.mode = mode
        self.waveforms_norm = waveforms_norm
        self.spectrograms = spectrograms
        self.thermal = thermal
        self.y_reg = y_reg.astype(np.float32)
        self.y_clf = y_clf.astype(np.int64)

    def __len__(self) -> int:
        return int(self.y_reg.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {
            "y_reg": torch.from_numpy(self.y_reg[idx]),
            "y_clf": torch.tensor(self.y_clf[idx]),
        }
        if self.mode in ("waveform", "fusion"):
            # (1, L) - single channel.
            item["waveform"] = torch.from_numpy(self.waveforms_norm[idx]).unsqueeze(0)
        if self.mode == "spectrogram":
            # (1, F, T) - single channel.
            item["spectrogram"] = torch.from_numpy(self.spectrograms[idx]).unsqueeze(0)
        if self.mode == "fusion":
            item["thermal"] = torch.from_numpy(self.thermal[idx])
        return item


# --------------------------------------------------------------------------- #
# Precompute + split + loaders
# --------------------------------------------------------------------------- #
def _stratified_split(
    n: int, strat: np.ndarray | None, seed: int, test_frac: float = 0.2, val_frac: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the baseline split: (train, val, test) index arrays.

    Mirrors ``models.baseline``: one stratified test split on ``wear_bin`` with
    ``random_state=seed``, then a further stratified carve of ``val_frac`` from
    train for early stopping. Falls back to non-stratified splits when a class is
    too small to stratify.
    """
    from sklearn.model_selection import train_test_split

    idx = np.arange(n)
    try:
        train_idx, test_idx = train_test_split(
            idx, test_size=test_frac, random_state=seed, stratify=strat
        )
    except ValueError:
        train_idx, test_idx = train_test_split(idx, test_size=test_frac, random_state=seed)
    strat_tr = strat[train_idx] if strat is not None else None
    try:
        fit_idx, val_idx = train_test_split(
            train_idx, test_size=val_frac, random_state=seed, stratify=strat_tr
        )
    except ValueError:
        fit_idx, val_idx = train_test_split(train_idx, test_size=val_frac, random_state=seed)
    return fit_idx, val_idx, test_idx


def prepare_dl_data(
    data_dir: Path,
    mode: str,
    *,
    processor: Any | None = None,
    target_len: int | None = 16384,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Load, precompute, split, and wrap the dataset into DataLoaders.

    Returns a dict with ``train_loader`` / ``val_loader`` / ``test_loader``,
    the split index arrays, ``class_names``, ``thermal_dim``, ``fs_hz``,
    ``input_len`` / spectrogram ``input_shape``, and the loaded ``manifest``
    DataFrame (so the caller can run the XGBoost baseline on the same test rows).
    """
    from dsp import SignalProcessor

    # Accept either a raw input-kind or a model name/alias.
    _KIND = {
        "1dcnn": "waveform", "cnn1d": "waveform", "vibration1dcnn": "waveform",
        "waveform": "waveform",
        "spectrogram": "spectrogram", "spectrogramcnn": "spectrogram", "cnn2d": "spectrogram",
        "fusion": "fusion", "fusionmodel": "fusion",
    }
    mode = _KIND.get(mode.strip().lower(), mode)

    data_dir = Path(data_dir)
    manifest = pd.read_parquet(data_dir / "manifest.parquet")
    sample_ids, waveforms = load_waveforms(data_dir / "records")
    if not np.array_equal(np.sort(manifest["sample_id"].to_numpy()), sample_ids):
        raise RuntimeError("records/ sample_ids do not match manifest.parquet - regenerate dataset.")
    manifest = manifest.sort_values("sample_id").reset_index(drop=True)

    if processor is None:
        processor = SignalProcessor()
    fs = float(manifest["vib_fs_hz"].iloc[0])

    n = len(manifest)
    y_reg = manifest[list(REGRESSION_LABEL_COLS)].to_numpy(dtype=np.float32)
    class_names = sorted(manifest[CLASSIFICATION_LABEL_COL].astype(str).unique().tolist())
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    y_clf = manifest[CLASSIFICATION_LABEL_COL].astype(str).map(cls_to_idx).to_numpy(dtype=np.int64)

    # --- Precompute normalized waveforms (waveform / fusion) ---
    waveforms_norm = None
    input_len = None
    if mode in ("waveform", "fusion"):
        wf_norm = np.zeros((n, target_len if target_len else waveforms[0].size), dtype=np.float32)
        for i in range(n):
            w = processor.get_normalized_waveform(waveforms[i], fs)
            wf_norm[i] = _crop_or_pad(w, target_len if target_len else w.size)
        waveforms_norm = wf_norm
        input_len = int(wf_norm.shape[1])

    # --- Precompute spectrograms (spectrogram) ---
    spectrograms = None
    input_shape = None
    if mode == "spectrogram":
        specs = []
        for i in range(n):
            w = _crop_or_pad(waveforms[i], target_len) if target_len else waveforms[i]
            specs.append(processor.compute_stft(processor._normalize(
                processor.preprocess(w, fs), method=processor.dl.normalize_for_dl
            ), fs)["power"])
        spectrograms = np.stack(specs).astype(np.float32)
        input_shape = tuple(int(s) for s in spectrograms.shape[1:])

    # --- Thermal scalars (fusion) ---
    thermal = None
    thermal_dim = len(THERMAL_FEATURES)
    if mode == "fusion":
        missing = [c for c in THERMAL_FEATURES if c not in manifest.columns]
        if missing:
            raise SystemExit(
                f"Fusion model needs thermal features {missing} - regenerate the "
                "dataset with `--extract-features`."
            )
        thermal = manifest[list(THERMAL_FEATURES)].to_numpy(dtype=np.float32)
        # Standardize thermal scalars (very different physical scales / units).
        mu, sigma = thermal.mean(axis=0), thermal.std(axis=0)
        sigma[sigma < 1e-9] = 1.0
        thermal = (thermal - mu) / sigma

    strat = manifest["wear_bin"].to_numpy() if "wear_bin" in manifest.columns else None
    fit_idx, val_idx, test_idx = _stratified_split(n, strat, seed)

    def _subset(idx: np.ndarray) -> ArgusDLDataset:
        return ArgusDLDataset(
            waveforms_norm[idx] if waveforms_norm is not None else None,
            spectrograms[idx] if spectrograms is not None else None,
            thermal[idx] if thermal is not None else None,
            y_reg[idx],
            y_clf[idx],
            mode,
        )

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        _subset(fit_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g
    )
    val_loader = DataLoader(_subset(val_idx), batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(_subset(test_idx), batch_size=batch_size, num_workers=num_workers)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "fit_idx": fit_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "class_names": class_names,
        "thermal_dim": thermal_dim,
        "fs_hz": fs,
        "input_len": input_len,
        "input_shape": input_shape,
        "manifest": manifest,
        "n_samples": n,
    }
