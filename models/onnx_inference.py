"""ONNX Runtime inference for the Argus Panoptes DL models (Day 3 / edge path).

A thin, dependency-light wrapper around ``onnxruntime`` that mirrors the
expected *streaming* input shapes of the exported models
(:func:`models.dl_models.export_to_onnx`):

* ``1dcnn``       -> ``waveform (B, 1, L)``
* ``spectrogram`` -> ``spectrogram (B, 1, F, T)``
* ``fusion``      -> ``waveform (B, 1, L)`` + ``thermal (B, thermal_dim)``

Outputs are ``regression (B, 3)`` and ``health_logits (B, C)``. Only
``onnxruntime`` (+ NumPy) is required here - torch is *not* needed for
inference, which is the whole point of exporting to ONNX for the edge.

Example
-------
>>> from models.onnx_inference import ONNXPerceptor
>>> perc = ONNXPerceptor("experiments/models/dl_1dcnn.onnx")
>>> import numpy as np
>>> out = perc.infer(np.random.randn(1, 1, 16384).astype("float32"))
>>> out["regression"].shape, out["health_logits"].shape
((1, 3), (1, 4))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


class ONNXPerceptor:
    """Load an exported Argus DL model and run CPU (or GPU) ONNX inference.

    Parameters
    ----------
    onnx_path:
        Path to a ``.onnx`` artifact from :func:`export_to_onnx`.
    providers:
        ONNX Runtime execution providers. Defaults to CPU; pass
        ``["CUDAExecutionProvider", "CPUExecutionProvider"]`` on a GPU host.
    """

    def __init__(self, onnx_path: str | Path, providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        self.onnx_path = str(onnx_path)
        self.session = ort.InferenceSession(
            self.onnx_path, providers=providers or ["CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def infer(self, *inputs: np.ndarray, **named: np.ndarray) -> dict[str, np.ndarray]:
        """Run a forward pass.

        Inputs may be passed positionally (in the model's input order) or by name
        (e.g. ``waveform=...``, ``thermal=...``). Returns a dict keyed by output
        name (``regression`` / ``health_logits``), all ``float32``.
        """
        if named:
            feeds = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in named.items()}
        else:
            if len(inputs) != len(self.input_names):
                raise ValueError(
                    f"Expected {len(self.input_names)} inputs {self.input_names}, "
                    f"got {len(inputs)}."
                )
            feeds = {
                name: np.ascontiguousarray(arr, dtype=np.float32)
                for name, arr in zip(self.input_names, inputs)
            }
        outputs = self.session.run(self.output_names, feeds)
        return dict(zip(self.output_names, outputs))

    def predict(self, *inputs: np.ndarray, **named: np.ndarray) -> dict[str, Any]:
        """Convenience: return regression vector + health-state class probabilities."""
        out = self.infer(*inputs, **named)
        probs = _softmax(out["health_logits"], axis=-1)
        return {
            "regression": out["regression"],
            "health_probs": probs,
            "health_class": np.argmax(probs, axis=-1),
        }


def infer_onnx(
    onnx_path: str | Path, *inputs: np.ndarray, providers: list[str] | None = None
) -> dict[str, np.ndarray]:
    """One-shot helper: load ``onnx_path`` and run a single forward pass."""
    return ONNXPerceptor(onnx_path, providers=providers).infer(*inputs)
