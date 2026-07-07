"""ML models & experiments for Argus Panoptes.

Day 2 (implemented): interpretable **XGBoost baselines + feature-group ablations**
on the physics-informed DSP features in ``manifest.parquet`` — see
:mod:`models.baseline` (run as ``python models/baseline.py``).

Day 3 (implemented): **deep learning** on the raw signals — :mod:`models.dl_models`
(``Vibration1DCNN``, ``SpectrogramCNN``, ``FusionModel`` + train / eval / ONNX
export utilities), :mod:`models.dl_data` (Parquet → PyTorch loaders reusing
:class:`dsp.SignalProcessor`), :mod:`models.train_dl` (training CLI with a
same-split XGBoost comparison), and :mod:`models.onnx_inference`
(``ONNXPerceptor`` for torch-free edge inference).

.. note::
   Every submodule here is imported **lazily** (nothing is imported from this
   package ``__init__``), so ``import models`` never requires torch or the ML
   stack. :mod:`models.baseline` needs the ``ml`` extra
   (``pip install -e ".[ml]"``); :mod:`models.dl_models` / :mod:`models.dl_data`
   / :mod:`models.train_dl` need the ``dl`` extra (``pip install -e ".[dl]"`` →
   torch, onnx, onnxruntime); :mod:`models.onnx_inference` only needs
   ``onnxruntime`` (no torch) for inference.
"""

__version__ = "0.1.0"
