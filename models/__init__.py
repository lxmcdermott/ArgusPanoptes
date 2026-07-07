"""ML models & experiments for Argus Panoptes.

Day 2 (implemented): interpretable **XGBoost baselines + feature-group ablations**
on the physics-informed DSP features in ``manifest.parquet`` — see
:mod:`models.baseline` (run as ``python models/baseline.py``).

Planned (Day 3+): 1D-CNN on raw vibration, spectrogram CNN, and sensor-fusion
(vibration + thermal) models, with ONNX export for edge inference.

.. note::
   :mod:`models.baseline` depends on the optional ``ml`` extra
   (``pip install -e ".[ml]"`` → scikit-learn, xgboost, joblib). It is imported
   lazily (not from this package ``__init__``) so importing :mod:`models` never
   requires the ML dependencies.
"""

__version__ = "0.1.0"
