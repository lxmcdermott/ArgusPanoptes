"""UI-agnostic building blocks for the Argus Panoptes operator dashboard.

The operator UI is the high-performance **NiceGUI + Plotly** app in
``app/nicegui_dashboard.py``; this package holds the reusable, framework-neutral
pieces it composes (the legacy Streamlit dashboard is archived under
``_legacy/``):

* :mod:`dashviz.theme` - dark industrial theme (CSS, colors, styled components,
  and a shared Plotly dark layout template).
* :mod:`dashviz.plots` - Plotly figure builders (waveform, FFT, STFT heatmap,
  gauges, trends, comparison bars) that all use the shared dark template.
* :mod:`dashviz.orchestrator` - the framework-agnostic
  :class:`~dashviz.orchestrator.SimulationOrchestrator`: a background
  simulation/DSP/inference loop with thread-safe :class:`Snapshot` state, live
  parameter updates, the unified :func:`run_inference` (direct + API modes), and
  display downsampling / STFT helpers.
* :mod:`dashviz.optimization` - transparent downstream production-impact model.
* :mod:`dashviz.scenarios` - pre-built demo scenarios for repeatable recordings.
* :mod:`dashviz.metrics` - cached loaders for the experiment metric JSONs.

Nothing here imports a UI framework at module import time, so every piece stays
importable and unit-testable headlessly.
"""

from __future__ import annotations

__all__ = [
    "theme",
    "plots",
    "orchestrator",
    "optimization",
    "scenarios",
    "metrics",
]

__version__ = "0.2.0"
