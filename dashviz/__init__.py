"""Helper package for the Argus Panoptes Streamlit dashboard (Day 5 ops layer).

The operator UI lives in the project-root ``dashboard.py`` script; this package
holds the reusable, testable building blocks it orchestrates:

* :mod:`dashviz.theme` - dark industrial theme (CSS, colors, styled components,
  and a shared Plotly dark layout template).
* :mod:`dashviz.plots` - Plotly figure builders (waveform, FFT, STFT heatmap,
  gauges, trends, comparison bars) that all use the shared dark template.
* :mod:`dashviz.infra` - session-state schema, cached resources (perceptor /
  HTTP client / simulators), the unified :func:`run_inference`, and downsampling.
* :mod:`dashviz.optimization` - transparent downstream production-impact model.
* :mod:`dashviz.scenarios` - pre-built demo scenarios for repeatable recordings.
* :mod:`dashviz.metrics` - cached loaders for the experiment metric JSONs.

Nothing here imports Streamlit at module import time beyond what is needed, so
the pure-Python pieces (plots, optimization, scenarios, metrics, downsample)
remain importable and unit-testable without a running Streamlit context.
"""

from __future__ import annotations

__all__ = [
    "theme",
    "plots",
    "infra",
    "optimization",
    "scenarios",
    "metrics",
]

__version__ = "0.1.0"
