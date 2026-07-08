"""Plotly figure builders for the Argus Panoptes dashboard.

Every builder returns a ready-to-render :class:`plotly.graph_objects.Figure`
styled via :func:`dashviz.theme.get_dark_layout`, so all charts share the dark
industrial look. Builders are pure (NumPy + Plotly only, no Streamlit) so they
can be unit-tested and reused in either operation mode.

Performance
-----------
All time-series inputs are passed through :func:`downsample` (default 2000
points) before plotting so 40.96 kHz waveforms render instantly without
shipping hundreds of thousands of points to the browser.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

from dashviz.theme import COLORS, get_dark_layout


def downsample(arr: np.ndarray, max_points: int = 2000) -> np.ndarray:
    """Uniformly subsample a 1-D array to at most ``max_points`` samples.

    Uses evenly spaced index selection (cheap, preserves overall shape and the
    endpoints). Returns the input unchanged when it is already short enough.

    Parameters
    ----------
    arr:
        1-D array-like to thin.
    max_points:
        Maximum number of points to keep (>= 2).
    """
    a = np.asarray(arr)
    n = a.shape[0]
    if n <= max_points or max_points < 2:
        return a
    idx = np.linspace(0, n - 1, max_points).astype(np.int64)
    return a[idx]


def _empty_figure(message: str = "No data yet", height: int = 260) -> go.Figure:
    """A styled placeholder figure shown before any data is available."""
    fig = go.Figure()
    fig.update_layout(**get_dark_layout(height=height))
    fig.add_annotation(
        text=message,
        showarrow=False,
        font={"color": COLORS.text_muted, "size": 14},
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# --------------------------------------------------------------------------- #
# Time-domain waveform
# --------------------------------------------------------------------------- #
def build_waveform_figure(
    vibration: np.ndarray,
    *,
    t: np.ndarray | None = None,
    fs: float | None = None,
    title: str = "Live Vibration Waveform",
    color: str | None = None,
    height: int = 260,
    max_points: int = 2000,
    uirevision: str | None = "argus_live_signals",
) -> go.Figure:
    """Line plot of a (downsampled) vibration chunk in g vs time."""
    y = np.asarray(vibration, dtype=np.float64).ravel()
    if y.size == 0:
        return _empty_figure("Waiting for signal\u2026", height)
    if t is None:
        t = np.arange(y.size) / fs if fs else np.arange(y.size, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()[: y.size]

    ys = downsample(y, max_points)
    ts = downsample(t, max_points)
    color = color or COLORS.accent_soft

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=ts,
            y=ys,
            mode="lines",
            line={"color": color, "width": 1.4},
            fill="tozeroy",
            fillcolor=_rgba(color, 0.22),
            hovertemplate="t=%{x:.4f}s<br>%{y:.3f} g<extra></extra>",
            name="accel",
        )
    )
    fig.update_layout(**get_dark_layout(height=height, title=title))
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Acceleration (g)")
    # Stable uirevision lets Plotly update traces in place across fragment ticks
    # (preserves zoom/pan and avoids a full redraw / flicker).
    if uirevision:
        fig.update_layout(uirevision=uirevision)
    return fig


# --------------------------------------------------------------------------- #
# Frequency-domain FFT with tooth-pass annotation
# --------------------------------------------------------------------------- #
def build_fft_figure(
    vibration: np.ndarray,
    fs: float,
    *,
    tpf_hz: float | None = None,
    title: str = "Vibration Spectrum (FFT)",
    freq_max: float | None = 8000.0,
    height: int = 260,
    max_points: int = 2000,
    uirevision: str | None = "argus_live_signals",
) -> go.Figure:
    """Single-sided amplitude spectrum with an optional tooth-pass marker.

    The tooth-pass frequency (TPF) and its dominant peak are the key blade-wear
    cues, so when ``tpf_hz`` is supplied we annotate the fundamental line.
    """
    x = np.asarray(vibration, dtype=np.float64).ravel()
    if x.size < 8 or fs <= 0:
        return _empty_figure("Waiting for signal\u2026", height)

    x = x - float(np.mean(x))
    window = np.hanning(x.size)
    spectrum = np.fft.rfft(x * window)
    mag = np.abs(spectrum) * (2.0 / np.sum(window))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)

    if freq_max is not None:
        keep = freqs <= freq_max
        freqs, mag = freqs[keep], mag[keep]
    if freqs.size == 0:
        return _empty_figure("Waiting for signal\u2026", height)

    fs_d = downsample(freqs, max_points)
    mag_d = downsample(mag, max_points)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=fs_d,
            y=mag_d,
            mode="lines",
            line={"color": COLORS.info, "width": 1.4},
            fill="tozeroy",
            fillcolor=_rgba(COLORS.info, 0.20),
            hovertemplate="%{x:.0f} Hz<br>%{y:.4f} g<extra></extra>",
            name="|FFT|",
        )
    )

    if tpf_hz and tpf_hz > 0 and (freq_max is None or tpf_hz <= freq_max):
        fig.add_vline(
            x=tpf_hz,
            line={"color": COLORS.warning, "width": 1.8, "dash": "dash"},
        )
        fig.add_annotation(
            x=tpf_hz,
            y=1.0,
            yref="paper",
            text=f"TPF {tpf_hz:.0f} Hz",
            showarrow=False,
            yshift=-4,
            font={"color": COLORS.warning, "size": 12},
            bgcolor=_rgba(COLORS.surface_2, 0.92),
            bordercolor=COLORS.warning,
            borderwidth=1,
            borderpad=4,
        )

    fig.update_layout(**get_dark_layout(height=height, title=title))
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Amplitude (g)")
    # Stable uirevision preserves zoom/pan and enables in-place updates.
    if uirevision:
        fig.update_layout(uirevision=uirevision)
    return fig


# --------------------------------------------------------------------------- #
# STFT heatmap
# --------------------------------------------------------------------------- #
def build_stft_heatmap(
    power: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    *,
    title: str = "STFT Spectrogram",
    freq_max: float | None = 8000.0,
    colorscale: str = "Inferno",
    height: int = 300,
    max_freq_bins: int = 260,
    max_time_bins: int = 240,
    uirevision: str | None = "argus_live_signals",
) -> go.Figure:
    """Time-frequency heatmap of STFT (log-)power (``power`` is ``(n_freq, n_time)``)."""
    p = np.asarray(power, dtype=np.float32)
    f = np.asarray(freqs, dtype=np.float64).ravel()
    tm = np.asarray(times, dtype=np.float64).ravel()
    if p.ndim != 2 or p.size == 0:
        return _empty_figure("Waiting for signal\u2026", height)

    if freq_max is not None:
        keep = f <= freq_max
        if np.count_nonzero(keep) >= 2:
            p, f = p[keep, :], f[keep]

    # Thin dense axes so the browser stays snappy.
    if f.size > max_freq_bins:
        fi = np.linspace(0, f.size - 1, max_freq_bins).astype(np.int64)
        p, f = p[fi, :], f[fi]
    if tm.size > max_time_bins:
        ti = np.linspace(0, tm.size - 1, max_time_bins).astype(np.int64)
        p, tm = p[:, ti], tm[ti]

    fig = go.Figure(
        go.Heatmap(
            z=p,
            x=tm,
            y=f,
            colorscale=colorscale,
            colorbar={
                "title": {"text": "dB", "font": {"color": COLORS.text, "size": 11}},
                "tickfont": {"color": COLORS.text_muted, "size": 10},
                "outlinewidth": 0,
                "thickness": 14,
                "bgcolor": _rgba(COLORS.surface, 0.6),
            },
            hovertemplate="t=%{x:.3f}s<br>f=%{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>",
        )
    )
    fig.update_layout(**get_dark_layout(height=height, title=title))
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Frequency (Hz)")
    # Stable uirevision preserves zoom/pan and enables in-place updates.
    if uirevision:
        fig.update_layout(uirevision=uirevision)
    return fig


# --------------------------------------------------------------------------- #
# Gauge
# --------------------------------------------------------------------------- #
def _gauge_indicator(
    value: float,
    title: str,
    *,
    min_val: float = 0.0,
    max_val: float = 1.0,
    thresholds: Sequence[float] | None = None,
    colors: Sequence[str] | None = None,
    number_suffix: str = "",
    number_format: str = ".2f",
    reference: float | None = None,
    domain: dict[str, Any] | None = None,
) -> go.Indicator:
    """Build a single styled ``go.Indicator`` gauge trace (no layout).

    Split out from :func:`build_gauge_figure` so several gauges can share one
    figure (see :func:`build_gauge_row_figure`) - fewer Streamlit/Plotly
    components means far less per-tick DOM churn on the live monitor.
    """
    v = float(np.clip(value, min_val, max_val)) if np.isfinite(value) else min_val
    if thresholds is None:
        thresholds = (min_val + 0.60 * (max_val - min_val), min_val + 0.85 * (max_val - min_val))
    t1, t2 = thresholds
    step_colors = colors or (COLORS.accent, COLORS.warning, COLORS.critical)

    # Bar color reflects which band the value falls in.
    bar_color = step_colors[0]
    if v >= t2:
        bar_color = step_colors[2]
    elif v >= t1:
        bar_color = step_colors[1]

    mode = "gauge+number+delta" if reference is not None else "gauge+number"
    ind: dict[str, Any] = {
        "mode": mode,
        "value": v,
        "title": {"text": title, "font": {"size": 13, "color": COLORS.text}},
        "number": {
            "suffix": number_suffix,
            "font": {"color": COLORS.text, "size": 24},
            "valueformat": number_format,
        },
        "gauge": {
            "axis": {
                "range": [min_val, max_val],
                "tickcolor": COLORS.border,
                "tickfont": {"color": COLORS.text_muted, "size": 10},
                "tickwidth": 1,
            },
            "bar": {"color": bar_color, "thickness": 0.32},
            "bgcolor": _rgba(COLORS.surface_2, 0.75),
            "borderwidth": 0,
            "steps": [
                {"range": [min_val, t1], "color": _rgba(step_colors[0], 0.38)},
                {"range": [t1, t2], "color": _rgba(step_colors[1], 0.38)},
                {"range": [t2, max_val], "color": _rgba(step_colors[2], 0.38)},
            ],
            "threshold": {
                "line": {"color": COLORS.text, "width": 2},
                "thickness": 0.75,
                "value": v,
            },
        },
    }
    if reference is not None:
        ind["delta"] = {
            "reference": reference,
            "increasing": {"color": COLORS.critical},
            "decreasing": {"color": COLORS.accent},
            "font": {"size": 12},
        }
    if domain is not None:
        ind["domain"] = domain
    return go.Indicator(**ind)


def build_gauge_figure(
    value: float,
    title: str,
    *,
    min_val: float = 0.0,
    max_val: float = 1.0,
    thresholds: Sequence[float] | None = None,
    colors: Sequence[str] | None = None,
    number_suffix: str = "",
    number_format: str = ".2f",
    reference: float | None = None,
    height: int = 200,
    uirevision: str | None = None,
) -> go.Figure:
    """A ``go.Indicator`` gauge with healthy/warning/critical colored steps.

    Parameters
    ----------
    thresholds:
        Two ascending values ``(t1, t2)`` splitting the range into
        healthy ``[min, t1)`` / warning ``[t1, t2)`` / critical ``[t2, max]``.
        Defaults to 60% / 85% of the range.
    colors:
        Three colors for the three steps (defaults to teal / amber / red).
    reference:
        Optional delta reference (renders a small delta under the number).
    uirevision:
        When set (stable across reruns), Plotly animates the needle/number in
        place on data updates instead of tearing the SVG down and rebuilding
        it - this is what makes live gauges update smoothly rather than flicker.
    """
    fig = go.Figure(
        _gauge_indicator(
            value, title, min_val=min_val, max_val=max_val, thresholds=thresholds,
            colors=colors, number_suffix=number_suffix, number_format=number_format,
            reference=reference,
        )
    )
    fig.update_layout(
        **get_dark_layout(height=height, margin={"l": 24, "r": 24, "t": 44, "b": 8})
    )
    if uirevision:
        fig.update_layout(uirevision=uirevision)
    return fig


def build_gauge_row_figure(
    gauges: Sequence[Mapping[str, Any]],
    *,
    height: int = 210,
    uirevision: str | None = "argus_gauges",
) -> go.Figure:
    """Render several gauges as ONE figure (a 1xN indicator grid).

    Collapsing the live KPI gauges into a single Plotly component (instead of
    one ``st.plotly_chart`` per gauge) removes most of the per-tick component
    churn on the Live Monitor, and the shared ``uirevision`` lets every needle
    animate in place - the single biggest smoothness win for the live view.

    Each item in ``gauges`` is a mapping accepted by :func:`_gauge_indicator`
    (``value``, ``title``, and optional ``thresholds`` / ``colors`` / ...).
    """
    n = max(1, len(gauges))
    fig = go.Figure()
    for i, g in enumerate(gauges):
        fig.add_trace(
            _gauge_indicator(
                float(g.get("value", 0.0)),
                str(g.get("title", "")),
                min_val=float(g.get("min_val", 0.0)),
                max_val=float(g.get("max_val", 1.0)),
                thresholds=g.get("thresholds"),
                colors=g.get("colors"),
                number_suffix=str(g.get("number_suffix", "")),
                number_format=str(g.get("number_format", ".2f")),
                reference=g.get("reference"),
                domain={"row": 0, "column": i},
            )
        )
    fig.update_layout(**get_dark_layout(height=height, margin={"l": 16, "r": 16, "t": 46, "b": 8}))
    fig.update_layout(grid={"rows": 1, "columns": n, "pattern": "independent"})
    if uirevision:
        fig.update_layout(uirevision=uirevision)
    return fig


# --------------------------------------------------------------------------- #
# Health probability bar
# --------------------------------------------------------------------------- #
def build_health_prob_bar(
    health_probs: Mapping[str, float],
    *,
    title: str = "Health-State Confidence",
    height: int = 200,
) -> go.Figure:
    """Horizontal bar of per-class health probabilities, colored by class."""
    order = ["healthy", "monitor", "warning", "critical"]
    labels = [c for c in order if c in health_probs] or list(health_probs)
    values = [float(health_probs.get(c, 0.0)) for c in labels]
    bar_colors = [COLORS.health_color(c) for c in labels]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=[c.capitalize() for c in labels],
            orientation="h",
            marker={"color": bar_colors, "line": {"width": 0}},
            text=[f"{v*100:.0f}%" for v in values],
            textposition="outside",
            textfont={"color": COLORS.text, "size": 11},
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    fig.update_layout(**get_dark_layout(height=height, title=title))
    fig.update_xaxes(range=[0, 1.08], tickformat=".0%", title_text="")
    fig.update_yaxes(title_text="")
    return fig


# --------------------------------------------------------------------------- #
# Trends
# --------------------------------------------------------------------------- #
def build_trend_figure(
    df: "Any",
    y_cols: Sequence[str],
    *,
    x_col: str | None = None,
    title: str = "Trend",
    height: int = 320,
    color_map: Mapping[str, str] | None = None,
    range_slider: bool = False,
    y_title: str = "",
    labels: Mapping[str, str] | None = None,
    uirevision: str | None = "argus_live_signals",
) -> go.Figure:
    """Multi-series line chart over a (time or index) axis for historical trends."""
    import pandas as pd  # local import; only needed here

    if df is None or len(df) == 0:
        return _empty_figure("No records to plot", height)
    labels = labels or {}
    x = df[x_col] if x_col and x_col in df.columns else pd.RangeIndex(len(df))

    fig = go.Figure()
    for i, col in enumerate(y_cols):
        if col not in df.columns:
            continue
        color = (color_map or {}).get(col, COLORS.series[i % len(COLORS.series)])
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=df[col],
                mode="lines",
                name=labels.get(col, col),
                line={"color": color, "width": 2.0},
                hovertemplate=f"{labels.get(col, col)}: %{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(**get_dark_layout(height=height, title=title, showlegend=True))
    fig.update_yaxes(title_text=y_title)
    if range_slider:
        fig.update_xaxes(rangeslider={"visible": True, "thickness": 0.06})
    # Stable uirevision preserves zoom/pan and enables in-place updates.
    if uirevision:
        fig.update_layout(uirevision=uirevision)
    return fig


def build_scatter_timeline(
    x: Sequence[Any],
    y: Sequence[float],
    colors: Sequence[str],
    *,
    text: Sequence[str] | None = None,
    title: str = "Anomaly Timeline",
    height: int = 240,
    y_title: str = "",
) -> go.Figure:
    """Scatter timeline (e.g. anomaly events colored by health state)."""
    fig = go.Figure(
        go.Scattergl(
            x=list(x),
            y=list(y),
            mode="markers",
            marker={
                "color": list(colors),
                "size": 9,
                "line": {"color": COLORS.surface, "width": 1.5},
                "opacity": 0.9,
            },
            text=list(text) if text is not None else None,
            hovertemplate="%{x}<br>%{y:.3f}<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(**get_dark_layout(height=height, title=title))
    fig.update_yaxes(title_text=y_title)
    return fig


# --------------------------------------------------------------------------- #
# Distributions / comparisons
# --------------------------------------------------------------------------- #
def build_histogram(
    values: Sequence[float],
    *,
    title: str = "Distribution",
    color: str | None = None,
    x_title: str = "",
    height: int = 260,
    nbins: int = 24,
) -> go.Figure:
    """Histogram of a batch of scalar values."""
    v = np.asarray(list(values), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return _empty_figure("No data", height)
    fig = go.Figure(
        go.Histogram(
            x=v,
            nbinsx=nbins,
            marker={"color": color or COLORS.accent_soft, "line": {"color": COLORS.surface, "width": 1}},
            opacity=0.85,
            hovertemplate=f"{x_title}: %{{x}}<br>count: %{{y}}<extra></extra>",
        )
    )
    fig.add_vline(
        x=float(np.mean(v)),
        line={"color": COLORS.warning, "width": 1.5, "dash": "dash"},
    )
    fig.update_layout(**get_dark_layout(height=height, title=title))
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text="Count")
    return fig


def build_grouped_bar(
    categories: Sequence[str],
    series: Mapping[str, Sequence[float]],
    *,
    title: str = "Comparison",
    colors: Sequence[str] | None = None,
    height: int = 300,
    y_title: str = "",
    text_format: str = ".2f",
) -> go.Figure:
    """Grouped bar chart (e.g. before/after or per-model comparison)."""
    fig = go.Figure()
    palette = list(colors) if colors else list(COLORS.series)
    for i, (name, vals) in enumerate(series.items()):
        fig.add_trace(
            go.Bar(
                name=name,
                x=list(categories),
                y=list(vals),
                marker={"color": palette[i % len(palette)]},
                text=[format(float(v), text_format) for v in vals],
                textposition="outside",
                textfont={"color": COLORS.text, "size": 11},
                hovertemplate="%{x}<br>" + name + ": %{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        **get_dark_layout(height=height, title=title, showlegend=len(series) > 1),
        barmode="group",
    )
    fig.update_yaxes(title_text=y_title)
    return fig


def build_pie(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str = "Distribution",
    color_map: Mapping[str, str] | None = None,
    height: int = 260,
) -> go.Figure:
    """Donut chart (e.g. health-state distribution)."""
    labels = list(labels)
    marker_colors = [
        (color_map or {}).get(lbl, COLORS.series[i % len(COLORS.series)])
        for i, lbl in enumerate(labels)
    ]
    fig = go.Figure(
        go.Pie(
            labels=[str(x_).capitalize() for x_ in labels],
            values=list(values),
            hole=0.55,
            marker={"colors": marker_colors, "line": {"color": COLORS.surface, "width": 2}},
            textfont={"color": COLORS.text, "size": 12},
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(**get_dark_layout(height=height, title=title, showlegend=True))
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    """Local ``#rrggbb`` -> ``rgba(...)`` helper (mirrors theme._rgba)."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
