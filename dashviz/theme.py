"""Dark industrial theme for the Argus Panoptes dashboard.

Central place for the color palette, global CSS, a shared Plotly dark layout
template, and small reusable "styled component" render helpers (KPI cards,
alert banners, recommendation cards, status pills). Keeping all styling here
means every tab / figure looks consistent and the palette can be retuned in
exactly one spot.

Palette rationale (industrial monitoring convention)
----------------------------------------------------
* deep slate/charcoal background, high-contrast light-gray text,
* **teal/cyan** for healthy / good states,
* **amber** for degraded / warning,
* **red** for critical / alerts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Palette:
    """Immutable color palette shared across CSS, Plotly figures, and gauges.

    Contrast targets (WCAG-ish on ``#0f172a`` / ``#1e293b`` surfaces):
    * primary text ≥ ~12:1, secondary labels ≥ ~7:1, captions ≥ ~4.5:1,
    * chart gridlines and borders visibly distinct from plot background,
    * semantic accents (teal / amber / red) bright enough for lines, text, and
      low-alpha tinted panels without washing out.
    """

    bg: str = "#0f172a"           # deep slate - app background
    bg_alt: str = "#111827"       # charcoal - secondary surfaces
    surface: str = "#1e293b"      # card surface
    surface_2: str = "#334155"    # elevated card / hover (was too close to surface)
    border: str = "#475569"       # visible borders / axis lines (slate-600)
    grid: str = "#3d4f66"         # plot gridlines — readable on dark bg

    text: str = "#f1f5f9"         # primary text (slate-100)
    text_muted: str = "#cbd5e1"   # secondary labels (slate-300)
    text_faint: str = "#94a3b8"   # captions (slate-400; was slate-500, too dim)

    accent: str = "#2dd4bf"       # teal-400 — healthy / primary accent
    accent_soft: str = "#5eead4"  # teal-300 — highlights on dark
    warning: str = "#fbbf24"      # amber-400 — degraded / warning
    critical: str = "#f87171"     # red-400 — critical / alert
    info: str = "#7dd3fc"         # sky-300 — informational / FFT lines
    purple: str = "#c4b5fd"        # violet-300 — extra series
    neutral: str = "#a8b4c8"      # neutral bars / baselines (not as dim as captions)

    #: Health-state -> color (matches HEALTH_CLASS_NAMES from the perceptor).
    health: dict[str, str] = field(
        default_factory=lambda: {
            "healthy": "#2dd4bf",
            "monitor": "#7dd3fc",
            "warning": "#fbbf24",
            "critical": "#f87171",
        }
    )

    #: Ordered categorical series colors for multi-line / multi-model plots.
    series: tuple[str, ...] = (
        "#5eead4",  # teal-300
        "#fbbf24",  # amber-400
        "#c4b5fd",  # violet-300
        "#7dd3fc",  # sky-300
        "#f9a8d4",  # pink-300
        "#86efac",  # green-300
    )

    def health_color(self, state: str | None) -> str:
        """Return the color for a health state (falls back to muted gray)."""
        return self.health.get((state or "").lower(), self.text_muted)


#: The single shared palette instance imported everywhere.
COLORS = Palette()


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to an ``rgba(r,g,b,alpha)`` CSS string."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def get_dark_layout(
    *,
    height: int | None = None,
    title: str | None = None,
    margin: dict[str, int] | None = None,
    showlegend: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    """Return a Plotly ``layout`` dict for the dark industrial theme.

    Every figure builder in :mod:`dashviz.plots` calls this so fonts, colors,
    gridlines, margins, and hover styling stay identical across the dashboard.

    Parameters
    ----------
    height:
        Optional fixed pixel height.
    title:
        Optional title text (rendered in muted, small caps-ish styling).
    margin:
        Override the default tight margins (``dict(l, r, t, b)``).
    showlegend:
        Whether to show the legend (default off for single-series plots).
    **overrides:
        Extra top-level layout keys merged last (win over defaults).
    """
    layout: dict[str, Any] = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": _rgba(COLORS.surface, 0.35),
        "font": {
            "family": "Inter, 'Segoe UI', system-ui, sans-serif",
            "color": COLORS.text,
            "size": 13,
        },
        "margin": margin or {"l": 56, "r": 20, "t": 40 if title else 16, "b": 44},
        "showlegend": showlegend,
        "legend": {
            "bgcolor": _rgba(COLORS.surface_2, 0.92),
            "bordercolor": COLORS.border,
            "borderwidth": 1,
            "font": {"size": 11, "color": COLORS.text},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
        "hoverlabel": {
            "bgcolor": COLORS.surface_2,
            "bordercolor": COLORS.accent_soft,
            "font": {"color": COLORS.text, "size": 12},
        },
        "xaxis": _axis_style(),
        "yaxis": _axis_style(),
        "colorway": list(COLORS.series),
    }
    if height is not None:
        layout["height"] = height
    if title is not None:
        layout["title"] = {
            "text": title,
            "font": {"size": 14, "color": COLORS.text},
            "x": 0.01,
            "xanchor": "left",
        }
    layout.update(overrides)
    return layout


def _axis_style() -> dict[str, Any]:
    """Consistent axis styling (visible gridlines, readable tick labels)."""
    return {
        "gridcolor": COLORS.grid,
        "gridwidth": 1,
        "zerolinecolor": COLORS.border,
        "zerolinewidth": 1,
        "linecolor": COLORS.border,
        "linewidth": 1,
        "tickfont": {"color": COLORS.text_muted, "size": 11},
        "title": {"font": {"color": COLORS.text, "size": 12}},
        "showgrid": True,
    }


#: Plotly ``config`` passed to analysis charts (Lab / History / System) for a
#: consistent, responsive, low-chrome interactive experience.
PLOTLY_CONFIG: dict[str, Any] = {
    "displayModeBar": True,
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
    "scrollZoom": False,
}

#: Config for the self-refreshing **Live Monitor** charts. The mode bar is
#: hidden (drag-to-zoom / double-click-reset still work) so there is no hover
#: toolbar to repaint every tick, and ``doubleClick`` resets against the stable
#: ``uirevision``. Fewer interactive widgets => noticeably smoother refresh.
PLOTLY_CONFIG_LIVE: dict[str, Any] = {
    "displayModeBar": False,
    "responsive": True,
    "displaylogo": False,
    "scrollZoom": False,
    "doubleClick": "reset",
    "showTips": False,
}


# --------------------------------------------------------------------------- #
# Global CSS
# --------------------------------------------------------------------------- #
def build_css() -> str:
    """Return the global ``<style>`` block for the dark industrial theme."""
    c = COLORS
    return f"""
<style>
    /* ---- Base surfaces ---- */
    .stApp {{
        background:
            radial-gradient(1200px 600px at 15% -10%, #16233a 0%, rgba(15,23,42,0) 55%),
            radial-gradient(1000px 500px at 100% 0%, #12202f 0%, rgba(15,23,42,0) 50%),
            {c.bg};
        color: {c.text};
    }}
    /* Top header/toolbar: kill the opaque white bar covering the page top */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        box-shadow: none !important;
    }}
    header[data-testid="stHeader"]::before {{ background: transparent !important; }}
    [data-testid="stToolbar"] {{ right: 0.5rem; }}
    [data-testid="stToolbar"] button,
    [data-testid="stToolbar"] svg,
    [data-testid="stMainMenu"] svg {{ color: {c.text_muted} !important; fill: {c.text_muted} !important; }}
    [data-testid="stDecoration"] {{ background: transparent !important; }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {c.bg_alt} 0%, {c.bg} 100%);
        border-right: 1px solid {c.border};
    }}
    section[data-testid="stSidebar"] * {{ color: {c.text}; }}

    /* ---- Typography ---- */
    h1, h2, h3, h4 {{
        color: {c.text} !important;
        font-family: Inter, 'Segoe UI', system-ui, sans-serif;
        letter-spacing: -0.01em;
    }}
    .block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1600px; }}

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid {c.border};
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        color: {c.text_muted};
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.92rem;
    }}
    .stTabs [aria-selected="true"] {{
        background: {c.surface};
        color: {c.accent_soft} !important;
        border-bottom: 2px solid {c.accent_soft};
    }}

    /* ---- Streamlit widgets (labels often too dim by default) ---- */
    label[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] span,
    .stSelectbox label p, .stSlider label p, .stRadio label p,
    .stMultiSelect label p, .stTextInput label p, .stNumberInput label p {{
        color: {c.text_muted} !important;
    }}
    .stCaption, [data-testid="stCaptionContainer"] p, .stMarkdown p {{
        color: {c.text_faint};
    }}
    div[data-testid="stExpander"] summary span {{
        color: {c.text} !important;
        font-weight: 600;
    }}
    /* Text / number inputs, textareas, selects: dark surface + light text so
       typed text is never white-on-white. */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] {{
        background-color: {c.surface} !important;
        border-color: {c.border} !important;
        color: {c.text} !important;
    }}
    .stTextInput input, .stNumberInput input, .stTextArea textarea,
    div[data-baseweb="input"] input, div[data-baseweb="select"] input {{
        color: {c.text} !important;
        background-color: transparent !important;
        -webkit-text-fill-color: {c.text} !important;
    }}
    .stTextInput input::placeholder, .stNumberInput input::placeholder,
    .stTextArea textarea::placeholder {{
        color: {c.text_faint} !important;
        -webkit-text-fill-color: {c.text_faint} !important;
    }}
    /* Number-input +/- steppers */
    .stNumberInput button {{
        background-color: {c.surface_2} !important;
        color: {c.text} !important;
        border-color: {c.border} !important;
    }}
    /* Dropdown popover options */
    div[data-baseweb="popover"] {{ background-color: {c.surface} !important; }}
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] div[role="option"] {{
        color: {c.text} !important;
        background-color: {c.surface} !important;
    }}
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] div[role="option"]:hover {{
        background-color: {c.surface_2} !important;
    }}
    /* Multiselect selected chips/tags (were white with pale text) */
    span[data-baseweb="tag"] {{
        background-color: {c.accent} !important;
        color: #04121a !important;
    }}
    span[data-baseweb="tag"] span, span[data-baseweb="tag"] svg {{
        color: #04121a !important;
        fill: #04121a !important;
    }}
    /* Radio / checkbox option labels */
    .stRadio div[role="radiogroup"] label p,
    .stCheckbox label p {{ color: {c.text} !important; }}
    .stProgress > div > div {{
        background-color: {c.accent_soft} !important;
    }}
    .stProgress > div {{
        background-color: {c.surface_2} !important;
    }}
    /* Plotly mode bar icons */
    .js-plotly-plot .plotly .modebar-btn path {{
        fill: {c.text_muted} !important;
    }}
    .js-plotly-plot .plotly .modebar-btn:hover path {{
        fill: {c.accent_soft} !important;
    }}

    /* ---- Native metric widgets ---- */
    div[data-testid="stMetric"] {{
        background: linear-gradient(160deg, {c.surface} 0%, {c.bg_alt} 100%);
        border: 1px solid {c.border};
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.35);
    }}
    div[data-testid="stMetricLabel"] p {{ color: {c.text_muted} !important; font-size: 0.78rem; }}
    div[data-testid="stMetricValue"] {{ color: {c.text} !important; }}

    /* ---- Buttons (regular + form-submit + download all styled) ---- */
    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button,
    .stLinkButton > a {{
        border-radius: 10px;
        border: 1px solid {c.border};
        background: {c.surface};
        color: {c.text} !important;
        font-weight: 600;
        transition: all 0.12s ease-in-out;
    }}
    .stButton > button p,
    .stFormSubmitButton > button p,
    .stDownloadButton > button p {{ color: {c.text} !important; }}
    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover,
    .stLinkButton > a:hover {{
        border-color: {c.accent};
        color: {c.accent_soft} !important;
        box-shadow: 0 0 0 1px {c.accent} inset;
    }}
    .stButton > button:hover p,
    .stFormSubmitButton > button:hover p,
    .stDownloadButton > button:hover p {{ color: {c.accent_soft} !important; }}
    /* Primary buttons: teal fill with dark text */
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {c.accent} 0%, #0d9488 100%);
        border: none;
        color: #04121a !important;
    }}
    .stButton > button[kind="primary"] p,
    .stFormSubmitButton > button[kind="primary"] p {{ color: #04121a !important; }}
    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button[kind="primary"]:hover {{
        box-shadow: 0 0 18px rgba(45,212,191,0.45);
        color: #04121a !important;
    }}

    /* ---- Cards & custom components ---- */
    .argus-card {{
        background: linear-gradient(160deg, {c.surface} 0%, {c.bg_alt} 100%);
        border: 1px solid {c.border};
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        height: 100%;
    }}
    .argus-kpi {{
        background: linear-gradient(160deg, {c.surface} 0%, {c.bg_alt} 100%);
        border: 1px solid {c.border};
        border-left: 4px solid {c.accent};
        border-radius: 12px;
        padding: 12px 14px 10px 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }}
    .argus-kpi .kpi-label {{
        color: {c.text_muted};
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        display: flex; align-items: center; gap: 6px;
    }}
    .argus-kpi .kpi-value {{
        color: {c.text};
        font-size: 1.7rem;
        font-weight: 700;
        line-height: 1.15;
        font-variant-numeric: tabular-nums;
    }}
    .argus-kpi .kpi-sub {{ color: {c.text_muted}; font-size: 0.74rem; }}

    /* ---- Status pill ---- */
    .argus-pill {{
        display: inline-flex; align-items: center; gap: 7px;
        padding: 4px 12px; border-radius: 999px;
        font-size: 0.8rem; font-weight: 700; letter-spacing: 0.02em;
    }}
    .argus-dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}
    /* Static glow (not a keyframe animation): the live status bar re-renders
       every fragment tick, and a restarting CSS animation reads as flicker.
       A steady halo communicates "live" without any per-tick animation reset. */
    .argus-dot.live {{ box-shadow: 0 0 0 3px rgba(45,212,191,0.28), 0 0 8px 2px rgba(45,212,191,0.45); }}

    /* ---- Alert banner ---- */
    .argus-alert {{
        border-radius: 12px; padding: 12px 16px; margin: 4px 0 6px 0;
        display: flex; align-items: center; gap: 12px;
        font-size: 0.92rem; font-weight: 600; border: 1px solid transparent;
    }}
    .argus-alert .a-icon {{ font-size: 1.25rem; }}
    .argus-alert .a-title {{ font-weight: 800; letter-spacing: 0.02em; }}
    .argus-alert .a-msg {{ font-weight: 500; color: {c.text}; opacity: 0.92; }}

    /* ---- Recommendation card ---- */
    .argus-rec {{
        background: linear-gradient(160deg, {c.surface} 0%, {c.bg_alt} 100%);
        border: 1px solid {c.border};
        border-radius: 14px; padding: 16px 18px;
    }}
    .argus-rec .rec-action {{ font-size: 1.35rem; font-weight: 800; letter-spacing: -0.01em; }}
    .argus-rec .rec-note {{ color: {c.text_muted}; font-size: 0.86rem; margin-top: 4px; }}
    .argus-rec .rec-flag {{
        margin-top: 12px; padding: 8px 12px; border-radius: 9px;
        font-weight: 700; font-size: 0.86rem;
    }}

    .argus-status-bar {{
        display: flex; flex-wrap: wrap; align-items: center; gap: 10px 22px;
        background: {c.bg_alt}; border: 1px solid {c.border};
        border-radius: 12px; padding: 10px 16px; margin-bottom: 12px;
    }}
    .argus-status-bar .sb-item {{ display: flex; align-items: center; gap: 7px; font-size: 0.84rem; }}
    .argus-status-bar .sb-label {{ color: {c.text_muted}; text-transform: uppercase; font-size: 0.68rem; letter-spacing: 0.06em; }}
    .argus-status-bar .sb-value {{ color: {c.text}; font-weight: 700; font-variant-numeric: tabular-nums; }}

    .argus-caption {{ color: {c.text_muted}; font-size: 0.78rem; }}
    hr {{ border-color: {c.border}; }}
    div[data-testid="stExpander"] {{ border: 1px solid {c.border}; border-radius: 12px; }}
    div[data-testid="stExpander"] details {{ background: {c.surface}; border-radius: 12px; }}
    div[data-testid="stDataFrame"] {{ border: 1px solid {c.border}; border-radius: 10px; }}
    div[data-testid="stForm"] {{
        border: 1px solid {c.border}; border-radius: 14px; background: {_rgba(c.surface, 0.4)};
    }}
    /* Code / JSON blocks: dark surface, light text */
    .stCode, .stJson, pre, code {{
        background-color: {c.bg_alt} !important;
        color: {c.text} !important;
    }}
</style>
"""


# --------------------------------------------------------------------------- #
# Styled component render helpers (return HTML strings for st.markdown)
# --------------------------------------------------------------------------- #
def kpi_card_html(
    label: str,
    value: str,
    *,
    sub: str = "",
    accent: str | None = None,
    icon: str = "",
) -> str:
    """HTML for a compact KPI card with a colored left accent bar."""
    accent = accent or COLORS.accent
    icon_html = f"<span>{icon}</span>" if icon else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="argus-kpi" style="border-left-color:{accent}">'
        f'<div class="kpi-label">{icon_html}{label}</div>'
        f'<div class="kpi-value" style="color:{accent}">{value}</div>'
        f"{sub_html}</div>"
    )


def status_pill_html(text: str, color: str, *, live: bool = False) -> str:
    """HTML for a colored status pill (optionally with a pulsing 'live' dot)."""
    dot_cls = "argus-dot live" if live else "argus-dot"
    bg = _rgba(color, 0.28)
    border = _rgba(color, 0.55)
    return (
        f'<span class="argus-pill" style="background:{bg};color:{color};'
        f'border:1px solid {border};">'
        f'<span class="{dot_cls}" style="background:{color}"></span>{text}</span>'
    )


def alert_banner_html(level: str, title: str, message: str) -> str:
    """HTML for an alert banner. ``level`` in {healthy, info, warning, critical}."""
    color_map = {
        "healthy": COLORS.accent,
        "ok": COLORS.accent,
        "info": COLORS.info,
        "warning": COLORS.warning,
        "critical": COLORS.critical,
    }
    icon_map = {
        "healthy": "\u2713",
        "ok": "\u2713",
        "info": "\u2139",
        "warning": "\u26a0",
        "critical": "\u26d4",
    }
    color = color_map.get(level, COLORS.info)
    icon = icon_map.get(level, "\u2139")
    return (
        f'<div class="argus-alert" style="background:{_rgba(color, 0.22)};'
        f'border-color:{_rgba(color, 0.65)};">'
        f'<span class="a-icon" style="color:{color}">{icon}</span>'
        f'<span><span class="a-title" style="color:{color}">{title}</span>'
        f'&nbsp;&nbsp;<span class="a-msg">{message}</span></span></div>'
    )


def recommendation_card_html(
    action: str,
    note: str,
    *,
    color: str,
    blade_change: bool,
) -> str:
    """HTML for the large recommendation card with a blade-change flag."""
    if blade_change:
        flag = (
            f'<div class="rec-flag" style="background:{_rgba(COLORS.critical, 0.28)};'
            f'color:{COLORS.critical};border:1px solid {_rgba(COLORS.critical, 0.65)};">'
            f"\u26a0 BLADE CHANGE SUGGESTED</div>"
        )
    else:
        flag = (
            f'<div class="rec-flag" style="background:{_rgba(COLORS.accent, 0.24)};'
            f'color:{COLORS.accent_soft};border:1px solid {_rgba(COLORS.accent, 0.55)};">'
            f"\u2713 Blade OK \u2014 no change needed</div>"
        )
    pretty = action.replace("_", " ").title()
    return (
        f'<div class="argus-rec">'
        f'<div class="argus-caption">Recommended action</div>'
        f'<div class="rec-action" style="color:{color}">{pretty}</div>'
        f'<div class="rec-note">{note}</div>'
        f"{flag}</div>"
    )


# _rgba is defined near COLORS (used by layout + HTML helpers above).
