"""
Brent Crude Price Forecaster
Monte Carlo simulation with scenario weighting.
Streamlit dashboard — deployable on Streamlit Community Cloud (free).
"""

import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex colour string to an rgba() string for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

from data_fetcher import (
    fetch_brent,
    calibrate_gbm,
    calibrate_jumps,
    calibrate_garch,
    calibrate_jumps_mle,
    calibrate_ou,
    get_latest_price,
    get_price_history_since,
)
from mc_engine import (
    SCENARIOS,
    HORIZONS,
    blended_terminal_prices,
    compute_stats,
    build_fan_data,
    run_scenario_gbm,
    run_scenario,
    run_scenario_garch,
    run_scenario_ou,
)

# ---------------------------------------------------------------------------
# Page config and theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Brent Crude Forecaster",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar and dark mode toggle
# ---------------------------------------------------------------------------

with st.sidebar:
    dark_mode = st.toggle("🌙 Dark mode", value=False, key="dark_mode")
    st.markdown("## 🛢️ Brent Crude Forecaster")
    st.markdown(
        "Monte Carlo simulation using Merton jump-diffusion. "
        "Calibrated on live Brent crude data."
    )
    st.divider()

    st.markdown("### Forecast horizon")
    horizon_label = st.radio(
        "Select horizon",
        list(HORIZONS.keys()),
        index=1,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Scenario weights")
    st.caption("Drag to set the weight of each supply scenario.")

    scenario_weights = {}
    raw_weights = {}
    for name, cfg in SCENARIOS.items():
        color_dot = f"<span style='color:{cfg['color']};'>●</span>"
        st.markdown(f"{color_dot} **{name}**", unsafe_allow_html=True)
        raw_weights[name] = st.slider(
            name,
            min_value=0,
            max_value=100,
            value=int(cfg["default_weight"] * 100),
            step=5,
            label_visibility="collapsed",
            key=f"w_{name}",
        )

    total_raw = sum(raw_weights.values()) or 1
    for name in raw_weights:
        scenario_weights[name] = raw_weights[name] / total_raw

    st.divider()
    st.markdown("### Simulation settings")
    n_paths = st.select_slider(
        "Number of paths",
        options=[5_000, 10_000, 20_000, 50_000],
        value=20_000,
    )
    jump_threshold = st.slider(
        "Jump detection threshold (|log-return|)",
        min_value=0.02,
        max_value=0.10,
        value=0.04,
        step=0.005,
        format="%.3f",
    )

    st.divider()
    run_button = st.button("Run simulation", type="primary", width='stretch')


# ---------------------------------------------------------------------------
# Theme colors and CSS
# ---------------------------------------------------------------------------

if dark_mode:
    BG       = "#0d1117"
    SIDEBAR  = "#161b22"
    CARD_BG  = "#21262d"
    TEXT     = "#e6edf3"
    BORDER   = "#30363d"
    MUTED    = "#8b949e"
    PLOT_BG  = "#0d1117"
    GRID_CLR = "#21262d"
    INPUT_BG = "#161b22"
else:
    BG       = "#f9f7f4"
    SIDEBAR  = "#f0ede8"
    CARD_BG  = "#ffffff"
    TEXT     = "#1a1a1a"
    BORDER   = "#d4cfc8"
    MUTED    = "#6b6560"
    PLOT_BG  = "#f9f7f4"
    GRID_CLR = "#e8e4de"
    INPUT_BG = "#ffffff"

# Custom CSS with full theme coverage
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&display=swap');

/* ---- Base ---- */
html, body, [class*="css"] {{
    font-family: 'Times New Roman', Times, serif;
    color: {TEXT};
}}
.stApp, .main, .block-container {{
    background: {BG} !important;
    color: {TEXT};
}}

/* ---- Header / toolbar strip ---- */
header[data-testid="stHeader"],
header[data-testid="stHeader"] > div,
header[data-testid="stHeader"] > div > div {{
    background: {BG} !important;
    border-bottom: 1px solid {BORDER};
}}
header[data-testid="stHeader"] button {{
    background: transparent !important;
    color: {TEXT} !important;
    border: none !important;
}}
header[data-testid="stHeader"] svg,
header[data-testid="stHeader"] button svg {{
    fill: {TEXT} !important;
    color: {TEXT} !important;
}}
[data-testid="stToolbar"],
[data-testid="stAppDeployButton"],
[data-testid="stDeployButton"],
[data-testid="stToolbarActions"] {{
    background: {BG} !important;
    color: {TEXT} !important;
}}
[data-testid="stToolbarActions"] svg,
[data-testid="stAppDeployButton"] svg,
[data-testid="stDeployButton"] svg {{
    fill: {TEXT} !important;
    color: {TEXT} !important;
}}
#MainMenu, #MainMenu button {{
    background: transparent !important;
    color: {TEXT} !important;
}}
#MainMenu svg, #MainMenu button svg {{
    fill: {TEXT} !important;
    color: {TEXT} !important;
}}
button[kind="header"],
button[data-testid="stBaseButton-header"] {{
    background: transparent !important;
    color: {TEXT} !important;
    border: none !important;
}}
button[kind="header"] svg,
button[data-testid="stBaseButton-header"] svg {{
    fill: {TEXT} !important;
    color: {TEXT} !important;
}}

/* ---- Sidebar collapse / expand button — always visible ---- */
[data-testid="collapsedControl"] {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 9999 !important;
    background: {SIDEBAR} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 0 4px 4px 0 !important;
    box-shadow: 2px 0 6px rgba(0,0,0,0.2) !important;
    min-width: 28px !important;
    min-height: 40px !important;
}}
[data-testid="collapsedControl"] svg {{
    fill: {TEXT} !important;
    color: {TEXT} !important;
}}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {{
    background: {SIDEBAR} !important;
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{
    color: {TEXT} !important;
}}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {{
    color: {TEXT} !important;
}}

/* ---- Tabs ---- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: {BG} !important;
    border-bottom: 1px solid {BORDER};
    overflow-x: auto;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    background: transparent !important;
    color: {MUTED} !important;
    white-space: nowrap;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    color: {TEXT} !important;
    border-bottom: 2px solid #f0b429 !important;
}}
[data-testid="stTabPanel"] {{
    background: {BG} !important;
}}

/* ---- Expanders ---- */
[data-testid="stExpander"] {{
    background: {CARD_BG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px;
}}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary > div,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary > div > p {{
    color: {TEXT} !important;
}}
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:hover span,
[data-testid="stExpander"] summary:hover p {{
    color: #f0b429 !important;
    cursor: pointer;
}}
[data-testid="stExpander"] summary svg {{
    fill: {TEXT} !important;
}}

/* ---- Metric cards (custom HTML) ---- */
.metric-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 10px;
}}
.metric-value {{
    font-family: 'Courier Prime', 'Courier New', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #f0b429;
}}
.metric-label {{
    font-size: 0.75rem;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}}
.metric-delta {{
    font-family: 'Courier Prime', 'Courier New', monospace;
    font-size: 0.85rem;
    margin-top: 4px;
}}

/* ---- Risk cards ---- */
.risk-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.prob-high {{ color: #f85149; }}
.prob-med  {{ color: #f0b429; }}
.prob-low  {{ color: #3fb950; }}

/* ---- Typography ---- */
h1 {{ font-family: 'Courier Prime', 'Courier New', monospace !important;
      letter-spacing: -0.02em; color: {TEXT} !important; }}
h2, h3 {{ font-family: 'Times New Roman', Times, serif !important;
           font-weight: 700; color: {TEXT} !important; }}
h4 {{ color: {TEXT} !important; }}
p, span, label {{ color: {TEXT}; }}
.stMarkdown, .stMarkdown p, .stMarkdown span {{ color: {TEXT}; }}
div[data-testid="stText"] {{ color: {TEXT}; }}
div[data-testid="stCaptionContainer"] p {{ color: {MUTED} !important; }}

/* ---- Native st.metric ---- */
div[data-testid="stMetric"] {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 14px 18px;
}}
div[data-testid="stMetric"] label,
div[data-testid="stMetricLabel"] p {{
    color: {MUTED} !important;
}}
div[data-testid="stMetricValue"] {{
    color: {TEXT} !important;
}}

/* ---- Sliders and inputs ---- */
.stSlider > div > div > div {{ background: #f0b429 !important; }}
.stSelectSlider [data-baseweb="slider"] div {{ background: #f0b429 !important; }}
[data-baseweb="input"] input, [data-baseweb="select"] {{
    background: {INPUT_BG} !important;
    color: {TEXT} !important;
    border-color: {BORDER} !important;
}}

/* ---- Dividers ---- */
hr {{ border-color: {BORDER} !important; opacity: 0.5; }}

/* ---- Dataframes ---- */
/* Only style the outer wrapper — let Streamlit's native renderer handle internal colours */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    overflow-x: auto;
}}

/* ---- Info / alert boxes ---- */
[data-testid="stInfo"] {{
    background: {CARD_BG} !important;
    border-color: {BORDER} !important;
    color: {TEXT} !important;
}}
[data-testid="stInfo"] p {{ color: {TEXT} !important; }}

/* ---- Mobile responsive ---- */
@media (max-width: 768px) {{
    /* Wrap metric columns 2×2 on tablet */
    [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
    }}
    [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"] {{
        flex: 1 1 48% !important;
        min-width: 48% !important;
    }}
    .metric-value {{ font-size: 1.1rem; }}
    .metric-card {{ padding: 10px 12px; }}
    .metric-label {{ font-size: 0.65rem; }}
    .metric-delta {{ font-size: 0.75rem; }}
    .risk-card {{ flex-direction: column; align-items: flex-start; gap: 4px; }}
    h1 {{ font-size: 1.3rem !important; }}
    [data-testid="stTabs"] [data-baseweb="tab"] span {{ font-size: 0.75rem !important; }}
}}
@media (max-width: 480px) {{
    .stApp .block-container {{ padding: 0.5rem 0.5rem 2rem !important; }}
    .metric-card {{ padding: 8px 10px; }}
    .metric-value {{ font-size: 1rem; }}
    /* Single-column stack on phone */
    [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"] {{
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab"] span {{ font-size: 0.65rem !important; }}
    [data-testid="stTabs"] [data-baseweb="tab"] {{ padding: 8px 6px !important; }}
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: Plotly dark layout defaults
# ---------------------------------------------------------------------------

_FONT = dict(family="Times New Roman, Times, serif", color=TEXT, size=12)
# Base axis style — no 'title' key so callers can add it without collision
_AXIS = dict(
    gridcolor=GRID_CLR, linecolor=BORDER, showgrid=True,
    tickfont=dict(family="Times New Roman, Times, serif", color=TEXT, size=11),
    zerolinecolor=BORDER,
)

def _ax(title_text: str) -> dict:
    """Return a fully-styled axis dict with the given title."""
    return {
        **_AXIS,
        "title": dict(
            text=title_text,
            font=dict(family="Times New Roman, Times, serif", color=TEXT, size=12),
        ),
    }

_MARGIN_DEFAULT = dict(l=60, r=110, t=60, b=60)
_MARGIN_WIDE_R  = dict(l=60, r=155, t=60, b=60)   # charts with right-side annotations

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=PLOT_BG,
    font=_FONT,
    # xaxis / yaxis / title / margin omitted — each chart passes them explicitly
    legend=dict(
        bgcolor=_hex_to_rgba(CARD_BG, 0.85),
        bordercolor=BORDER,
        borderwidth=1,
        font=dict(family="Times New Roman, Times, serif", color=TEXT, size=11),
    ),
)

AMBER  = "#f0b429"
RED    = "#f85149"
GREEN  = "#3fb950"
PURPLE = "#a371f7"
BLUE   = "#58a6ff"


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    df = fetch_brent(lookback_days=730)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_gbm_params(df_json: str, window: int = 60):
    df = pd.read_json(io.StringIO(df_json))
    df.index = pd.to_datetime(df.index)
    return calibrate_gbm(df, window=window)


@st.cache_data(ttl=3600, show_spinner=False)
def get_jump_params(df_json: str, threshold: float):
    df = pd.read_json(io.StringIO(df_json))
    df.index = pd.to_datetime(df.index)
    return calibrate_jumps(df, jump_threshold=threshold)


@st.cache_data(ttl=3600, show_spinner=False)
def get_garch_params(df_json: str):
    df = pd.read_json(io.StringIO(df_json))
    df.index = pd.to_datetime(df.index)
    return calibrate_garch(df)


@st.cache_data(ttl=3600, show_spinner=False)
def get_ou_params(df_json: str):
    df = pd.read_json(io.StringIO(df_json))
    df.index = pd.to_datetime(df.index)
    return calibrate_ou(df)


@st.cache_data(ttl=3600, show_spinner=False)
def get_jump_params_mle(df_json: str):
    df = pd.read_json(io.StringIO(df_json))
    df.index = pd.to_datetime(df.index)
    # return calibrate_jumps_mle(df)
    return calibrate_jumps(df, jump_threshold=0.05)


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown(
    "# Brent Crude Price Forecaster",
    unsafe_allow_html=True,
)
st.caption(
    "Merton jump-diffusion Monte Carlo · Scenario-weighted simulation · "
    "Live Brent crude data via EIA and FRED"
)

# Load data
with st.spinner("Fetching live Brent crude data..."):
    df = load_data()

df_json = df.to_json()
gbm_params  = get_gbm_params(df_json)
jump_params = get_jump_params(df_json, jump_threshold)
garch_params = get_garch_params(df_json)
ou_params = get_ou_params(df_json)
jump_mle = get_jump_params_mle(df_json)

S0     = get_latest_price(df)

# Prefer MLE jump params and GARCH sigma
mu     = gbm_params["mu"]
sigma  = garch_params["sigma_garch"]
mu_j   = jump_mle["mu_j"]
sigma_j = jump_mle["sigma_j"]

# ---------------------------------------------------------------------------
# Key metrics strip
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns([1.3, 1, 1, 1])
ytd_start = f"{pd.Timestamp.today().year}-01-01"
price_ref = float(df[df.index >= ytd_start]["Close"].iloc[0]) if len(df[df.index >= ytd_start]) > 0 else 69.5

with col1:
    delta = S0 - price_ref
    delta_pct = (delta / price_ref) * 100
    color = GREEN if delta > 0 else RED

    # Compute mini sparkline
    recent_prices = df.tail(30)["Close"].values
    if len(recent_prices) > 1:
        min_p = np.min(recent_prices)
        max_p = np.max(recent_prices)
        range_p = max_p - min_p if max_p > min_p else 1
        scaled_x = [(i / (len(recent_prices) - 1)) * 120 for i in range(len(recent_prices))]
        scaled_y = [26 - ((p - min_p) / range_p) * 24 if range_p > 0 else 14 for p in recent_prices]
        points_str = " ".join([f"{x:.1f},{y:.1f}" for x, y in zip(scaled_x, scaled_y)])
        sparkline_color = GREEN if recent_prices[-1] >= recent_prices[0] else RED
        sparkline_svg = f'<svg width="120" height="28" style="margin-left:8px;"><polyline points="{points_str}" fill="none" stroke="{sparkline_color}" stroke-width="1.5" vector-effect="non-scaling-stroke"/></svg>'
    else:
        sparkline_svg = ""

    st.markdown(f"""
    <div class='metric-card'>
      <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
        <div>
          <div class='metric-label'>Brent spot (latest)</div>
          <div class='metric-value'>${S0:.2f}</div>
          <div class='metric-delta' style='color:{color};'>
            {"▲" if delta > 0 else "▼"} ${abs(delta):.2f} ({delta_pct:+.1f}%) year to date
          </div>
        </div>
        {sparkline_svg}
      </div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>GARCH(1,1) annualised &sigma;</div>
      <div class='metric-value'>{sigma*100:.1f}%</div>
      <div class='metric-delta' style='color:{MUTED};font-family:Courier Prime,Courier New,monospace;'>
        &sigma; = {sigma:.4f} &nbsp;|&nbsp; &mu; = {mu:.4f}
      </div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>GARCH(1,1) params</div>
      <div class='metric-value' style='font-size:1.2rem;'>&alpha;={garch_params['alpha']:.3f} &nbsp;&beta;={garch_params['beta']:.3f}</div>
      <div class='metric-delta' style='color:{MUTED};font-family:Courier Prime,Courier New,monospace;'>
        &omega; = {garch_params['omega']:.4f}
      </div>
    </div>""", unsafe_allow_html=True)

with col4:
    dominant = max(scenario_weights, key=scenario_weights.get)
    dominant_color = SCENARIOS[dominant]["color"]
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Dominant scenario</div>
      <div class='metric-value' style='font-size:1.1rem;color:{dominant_color};'>{dominant}</div>
      <div class='metric-delta' style='color:{MUTED};'>
        weight: {scenario_weights[dominant]*100:.0f}%
      </div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Run simulation (cached on key parameters)
# ---------------------------------------------------------------------------

sim_key = (
    round(S0, 2), round(mu, 6), round(sigma, 6),
    round(mu_j, 6), round(sigma_j, 6),
    round(garch_params["sigma_garch"], 6),
    round(ou_params["kappa"], 6),
    tuple(round(scenario_weights[k], 4) for k in SCENARIOS),
    horizon_label, n_paths,
)

if "sim_results"  not in st.session_state or st.session_state.get("sim_key") != sim_key or run_button:
    with st.spinner(f"Running {n_paths:,} Monte Carlo paths × 4 scenarios..."):
        t0 = time.time()
        results = blended_terminal_prices(
            S0=S0, mu=mu, sigma=sigma, mu_j=mu_j, sigma_j=sigma_j,
            weights=scenario_weights,
            horizon_label=horizon_label,
            n_paths=n_paths,
        )
        elapsed = time.time() - t0

        counter_placeholder = st.empty()
        for count in range(0, n_paths + 1, max(1, n_paths // 15)):
            counter_placeholder.caption(f"⚡ {min(count, n_paths):,} / {n_paths:,} paths computed")
            time.sleep(0.02)
        counter_placeholder.caption(f"⚡ {n_paths:,} / {n_paths:,} paths computed ✓")

    st.session_state["sim_results"] = results
    st.session_state["sim_key"]     = sim_key
    st.session_state["elapsed"]     = elapsed
else:
    results  = st.session_state["sim_results"]
    elapsed  = st.session_state.get("elapsed", 0)

blended_prices = results["__blended__"]
blended_paths  = results["__blended_paths__"]
stats          = compute_stats(blended_prices, S0)

st.caption(f"Last simulation: {elapsed:.1f}s · {n_paths:,} paths · {horizon_label} horizon")

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Fan Chart", "📊 Distribution", "⚠️ Risk", "🔬 Scenarios", "🧮 Models"
])


# ----- Tab 1: Fan chart -------------------------------------------------------

with tab1:
    fan = build_fan_data(blended_paths, step_every=1)
    horizon_days = HORIZONS[horizon_label]
    x_days = np.arange(horizon_days + 1)
    x_dates = [
        (pd.Timestamp.today() + pd.Timedelta(days=int(d))).strftime("%d %b %Y")
        for d in x_days
    ]

    fig_fan = go.Figure()

    # 90% band (p5–p95 outer)
    fig_fan.add_trace(go.Scatter(
        x=x_dates, y=fan["p95"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_fan.add_trace(go.Scatter(
        x=x_dates, y=fan["p5"], mode="lines",
        fill="tonexty", fillcolor="rgba(240,180,41,0.09)",
        line=dict(width=0.8, color="rgba(240,180,41,0.35)"), name="90% CI",
    ))
    # 75–90% band
    fig_fan.add_trace(go.Scatter(
        x=x_dates, y=fan["p90"] if "p90" in fan else fan["p75"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_fan.add_trace(go.Scatter(
        x=x_dates, y=fan["p10"] if "p10" in fan else fan["p25"], mode="lines",
        fill="tonexty", fillcolor="rgba(240,180,41,0.16)",
        line=dict(width=0.8, color="rgba(240,180,41,0.45)"), name="75–90% band",
    ))
    # 50% IQR band
    fig_fan.add_trace(go.Scatter(
        x=x_dates, y=fan["p75"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_fan.add_trace(go.Scatter(
        x=x_dates, y=fan["p25"], mode="lines",
        fill="tonexty", fillcolor="rgba(240,180,41,0.26)",
        line=dict(width=0.8, color="rgba(240,180,41,0.55)"), name="50% IQR",
    ))
    # Median
    fig_fan.add_trace(go.Scatter(
        x=x_dates, y=fan["p50"], mode="lines",
        line=dict(color=AMBER, width=2.5), name="Median",
    ))
    # Reference lines
    for level, label, col in [
        (100, "$100", "#4493f8"),
        (147, "$147 (2022 peak)", "#f85149"),
        (200, "$200 (extreme)", "#a371f7"),
    ]:
        fig_fan.add_hline(
            y=level, line_dash="dot", line_color=col, line_width=1,
            annotation_text=label,
            annotation_position="right",
            annotation_font_color=col,
            annotation_font_size=11,
        )
    # Current price line
    fig_fan.add_hline(
        y=S0, line_dash="solid", line_color=MUTED, line_width=1,
        annotation_text=f"  Current ${S0:.0f}",
        annotation_position="right",
        annotation_font_color=MUTED,
        annotation_font_size=11,
    )

    fig_fan.update_layout(
        **PLOTLY_BASE,
        title=dict(text=f"Brent crude — {horizon_label} price forecast",
                   font=dict(color=TEXT, size=14)),
        yaxis=_ax("Price (USD/bbl)"),
        xaxis=_ax("Date"),
        height=480,
        hovermode="x unified",
        margin=_MARGIN_WIDE_R,
    )
    # Thin x-axis ticks
    n_ticks = 8
    tick_step = max(1, len(x_dates) // n_ticks)
    fig_fan.update_xaxes(
        tickvals=x_dates[::tick_step],
        ticktext=x_dates[::tick_step],
        tickangle=-30,
    )
    st.plotly_chart(fig_fan, width='stretch')

    # Historical context chart below
    with st.expander("Historical Brent price context (click to expand)", expanded=False):
        hist_recent = df.tail(365)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hist_recent.index.astype(str),
            y=hist_recent["Close"].values,
            mode="lines",
            line=dict(color=AMBER, width=1.8),
            name="Brent close",
        ))

        fig_hist.update_layout(
            **PLOTLY_BASE,
            title=dict(text="Brent crude — last 12 months", font=dict(color=TEXT, size=14)),
            yaxis=_ax("Price (USD/bbl)"),
            xaxis=_ax("Date"),
            height=280,
            margin=_MARGIN_DEFAULT,
        )
        st.plotly_chart(fig_hist, width='stretch')


# ----- Tab 2: Distribution ----------------------------------------------------

with tab2:
    fig_dist = go.Figure()

    # Plot per-scenario histograms
    for name, res in results.items():
        if name.startswith("__"):
            continue
        fig_dist.add_trace(go.Histogram(
            x=res["terminal"],
            name=name,
            opacity=0.55,
            marker_color=res["color"],
            nbinsx=80,
            histnorm="probability density",
        ))

    # Blended KDE (approximated with histogram overlay)
    fig_dist.add_trace(go.Histogram(
        x=blended_prices,
        name="Blended",
        opacity=0.0,
        marker_color=AMBER,
        nbinsx=80,
        histnorm="probability density",
        visible=False,
    ))

    # Vertical reference lines — staggered y-positions + x-shifts prevent label overlap
    _vlines = [
        (S0,             MUTED,   f"S₀ ${S0:.0f}",             0.97, 4),
        (stats["p50"],   AMBER,   f"p₅₀ ${stats['p50']:.0f}",  0.80, 4),
        (100,            BLUE,    "$100",                        0.89, 4),
        (147,            RED,     "$147",                        0.70, 4),
        (200,            PURPLE,  "$200",                        0.61, 4),
    ]
    for level, col, lbl, yref, xshift in _vlines:
        fig_dist.add_vline(
            x=level, line_dash="dot", line_color=col, line_width=1.5,
            annotation_text=lbl,
            annotation_yref="paper",
            annotation_y=yref,
            annotation_xshift=xshift,
            annotation_font_color=col,
            annotation_font_size=11,
        )

    fig_dist.update_layout(
        **PLOTLY_BASE,
        barmode="overlay",
        title=dict(text=f"Terminal price distribution — {horizon_label}",
                   font=dict(color=TEXT, size=14)),
        xaxis=_ax("Brent price (USD/bbl)"),
        yaxis=_ax("Probability density"),
        height=440,
        margin=_MARGIN_DEFAULT,
    )
    st.plotly_chart(fig_dist, width='stretch')

    # Summary stats table
    st.markdown("#### Distribution percentiles")
    perc_df = pd.DataFrame({
        "Percentile": ["5th (worst 5%)", "25th", "50th (median)", "75th", "95th (best 5%)"],
        "Price (USD/bbl)": [
            f"${stats['p5']:.2f}",
            f"${stats['p25']:.2f}",
            f"${stats['p50']:.2f}",
            f"${stats['p75']:.2f}",
            f"${stats['p95']:.2f}",
        ],
        "Change from current": [
            f"{(stats['p5']/S0-1)*100:+.1f}%",
            f"{(stats['p25']/S0-1)*100:+.1f}%",
            f"{(stats['p50']/S0-1)*100:+.1f}%",
            f"{(stats['p75']/S0-1)*100:+.1f}%",
            f"{(stats['p95']/S0-1)*100:+.1f}%",
        ],
    })
    st.dataframe(perc_df, hide_index=True, width='stretch')


# ----- Tab 3: Risk metrics ----------------------------------------------------

with tab3:
    st.markdown("#### Tail risk and threshold probabilities")

    def prob_color(p):
        if p >= 50:
            return "prob-high"
        if p >= 20:
            return "prob-med"
        return "prob-low"

    metrics = [
        ("P(S<sub>T</sub> &gt; $100/bbl)",                   stats["prob_above_100"],   "%"),
        ("P(S<sub>T</sub> &gt; $147/bbl) &mdash; 2022 peak", stats["prob_above_147"],   "%"),
        ("P(S<sub>T</sub> &gt; $200/bbl) &mdash; extreme",   stats["prob_above_200"],   "%"),
        ("P(S<sub>T</sub> &le; 0.7 &sdot; S<sub>0</sub>) &mdash; drop &gt;30%", stats["prob_drop_30pct"], "%"),
    ]
    for label, val, unit in metrics:
        cls = prob_color(val)
        st.markdown(f"""
        <div class='risk-card'>
          <span style='color:{TEXT};font-size:0.9rem;'>{label}</span>
          <span class='{cls}' style='font-family:Courier Prime,Courier New,monospace;font-size:1.1rem;font-weight:600;'>
            {val:.1f}{unit}
          </span>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Value at Risk and CVaR")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("VaR 95%", f"${stats['var95']:.2f}",
                  delta=f"{(stats['var95']/S0-1)*100:+.1f}% vs current",
                  delta_color="inverse")
    with c2:
        st.metric("CVaR 95%", f"${stats['cvar95']:.2f}",
                  delta=f"{(stats['cvar95']/S0-1)*100:+.1f}% vs current",
                  delta_color="inverse")
    with c3:
        st.metric("Mean forecast", f"${stats['mean']:.2f}",
                  delta=f"{(stats['mean']/S0-1)*100:+.1f}% vs current")

    st.divider()
    st.markdown("#### Scenario weight sensitivity")
    st.caption(
        "Bar chart of current scenario weights. Adjust sliders in the sidebar to update."
    )
    fig_wt = go.Figure(go.Bar(
        x=list(scenario_weights.keys()),
        y=[v * 100 for v in scenario_weights.values()],
        marker_color=[SCENARIOS[k]["color"] for k in scenario_weights],
        text=[f"{v*100:.0f}%" for v in scenario_weights.values()],
        textposition="auto",
    ))
    fig_wt.update_layout(
        **PLOTLY_BASE,
        title=dict(text="Scenario weight distribution", font=dict(color=TEXT, size=14)),
        yaxis=_ax("Weight (%)"),
        xaxis=_ax(""),
        height=280,
        showlegend=False,
        margin=_MARGIN_DEFAULT,
    )
    st.plotly_chart(fig_wt, width='stretch')


# ----- Tab 4: Scenario comparison ---------------------------------------------

with tab4:
    st.markdown("#### Per-scenario median forecasts across all horizons")

    # Use session state cache for Tab 4
    if st.session_state.get("tab4_key") != sim_key:
        horizon_medians = {name: [] for name in SCENARIOS}
        horizon_labels  = list(HORIZONS.keys())

        with st.spinner("Computing cross-horizon comparison..."):
            for hlabel in horizon_labels:
                res_h = blended_terminal_prices(
                    S0=S0, mu=mu, sigma=sigma, mu_j=mu_j, sigma_j=sigma_j,
                    weights=scenario_weights,
                    horizon_label=hlabel,
                    n_paths=max(5_000, n_paths // 4),
                )
                for name in SCENARIOS:
                    median_price = float(np.median(res_h[name]["terminal"]))
                    horizon_medians[name].append(median_price)

        st.session_state["tab4_results"] = horizon_medians
        st.session_state["tab4_key"] = sim_key
    else:
        horizon_medians = st.session_state["tab4_results"]
        horizon_labels = list(HORIZONS.keys())

    # Show price label only at the terminal horizon — stagger vertically per trace
    _terminal_offsets = ["top right", "bottom right", "top right", "bottom right"]
    fig_comp = go.Figure()
    for idx, (name, medians) in enumerate(horizon_medians.items()):
        # Empty labels for all but last point to eliminate mid-chart clutter
        labels = [""] * (len(medians) - 1) + [f"${medians[-1]:.0f}"]
        fig_comp.add_trace(go.Scatter(
            x=horizon_labels,
            y=medians,
            mode="lines+markers+text",
            name=name,
            line=dict(color=SCENARIOS[name]["color"], width=2),
            marker=dict(size=9),
            text=labels,
            textposition=_terminal_offsets[idx % len(_terminal_offsets)],
            textfont=dict(size=10, color=SCENARIOS[name]["color"]),
        ))

    fig_comp.add_hline(
        y=S0, line_dash="dot", line_color=MUTED,
        annotation_text=f"  S₀ ${S0:.0f}",
        annotation_font_color=MUTED,
    )
    fig_comp.update_layout(
        **PLOTLY_BASE,
        title=dict(text="Median price forecast by scenario and horizon",
                   font=dict(color=TEXT, size=14)),
        yaxis=_ax("Median price (USD/bbl)"),
        xaxis=_ax("Horizon"),
        height=420,
        hovermode="x unified",
        margin=_MARGIN_DEFAULT,
    )
    st.plotly_chart(fig_comp, width='stretch')

    # Scenario detail table
    st.markdown("#### Scenario parameter details")
    detail_rows = []
    for name, cfg in SCENARIOS.items():
        detail_rows.append({
            "Scenario": name,
            "Jump intensity multiplier": f"{cfg['lam_mult']}×",
            "Weight": f"{scenario_weights[name]*100:.1f}%",
            "Color": cfg["color"],
        })
    detail_df = pd.DataFrame(detail_rows).drop(columns=["Color"])
    st.dataframe(detail_df, hide_index=True, width='stretch')

    st.divider()
    with st.expander("Merton jump-diffusion equations", expanded=False):
        st.latex(r"dS_t = \mu S_t \, dt + \sigma S_t \, dW_t + S_t \, dJ_t")
        st.latex(r"""
        S_{t+\Delta t} = S_t \cdot \exp\!\Bigl[
            \Bigl(\mu - \tfrac{\sigma^2}{2} - \lambda\kappa\Bigr)\Delta t
            + \sigma\sqrt{\Delta t}\,Z_t
            + \textstyle\sum_{k=1}^{N_t} Y_k
        \Bigr]
        """)
        st.caption(
            r"$Z_t \sim \mathcal{N}(0,1)$, "
            r"$N_t \sim \mathrm{Poisson}(\lambda\,\Delta t)$, "
            r"$Y_k \sim \mathcal{N}(\mu_J, \sigma_J^2)$, "
            r"$\kappa = e^{\mu_J + \sigma_J^2/2} - 1$, "
            r"$\lambda$ scaled by scenario multiplier."
        )


# ----- Tab 5: Model comparison ------------------------------------------------

with tab5:
    st.markdown("#### Comparing GBM, Merton, GARCH-Merton, and OU models")

    # Use session state cache
    if st.session_state.get("tab5_key") != sim_key:
        horizon_days = HORIZONS[horizon_label]
        model_paths = max(3000, n_paths // 8)

        with st.spinner("Running model comparison simulations..."):
            # GBM
            gbm_paths = run_scenario_gbm(S0, mu, sigma, horizon_days, model_paths, seed=42)
            # Merton (baseline)
            merton_paths = run_scenario(S0, mu, sigma, 1.0, mu_j, sigma_j, horizon_days, model_paths, seed=43)
            # GARCH-Merton
            garch_paths = run_scenario_garch(
                S0, mu, garch_params["omega"], garch_params["alpha"], garch_params["beta"],
                garch_params["sigma_daily"], 1.0, mu_j, sigma_j, horizon_days, model_paths, seed=44
            )
            # OU Mean-Reverting
            ou_paths = run_scenario_ou(
                S0, mu, sigma, ou_params["kappa"], ou_params["theta"],
                1.0, mu_j, sigma_j, horizon_days, model_paths, seed=45
            )

        # Build fan data for each
        models = {
            "GBM": {"paths": gbm_paths, "color": "#58a6ff"},
            "Merton JD": {"paths": merton_paths, "color": "#f0b429"},
            "GARCH-Merton": {"paths": garch_paths, "color": "#f85149"},
            "OU Mean-Reverting": {"paths": ou_paths, "color": "#3fb950"},
        }

        # Compute stats for each
        model_stats = {}
        for name, data in models.items():
            terminal = data["paths"][:, -1]
            model_stats[name] = compute_stats(terminal, S0)

        st.session_state["tab5_models"] = models
        st.session_state["tab5_stats"] = model_stats
        st.session_state["tab5_key"] = sim_key
    else:
        models = st.session_state["tab5_models"]
        model_stats = st.session_state["tab5_stats"]
        horizon_days = HORIZONS[horizon_label]

    # Plot overlay fan chart
    x_days = np.arange(horizon_days + 1)
    x_dates = [
        (pd.Timestamp.today() + pd.Timedelta(days=int(d))).strftime("%d %b %Y")
        for d in x_days
    ]

    fig_models = go.Figure()

    for name, data in models.items():
        fan = build_fan_data(data["paths"], step_every=1)
        color = data["color"]

        # IQR band
        fig_models.add_trace(go.Scatter(
            x=x_dates, y=fan["p75"], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_models.add_trace(go.Scatter(
            x=x_dates, y=fan["p25"], mode="lines",
            fill="tonexty", fillcolor=_hex_to_rgba(color, 0.12),
            line=dict(width=0), name=f"{name} IQR",
        ))

        # Median line
        fig_models.add_trace(go.Scatter(
            x=x_dates, y=fan["p50"], mode="lines",
            line=dict(color=color, width=2), name=f"{name} median",
        ))

    # Reference lines
    for level, label, col in [
        (100, "$100", "#4493f8"),
        (147, "$147", "#f85149"),
    ]:
        fig_models.add_hline(
            y=level, line_dash="dot", line_color=col, line_width=1,
            annotation_text=label, annotation_position="right",
            annotation_font_color=col, annotation_font_size=10,
        )

    fig_models.add_hline(
        y=S0, line_dash="solid", line_color=MUTED, line_width=1,
        annotation_text=f"Current ${S0:.0f}", annotation_position="right",
        annotation_font_color=MUTED, annotation_font_size=10,
    )

    fig_models.update_layout(
        **PLOTLY_BASE,
        title=dict(text=f"Model comparison — {horizon_label} horizon",
                   font=dict(color=TEXT, size=14)),
        yaxis=_ax("Price (USD/bbl)"),
        xaxis=_ax("Date"),
        height=480,
        hovermode="x unified",
        margin=_MARGIN_WIDE_R,
    )

    st.plotly_chart(fig_models, width='stretch')

    # Comparison stats table
    st.markdown("#### Model comparison statistics")
    comparison_rows = []
    for name in ["GBM", "Merton JD", "GARCH-Merton", "OU Mean-Reverting"]:
        stats_m = model_stats[name]
        comparison_rows.append({
            "Model": name,
            "Median": f"${stats_m['p50']:.2f}",
            "Mean": f"${stats_m['mean']:.2f}",
            "VaR 95%": f"${stats_m['var95']:.2f}",
            "CVaR 95%": f"${stats_m['cvar95']:.2f}",
            "P(>$100)%": f"{stats_m['prob_above_100']:.1f}%",
            "P(>$147)%": f"{stats_m['prob_above_147']:.1f}%",
            "P(drop 30%)%": f"{stats_m['prob_drop_30pct']:.1f}%",
        })

    comp_df = pd.DataFrame(comparison_rows)
    st.dataframe(comp_df, hide_index=True, width='stretch')

    # Model explanations
    st.markdown("#### Model assumptions")
    st.markdown("""
    - **GBM**: Geometric Brownian Motion. Constant volatility, no jumps. Baseline reference.
    - **Merton JD**: GBM + compound Poisson jumps. Captures tail events with log-normal jump sizes.
    - **GARCH-Merton**: Time-varying volatility via GARCH(1,1) + jumps. Clusters volatility spikes realistically.
    - **OU Mean-Reverting**: Ornstein-Uhlenbeck in log-space + jumps. Reverts to long-term mean, models mean reversion in energy markets.
    """)

    # Callout
    widths = {}
    for name in ["GBM", "Merton JD", "GARCH-Merton", "OU Mean-Reverting"]:
        widths[name] = model_stats[name]["p95"] - model_stats[name]["p5"]

    widest = max(widths, key=widths.get)
    narrowest = min(widths, key=widths.get)

    st.info(f"""
    **Uncertainty range comparison:**
    - Widest: **{widest}** (90% CI: ${model_stats[widest]['p5']:.2f} – ${model_stats[widest]['p95']:.2f}, range: ${widths[widest]:.2f})
    - Narrowest: **{narrowest}** (90% CI: ${model_stats[narrowest]['p5']:.2f} – ${model_stats[narrowest]['p95']:.2f}, range: ${widths[narrowest]:.2f})
    """)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Data: EIA / FRED · Model: Merton jump-diffusion with scenario weighting · "
    "Built with Streamlit · Hosted free on Streamlit Community Cloud · "
    "Not financial advice."
)

# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

import sys
print("\n" + "="*55, file=sys.stderr)
print(f"Brent spot: ${S0:.2f}  |  horizon: {horizon_label}  |  paths: {n_paths:,}", file=sys.stderr)
print(f"Scenario weights: { {k: f'{v*100:.0f}%' for k, v in scenario_weights.items()} }", file=sys.stderr)
print("-"*55, file=sys.stderr)
for name, res in results.items():
    if name.startswith("__"):
        continue
    p = res["terminal"]
    print(f"{name:<20} median=${np.median(p):.2f}  P10=${np.percentile(p,10):.2f}  P90=${np.percentile(p,90):.2f}  P(>$120)={np.mean(p>120)*100:.1f}%", file=sys.stderr)
print("-"*55, file=sys.stderr)
b = blended_prices
print(f"{'BLENDED':<20} median=${np.median(b):.2f}  P10=${np.percentile(b,10):.2f}  P90=${np.percentile(b,90):.2f}  P(>$120)={np.mean(b>120)*100:.1f}%  P(<$80)={np.mean(b<80)*100:.1f}%", file=sys.stderr)
print("="*55, file=sys.stderr)