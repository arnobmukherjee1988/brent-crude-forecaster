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

# Custom CSS with theme
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Times New Roman', Times, serif;
    background-color: {BG};
    color: {TEXT};
}}
.stApp {{
    background: {BG};
    color: {TEXT};
}}
section[data-testid="stSidebar"] {{
    background: {SIDEBAR};
    border-right: 1px solid {BORDER};
}}
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
h1 {{ font-family: 'Courier Prime', 'Courier New', monospace !important; letter-spacing: -0.02em; color: {TEXT}; }}
h2, h3 {{ font-family: 'Times New Roman', Times, serif !important; font-weight: 700; color: {TEXT}; }}
p, span, label {{ color: {TEXT}; }}
.stMarkdown {{ color: {TEXT}; }}
div[data-testid="stText"] {{ color: {TEXT}; }}

.stSlider > div > div > div {{ background: #f0b429 !important; }}
div[data-testid="stMetric"] {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 14px 18px;
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: Plotly dark layout defaults
# ---------------------------------------------------------------------------

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=PLOT_BG,
    font=dict(family="Times New Roman, Times, serif", color=TEXT, size=12),
    xaxis=dict(gridcolor=GRID_CLR, linecolor=BORDER, showgrid=True),
    yaxis=dict(gridcolor=GRID_CLR, linecolor=BORDER, showgrid=True),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1),
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
    return calibrate_jumps_mle(df)


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
      <div class='metric-label'>Annualised vol (60-day)</div>
      <div class='metric-value'>{sigma*100:.1f}%</div>
      <div class='metric-delta' style='color:{MUTED};'>σ = {sigma:.4f} | μ = {mu:.4f}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>GARCH(1,1) volatility</div>
      <div class='metric-value'>{garch_params['sigma_garch']*100:.1f}%</div>
      <div class='metric-delta' style='color:{MUTED};'>α={garch_params['alpha']:.3f} β={garch_params['beta']:.3f}</div>
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
    "📈 Fan chart", "📊 Distribution", "⚠️ Risk metrics", "🔬 Scenario comparison", "🧮 Model comparison"
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

    # 90% band (p90-p95 outer)
    fig_fan.add_trace(go.Scatter(
        x=x_dates, y=fan["p95"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_fan.add_trace(go.Scatter(
        x=x_dates, y=fan["p5"], mode="lines",
        fill="tonexty", fillcolor="rgba(240,180,41,0.04)",
        line=dict(width=0), name="90% CI",
    ))
    # 75-90 band
    fig_fan.add_trace(go.Scatter(
        x=x_dates, y=fan["p90"] if "p90" in fan else fan["p75"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_fan.add_trace(go.Scatter(
        x=x_dates, y=fan["p10"] if "p10" in fan else fan["p25"], mode="lines",
        fill="tonexty", fillcolor="rgba(240,180,41,0.10)",
        line=dict(width=0), name="75-90 band",
    ))
    # 50% band (IQR)
    fig_fan.add_trace(go.Scatter(
        x=x_dates, y=fan["p75"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_fan.add_trace(go.Scatter(
        x=x_dates, y=fan["p25"], mode="lines",
        fill="tonexty", fillcolor="rgba(240,180,41,0.20)",
        line=dict(width=0), name="50% CI",
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
        title=f"Brent crude price fan chart — {horizon_label} forecast",
        yaxis_title="Price (USD/bbl)",
        xaxis_title="Date",
        height=480,
        hovermode="x unified",
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
            title="Brent crude: last 12 months",
            yaxis_title="Price (USD/bbl)",
            height=280,
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

    # Vertical lines for key levels
    for level, col, lbl in [
        (S0,  MUTED, f"Current ${S0:.0f}"),
        (stats["p50"], AMBER, f"Median ${stats['p50']:.0f}"),
        (100, BLUE,  "$100"),
        (147, RED,   "$147"),
        (200, PURPLE,"$200"),
    ]:
        fig_dist.add_vline(
            x=level, line_dash="dot", line_color=col, line_width=1.5,
            annotation_text=lbl, annotation_position="top",
            annotation_font_color=col, annotation_font_size=11,
        )

    fig_dist.update_layout(
        **PLOTLY_BASE,
        barmode="overlay",
        title=f"Terminal price distribution — {horizon_label}",
        xaxis_title="Brent price (USD/bbl)",
        yaxis_title="Probability density",
        height=440,
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
        ("Prob. price above $100/bbl",      stats["prob_above_100"],   "%"),
        ("Prob. price above $147/bbl (2022 peak)", stats["prob_above_147"], "%"),
        ("Prob. price above $200/bbl (extreme scenario)", stats["prob_above_200"], "%"),
        ("Prob. price drops >30% from current", stats["prob_drop_30pct"], "%"),
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
        title="Scenario weight distribution",
        yaxis_title="Weight (%)",
        height=280,
        showlegend=False,
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

    fig_comp = go.Figure()
    for name, medians in horizon_medians.items():
        fig_comp.add_trace(go.Scatter(
            x=horizon_labels,
            y=medians,
            mode="lines+markers+text",
            name=name,
            line=dict(color=SCENARIOS[name]["color"], width=2),
            marker=dict(size=9),
            text=[f"${m:.0f}" for m in medians],
            textposition="top center",
            textfont=dict(size=11, color=SCENARIOS[name]["color"]),
        ))

    fig_comp.add_hline(
        y=S0, line_dash="dot", line_color=MUTED,
        annotation_text=f"  Current ${S0:.0f}",
        annotation_font_color=MUTED,
    )
    fig_comp.update_layout(
        **PLOTLY_BASE,
        title="Median price forecast by scenario and horizon",
        yaxis_title="Median price (USD/bbl)",
        height=420,
        hovermode="x unified",
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
    st.markdown("#### Model equations")
    st.latex(r"""
    dS_t = \mu S_t \, dt + \sigma S_t \, dW_t + S_t \, dJ_t
    """)
    st.latex(r"""
    S_{t+\Delta t} = S_t \cdot \exp\!\left[
        \left(\mu - \tfrac{\sigma^2}{2} - \lambda\kappa\right)\Delta t
        + \sigma\sqrt{\Delta t}\,Z_t
        + \sum_{k=1}^{N_t} Y_k
    \right]
    """)
    st.caption(
        "where $Z_t \\sim \\mathcal{N}(0,1)$, "
        "$N_t \\sim \\text{Poisson}(\\lambda \\Delta t)$, "
        "$Y_k \\sim \\mathcal{N}(\\mu_J, \\sigma_J^2)$, "
        "and $\\kappa = e^{\\mu_J + \\sigma_J^2/2} - 1$."
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
        title=f"Model comparison: fan charts — {horizon_label} horizon",
        yaxis_title="Price (USD/bbl)",
        xaxis_title="Date",
        height=480,
        hovermode="x unified",
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
