"""
Brent Crude Price Forecaster
Monte Carlo simulation with scenario weighting.
Streamlit dashboard — deployable on Streamlit Community Cloud (free).
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from data_fetcher import (
    fetch_brent,
    calibrate_gbm,
    calibrate_jumps,
    get_latest_price,
    get_price_history_since,
)
from mc_engine import (
    SCENARIOS,
    HORIZONS,
    blended_terminal_prices,
    compute_stats,
    build_fan_data,
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

# Custom CSS: dark petroleum aesthetic
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Times New Roman', Times, serif;
}
.stApp {
    background: #f9f7f4;
    color: #1a1a1a;
}
section[data-testid="stSidebar"] {
    background: #f0ede8;
    border-right: 1px solid #d4cfc8;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #d4cfc8;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.metric-value {
    font-family: 'Courier Prime', 'Courier New', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #f0b429;
}
.metric-label {
    font-size: 0.75rem;
    color: #6b6560;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.metric-delta {
    font-family: 'Courier Prime', 'Courier New', monospace;
    font-size: 0.85rem;
    margin-top: 4px;
}
.risk-card {
    background: #ffffff;
    border: 1px solid #d4cfc8;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.prob-high { color: #f85149; }
.prob-med  { color: #f0b429; }
.prob-low  { color: #3fb950; }
h1 { font-family: 'Courier Prime', 'Courier New', monospace !important; letter-spacing: -0.02em; }
h2, h3 { font-family: 'Times New Roman', Times, serif !important; font-weight: 700; }

.stSlider > div > div > div { background: #f0b429 !important; }
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #d4cfc8;
    border-radius: 8px;
    padding: 14px 18px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: Plotly dark layout defaults
# ---------------------------------------------------------------------------

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#f9f7f4",
    font=dict(family="Times New Roman, Times, serif", color="#1a1a1a", size=12),
    xaxis=dict(gridcolor="#e8e4de", linecolor="#d4cfc8", showgrid=True),
    yaxis=dict(gridcolor="#e8e4de", linecolor="#d4cfc8", showgrid=True),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d", borderwidth=1),
)

AMBER  = "#f0b429"
RED    = "#f85149"
GREEN  = "#3fb950"
PURPLE = "#a371f7"
BLUE   = "#58a6ff"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
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
    run_button = st.button("Run simulation", type="primary", width="stretch")


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

S0     = get_latest_price(df)
mu     = gbm_params["mu"]
sigma  = gbm_params["sigma"]
mu_j   = jump_params["mu_j"]
sigma_j = jump_params["sigma_j"]

# ---------------------------------------------------------------------------
# Key metrics strip
# ---------------------------------------------------------------------------


col1, col2, col3, col4 = st.columns(4)
price_ref   = float(df[df.index >= "2026-01-01"]["Close"].iloc[0]) if len(df[df.index >= "2026-01-01"]) > 0 else 69.5

with col1:
    delta = S0 - price_ref
    delta_pct = (delta / price_ref) * 100
    color = RED if delta > 0 else GREEN
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Brent spot (latest)</div>
      <div class='metric-value'>${S0:.2f}</div>
      <div class='metric-delta' style='color:{color};'>
        {"▲" if delta > 0 else "▼"} ${abs(delta):.2f} ({delta_pct:+.1f}%) year to date
      </div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Annualised vol (60-day)</div>
      <div class='metric-value'>{sigma*100:.1f}%</div>
      <div class='metric-delta' style='color:#8b949e;'>σ = {sigma:.4f} | μ = {mu:.4f}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Jump parameters (MLE)</div>
      <div class='metric-value'>μ<sub>J</sub> = {mu_j:.3f}</div>
      <div class='metric-delta' style='color:#8b949e;'>σ<sub>J</sub> = {sigma_j:.3f}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    dominant = max(scenario_weights, key=scenario_weights.get)
    dominant_color = SCENARIOS[dominant]["color"]
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Dominant scenario</div>
      <div class='metric-value' style='font-size:1.1rem;color:{dominant_color};'>{dominant}</div>
      <div class='metric-delta' style='color:#8b949e;'>
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

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Fan chart", "📊 Distribution", "⚠️ Risk metrics", "🔬 Scenario comparison"
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

    # 90% band
    fig_fan.add_trace(go.Scatter(
        x=x_dates, y=fan["p95"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_fan.add_trace(go.Scatter(
        x=x_dates, y=fan["p5"], mode="lines",
        fill="tonexty", fillcolor="rgba(240,180,41,0.08)",
        line=dict(width=0), name="90% CI",
    ))
    # 50% band
    fig_fan.add_trace(go.Scatter(
        x=x_dates, y=fan["p75"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig_fan.add_trace(go.Scatter(
        x=x_dates, y=fan["p25"], mode="lines",
        fill="tonexty", fillcolor="rgba(240,180,41,0.18)",
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
        y=S0, line_dash="solid", line_color="#8b949e", line_width=1,
        annotation_text=f"  Current ${S0:.0f}",
        annotation_position="right",
        annotation_font_color="#8b949e",
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
    st.plotly_chart(fig_fan, width="stretch")

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
        st.plotly_chart(fig_hist, width="stretch")


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
        (S0,  "#8b949e", f"Current ${S0:.0f}"),
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
    st.plotly_chart(fig_dist, width="stretch")

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
    st.dataframe(perc_df, hide_index=True, width="stretch")


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
          <span style='color:#1a1a1a;font-size:0.9rem;'>{label}</span>
          <span class='{cls}' style='font-family:'Courier Prime','Courier New',monospace;font-size:1.1rem;font-weight:600;'>
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
    st.plotly_chart(fig_wt, width="stretch")


# ----- Tab 4: Scenario comparison ---------------------------------------------

with tab4:
    st.markdown("#### Per-scenario median forecasts across all horizons")

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
        y=S0, line_dash="dot", line_color="#8b949e",
        annotation_text=f"  Current ${S0:.0f}",
        annotation_font_color="#8b949e",
    )
    fig_comp.update_layout(
        **PLOTLY_BASE,
        title="Median price forecast by scenario and horizon",
        yaxis_title="Median price (USD/bbl)",
        height=420,
        hovermode="x unified",
    )
    st.plotly_chart(fig_comp, width="stretch")

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
    st.dataframe(detail_df, hide_index=True, width="stretch")

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


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Data: EIA / FRED · Model: Merton jump-diffusion · "
    "Built with Streamlit · Hosted free on Streamlit Community Cloud · "
    "Not financial advice."
)
