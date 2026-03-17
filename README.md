## brent-crude-forecaster

Monte Carlo simulation of Brent crude oil prices using multiple stochastic models.
Built with Streamlit. Developed in the context of the 2026 US-Iran geopolitical risk
and Strait of Hormuz disruption scenarios.

---

### What it does

- Fetches live Brent crude price data via EIA API or FRED (with disk cache fallback)
- Calibrates four models from historical daily returns: Merton jump-diffusion, GARCH(1,1)-Merton, OU mean-reverting, and plain GBM
- Runs up to 50,000 Monte Carlo paths across four geopolitical scenarios
- Produces 3, 6, and 12-month price forecasts with confidence intervals
- Computes risk metrics: VaR, CVaR, and threshold exceedance probabilities
- Compares all four models side by side on a dedicated tab

---

### Project structure

    app.py              Streamlit dashboard (5 tabs, dark/light mode)
    mc_engine.py        Simulation engine (Numba-compiled, antithetic variates)
    data_fetcher.py     Data fetching, GARCH/MLE/OU/GBM calibration
    requirements.txt    Python dependencies

---

### Run locally

    pip install -r requirements.txt
    streamlit run app.py

Opens at http://localhost:8501

---

### Deploy on Streamlit Community Cloud

1. Push to GitHub
2. Go to https://share.streamlit.io, connect the repo, set main file to app.py
3. Deploy

Free tier (1 GB RAM) is sufficient.

---

### Models

Merton jump-diffusion (primary):

    dS = mu * S * dt  +  sigma * S * dW  +  S * dJ

where dJ is a compound Poisson process with log-normal jump sizes.

GARCH(1,1)-Merton: time-varying conditional volatility fitted on percentage log-returns.

OU mean-reverting: Ornstein-Uhlenbeck drift in log-price space, calibrated via OLS.

GBM: constant-volatility baseline with no jumps.

Parameters are calibrated via MLE (jumps) and GARCH(1,1) fitting (volatility).
Jump parameters use the full available history; volatility uses a one-step-ahead GARCH forecast.

---

### Scenarios

    Low disruption      25%
    Base case           40%
    High disruption     25%
    Extreme shock       10%

Weights are adjustable in real time from the sidebar.

---

### Stack

    Python 3.11+, Streamlit, NumPy, pandas, SciPy
    Numba (JIT-compiled parallel simulation)
    arch (GARCH fitting)
    Plotly (interactive charts)

---

### Disclaimer

For educational and research purposes only. Not financial advice.
