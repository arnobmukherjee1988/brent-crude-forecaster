## brent-crude-forecaster

Monte Carlo simulation of Brent crude oil prices using the Merton jump-diffusion model.
Built with Streamlit. Developed in the context of the 2026 US-Iran war and Strait of Hormuz disruption.

---

### What it does

- Fetches live Brent crude price data from Yahoo Finance
- Calibrates a stochastic model from recent daily returns
- Runs 50,000 simulated price paths across four geopolitical scenarios
- Produces 3-month, 6-month, and 12-month price forecasts with confidence intervals
- Computes risk metrics: Value at Risk, Conditional VaR, and threshold probabilities

---

### Project structure

    app.py              Main Streamlit dashboard
    mc_engine.py        Monte Carlo simulation engine (Numba-compiled)
    data_fetcher.py     Live data fetching and parameter calibration
    requirements.txt    Python dependencies

---

### Run locally

    pip install -r requirements.txt
    streamlit run app.py

The dashboard opens at http://localhost:8501

---

### Deploy for free on Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to https://share.streamlit.io and sign in with GitHub
3. Select this repository, set the main file to app.py
4. Click Deploy

Your app gets a permanent public URL at no cost. The free tier provides 1 GB RAM,
which is sufficient for this simulation. The app sleeps after inactivity and wakes
automatically when someone visits the URL.

No server, no Docker, no payment required.

---

### Model

The price follows a Merton jump-diffusion process:

    dS = mu * S * dt  +  sigma * S * dW  +  S * dJ

where dW is a Wiener process and dJ is a compound Poisson jump process.
Parameters mu, sigma, mu_J, and sigma_J are calibrated from the most recent
60 days of Brent closing prices. Jump intensity is scaled by scenario.

---

### Geopolitical scenarios

    Ceasefire (1-2 months)        default weight 25%
    Prolonged war (4-6 months)    default weight 40%
    Escalation (two straits)      default weight 25%
    Extreme shock ($200+/bbl)     default weight 10%

Scenario weights are adjustable in real time from the sidebar.

---

### Stack

    Python 3.11+
    Streamlit
    NumPy, pandas, SciPy
    Numba (JIT compilation for simulation speed)
    Plotly (interactive charts)
    yfinance (live market data)

---

### Disclaimer

This project is for educational and research purposes only.
It is not financial advice.
