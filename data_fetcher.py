"""
Live Brent crude data fetching and parameter calibration.

Source priority:
  1. Local disk cache        (if less than 1 hour old, no network call)
  2. EIA API                 (primary, requires EIA_API_KEY in .streamlit/secrets.toml)
  3. FRED CSV                (fallback, no key required)
  4. Stale disk cache        (if all network sources fail)
  5. Synthetic series        (absolute last resort, app still runs)
"""

import io
import time
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
from math import factorial


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EIA series: Europe Brent Spot Price FOB (Dollars per Barrel), daily
EIA_SERIES   = "PET.RBRTE.D"
EIA_BASE_URL = "https://api.eia.gov/v2/seriesid/{series}?api_key={key}&data[]=value&frequency=daily&sort[0][column]=period&sort[0][direction]=asc&length=1000"

# FRED fallback: same series, no key required
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"

CACHE_FILE        = Path(__file__).parent / ".streamlit" / "price_cache.json"
CACHE_TTL_SECONDS = 3600


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_is_fresh() -> bool:
    if not CACHE_FILE.exists():
        return False
    return (time.time() - CACHE_FILE.stat().st_mtime) < CACHE_TTL_SECONDS


def _save_cache(df: pd.DataFrame) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "index": [str(d.date()) for d in df.index],
        "close": [round(float(v), 4) for v in df["Close"]],
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(payload, f)


def _load_cache() -> pd.DataFrame:
    with open(CACHE_FILE, "r") as f:
        payload = json.load(f)
    return pd.DataFrame(
        {"Close": payload["close"]},
        index=pd.to_datetime(payload["index"]),
    )


# ---------------------------------------------------------------------------
# EIA fetch
# ---------------------------------------------------------------------------

def _get_eia_key() -> str:
    """
    Read EIA API key from Streamlit secrets.
    Returns empty string if the key is not configured,
    so the caller can fall through to FRED gracefully.
    """
    try:
        import streamlit as st
        return st.secrets.get("EIA_API_KEY", "")
    except Exception:
        return ""


def _fetch_from_eia(lookback_days: int) -> pd.DataFrame:
    """
    Fetch Brent crude daily prices from the EIA API v2.
    Raises an exception if the key is missing or the request fails.
    """
    key = _get_eia_key()
    if not key:
        raise ValueError("EIA_API_KEY not found in secrets.")

    url      = EIA_BASE_URL.format(series=EIA_SERIES, key=key)
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    data = response.json()

    # EIA v2 response structure:
    # data["response"]["data"] is a list of dicts, each with "period" and "value"
    rows = data.get("response", {}).get("data", [])
    if not rows:
        raise ValueError("EIA returned empty data.")

    df = pd.DataFrame(rows)[["period", "value"]]
    df.columns    = ["date", "Close"]
    df["date"]    = pd.to_datetime(df["date"])
    df["Close"]   = pd.to_numeric(df["Close"], errors="coerce")
    df            = df.dropna().set_index("date").sort_index()

    cutoff = pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
    df     = df[df.index >= cutoff]

    if len(df) < 30:
        raise ValueError(f"EIA returned only {len(df)} rows.")

    return df


# ---------------------------------------------------------------------------
# FRED fetch
# ---------------------------------------------------------------------------

def _fetch_from_fred(lookback_days: int) -> pd.DataFrame:
    """
    Fetch Brent crude daily prices from FRED. No key required.
    """
    response = requests.get(FRED_URL, timeout=15)
    response.raise_for_status()

    df = pd.read_csv(
        io.StringIO(response.text),
        parse_dates=["DATE"],
        index_col="DATE",
        na_values=".",
    )
    df.index.name = None
    df.columns    = ["Close"]
    df            = df.dropna().sort_index()

    cutoff = pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
    df     = df[df.index >= cutoff]

    if len(df) < 30:
        raise ValueError(f"FRED returned only {len(df)} rows.")

    return df


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def fetch_brent(lookback_days: int = 730) -> pd.DataFrame:
    """
    Return daily Brent crude closing prices as a DataFrame
    with a DatetimeIndex and a single column 'Close'.
    """
    # 1. Fresh cache — zero network calls
    if _cache_is_fresh():
        return _load_cache()

    # 2. EIA — primary source
    try:
        df = _fetch_from_eia(lookback_days)
        _save_cache(df)
        return df
    except Exception:
        pass

    # 3. FRED — fallback, no key needed
    try:
        df = _fetch_from_fred(lookback_days)
        _save_cache(df)
        return df
    except Exception:
        pass

    # 4. Stale cache — network is down
    if CACHE_FILE.exists():
        return _load_cache()

    # 5. Synthetic — app always runs
    end    = datetime.today()
    start  = end - timedelta(days=lookback_days)
    dates  = pd.date_range(start=start, end=end, freq="B")
    rng    = np.random.default_rng(0)
    closes = 97.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(dates))))
    return pd.DataFrame({"Close": closes}, index=dates)


# ---------------------------------------------------------------------------
# Parameter calibration
# ---------------------------------------------------------------------------

def calibrate_gbm(df: pd.DataFrame, window: int = 60) -> dict:
    """
    Calibrate annualised GBM drift (mu) and volatility (sigma)
    from the most recent `window` daily log-returns.
    """
    closes      = df["Close"].values.astype(float)
    log_returns = np.diff(np.log(closes[-window:]))
    mu_daily    = float(np.mean(log_returns))
    sigma_daily = float(np.std(log_returns, ddof=1))
    return {
        "mu":    mu_daily    * 252,
        "sigma": sigma_daily * np.sqrt(252),
    }


def calibrate_jumps(df: pd.DataFrame, jump_threshold: float = 0.04) -> dict:
    """
    Estimate Merton jump parameters.
    Days with |log-return| > jump_threshold are classified as jumps.
    """
    closes       = df["Close"].values.astype(float)
    lr           = np.diff(np.log(closes))
    jump_mask    = np.abs(lr) > jump_threshold
    jump_returns = lr[jump_mask]

    if len(jump_returns) < 3:
        return {"mu_j": 0.04, "sigma_j": 0.06}

    mu_j    = float(np.mean(jump_returns))
    sigma_j = float(np.std(jump_returns, ddof=1))
    return {"mu_j": mu_j, "sigma_j": max(sigma_j, 0.02)}


def calibrate_garch(df: pd.DataFrame) -> dict:
    """
    Fit GARCH(1,1) on log-returns. Returns annualised sigma and GARCH params.
    Falls back to calibrate_gbm if arch is unavailable.
    """
    try:
        from arch import arch_model
        closes = df["Close"].values.astype(float)
        lr = np.diff(np.log(closes)) * 100
        am = arch_model(lr, vol="Garch", p=1, q=1, dist="Normal", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        forecast = res.forecast(horizon=1, reindex=False)
        sigma_daily_pct = float(np.sqrt(forecast.variance.iloc[-1, 0]))
        sigma_daily = sigma_daily_pct / 100.0
        omega = float(res.params.get("omega", 0.0))
        alpha = float(res.params.get("alpha[1]", 0.0))
        beta  = float(res.params.get("beta[1]", 0.0))
        return {
            "sigma_garch": sigma_daily * np.sqrt(252),
            "omega": omega,
            "alpha": alpha,
            "beta":  beta,
            "sigma_daily": sigma_daily,
            "available": True,
        }
    except Exception:
        gbm = calibrate_gbm(df)
        sigma_d = gbm["sigma"] / np.sqrt(252)
        return {
            "sigma_garch": gbm["sigma"],
            "omega": sigma_d**2 * 0.05,
            "alpha": 0.10,
            "beta": 0.85,
            "sigma_daily": sigma_d,
            "available": False,
        }


def calibrate_jumps_mle(df: pd.DataFrame, lambda_init: float = 0.6) -> dict:
    """
    MLE for Merton jump-diffusion jump parameters.
    Falls back to threshold method if optimisation fails.
    """
    try:
        from scipy.optimize import minimize
        from scipy.stats import norm as sp_norm

        closes = df["Close"].values.astype(float)
        lr = np.diff(np.log(closes))
        mu_total = float(np.mean(lr))
        sigma_total = float(np.std(lr, ddof=1))

        def neg_log_lik(params):
            lam, mu_j, sigma_j, sigma_diff = params
            if sigma_diff <= 1e-6 or sigma_j <= 1e-6 or lam < 0:
                return 1e10
            dt = 1.0 / 252.0
            lam_dt = lam * dt
            log_lik = 0.0
            for r in lr:
                total = 0.0
                for k in range(6):
                    p_k = np.exp(-lam_dt) * (lam_dt ** k) / max(1, factorial(k))
                    mu_k = (mu_total - 0.5 * sigma_diff**2 + k * mu_j) * dt
                    sig_k = np.sqrt(sigma_diff**2 * dt + k * sigma_j**2 + 1e-12)
                    pdf_val = (1.0 / (sig_k * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((r - mu_k) / sig_k)**2)
                    total += p_k * pdf_val
                log_lik += np.log(max(total, 1e-300))
            return -log_lik

        x0 = [lambda_init, 0.0, sigma_total, sigma_total * 0.7]
        bounds = [(0.01, 10), (-0.5, 0.5), (0.005, 0.5), (0.005, 0.5)]
        result = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 300, "ftol": 1e-9})
        if result.success:
            lam_mle, mu_j_mle, sigma_j_mle, _ = result.x
            return {
                "mu_j": float(mu_j_mle),
                "sigma_j": float(max(sigma_j_mle, 0.02)),
                "lam_mle": float(lam_mle),
                "method": "mle",
            }
    except Exception:
        pass
    result = calibrate_jumps(df)
    result["method"] = "threshold"
    result["lam_mle"] = 0.6
    return result


def calibrate_ou(df: pd.DataFrame) -> dict:
    """
    Estimate OU mean-reversion parameters kappa and theta via OLS on log-prices.
    """
    closes = df["Close"].values.astype(float)
    log_p = np.log(closes)
    x = log_p[:-1]
    y = log_p[1:]
    b, a = np.polyfit(x, y, 1)
    b = float(np.clip(b, 0.001, 0.9999))
    dt = 1.0 / 252.0
    kappa = float(-np.log(b) / dt)
    theta = float(np.exp(a / (1.0 - b)))
    return {"kappa": max(kappa, 0.1), "theta": theta}


def get_latest_price(df: pd.DataFrame) -> float:
    return float(df["Close"].iloc[-1])


def get_price_history_since(df: pd.DataFrame, since_date: str) -> pd.DataFrame:
    return df[df.index >= pd.Timestamp(since_date)]
