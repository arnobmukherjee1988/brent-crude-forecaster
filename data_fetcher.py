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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EIA series: Europe Brent Spot Price FOB (Dollars per Barrel), daily
EIA_SERIES   = "PET.RBRTE.D"
EIA_BASE_URL = "https://api.eia.gov/v2/seriesid/{series}?api_key={key}&data[]=value&frequency=daily&sort[0][column]=period&sort[0][direction]=asc&length=1000"

# FRED fallback: same series, no key required
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"

CACHE_FILE        = Path(".streamlit/price_cache.json")
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


def get_latest_price(df: pd.DataFrame) -> float:
    return float(df["Close"].iloc[-1])


def get_price_history_since(df: pd.DataFrame, since_date: str) -> pd.DataFrame:
    return df[df.index >= pd.Timestamp(since_date)]
