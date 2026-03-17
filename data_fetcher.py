"""
Live Brent crude data fetching and parameter calibration.

Source priority:
  1. Local disk cache        (if less than 15 minutes old, no network call)
  2. Stooq CSV               (primary, no key, no rate limit, ~15 min delay)
  3. Yahoo Finance           (fallback, no key, may rate-limit)
  4. EIA API                 (fallback, requires EIA_API_KEY in .streamlit/secrets.toml)
  5. FRED CSV                (fallback, no key required, 1-2 day lag)
  6. Stale disk cache        (if all network sources fail)
  7. Synthetic series        (absolute last resort, app still runs)
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

STOOQ_URL    = "https://stooq.com/q/d/l/?s=lco.f&i=d"
YAHOO_TICKER = "BZ=F"
EIA_SERIES   = "PET.RBRTE.D"
EIA_BASE_URL = "https://api.eia.gov/v2/seriesid/{series}?api_key={key}&data[]=value&frequency=daily&sort[0][column]=period&sort[0][direction]=asc&length=1000"
FRED_URL     = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"

CACHE_FILE        = Path(__file__).parent / ".streamlit" / "price_cache.json"
CACHE_TTL_SECONDS = 900


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
# Stooq fetch (primary)
# ---------------------------------------------------------------------------

def _fetch_from_stooq(lookback_days: int) -> pd.DataFrame:
    headers  = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(STOOQ_URL, headers=headers, timeout=15)
    response.raise_for_status()

    text = response.text.strip()
    if not text or "No data" in text or len(text) < 50:
        raise ValueError("Stooq returned empty or invalid data.")

    df = pd.read_csv(io.StringIO(text), parse_dates=["Date"], index_col="Date")

    if "Close" not in df.columns:
        raise ValueError(f"Stooq missing Close column. Got: {df.columns.tolist()}")

    df     = df[["Close"]].dropna().sort_index()
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
    df     = df[df.index >= cutoff]

    if len(df) < 30:
        raise ValueError(f"Stooq returned only {len(df)} rows.")

    return df


# ---------------------------------------------------------------------------
# Yahoo Finance fetch (second source)
# ---------------------------------------------------------------------------

def _fetch_from_yahoo(lookback_days: int) -> pd.DataFrame:
    try:
        import yfinance as yf
        time.sleep(1)
        ticker = yf.Ticker(YAHOO_TICKER)
        raw    = ticker.history(period=f"{lookback_days}d", interval="1d", auto_adjust=True)
        if raw.empty:
            raise ValueError("yfinance returned empty data.")
        df = raw[["Close"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)

    except ImportError:
        end  = datetime.today()
        p1   = int((end - timedelta(days=lookback_days + 10)).timestamp())
        p2   = int(end.timestamp())
        url  = (
            f"https://query1.finance.yahoo.com/v7/finance/download/{YAHOO_TICKER}"
            f"?period1={p1}&period2={p2}&interval=1d&events=history"
        )
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), parse_dates=["Date"], index_col="Date")
        df = df[["Close"]]

    df     = df.dropna().sort_index()
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
    df     = df[df.index >= cutoff]

    if len(df) < 30:
        raise ValueError(f"Yahoo returned only {len(df)} rows.")

    return df


# ---------------------------------------------------------------------------
# EIA fetch (third source)
# ---------------------------------------------------------------------------

def _get_eia_key() -> str:
    try:
        import streamlit as st
        return st.secrets.get("EIA_API_KEY", "")
    except Exception:
        return ""


def _fetch_from_eia(lookback_days: int) -> pd.DataFrame:
    key = _get_eia_key()
    if not key:
        raise ValueError("EIA_API_KEY not found.")

    response = requests.get(EIA_BASE_URL.format(series=EIA_SERIES, key=key), timeout=15)
    response.raise_for_status()

    rows = response.json().get("response", {}).get("data", [])
    if not rows:
        raise ValueError("EIA returned empty data.")

    df = pd.DataFrame(rows)[["period", "value"]]
    df.columns  = ["date", "Close"]
    df["date"]  = pd.to_datetime(df["date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df          = df.dropna().set_index("date").sort_index()

    cutoff = pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
    df     = df[df.index >= cutoff]

    if len(df) < 30:
        raise ValueError(f"EIA returned only {len(df)} rows.")

    return df


# ---------------------------------------------------------------------------
# FRED fetch (fourth source)
# ---------------------------------------------------------------------------

def _fetch_from_fred(lookback_days: int) -> pd.DataFrame:
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
    if _cache_is_fresh():
        return _load_cache()

    for fetcher in [
        lambda: _fetch_from_stooq(lookback_days),
        lambda: _fetch_from_yahoo(lookback_days),
        lambda: _fetch_from_eia(lookback_days),
        lambda: _fetch_from_fred(lookback_days),
    ]:
        try:
            df = fetcher()
            _save_cache(df)
            return df
        except Exception:
            pass

    if CACHE_FILE.exists():
        return _load_cache()

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
    closes      = df["Close"].values.astype(float)
    log_returns = np.diff(np.log(closes[-window:]))
    mu_daily    = float(np.mean(log_returns))
    sigma_daily = float(np.std(log_returns, ddof=1))
    return {
        "mu":    mu_daily    * 252,
        "sigma": sigma_daily * np.sqrt(252),
    }


def calibrate_jumps(df: pd.DataFrame, jump_threshold: float = 0.05) -> dict:
    """
    Estimate Merton jump parameters using threshold method.
    Days with |log-return| > jump_threshold are classified as jumps.
    Default threshold 5% gives physically sensible parameters.
    """
    closes       = df["Close"].values.astype(float)
    lr           = np.diff(np.log(closes))
    jump_mask    = np.abs(lr) > jump_threshold
    jump_returns = lr[jump_mask]

    if len(jump_returns) < 3:
        return {
            "mu_j":    0.00,
            "sigma_j": 0.06,
            "lam_mle": 3.87,
            "method":  "threshold_default",
        }

    mu_j = float(np.mean(jump_returns))
    sigma_j = float(np.std(jump_returns, ddof=1))
    lam     = float(jump_mask.sum() / len(lr) * 252)

    return {
        "mu_j":    mu_j,
        "sigma_j": max(sigma_j, 0.02),
        "lam_mle": lam,
        "method":  "threshold",
    }


def calibrate_garch(df: pd.DataFrame) -> dict:
    try:
        from arch import arch_model
        closes = df["Close"].values.astype(float)
        lr = np.diff(np.log(closes)) * 100
        am = arch_model(lr, vol="Garch", p=1, q=1, dist="Normal", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        forecast = res.forecast(horizon=1, reindex=False)
        sigma_daily_pct = float(np.sqrt(forecast.variance.iloc[-1, 0]))
        sigma_daily = sigma_daily_pct / 100.0
        return {
            "sigma_garch": sigma_daily * np.sqrt(252),
            "omega":       float(res.params.get("omega", 0.0)),
            "alpha":       float(res.params.get("alpha[1]", 0.0)),
            "beta":        float(res.params.get("beta[1]", 0.0)),
            "sigma_daily": sigma_daily,
            "available":   True,
        }
    except Exception:
        gbm = calibrate_gbm(df)
        sigma_d = gbm["sigma"] / np.sqrt(252)
        return {
            "sigma_garch": gbm["sigma"],
            "omega":       sigma_d**2 * 0.05,
            "alpha":       0.10,
            "beta":        0.85,
            "sigma_daily": sigma_d,
            "available":   False,
        }


def calibrate_jumps_mle(df: pd.DataFrame, lambda_init: float = 0.6) -> dict:
    """Delegates to threshold method. Kept for API compatibility."""
    return calibrate_jumps(df, jump_threshold=0.05)


def calibrate_ou(df: pd.DataFrame) -> dict:
    closes = df["Close"].values.astype(float)
    log_p  = np.log(closes)
    b, a   = np.polyfit(log_p[:-1], log_p[1:], 1)
    b      = float(np.clip(b, 0.001, 0.9999))
    dt     = 1.0 / 252.0
    kappa  = float(-np.log(b) / dt)
    theta  = float(np.exp(a / (1.0 - b)))
    return {"kappa": max(kappa, 0.1), "theta": theta}


def get_latest_price(df: pd.DataFrame) -> float:
    return float(df["Close"].iloc[-1])


def get_price_history_since(df: pd.DataFrame, since_date: str) -> pd.DataFrame:
    return df[df.index >= pd.Timestamp(since_date)]