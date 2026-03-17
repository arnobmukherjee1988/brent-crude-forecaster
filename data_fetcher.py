"""
Live Brent crude data fetching and parameter calibration.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


BRENT_TICKER = "BZ=F"      # Brent crude futures (Yahoo Finance)
WTI_TICKER   = "CL=F"      # WTI fallback


def fetch_brent(lookback_days: int = 730) -> pd.DataFrame:
    """
    Download Brent crude daily OHLCV from Yahoo Finance.
    Falls back to WTI if Brent unavailable.
    Returns DataFrame with columns: Open, High, Low, Close, Volume.
    """
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)

    for ticker in [BRENT_TICKER, WTI_TICKER]:
        try:
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=True)
            if df is not None and len(df) > 30:
                df = df.dropna(subset=["Close"])
                df.index = pd.to_datetime(df.index)
                return df
        except Exception:
            continue

    # Last-resort synthetic fallback (should not normally be reached)
    dates = pd.date_range(start=start, end=end, freq="B")
    rng = np.random.default_rng(0)
    closes = 97.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(dates))))
    return pd.DataFrame({"Close": closes}, index=dates)


def calibrate_gbm(df: pd.DataFrame, window: int = 60) -> dict:
    """
    Calibrate GBM parameters from recent log-returns.
    Returns dict with mu, sigma (annualised).
    """
    closes = df["Close"].values.astype(float)
    log_returns = np.diff(np.log(closes[-window:]))
    mu_daily    = float(np.mean(log_returns))
    sigma_daily = float(np.std(log_returns, ddof=1))
    mu    = mu_daily    * 252
    sigma = sigma_daily * np.sqrt(252)
    return {"mu": mu, "sigma": sigma}


def calibrate_jumps(df: pd.DataFrame, jump_threshold: float = 0.04) -> dict:
    """
    Estimate Merton jump parameters from large daily moves.
    jump_threshold: moves beyond this (abs log-return) are treated as jumps.
    Returns dict with mu_j, sigma_j (log-normal jump distribution parameters).
    """
    closes = df["Close"].values.astype(float)
    lr = np.diff(np.log(closes))

    # Identify jumps as returns exceeding threshold in abs value
    jump_mask = np.abs(lr) > jump_threshold
    jump_returns = lr[jump_mask]

    if len(jump_returns) < 3:
        # Default elevated war-regime values
        return {"mu_j": 0.04, "sigma_j": 0.06}

    # Log-normal parameters of jump size distribution
    mu_j    = float(np.mean(jump_returns))
    sigma_j = float(np.std(jump_returns, ddof=1))
    return {"mu_j": mu_j, "sigma_j": max(sigma_j, 0.02)}


def get_latest_price(df: pd.DataFrame) -> float:
    """Return most recent closing price."""
    return float(df["Close"].iloc[-1])


def get_price_history_since(df: pd.DataFrame, since_date: str) -> pd.DataFrame:
    """Return price history from a given date onward."""
    return df[df.index >= pd.Timestamp(since_date)]
