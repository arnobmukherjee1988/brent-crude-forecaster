"""
Monte Carlo engine for Brent crude price forecasting.
Merton jump-diffusion model with geopolitical scenario weighting.
"""

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# Core simulation (Numba-compiled for speed)
# ---------------------------------------------------------------------------

@njit(parallel=True)
def _simulate_paths(
    S0: float,
    mu: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """
    Merton jump-diffusion paths.

    dS = mu*S*dt + sigma*S*dW + S*dJ
    where dJ is compound Poisson with log-normal jumps.

    Returns array of shape (n_paths, n_steps + 1).
    """
    np.random.seed(seed)
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    kappa = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    drift = (mu - 0.5 * sigma ** 2 - lam * kappa) * dt

    for i in prange(n_paths):
        paths[i, 0] = S0
        price = S0
        for t in range(n_steps):
            # Diffusion component
            z = np.random.standard_normal()
            diffusion = sigma * np.sqrt(dt) * z

            # Jump component: Poisson number of jumps
            n_jumps = 0
            u = np.random.uniform(0.0, 1.0)
            lam_dt = lam * dt
            p = np.exp(-lam_dt)
            cumulative = p
            while u > cumulative and n_jumps < 20:
                n_jumps += 1
                p *= lam_dt / n_jumps
                cumulative += p

            jump = 0.0
            for _ in range(n_jumps):
                y = mu_j + sigma_j * np.random.standard_normal()
                jump += y

            price = price * np.exp(drift + diffusion + jump)
            paths[i, t + 1] = price

    return paths


def run_scenario(
    S0: float,
    mu: float,
    sigma: float,
    lam_multiplier: float,
    mu_j: float,
    sigma_j: float,
    horizon_days: int,
    n_paths: int = 50_000,
    dt: float = 1.0 / 252.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Run simulation for one scenario. Returns terminal prices (shape: n_paths).
    """
    base_lam = 0.6          # ~0.6 jumps/year baseline (elevated war regime)
    lam = base_lam * lam_multiplier
    n_steps = horizon_days
    paths = _simulate_paths(
        S0, mu, sigma, lam, mu_j, sigma_j,
        dt, n_steps, n_paths, seed
    )
    return paths                # shape (n_paths, n_steps+1)


# ---------------------------------------------------------------------------
# Multi-scenario blended forecast
# ---------------------------------------------------------------------------

SCENARIOS = {
    "Ceasefire (1-2 months)":  {"lam_mult": 0.5,  "color": "#22c55e", "default_weight": 0.25},
    "Prolonged war (4-6 months)": {"lam_mult": 1.0, "color": "#f59e0b", "default_weight": 0.40},
    "Escalation (two straits)": {"lam_mult": 1.8,  "color": "#ef4444", "default_weight": 0.25},
    "Extreme shock ($200+)":    {"lam_mult": 3.0,  "color": "#7c3aed", "default_weight": 0.10},
}

HORIZONS = {
    "3 months":  63,
    "6 months":  126,
    "12 months": 252,
}


def blended_terminal_prices(
    S0: float,
    mu: float,
    sigma: float,
    mu_j: float,
    sigma_j: float,
    weights: dict,
    horizon_label: str,
    n_paths: int = 50_000,
) -> dict:
    """
    Returns a dict with per-scenario terminal price arrays and a blended sample.
    """
    horizon_days = HORIZONS[horizon_label]
    results = {}
    blended_prices = []
    blended_paths_list = []

    total_w = sum(weights.values())
    norm_weights = {k: v / total_w for k, v in weights.items()}

    for i, (name, cfg) in enumerate(SCENARIOS.items()):
        n_s = max(1, int(n_paths * norm_weights[name]))
        paths = run_scenario(
            S0=S0,
            mu=mu,
            sigma=sigma,
            lam_multiplier=cfg["lam_mult"],
            mu_j=mu_j,
            sigma_j=sigma_j,
            horizon_days=horizon_days,
            n_paths=n_s,
            seed=42 + i,
        )
        results[name] = {
            "terminal": paths[:, -1],
            "paths": paths,
            "color": cfg["color"],
            "weight": norm_weights[name],
        }
        blended_prices.append(paths[:, -1])
        blended_paths_list.append(paths)

    results["__blended__"] = np.concatenate(blended_prices)
    results["__blended_paths__"] = np.vstack(blended_paths_list)
    return results


def compute_stats(prices: np.ndarray, S0: float) -> dict:
    """Compute risk statistics for a terminal price array."""
    p5, p25, p50, p75, p95 = np.percentile(prices, [5, 25, 50, 75, 95])
    mean = float(np.mean(prices))
    var95 = float(p5)
    cvar95 = float(np.mean(prices[prices <= p5]))
    prob_100 = float(np.mean(prices >= 100.0)) * 100
    prob_147 = float(np.mean(prices >= 147.0)) * 100
    prob_200 = float(np.mean(prices >= 200.0)) * 100
    prob_drop = float(np.mean(prices <= S0 * 0.70)) * 100
    return {
        "mean": mean,
        "p5": float(p5),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "var95": var95,
        "cvar95": cvar95,
        "prob_above_100": prob_100,
        "prob_above_147": prob_147,
        "prob_above_200": prob_200,
        "prob_drop_30pct": prob_drop,
    }


def build_fan_data(paths: np.ndarray, step_every: int = 1) -> dict:
    """Compute percentile bands across all paths for fan chart."""
    n_paths, n_steps = paths.shape
    idx = np.arange(0, n_steps, step_every)
    sub = paths[:, idx]
    return {
        "steps": idx,
        "p5":   np.percentile(sub, 5,  axis=0),
        "p25":  np.percentile(sub, 25, axis=0),
        "p50":  np.percentile(sub, 50, axis=0),
        "p75":  np.percentile(sub, 75, axis=0),
        "p95":  np.percentile(sub, 95, axis=0),
    }
