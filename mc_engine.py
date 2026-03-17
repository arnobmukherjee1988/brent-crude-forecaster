"""
Monte Carlo engine for Brent crude price forecasting.
Merton jump-diffusion model with scenario weighting.
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
    Merton jump-diffusion paths with antithetic variates.

    dS = mu*S*dt + sigma*S*dW + S*dJ
    where dJ is compound Poisson with log-normal jumps.

    Returns array of shape (n_paths, n_steps + 1).
    """
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    kappa = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    drift = (mu - 0.5 * sigma ** 2 - lam * kappa) * dt

    for i in prange(n_paths):
        np.random.seed(seed + i)
        paths[i, 0] = S0
        price = S0
        for t in range(n_steps):
            # Antithetic variates: use z and -z for even/odd paths
            z_raw = np.random.standard_normal()
            z = z_raw if (i % 2 == 0) else -z_raw

            # Diffusion component
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


@njit(parallel=True)
def _simulate_paths_gbm_only(
    S0: float,
    mu: float,
    sigma: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Plain GBM — no jumps."""
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    drift = (mu - 0.5 * sigma ** 2) * dt
    for i in prange(n_paths):
        np.random.seed(seed + i)
        paths[i, 0] = S0
        price = S0
        for t in range(n_steps):
            z = np.random.standard_normal()
            price = price * np.exp(drift + sigma * np.sqrt(dt) * z)
            paths[i, t + 1] = price
    return paths


@njit(parallel=True)
def _simulate_paths_ou(
    S0: float,
    kappa: float,
    theta_log: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """OU mean-reverting jump-diffusion (in log-price space)."""
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    kappa_e = np.exp(-kappa * dt)
    kappa_j = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0

    for i in prange(n_paths):
        np.random.seed(seed + i)
        paths[i, 0] = S0
        price = S0
        for t in range(n_steps):
            log_p = np.log(max(price, 1e-6))
            mean_log = kappa_e * log_p + (1.0 - kappa_e) * theta_log
            var_log = sigma ** 2 * (1.0 - kappa_e ** 2) / (2.0 * kappa + 1e-9)
            z = np.random.standard_normal()
            diffusion = np.sqrt(max(var_log, 0.0)) * z

            n_jumps = 0
            u = np.random.uniform(0.0, 1.0)
            lam_dt = lam * dt
            p_j = np.exp(-lam_dt)
            cumulative = p_j
            while u > cumulative and n_jumps < 20:
                n_jumps += 1
                p_j *= lam_dt / n_jumps
                cumulative += p_j

            jump = 0.0
            for _ in range(n_jumps):
                y = mu_j + sigma_j * np.random.standard_normal()
                jump += y

            price = np.exp(mean_log + diffusion + jump)
            paths[i, t + 1] = price
    return paths


def _simulate_paths_garch(
    S0: float,
    mu: float,
    omega: float,
    alpha: float,
    beta: float,
    sigma0_daily: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    dt: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """
    GARCH(1,1) + Merton jump-diffusion. Pure NumPy (not JIT-compiled).

    The GARCH model is calibrated on percentage log-returns, so omega, alpha, beta
    are in %^2 units. We maintain h in %^2 internally and convert to decimal for
    the price evolution step.

    Discrete-time Ito step (GARCH):
        log(S_{t+1}/S_t) = mu*dt - 0.5*sigma_t_dec^2 + sigma_t_dec*z + jump
    where sigma_t_dec = sqrt(h_pct2) / 100 is the per-day (not annualised) sigma.
    """
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    kappa = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    # Convert initial daily sigma to %^2 (consistent with GARCH calibration space)
    h_init_pct2 = (sigma0_daily * 100.0) ** 2

    for i in range(n_paths):
        paths[i, 0] = S0
        price = S0
        h_pct2 = h_init_pct2
        for t in range(n_steps):
            sigma_t_pct = np.sqrt(max(h_pct2, 1e-10))  # daily sigma in %
            sigma_t_dec = sigma_t_pct / 100.0            # daily sigma in decimal

            # Ito drift: mu is annualised, sigma_t_dec^2 is daily variance
            drift = mu * dt - 0.5 * sigma_t_dec ** 2 - lam * kappa * dt

            z = rng.standard_normal()
            # Diffusion: sigma_t_dec * z (per-day step — no extra sqrt(dt))
            diffusion = sigma_t_dec * z

            n_jumps = int(rng.poisson(lam * dt))
            jump = float(np.sum(rng.normal(mu_j, sigma_j, n_jumps))) if n_jumps > 0 else 0.0

            price = price * np.exp(drift + diffusion + jump)
            paths[i, t + 1] = max(price, 1e-6)

            # GARCH update: stays in %^2 to match calibrated omega/alpha/beta
            h_pct2 = omega + alpha * (z * sigma_t_pct) ** 2 + beta * h_pct2
            h_pct2 = max(h_pct2, 1e-10)

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
    base_lam = 0.6          # ~0.6 jumps/year baseline
    lam = base_lam * lam_multiplier
    n_steps = horizon_days
    paths = _simulate_paths(
        S0, mu, sigma, lam, mu_j, sigma_j,
        dt, n_steps, n_paths, seed
    )
    return paths                # shape (n_paths, n_steps+1)


def run_scenario_gbm(
    S0: float, mu: float, sigma: float,
    horizon_days: int, n_paths: int = 10_000,
    dt: float = 1.0 / 252.0, seed: int = 42,
) -> np.ndarray:
    """Plain GBM model."""
    return _simulate_paths_gbm_only(S0, mu, sigma, dt, horizon_days, n_paths, seed)


def run_scenario_ou(
    S0: float, mu: float, sigma: float,
    kappa: float, theta: float,
    lam_multiplier: float, mu_j: float, sigma_j: float,
    horizon_days: int, n_paths: int = 10_000,
    dt: float = 1.0 / 252.0, seed: int = 42,
) -> np.ndarray:
    """OU mean-reverting jump-diffusion model."""
    base_lam = 0.6
    lam = base_lam * lam_multiplier
    theta_log = np.log(max(theta, 1e-6))
    return _simulate_paths_ou(S0, kappa, theta_log, sigma, lam, mu_j, sigma_j, dt, horizon_days, n_paths, seed)


def run_scenario_garch(
    S0: float, mu: float,
    omega: float, alpha: float, beta: float, sigma0_daily: float,
    lam_multiplier: float, mu_j: float, sigma_j: float,
    horizon_days: int, n_paths: int = 10_000,
    dt: float = 1.0 / 252.0, seed: int = 42,
) -> np.ndarray:
    """GARCH-Merton model."""
    base_lam = 0.6
    lam = base_lam * lam_multiplier
    return _simulate_paths_garch(S0, mu, omega, alpha, beta, sigma0_daily, lam, mu_j, sigma_j, dt, horizon_days, n_paths, seed)


# ---------------------------------------------------------------------------
# Multi-scenario blended forecast
# ---------------------------------------------------------------------------

SCENARIOS = {
    "Low disruption":          {"lam_mult": 0.5,  "color": "#22c55e", "default_weight": 0.25},
    "Base case":               {"lam_mult": 1.0,  "color": "#f59e0b", "default_weight": 0.40},
    "High disruption":         {"lam_mult": 1.8,  "color": "#ef4444", "default_weight": 0.25},
    "Extreme shock":           {"lam_mult": 3.0,  "color": "#7c3aed", "default_weight": 0.10},
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

    # Compute path counts for each scenario
    path_counts = {}
    total_allocated = 0
    for name in SCENARIOS:
        n_s = max(1, int(n_paths * norm_weights[name]))
        path_counts[name] = n_s
        total_allocated += n_s

    # Adjust "Base case" to make total exactly n_paths
    if total_allocated != n_paths:
        adjustment = n_paths - total_allocated
        path_counts["Base case"] += adjustment

    for i, (name, cfg) in enumerate(SCENARIOS.items()):
        n_s = path_counts[name]
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
