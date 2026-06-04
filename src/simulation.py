"""
simulation.py
Soil moisture simulation using Euler and Runge-Kutta methods, and Monte Carlo analysis.
"""

import numpy as np


def compute_et(T, W, Solar, H):
    """Simplified evapotranspiration: ET = max(0, 0.12*T + 0.35*W + 2.4*Solar - 0.025*H)"""
    et = 0.12 * T + 0.35 * W + 2.4 * Solar - 0.025 * H
    return max(0.0, float(et)) if np.isscalar(et) else np.maximum(0.0, et)


def soil_moisture_derivative(S, t, R, ET, dc, fc):
    """
    dS/dt for soil moisture ODE:
        dS/dt = R - ET - D(S)
    where D(S) = dc * max(0, S - fc)
    """
    drainage = dc * max(0.0, S - fc)
    return R - ET - drainage


def euler_simulate(S0, rainfall, et_series, dc, fc, dt=1.0):
    """
    Simulate soil moisture over N days using Euler method.

    Parameters
    ----------
    S0 : float — initial soil moisture (%)
    rainfall : array — daily rainfall values
    et_series : array — daily ET values
    dc : float — drainage coefficient
    fc : float — field capacity (%)
    dt : float — time step (days)

    Returns array of length N+1 (including initial condition).
    """
    N = len(rainfall)
    S = np.zeros(N + 1)
    S[0] = S0
    for t in range(N):
        dSdt = soil_moisture_derivative(S[t], t, rainfall[t], et_series[t], dc, fc)
        S[t + 1] = max(0.0, S[t] + dt * dSdt)
    return S


def rk4_simulate(S0, rainfall, et_series, dc, fc, dt=1.0):
    """
    Simulate soil moisture over N days using 4th-order Runge-Kutta method.
    Returns array of length N+1.
    """
    N = len(rainfall)
    S = np.zeros(N + 1)
    S[0] = S0

    for t in range(N):
        R, ET = rainfall[t], et_series[t]
        k1 = dt * soil_moisture_derivative(S[t], t, R, ET, dc, fc)
        k2 = dt * soil_moisture_derivative(S[t] + k1 / 2, t + dt / 2, R, ET, dc, fc)
        k3 = dt * soil_moisture_derivative(S[t] + k2 / 2, t + dt / 2, R, ET, dc, fc)
        k4 = dt * soil_moisture_derivative(S[t] + k3, t + dt, R, ET, dc, fc)
        S[t + 1] = max(0.0, S[t] + (k1 + 2 * k2 + 2 * k3 + k4) / 6)
    return S


def monte_carlo_rainfall(observed_rainfall, n_scenarios=1000, seed=42):
    """
    Generate Monte Carlo rainfall scenarios by sampling from a distribution
    fitted to observed data.

    Returns array of shape (n_scenarios, n_days).
    """
    rng = np.random.default_rng(seed)
    mean_r = observed_rainfall.mean()
    std_r = observed_rainfall.std()
    n_days = len(observed_rainfall)

    # Use gamma distribution (non-negative, right-skewed like rainfall)
    # Fit shape and scale from observed mean and variance
    if std_r > 0 and mean_r > 0:
        shape = (mean_r / std_r) ** 2
        scale = (std_r ** 2) / mean_r
    else:
        shape, scale = 1.0, mean_r

    scenarios = rng.gamma(shape, scale, size=(n_scenarios, n_days))

    # Preserve dry-day probability from observed data
    p_dry = np.mean(observed_rainfall == 0)
    dry_mask = rng.random(size=(n_scenarios, n_days)) < p_dry
    scenarios[dry_mask] = 0.0

    return scenarios
