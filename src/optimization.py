"""
optimization.py
Irrigation schedule optimization — minimize water use while respecting moisture constraints.
"""

import numpy as np
from simulation import soil_moisture_derivative


def greedy_irrigation_schedule(S0, rainfall, et_series, dc, fc, min_moisture, target_moisture, dt=1.0):
    """
    Greedy irrigation optimizer: apply water only when next-day moisture
    would drop below min_moisture, and only enough to reach target_moisture.

    Returns
    -------
    irrigation : array of daily irrigation amounts
    moisture : array of soil moisture trajectory (length N+1)
    """
    N = len(rainfall)
    irrigation = np.zeros(N)
    moisture = np.zeros(N + 1)
    moisture[0] = S0

    for t in range(N):
        # Project next-day moisture without irrigation
        dSdt = soil_moisture_derivative(moisture[t], t, rainfall[t], et_series[t], dc, fc)
        S_proj = max(0.0, moisture[t] + dt * dSdt)

        if S_proj < min_moisture:
            # Irrigate just enough to reach target
            deficit = target_moisture - S_proj
            irrigation[t] = max(0.0, deficit)
            moisture[t + 1] = max(0.0, S_proj + irrigation[t])
        else:
            moisture[t + 1] = S_proj

    return irrigation, moisture


def optimize_schedule_gradient(S0, rainfall, et_series, dc, fc, min_moisture,
                                target_moisture, max_daily=15.0,
                                learning_rate=0.1, n_iterations=500):
    """
    Gradient-descent irrigation optimizer.
    Minimizes total water use subject to a penalty for moisture below min_moisture.

    Returns
    -------
    irrigation : optimized daily irrigation array
    moisture : resulting soil moisture trajectory
    cost_history : list of cost values per iteration
    """
    N = len(rainfall)
    irrigation = np.zeros(N)
    cost_history = []
    penalty_weight = 50.0

    for iteration in range(n_iterations):
        # Forward pass: simulate moisture
        moisture = np.zeros(N + 1)
        moisture[0] = S0
        for t in range(N):
            R_eff = rainfall[t] + irrigation[t]
            dSdt = soil_moisture_derivative(moisture[t], t, R_eff, et_series[t], dc, fc)
            moisture[t + 1] = max(0.0, moisture[t] + dSdt)

        # Compute cost: total water + penalty for violations
        water_cost = np.sum(irrigation)
        violations = np.maximum(0, min_moisture - moisture[1:])
        penalty = penalty_weight * np.sum(violations ** 2)
        total_cost = water_cost + penalty
        cost_history.append(total_cost)

        # Numerical gradient for each day
        grad = np.zeros(N)
        eps = 0.01
        for t in range(N):
            irrigation[t] += eps
            m_plus = np.zeros(N + 1)
            m_plus[0] = S0
            for k in range(N):
                R_eff = rainfall[k] + irrigation[k]
                dSdt = soil_moisture_derivative(m_plus[k], k, R_eff, et_series[k], dc, fc)
                m_plus[k + 1] = max(0.0, m_plus[k] + dSdt)

            w_plus = np.sum(irrigation)
            v_plus = np.maximum(0, min_moisture - m_plus[1:])
            cost_plus = w_plus + penalty_weight * np.sum(v_plus ** 2)
            grad[t] = (cost_plus - total_cost) / eps
            irrigation[t] -= eps

        # Update
        irrigation = irrigation - learning_rate * grad
        irrigation = np.clip(irrigation, 0, max_daily)

    # Final simulation
    moisture = np.zeros(N + 1)
    moisture[0] = S0
    for t in range(N):
        R_eff = rainfall[t] + irrigation[t]
        dSdt = soil_moisture_derivative(moisture[t], t, R_eff, et_series[t], dc, fc)
        moisture[t + 1] = max(0.0, moisture[t] + dSdt)

    return irrigation, moisture, cost_history
