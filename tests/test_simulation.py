"""Tests for simulation functions: Euler, RK4, Monte Carlo, and data cleaning."""

import pytest
import numpy as np
import sys
sys.path.insert(0, '..')

from src.simulation import compute_et, euler_simulate, rk4_simulate, monte_carlo_rainfall
from src.data_cleaning import detect_outliers_iqr, detect_outliers_zscore


class TestComputeET:
    def test_positive_et(self):
        et = compute_et(25.0, 2.0, 0.7, 65.0)
        assert et > 0

    def test_zero_floor(self):
        """ET should never be negative."""
        et = compute_et(0.0, 0.0, 0.0, 100.0)
        assert et == 0.0

    def test_vectorized(self):
        T = np.array([20, 25, 30])
        W = np.array([1, 2, 3])
        S = np.array([0.5, 0.7, 0.8])
        H = np.array([60, 65, 70])
        et = compute_et(T, W, S, H)
        assert len(et) == 3
        assert all(et >= 0)


class TestEulerSimulate:
    def test_constant_inputs(self):
        """With constant rain > ET and no drainage, moisture should rise."""
        S = euler_simulate(20.0, np.full(10, 5.0), np.full(10, 2.0), 0.0, 100.0)
        assert S[-1] > S[0]

    def test_no_rain_decreases_moisture(self):
        """With zero rain, moisture should decline."""
        S = euler_simulate(30.0, np.zeros(10), np.full(10, 3.0), 0.1, 40.0)
        assert S[-1] < S[0]

    def test_moisture_non_negative(self):
        """Moisture should never go below zero."""
        S = euler_simulate(5.0, np.zeros(30), np.full(30, 10.0), 0.2, 40.0)
        assert all(S >= 0)

    def test_output_length(self):
        S = euler_simulate(25.0, np.ones(30), np.ones(30), 0.1, 40.0)
        assert len(S) == 31  # N+1


class TestRK4Simulate:
    def test_matches_euler_approximately(self):
        """For smooth inputs, RK4 and Euler should give similar results."""
        rain = np.random.default_rng(0).uniform(0, 5, 30)
        et = np.full(30, 3.0)
        S_euler = euler_simulate(30.0, rain, et, 0.15, 41.0)
        S_rk4 = rk4_simulate(30.0, rain, et, 0.15, 41.0)
        assert np.max(np.abs(S_euler - S_rk4)) < 2.0

    def test_moisture_non_negative(self):
        S = rk4_simulate(5.0, np.zeros(30), np.full(30, 10.0), 0.2, 40.0)
        assert all(S >= 0)


class TestMonteCarlo:
    def test_shape(self):
        scenarios = monte_carlo_rainfall(np.array([1.0, 2.0, 0.0, 5.0]), n_scenarios=100)
        assert scenarios.shape == (100, 4)

    def test_non_negative(self):
        scenarios = monte_carlo_rainfall(np.array([3.0, 0.0, 7.0, 1.0]), n_scenarios=500)
        assert np.all(scenarios >= 0)

    def test_preserves_dry_days(self):
        """Some generated days should be dry (zero rainfall)."""
        rain = np.array([0, 0, 5, 3, 0, 2, 0, 0, 1, 0])
        scenarios = monte_carlo_rainfall(rain, n_scenarios=1000)
        dry_frac = np.mean(scenarios == 0)
        assert dry_frac > 0.1  # should have some dry days


class TestOutlierDetection:
    def test_iqr_catches_extreme(self):
        data = pd.Series([10, 11, 12, 13, 14, 100])
        mask = detect_outliers_iqr(data)
        assert mask.iloc[-1] == True

    def test_zscore_catches_extreme(self):
        data = pd.Series([10, 11, 12, 13, 14, 100])
        mask = detect_outliers_zscore(data, threshold=2.0)
        assert mask.iloc[-1] == True


# Need pandas for outlier tests
import pandas as pd
