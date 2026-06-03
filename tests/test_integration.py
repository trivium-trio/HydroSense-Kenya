"""Tests for numerical integration: trapezoidal and Simpson's rule."""

import pytest
import numpy as np
import sys
sys.path.insert(0, '..')

from src.numerical_methods import trapezoidal_rule, simpsons_rule


class TestTrapezoidal:
    def test_constant_function(self):
        """Integral of f(x)=5 over [0,4] should be 20."""
        y = np.array([5.0] * 5)
        assert abs(trapezoidal_rule(y, h=1.0) - 20.0) < 1e-10

    def test_linear_function(self):
        """Integral of f(x)=x over [0,4] should be 8. Trapezoidal is exact for linear."""
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert abs(trapezoidal_rule(y, h=1.0) - 8.0) < 1e-10

    def test_quadratic(self):
        """Integral of x^2 from 0 to 2 = 8/3 ≈ 2.6667."""
        x = np.linspace(0, 2, 101)
        y = x ** 2
        result = trapezoidal_rule(y, h=x[1] - x[0])
        assert abs(result - 8 / 3) < 0.001

    def test_sine(self):
        """Integral of sin(x) from 0 to pi = 2."""
        x = np.linspace(0, np.pi, 1001)
        y = np.sin(x)
        result = trapezoidal_rule(y, h=x[1] - x[0])
        assert abs(result - 2.0) < 1e-5


class TestSimpsons:
    def test_constant_function(self):
        y = np.array([5.0] * 5)
        assert abs(simpsons_rule(y, h=1.0) - 20.0) < 1e-10

    def test_linear_function(self):
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert abs(simpsons_rule(y, h=1.0) - 8.0) < 1e-10

    def test_quadratic_exact(self):
        """Simpson's rule is exact for polynomials up to degree 3."""
        y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])  # x^2 at x=0,1,2,3,4
        result = simpsons_rule(y, h=1.0)
        expected = 64 / 3  # exact integral of x^2 from 0 to 4
        assert abs(result - expected) < 1e-10

    def test_odd_intervals_raises(self):
        """Simpson's requires even number of intervals (odd points)."""
        y = np.array([1.0, 2.0, 3.0, 4.0])  # 3 intervals = odd
        with pytest.raises(ValueError):
            simpsons_rule(y, h=1.0)

    def test_more_accurate_than_trapezoidal(self):
        """Simpson's should be more accurate than trapezoidal for smooth functions."""
        x = np.linspace(0, np.pi, 21)  # coarse grid
        y = np.sin(x)
        h = x[1] - x[0]
        trap_err = abs(trapezoidal_rule(y, h) - 2.0)
        simp_err = abs(simpsons_rule(y, h) - 2.0)
        assert simp_err < trap_err
