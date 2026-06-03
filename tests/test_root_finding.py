"""Tests for root-finding methods: bisection, Newton-Raphson, secant."""

import pytest
import sys
sys.path.insert(0, '..')

from src.numerical_methods import bisection, newton_raphson, secant


# --- Test functions ---
def f_quadratic(x):
    return x**2 - 4  # roots at x = ±2

def f_quadratic_prime(x):
    return 2 * x

def f_cubic(x):
    return x**3 - x - 2  # root near x ≈ 1.5214

def f_cubic_prime(x):
    return 3 * x**2 - 1

def f_linear(x):
    return 3 * x - 6  # root at x = 2

def f_linear_prime(x):
    return 3


class TestBisection:
    def test_quadratic_root(self):
        result = bisection(f_quadratic, 0, 3, tol=1e-6)
        assert result['converged'] is True
        assert abs(result['root'] - 2.0) < 1e-5

    def test_negative_root(self):
        result = bisection(f_quadratic, -3, 0, tol=1e-6)
        assert result['converged'] is True
        assert abs(result['root'] - (-2.0)) < 1e-5

    def test_linear_root(self):
        result = bisection(f_linear, 0, 5, tol=1e-6)
        assert abs(result['root'] - 2.0) < 1e-5

    def test_invalid_bracket_raises(self):
        with pytest.raises(ValueError):
            bisection(f_quadratic, 3, 5)  # both positive, no sign change

    def test_history_recorded(self):
        result = bisection(f_quadratic, 0, 3, tol=1e-6)
        assert len(result['history']) == result['iterations']
        assert all('error' in h for h in result['history'])


class TestNewtonRaphson:
    def test_quadratic_root(self):
        result = newton_raphson(f_quadratic, f_quadratic_prime, 3.0, tol=1e-6)
        assert result['converged'] is True
        assert abs(result['root'] - 2.0) < 1e-5

    def test_cubic_root(self):
        result = newton_raphson(f_cubic, f_cubic_prime, 1.5, tol=1e-8)
        assert result['converged'] is True
        assert abs(f_cubic(result['root'])) < 1e-7

    def test_fewer_iterations_than_bisection(self):
        nr = newton_raphson(f_quadratic, f_quadratic_prime, 3.0, tol=1e-6)
        bi = bisection(f_quadratic, 0, 3, tol=1e-6)
        assert nr['iterations'] < bi['iterations']


class TestSecant:
    def test_quadratic_root(self):
        result = secant(f_quadratic, 1, 3, tol=1e-6)
        assert result['converged'] is True
        assert abs(result['root'] - 2.0) < 1e-5

    def test_cubic_root(self):
        result = secant(f_cubic, 1, 2, tol=1e-8)
        assert result['converged'] is True
        assert abs(f_cubic(result['root'])) < 1e-7

    def test_no_derivative_needed(self):
        """Secant should converge without requiring f'(x)."""
        result = secant(f_cubic, 1, 2, tol=1e-6)
        assert result['converged'] is True
