"""Tests for linear system solvers: Gaussian elimination and LU decomposition."""

import pytest
import numpy as np
import sys
sys.path.insert(0, '..')

from src.numerical_methods import gaussian_elimination, lu_decomposition, lu_solve


class TestGaussianElimination:
    def test_simple_2x2(self):
        A = [[2, 1], [1, 3]]
        b = [5, 10]
        x = gaussian_elimination(A, b)
        assert abs(x[0] - 1.0) < 1e-10
        assert abs(x[1] - 3.0) < 1e-10

    def test_3x3_system(self):
        A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        b = [6, 15, 25]
        x = gaussian_elimination(A, b)
        # Verify Ax = b
        residual = np.dot(A, x) - b
        assert np.max(np.abs(residual)) < 1e-10

    def test_matches_numpy(self):
        A = [[3, 1, -1], [2, 4, 1], [-1, 2, 5]]
        b = [4, 1, 1]
        x_ours = gaussian_elimination(A, b)
        x_numpy = np.linalg.solve(A, b)
        assert np.max(np.abs(x_ours - x_numpy)) < 1e-10

    def test_irrigation_allocation(self):
        """The actual 3-zone water allocation problem from Level 3."""
        A = [[1, 1, 1], [1/120, -1/90, 0], [0, 1/90, -1/180]]
        b = [50, 0, 0]  # simplified equal-deficit case
        x = gaussian_elimination(A, b)
        assert abs(sum(x) - 50.0) < 1e-10
        assert all(xi >= -1e-10 for xi in x)  # no negative allocation


class TestLUDecomposition:
    def test_decomposition_valid(self):
        A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
        L, U, P = lu_decomposition(A)
        # Verify PA = LU
        PA = P @ A
        LU = L @ U
        assert np.max(np.abs(PA - LU)) < 1e-10

    def test_lu_solve_matches_gaussian(self):
        A = [[3, 1, -1], [2, 4, 1], [-1, 2, 5]]
        b = [4, 1, 1]
        x_gauss = gaussian_elimination(A, b)
        L, U, P = lu_decomposition(A)
        x_lu = lu_solve(L, U, P, b)
        assert np.max(np.abs(x_gauss - x_lu)) < 1e-10

    def test_identity_matrix(self):
        A = np.eye(3)
        b = [1, 2, 3]
        L, U, P = lu_decomposition(A)
        x = lu_solve(L, U, P, b)
        assert np.allclose(x, b)
