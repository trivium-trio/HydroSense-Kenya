"""
numerical_methods.py
Core numerical methods for HydroSense-Kenya: root finding, differentiation,
integration, and linear systems — all implemented manually.
"""

import numpy as np


# ─────────────────────────────────────────────
# 1. ROOT FINDING
# ─────────────────────────────────────────────

def bisection(f, a, b, tol=1e-6, max_iter=100):
    """
    Bisection method for finding a root of f(x) = 0 on [a, b].
    Requires f(a) and f(b) to have opposite signs.

    Returns dict with keys: root, iterations, error, converged, history.
    """
    if f(a) * f(b) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")

    history = []
    for i in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = f(c)
        error = abs(b - a) / 2.0
        history.append({'iteration': i, 'root_estimate': c, 'f(x)': fc, 'error': error})

        if abs(fc) < tol or error < tol:
            return {'root': c, 'iterations': i, 'error': error, 'converged': True, 'history': history}

        if f(a) * fc < 0:
            b = c
        else:
            a = c

    return {'root': c, 'iterations': max_iter, 'error': error, 'converged': False, 'history': history}


def newton_raphson(f, f_prime, x0, tol=1e-6, max_iter=100):
    """
    Newton-Raphson method for finding a root of f(x) = 0.
    Requires the derivative f'(x).

    Returns dict with keys: root, iterations, error, converged, history.
    """
    x = x0
    history = []
    for i in range(1, max_iter + 1):
        fx = f(x)
        fpx = f_prime(x)
        if abs(fpx) < 1e-15:
            return {'root': x, 'iterations': i, 'error': abs(fx), 'converged': False, 'history': history}

        x_new = x - fx / fpx
        error = abs(x_new - x)
        history.append({'iteration': i, 'root_estimate': x_new, 'f(x)': f(x_new), 'error': error})

        if error < tol or abs(f(x_new)) < tol:
            return {'root': x_new, 'iterations': i, 'error': error, 'converged': True, 'history': history}
        x = x_new

    return {'root': x, 'iterations': max_iter, 'error': error, 'converged': False, 'history': history}


def secant(f, x0, x1, tol=1e-6, max_iter=100):
    """
    Secant method for finding a root of f(x) = 0.
    Does not require the derivative.

    Returns dict with keys: root, iterations, error, converged, history.
    """
    history = []
    for i in range(1, max_iter + 1):
        f0, f1 = f(x0), f(x1)
        if abs(f1 - f0) < 1e-15:
            return {'root': x1, 'iterations': i, 'error': abs(f1), 'converged': False, 'history': history}

        x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
        error = abs(x_new - x1)
        history.append({'iteration': i, 'root_estimate': x_new, 'f(x)': f(x_new), 'error': error})

        if error < tol or abs(f(x_new)) < tol:
            return {'root': x_new, 'iterations': i, 'error': error, 'converged': True, 'history': history}

        x0, x1 = x1, x_new

    return {'root': x1, 'iterations': max_iter, 'error': error, 'converged': False, 'history': history}


# ─────────────────────────────────────────────
# 2. NUMERICAL DIFFERENTIATION
# ─────────────────────────────────────────────

def forward_difference(y, h):
    """Forward difference approximation of dy/dt. Returns array of length len(y)-1."""
    y = np.asarray(y, dtype=float)
    return (y[1:] - y[:-1]) / h


def backward_difference(y, h):
    """Backward difference approximation of dy/dt. Returns array of length len(y)-1."""
    y = np.asarray(y, dtype=float)
    return (y[1:] - y[:-1]) / h


def central_difference(y, h):
    """Central difference approximation of dy/dt. Returns array of length len(y)-2."""
    y = np.asarray(y, dtype=float)
    return (y[2:] - y[:-2]) / (2 * h)


# ─────────────────────────────────────────────
# 3. NUMERICAL INTEGRATION
# ─────────────────────────────────────────────

def trapezoidal_rule(y, h):
    """Composite trapezoidal rule for uniformly spaced data."""
    y = np.asarray(y, dtype=float)
    n = len(y) - 1
    return h * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2)


def simpsons_rule(y, h):
    """
    Composite Simpson's 1/3 rule for uniformly spaced data.
    Requires an odd number of points (even number of intervals).
    """
    y = np.asarray(y, dtype=float)
    n = len(y) - 1
    if n % 2 != 0:
        raise ValueError(f"Simpson's rule requires even number of intervals, got {n}")
    return (h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))


# ─────────────────────────────────────────────
# 4. LINEAR SYSTEMS
# ─────────────────────────────────────────────

def gaussian_elimination(A, b):
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.
    A: n×n coefficient matrix, b: n-vector.
    Returns solution vector x.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # Augmented matrix
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # Forward elimination with partial pivoting
    for k in range(n):
        # Find pivot
        max_row = k + np.argmax(np.abs(Ab[k:, k]))
        Ab[[k, max_row]] = Ab[[max_row, k]]

        if abs(Ab[k, k]) < 1e-12:
            raise ValueError("Matrix is singular or near-singular")

        for i in range(k + 1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= factor * Ab[k, k:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]

    return x


def lu_decomposition(A):
    """
    LU decomposition with partial pivoting: PA = LU.
    Returns (L, U, P) where P is a permutation matrix.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    P = np.eye(n)

    for k in range(n):
        max_row = k + np.argmax(np.abs(U[k:, k]))
        if max_row != k:
            U[[k, max_row]] = U[[max_row, k]]
            P[[k, max_row]] = P[[max_row, k]]
            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return L, U, P


def lu_solve(L, U, P, b):
    """Solve Ax = b given LU decomposition PA = LU."""
    b = np.array(b, dtype=float)
    Pb = P @ b
    n = len(b)

    # Forward substitution: Ly = Pb
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])

    # Back substitution: Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x
