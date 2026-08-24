"""
American option pricing via the Longstaff-Schwartz Least-Squares Monte Carlo
(LSM) algorithm (Longstaff & Schwartz, 2001).

Idea: simulate full GBM paths, then work backwards from maturity, at each
exercise date regressing the continuation value on the current spot (using
only in-the-money paths) to decide whether early exercise is optimal.
"""

import numpy as np


def simulate_paths(S0: float, r: float, q: float, sigma: float, T: float, n_steps: int, n_paths: int, seed=None):
    """Simulate GBM paths on a discrete time grid, shape (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))

    increments = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    S = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))
    return S


def _payoff(S, K, option_type):
    if option_type == "call":
        return np.maximum(S - K, 0.0)
    elif option_type == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def lsm_price(
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    option_type: str = "put",
    n_steps: int = 50,
    n_paths: int = 50_000,
    poly_degree: int = 2,
    seed: int | None = None,
):
    """
    Price an American option using Longstaff-Schwartz LSM.

    Notes
    -----
    American puts are the classic non-trivial case (calls on non-dividend
    paying stock are never optimal to exercise early, so American call ==
    European call in that setting). Keep option_type="put" as the default
    to price the case that actually matters.
    """
    dt = T / n_steps
    discount = np.exp(-r * dt)

    S = simulate_paths(S0, r, q, sigma, T, n_steps, n_paths, seed=seed)
    cashflow = _payoff(S[:, -1], K, option_type)

    for t in range(n_steps - 1, 0, -1):
        in_the_money = _payoff(S[:, t], K, option_type) > 0
        if not np.any(in_the_money):
            cashflow *= discount
            continue

        X = S[in_the_money, t]
        Y = cashflow[in_the_money] * discount

        # Regress continuation value on a polynomial basis of the spot price
        basis = np.polynomial.polynomial.polyvander(X, poly_degree)
        coeffs, *_ = np.linalg.lstsq(basis, Y, rcond=None)
        continuation_value = basis @ coeffs

        exercise_value = _payoff(X, K, option_type)
        exercise_now = exercise_value > continuation_value

        # Update cashflows: exercise immediately where optimal, else discount
        cashflow[in_the_money] = np.where(exercise_now, exercise_value, Y)
        cashflow[~in_the_money] *= discount

    price = cashflow.mean() * discount  # discount the final step back to t=0
    std_error = cashflow.std(ddof=1) / np.sqrt(n_paths)
    return {"price": price, "std_error": std_error, "ci_95": (price - 1.96 * std_error, price + 1.96 * std_error)}
