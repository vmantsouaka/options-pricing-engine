"""
Sanity-check tests. Run with: pytest tests/

Core idea: MC price should converge to the closed-form BS price within a
few standard errors as n_paths grows, and put-call parity must hold exactly
for the analytical BS formula.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from black_scholes import bs_price
from monte_carlo import mc_price
from american_options import lsm_price

S0, K, r, q, sigma, T = 100.0, 100.0, 0.03, 0.01, 0.20, 1.0


def test_put_call_parity():
    call = bs_price(S0, K, r, q, sigma, T, "call")
    put = bs_price(S0, K, r, q, sigma, T, "put")

    # Put-call parity: C - P = S0*exp(-qT) - K*exp(-rT)

    lhs = call - put
    rhs = S0 * (2.718281828 ** (-q * T)) - K * (2.718281828 ** (-r * T))  # right hand side
    assert abs(lhs - rhs) < 1e-6


def test_mc_converges_to_bs():
    bs_ref = bs_price(S0, K, r, q, sigma, T, "call")
    result = mc_price(S0, K, r, q, sigma, T, "call", n_paths=200_000, seed=42)
    # Should be within ~4 standard errors most of the time
    assert abs(result.price - bs_ref) < 4 * result.std_error


def test_mc_variance_reduction_lowers_std_error():
    plain = mc_price(S0, K, r, q, sigma, T, "call", n_paths=50_000, antithetic=False, seed=1)
    reduced = mc_price(S0, K, r, q, sigma, T, "call", n_paths=50_000, antithetic=True, control_variate=True, seed=1)
    assert reduced.std_error < plain.std_error


def test_american_put_geq_european_put():
    european_put = bs_price(S0, K, r, q, sigma, T, "put")
    american_put = lsm_price(S0, K, r, q, sigma, T, option_type="put", n_paths=20_000, seed=7)
    # Early exercise premium should make the American put at least as valuable
    assert american_put["price"] >= european_put - 3 * american_put["std_error"]
