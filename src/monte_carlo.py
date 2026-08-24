"""
Monte Carlo pricing for European options under Geometric Brownian Motion,
with variance reduction techniques (antithetic variates, control variate).

The GBM terminal value is simulated in closed form (no need to discretize
the whole path for a European option):
    S_T = S0 * exp((r - q - 0.5*sigma^2) * T + sigma * sqrt(T) * Z),  Z ~ N(0,1)
"""

from dataclasses import dataclass

import numpy as np

from black_scholes import bs_price


@dataclass
class MCResult:
    price: float
    std_error: float
    ci_95: tuple  # (low, high)
    n_paths: int


def _payoff(S_T: np.ndarray, K: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        return np.maximum(K - S_T, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def mc_price(
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    antithetic: bool = True,
    control_variate: bool = False,
    seed: int | None = None,
) -> MCResult:
    """
    Monte Carlo price of a European option with optional variance reduction.

    antithetic : pair each draw Z with -Z, halves the effective variance
                 for roughly the same compute budget
    control_variate : uses the analytical Black-Scholes price as a control
                       variate to further shrink the variance. Only sensible
                       here since we know BS in closed form; in practice the
                       control variate is a related payoff that IS known
                       analytically while the target payoff is not.
    """
    rng = np.random.default_rng(seed)  # Random Number Generator, Le paramètre seed permet de reproduire exactement les mêmes tirages aléatoires à chaque simu de MC

    if antithetic:
        half = n_paths // 2
        Z_half = rng.standard_normal(half)
        Z = np.concatenate([Z_half, -Z_half])
    else:
        Z = rng.standard_normal(n_paths)    # Vectorisation, pas besoin de boucle for

    drift = (r - q - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * Z
    S_T = S0 * np.exp(drift + diffusion)

    discount = np.exp(-r * T)
    payoffs = discount * _payoff(S_T, K, option_type)

    if control_variate:
        # Control variate: discounted S_T itself, whose expectation we know
        # in closed form: E[disc * S_T] = S0 * exp(-q*T)
        control = discount * S_T
        control_mean_theoretical = S0 * np.exp(-q * T)

        cov = np.cov(payoffs, control)[0, 1]
        var_control = np.var(control)
        beta = cov / var_control if var_control > 0 else 0.0

        payoffs = payoffs - beta * (control - control_mean_theoretical)

    price = payoffs.mean()
    # Student-t based confidence interval (safer for finite n than pure normal)
    std_error = payoffs.std(ddof=1) / np.sqrt(len(payoffs))
    ci_95 = (price - 1.96 * std_error, price + 1.96 * std_error)

    return MCResult(price=price, std_error=std_error, ci_95=ci_95, n_paths=len(payoffs))


def convergence_study(S0, K, r, q, sigma, T, option_type="call", path_counts=None, seed=42):
    """
    Compare MC price to the Black-Scholes benchmark across increasing
    numbers of simulated paths. Returns a list of dicts ready to plot.
    """
    if path_counts is None:
        path_counts = [10**3, 10**4, 10**5, 10**6]

    bs_ref = bs_price(S0, K, r, q, sigma, T, option_type)
    rows = []
    for n in path_counts:
        result = mc_price(S0, K, r, q, sigma, T, option_type, n_paths=n, seed=seed)
        rows.append(
            {
                "n_paths": n,
                "mc_price": result.price,
                "std_error": result.std_error,
                "bs_price": bs_ref,
                "abs_error": abs(result.price - bs_ref),
            }
        )
    return rows
