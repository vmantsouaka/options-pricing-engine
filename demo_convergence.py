"""
Demo script : plots MC price convergence to the Black-Scholes benchmark
as the number of simulated paths grows.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt

from monte_carlo import convergence_study

S0, K, r, q, sigma, T = 100.0, 100.0, 0.03, 0.01, 0.20, 1.0

rows = convergence_study(S0, K, r, q, sigma, T, option_type="call")

n_paths = [row["n_paths"] for row in rows]
mc_prices = [row["mc_price"] for row in rows]
bs_ref = rows[0]["bs_price"]
errors = [row["std_error"] for row in rows]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.errorbar(n_paths, mc_prices, yerr=[1.96 * e for e in errors], fmt="o-", label="Monte Carlo (95% CI)")
ax.axhline(bs_ref, color="black", linestyle="--", label=f"Black-Scholes = {bs_ref:.4f}")
ax.set_xscale("log")
ax.set_xlabel("Number of simulated paths")
ax.set_ylabel("Option price")
ax.set_title("Monte Carlo convergence to Black-Scholes (European call)")
ax.legend()
fig.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), "..", "convergence_plot.png")
fig.savefig(output_path, dpi=150)
print(f"Saved plot to {output_path}")

for row in rows:
    print(row)
