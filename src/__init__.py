from .black_scholes import bs_price, bs_greeks                   # imports relatifs (.) à l'intérieur du package
from .monte_carlo import mc_price, convergence_study, MCResult   # pas de .py parce que Python n'importe pas des fichiers, il importe des modules.
from .american_options import lsm_price, simulate_paths

__all__ = [
    "bs_price",
    "bs_greeks",
    "mc_price",
    "convergence_study",
    "MCResult",
    "lsm_price",
    "simulate_paths",
]
