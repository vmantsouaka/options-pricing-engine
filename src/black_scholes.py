"""
Black-Scholes closed-form pricing for European options.


Pour une option européenne (Call ou Put) sur un actif ne versant pas de dividende,
le prix à l'instant t=0 est donné par les formules exactes suivantes :

    Call :  C = S0 * N(d1) - K * e^(-rT) * N(d2)
    Put  :  P = K * e^(-rT) * N(-d2) - S0 * N(-d1)

Avec :
    d1 = [ ln(S0/K) + (r - q + 0.5 * σ²) * T ] / (σ * √T)
    d2 = d1 - σ * √T

Où :
    S0  = prix spot de l'actif aujourd'hui
    K   = prix d'exercice (strike)
    r   = taux d'intérêt sans risque (annualisé, continûment composé)
    q   = taux de dividende continu (continuous dividend yield) de l'actif sous-jacent.
    σ   = volatilité annualisée du sous-jacent
    T   = temps jusqu'à l'échéance (en années)
    N(.)= fonction de répartition de la loi normale centrée réduite

NOTE : 
- Cette formule repose sur l'hypothèse que le sous-jacent suit un 
  Mouvement Brownien Géométrique (MBG) à volatilité et taux constants.
- Elle est valable UNIQUEMENT pour les options de type EUROPEEN (exercice 
  uniquement à l'échéance). Pour les Américaines, il n'existe pas de formule 
  fermée simple (sauf cas particuliers), et on utilise des arbres binomiaux 
  ou des méthodes numériques.

IMPLEMENTATION (avec dividendes q)

"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S0: float, K: float, r: float, q: float, sigma: float, T: float):
    """Compute d1 and d2 used throughout Black-Scholes formulas."""
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S0: float, K: float, r: float, q: float, sigma: float, T: float, option_type: str = "call") -> float:
    """
    Closed-form Black-Scholes price for a European call or put.

    Parameters
    ----------
    S0 : spot price of the underlying
    K : strike price
    r : risk-free rate (annualized, continuous compounding)
    q : continuous dividend yield
    sigma : volatility (annualized)
    T : time to maturity in years
    option_type : "call" or "put"

    Returns
    -------
    price : float
    """
    if T <= 0:
        # At maturity, price collapses to intrinsic value
        if option_type == "call":
            return max(S0 - K, 0.0)
        return max(K - S0, 0.0)

    d1, d2 = _d1_d2(S0, K, r, q, sigma, T)

    if option_type == "call":
        price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price

"""
================================================================================
RAPPEL DES GREEKS (BLACK-SCHOLES)
================================================================================

Les Greeks mesurent la sensibilité du prix d'une option aux différents paramètres.
Ils sont essentiels pour la couverture (hedging) et la gestion des risques.

--------------------------------------------------------------------------------
1. DELTA (Δ) - Sensibilité au prix du sous-jacent
--------------------------------------------------------------------------------
    Δ = ∂V / ∂S

    Interprétation : De combien varie le prix de l'option si le sous-jacent 
    augmente de 1€ (toutes choses égales par ailleurs).

    Call :  Δ = e^(-qT) * N(d1)              ∈ [0, 1]
    Put  :  Δ = -e^(-qT) * N(-d1)             ∈ [-1, 0]

    Remarque : Pour une option ATM (At-The-Money), Δ_call ≈ 0.5, Δ_put ≈ -0.5.
              C'est le Greek le plus important pour le hedging directionnel.

--------------------------------------------------------------------------------
2. GAMMA (Γ) - Sensibilité du Delta au prix du sous-jacent
--------------------------------------------------------------------------------
    Γ = ∂²V / ∂S² = ∂Δ / ∂S

    Interprétation : De combien varie le Delta si le sous-jacent augmente de 1€.
    C'est une mesure de la convexité de l'option.

    Formule (commun aux Calls et Puts) :
    Γ = e^(-qT) * N'(d1) / (S0 * σ * √T)

    Remarque : Le Gamma est maximal pour les options ATM proches de l'échéance.
              Un Gamma élevé signifie que le Delta change rapidement -> 
              risque directionnel accru et coûts de ré-hedging importants.

--------------------------------------------------------------------------------
3. VEGA (ν) - Sensibilité à la volatilité
--------------------------------------------------------------------------------
    ν = ∂V / ∂σ

    Interprétation : De combien varie le prix de l'option si la volatilité 
    implicite augmente de 1 point (ex: de 20% à 21%).

    Formule (commun aux Calls et Puts) :
    ν = S0 * e^(-qT) * N'(d1) * √T

    Remarque : Le Vega est maximal pour les options ATM avec une longue durée 
              de vie. Attention : le Vega n'est pas une lettre grecque à proprement 
              parler (c'est une mesure de volatilité), mais il est indispensable.

--------------------------------------------------------------------------------
4. THETA (Θ) - Sensibilité au temps (décroissance temporelle)
--------------------------------------------------------------------------------
    Θ = -∂V / ∂T   (ou plus souvent ∂V/∂t en continu)

    Interprétation : De combien l'option perd de la valeur lorsque le temps 
    s'écoule d'un jour (si on divise par 365).

    Call :  Θ = -S0*N'(d1)*σ*e^(-qT)/(2√T) - r*K*e^(-rT)*N(d2) + q*S0*e^(-qT)*N(d1)
    Put  :  Θ = -S0*N'(d1)*σ*e^(-qT)/(2√T) + r*K*e^(-rT)*N(-d2) - q*S0*e^(-qT)*N(-d1)

    Remarque : Le Θ est généralement négatif (l'option perd de la valeur avec 
              le temps). C'est le "coût du temps" pour l'acheteur et le gain 
              pour le vendeur. Pour les options ITM avec dividendes, Θ peut 
              être positif.

--------------------------------------------------------------------------------
5. RHO (ρ) - Sensibilité au taux d'intérêt sans risque
--------------------------------------------------------------------------------
    ρ = ∂V / ∂r

    Interprétation : De combien varie le prix de l'option si le taux sans 
    risque augmente de 1 point (ex: de 2% à 3%).

    Call :  ρ = K * T * e^(-rT) * N(d2)
    Put  :  ρ = -K * T * e^(-rT) * N(-d2)

    Remarque : Le Rho est généralement faible pour les options court terme, 
              mais devient significatif pour les options long terme (LEAPS). 
              Il est positif pour un Call et négatif pour un Put.

--------------------------------------------------------------------------------
6. AUTRES GREEKS (moins courants mais utiles)
--------------------------------------------------------------------------------
    - Vanna    : ∂Δ/∂σ = ∂ν/∂S  (sensibilité croisée Delta/Vega)
    - Volga    : ∂ν/∂σ = ∂²V/∂σ² (convexité en volatilité, aussi appelée Vomma)
    - Charm    : ∂Δ/∂T  (décroissance du Delta dans le temps)
    - Speed    : ∂Γ/∂S  (décroissance du Gamma, dérivée 3ème en S)

    Ces Greeks d'ordre supérieur sont utilisés pour des couvertures plus 
    fines, notamment dans les book de volatilité exotique.

--------------------------------------------------------------------------------
7. RÉSUMÉ DES FORMULES AVEC q (dividendes)
--------------------------------------------------------------------------------
    N'(d1) = (1/√(2π)) * exp(-d1²/2)   (densité de la loi normale)

    d1 = [ ln(S0/K) + (r - q + 0.5*σ²)*T ] / (σ*√T)
    d2 = d1 - σ*√T

    où :
    S0   = spot
    K    = strike
    r    = taux sans risque
    q    = taux de dividende continu
    σ    = volatilité
    T    = temps jusqu'à l'échéance

================================================================================
INTERPRÉTATION PRATIQUE POUR UN PORTEFEUILLE
================================================================================
    - Un portefeuille avec Δ ≈ 0 est "delta-neutre" (immunisé contre les petits 
      mouvements du sous-jacent).
    - Un portefeuille avec Γ ≈ 0 est "gamma-neutre" (le Delta ne varie pas 
      rapidement -> hedging moins coûteux).
    - Un portefeuille avec ν ≈ 0 est "vega-neutre" (immunisé contre les chocs 
      de volatilité).
    - Un portefeuille avec Θ > 0 génère du "cash" chaque jour (position vendeuse).
"""

def bs_greeks(S0: float, K: float, r: float, q: float, sigma: float, T: float, option_type: str = "call") -> dict:
    """
    Analytical Greeks for a European option under Black-Scholes.

    Returns a dict with delta, gamma, vega, theta, rho.
    Theta is expressed per year (divide by 365 for per-day theta).
    Vega is expressed per unit of volatility (divide by 100 for per-vol-point).
    """
    d1, d2 = _d1_d2(S0, K, r, q, sigma, T)
    pdf_d1 = norm.pdf(d1)

    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            -S0 * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S0 * np.exp(-q * T) * norm.cdf(d1)
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (
            -S0 * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S0 * np.exp(-q * T) * norm.cdf(-d1)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = np.exp(-q * T) * pdf_d1 / (S0 * sigma * np.sqrt(T))
    vega = S0 * np.exp(-q * T) * pdf_d1 * np.sqrt(T)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
