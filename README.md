# Options Pricing Engine

Un moteur de pricing d'options vanille et américaines, construit pour comparer
trois approches - formule fermée, Monte Carlo, arbre/simulation américaine —
et pour quantifier proprement l'erreur numérique de chacune.

## Pourquoi ce projet

N'importe qui peut recopier la formule de Black-Scholes. L'objectif ici est
différent : montrer une vraie maîtrise du calcul stochastique appliqué au
pricing — savoir *quantifier* l'erreur d'une approximation Monte Carlo,
*réduire sa variance* proprement, et gérer le cas non-trivial de l'exercice
anticipé américain via Longstaff-Schwartz.

## Contenu

|Module|Description|
|-|-|
|`src/black\\\_scholes.py`|Prix fermé + Greeks analytiques (delta, gamma, vega, theta, rho)|
|`src/monte\\\_carlo.py`|Pricing MC avec variables antithétiques et variable de contrôle, intervalle de confiance|
|`src/american\\\_options.py`|Pricing américain via Longstaff-Schwartz (Least-Squares Monte Carlo)|
|`demo\\\_convergence.py`|Génère le graphique de convergence MC → Black-Scholes|
|`test\\\_pricing.py`|Tests de cohérence : parité call-put, convergence MC, réduction de variance|

## Résultats clés

* **Convergence Monte Carlo** : le prix MC converge vers le prix Black-Scholes
à une vitesse en O(1/√n), visible sur `convergence\\\_plot.png` (généré par
le script de démo).
* **Réduction de variance** : la combinaison variables antithétiques +
variable de contrôle réduit significativement l'écart-type de l'estimateur
à budget de calcul égal (voir `test\\\_mc\\\_variance\\\_reduction\\\_lowers\\\_std\\\_error`).
* **Prime d'exercice anticipé** : le put américain est valorisé au-dessus de
son équivalent européen, ce qui matérialise la prime d'exercice anticipé.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation rapide

```python
from src.black\\\_scholes import bs\\\_price, bs\\\_greeks
from src.monte\\\_carlo import mc\\\_price
from src.american\\\_options import lsm\\\_price

# Prix Black-Scholes d'un call européen
price = bs\\\_price(S0=100, K=100, r=0.03, q=0.01, sigma=0.20, T=1.0, option\\\_type="call")

# Greeks analytiques
greeks = bs\\\_greeks(S0=100, K=100, r=0.03, q=0.01, sigma=0.20, T=1.0, option\\\_type="call")

# Pricing Monte Carlo avec réduction de variance
result = mc\\\_price(S0=100, K=100, r=0.03, q=0.01, sigma=0.20, T=1.0,
                   option\\\_type="call", n\\\_paths=200\\\_000,
                   antithetic=True, control\\\_variate=True)
print(result.price, result.ci\\\_95)

# Option américaine (put) via Longstaff-Schwartz
american = lsm\\\_price(S0=100, K=100, r=0.03, q=0.01, sigma=0.20, T=1.0,
                      option\\\_type="put", n\\\_paths=50\\\_000)
print(american\\\["price"])
```

```



## Prochaines étapes possibles

* Calibration de la surface de volatilité implicite sur des vraies chaînes
d'options (via `yfinance`)
* Modèle à volatilité stochastique (Heston) pour comparer avec Black-Scholes
* Pricing d'options exotiques (barrière, asiatique) en Monte Carlo

