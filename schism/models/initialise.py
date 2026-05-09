"""Parameter initialisation: KMeans emission centres + bootstrap τ (§5.2)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

_EPS = 1e-10
_N_BOOT = 200


def init_params(
    O: np.ndarray,
    K: int,
    M: int,
    random_state: int,
    n_boot: int = _N_BOOT,
    tau_percentile: float = 95.0,
    sticky_kappa: float = 1.0,
) -> dict:
    """
    Initialise IOHMM parameters from data.

    Returns dict with keys: pi, mu, sigma, tau, alpha, beta.

    τ is the `tau_percentile`-th percentile of Tr(Σ̂_pooled) estimated via
    bootstrap (§5.2). Read from model_config.yaml → tau_percentile.
    """
    T, D = O.shape
    rng = np.random.RandomState(random_state)

    # K-means emission centres
    km = KMeans(n_clusters=K, random_state=random_state, n_init=10)
    km.fit(O)
    mu = km.cluster_centers_.copy()                    # (K, D)

    pooled = np.cov(O.T) + _EPS * np.eye(D)           # (D, D)
    sigma = np.stack([pooled.copy() for _ in range(K)])  # (K, D, D)

    # Bootstrap Tr(Σ̂_pooled) → τ = tau_percentile-th percentile
    boot_traces = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(T, size=T, replace=True)
        cov_b = np.cov(O[idx].T) + _EPS * np.eye(D)
        boot_traces[b] = np.trace(cov_b)
    tau = float(np.percentile(boot_traces, tau_percentile))

    pi = np.ones(K) / K                               # uniform
    # Sticky prior: log(sticky_kappa) on diagonal → higher initial self-transition.
    # sticky_kappa=1 keeps uniform init; >1 biases EM toward persistent regimes.
    alpha = np.zeros((K, K))
    if sticky_kappa > 1.0:
        np.fill_diagonal(alpha, np.log(sticky_kappa))
    beta = np.zeros((K, K, M))

    return {
        "pi": pi,
        "mu": mu,
        "sigma": sigma,
        "tau": tau,
        "alpha": alpha,
        "beta": beta,
    }
