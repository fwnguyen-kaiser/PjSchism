"""Gaussian emission log-probabilities and M-step update (§5.2)."""

from __future__ import annotations

import numpy as np

_EPS = 1e-10


def log_emission(
    O: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Compute (T, K) log-emission matrix: log N(O_t; μ_k, Σ_k).

    Uses Cholesky decomposition for numerical stability; falls back to
    diagonal approximation if Σ_k is not positive-definite.
    """
    T, D = O.shape
    K = mu.shape[0]
    log_b = np.zeros((T, K))
    log_2pi_D = D * np.log(2.0 * np.pi)

    for k in range(K):
        diff = O - mu[k]                        # (T, D)
        try:
            L = np.linalg.cholesky(sigma[k])
            log_det = 2.0 * np.sum(np.log(np.diag(L) + _EPS))
            sol = np.linalg.solve(L, diff.T)    # (D, T)
            maha = (sol ** 2).sum(axis=0)       # (T,)
        except np.linalg.LinAlgError:
            var = np.diag(sigma[k]) + _EPS
            log_det = np.sum(np.log(var))
            maha = ((diff ** 2) / var).sum(axis=1)
        log_b[:, k] = -0.5 * (log_2pi_D + log_det + maha)

    return log_b


def m_step_emission(
    O: np.ndarray,
    gamma: np.ndarray,
    tau: float,
    cov_floor: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Closed-form M-step for emission parameters.

    Applies trace constraint Tr(Σ_k) ≤ τ after weighted MLE (§5.2).
    `cov_floor` is added to the diagonal for positive-definiteness — read from
    model_config.yaml → covariance_floor.

    Returns
    -------
    mu    : (K, D) updated emission means
    sigma : (K, D, D) updated covariance matrices
    pi    : (K,) updated initial-state distribution
    """
    T, D = O.shape
    K = gamma.shape[1]
    mu = np.zeros((K, D))
    sigma = np.zeros((K, D, D))

    for k in range(K):
        w = gamma[:, k]                         # (T,)
        w_sum = w.sum() + _EPS

        mu[k] = (w @ O) / w_sum

        diff = O - mu[k]                        # (T, D)
        sigma[k] = (w[:, None] * diff).T @ diff / w_sum
        sigma[k] += cov_floor * np.eye(D)

        tr = np.trace(sigma[k])
        if tr > tau:
            sigma[k] *= tau / tr

    pi = gamma[0] + _EPS
    pi /= pi.sum()

    return mu, sigma, pi
