"""
Forward/backward algorithm, E-step, Viterbi decoding, and online filter.

All computations are in log-space for numerical stability.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

_EPS = 1e-10


# ── Data helper ───────────────────────────────────────────────────────────────

def safe_U(U: np.ndarray) -> np.ndarray:
    """Replace NaN in U with 0 — missing exogenous component → no influence."""
    out = U.copy()
    out[np.isnan(out)] = 0.0
    return out


# ── Forward / Backward ────────────────────────────────────────────────────────

def forward(
    log_b: np.ndarray,
    log_A: np.ndarray,
    log_pi: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Log-space forward pass (Baum-Welch).

    Parameters
    ----------
    log_b  : (T, K) log-emission probabilities
    log_A  : (T, K, K) log-transition probabilities; log_A[t, i, j] = log P(j|i, U_t)
    log_pi : (K,) log initial-state distribution

    Returns
    -------
    log_alpha : (T, K) log forward variables
    log_ll    : scalar log-likelihood log P(O_{1:T})
    """
    T, K = log_b.shape
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi + log_b[0]

    for t in range(1, T):
        # logsumexp over i: (K,1) + (K,K) → logsumexp axis=0 → (K,)
        log_alpha[t] = log_b[t] + logsumexp(
            log_alpha[t - 1, :, None] + log_A[t], axis=0
        )

    return log_alpha, float(logsumexp(log_alpha[-1]))


def backward(log_b: np.ndarray, log_A: np.ndarray) -> np.ndarray:
    """
    Log-space backward pass.

    Returns log_beta (T, K); log_beta[-1] = log(1) = 0 by convention.
    """
    T, K = log_b.shape
    log_beta = np.zeros((T, K))

    for t in range(T - 2, -1, -1):
        # logsumexp over j: (K,K) + (1,K) + (1,K) → logsumexp axis=1 → (K,)
        log_beta[t] = logsumexp(
            log_A[t + 1] + log_b[t + 1][None, :] + log_beta[t + 1][None, :],
            axis=1,
        )

    return log_beta


# ── E-step ────────────────────────────────────────────────────────────────────

def e_step(
    log_b: np.ndarray,
    log_A: np.ndarray,
    log_pi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Full E-step: compute γ (T,K) and ξ (T-1,K,K) posteriors.

    γ_t(k)   = P(S_t=k | O_{1:T}, U_{1:T})
    ξ_t(i,j) = P(S_{t-1}=i, S_t=j | O_{1:T}, U_{1:T})

    Returns (gamma, xi, log_likelihood).
    """
    T, K = log_b.shape

    log_alpha, log_ll = forward(log_b, log_A, log_pi)
    log_beta = backward(log_b, log_A)

    # γ
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    # ξ — fully vectorised
    log_xi = (
        log_alpha[:-1, :, None]       # (T-1, K, 1)
        + log_A[1:]                    # (T-1, K, K)
        + log_b[1:, None, :]           # (T-1, 1, K)
        + log_beta[1:, None, :]        # (T-1, 1, K)
    )
    log_xi -= logsumexp(log_xi.reshape(T - 1, K * K), axis=1)[:, None, None]
    xi = np.exp(log_xi)

    return gamma, xi, log_ll


# ── Viterbi decoding ──────────────────────────────────────────────────────────

def viterbi(
    log_b: np.ndarray,
    log_A: np.ndarray,
    log_pi: np.ndarray,
) -> np.ndarray:
    """
    Viterbi algorithm: most-likely state sequence.

    Returns (T,) integer state indices.
    """
    T, K = log_b.shape
    log_delta = np.empty((T, K))
    psi = np.zeros((T, K), dtype=int)
    log_delta[0] = log_pi + log_b[0]

    for t in range(1, T):
        vals = log_delta[t - 1, :, None] + log_A[t]   # (K, K)
        psi[t] = vals.argmax(axis=0)
        log_delta[t] = vals.max(axis=0) + log_b[t]

    states = np.empty(T, dtype=int)
    states[-1] = log_delta[-1].argmax()
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


# ── Online forward filter ─────────────────────────────────────────────────────

def filter_sequence(
    log_b: np.ndarray,
    log_A: np.ndarray,
    log_pi: np.ndarray,
) -> np.ndarray:
    """
    Forward-only filter for a full sequence.
    Returns γ_t = P(S_t | O_{1:t}, U_{1:t}) as (T, K).
    """
    log_alpha, _ = forward(log_b, log_A, log_pi)
    log_gamma = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_gamma)


def filter_step(
    log_alpha_prev: np.ndarray,
    log_b_t: np.ndarray,
    log_A_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-step online forward update.

    Parameters
    ----------
    log_alpha_prev : (K,) unnormalised log forward variable from t-1
    log_b_t        : (K,) log-emission at time t
    log_A_t        : (K, K) log-transition at time t; log_A_t[i,j] = log P(j|i, U_t)

    Returns
    -------
    log_alpha_t : (K,) updated unnormalised log forward variable
    gamma_t     : (K,) P(S_t | O_{1:t}, U_{1:t})
    """
    log_alpha_t = log_b_t + logsumexp(log_alpha_prev[:, None] + log_A_t, axis=0)
    log_gamma_t = log_alpha_t - logsumexp(log_alpha_t)
    return log_alpha_t, np.exp(log_gamma_t)
