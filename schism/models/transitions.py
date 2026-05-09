"""Softmax transition log-probabilities and L-BFGS M-step (§5.1, §5.3)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def log_transition(
    U_safe: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """
    Compute (T, K, K) log-transition tensor.

    logit_{t,i,j} = α_{ij} + β_{ij}^T U_t
    log A_{t,i,j} = logit_{t,i,j} − log Σ_k exp(logit_{t,i,k})
    """
    # alpha: (K, K), beta: (K, K, M), U_safe: (T, M)
    logits = alpha[None, :, :] + np.einsum("ijm,tm->tij", beta, U_safe)
    return logits - logsumexp(logits, axis=2, keepdims=True)


def m_step_transition(
    U_safe: np.ndarray,
    xi: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    lambda_reg: float,
    sticky_kappa: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Maximise Q_transition via L-BFGS-B on (α, β) jointly (§5.3).

    Q_trans = Σ_{t,i,j} ξ_t(i,j) log A_t(i,j) − λ Σ_{ij} ||β_{ij}||²

    sticky_kappa > 0: augment the diagonal of xi with `sticky_kappa` total
    pseudo-self-transition counts per state before optimisation — equivalent to
    a Dirichlet prior concentrated on self-transitions (Fox et al. 2011 §sticky-HMM).
    Read from model_config.yaml → sticky_kappa.

    Gradient of d(Q_trans)/d(logit_{t,i,j}):
        = ξ_t(i,j) − [Σ_j ξ_t(i,:)] × A_t(i,j)

    Returns
    -------
    alpha : (K, K) updated transition intercepts
    beta  : (K, K, M) updated transition slopes
    """
    K, M = alpha.shape[0], beta.shape[2]
    U_t = U_safe[1:]           # transitions use U at t=1..T-1

    # Sticky-HMM prior: add pseudo-counts only on the diagonal.
    # Total self-transition pseudo-weight per state = sticky_kappa.
    # Distributed uniformly across timesteps so the objective scale is preserved.
    if sticky_kappa > 0.0:
        xi = xi.copy()
        per_step = sticky_kappa / max(1, len(xi))
        xi[:, np.arange(K), np.arange(K)] += per_step

    def _pack(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.concatenate([a.ravel(), b.ravel()])

    def _unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = params[: K * K].reshape(K, K)
        b = params[K * K :].reshape(K, K, M)
        return a, b

    def neg_Q_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        a, b = _unpack(params)
        logits = a[None, :, :] + np.einsum("ijm,tm->tij", b, U_t)
        log_A = logits - logsumexp(logits, axis=2, keepdims=True)
        A = np.exp(log_A)

        Q_trans = float((xi * log_A).sum())
        reg = float(lambda_reg * (b ** 2).sum())

        xi_row = xi.sum(axis=2, keepdims=True)         # (T-1, K, 1)
        d_logit = xi - xi_row * A                      # (T-1, K, K)

        grad_a = d_logit.sum(axis=0)                   # (K, K)
        grad_b = np.einsum("tij,tm->ijm", d_logit, U_t)
        grad_b -= 2.0 * lambda_reg * b

        return -(Q_trans - reg), -_pack(grad_a, grad_b)

    result = minimize(
        neg_Q_and_grad,
        _pack(alpha, beta),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 50, "ftol": 1e-10, "gtol": 1e-7},
    )
    return _unpack(result.x)
