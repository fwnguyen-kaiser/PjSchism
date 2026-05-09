"""
Hungarian algorithm for post-refit state label alignment.

Spec §8, Eq.(6):
    C_{ij} = ||μ_i^old - μ_j^new||²
    π* = argmin_π Σ_i C_{i,π(i)}    [O(K³) via lapjv]

If min_i C_{i,π*(i)} > δ_align for any state → drift_alert = True.
"""

from __future__ import annotations

import numpy as np
import lapjv

from schism.utils.logger import ingestion_logger

_LOG = ingestion_logger


def align_states(
    mu_old: np.ndarray,
    mu_new: np.ndarray,
    delta_align: float | None = None,
) -> tuple[np.ndarray, bool]:
    """
    Compute the permutation that aligns new states to old states by
    minimising the sum of squared distances between emission means.

    Parameters
    ----------
    mu_old      : (K, D) emission means from the previous model
    mu_new      : (K, D) emission means from the newly fitted model
    delta_align : drift alert threshold on max assignment cost (default: 10× median cost)

    Returns
    -------
    perm        : (K,) integer array — new_state[perm[i]] maps to old_state[i]
    drift_alert : True if any matched-pair distance exceeds delta_align
    """
    K = mu_old.shape[0]

    # Cost matrix C[i, j] = ||mu_old[i] - mu_new[j]||²
    diff = mu_old[:, None, :] - mu_new[None, :, :]   # (K, K, D)
    C = (diff ** 2).sum(axis=2).astype(np.float64)    # (K, K)

    # lapjv returns (row_ind, col_ind, cost)
    # col_ind[i] = column assigned to row i
    _, col_ind, _ = lapjv.lapjv(C)
    perm = col_ind.astype(int)   # new state j = perm[i] maps to old state i

    assignment_costs = C[np.arange(K), perm]
    max_cost = float(assignment_costs.max())

    if delta_align is None:
        # Default: alert if any cost > 10× median
        delta_align = 10.0 * float(np.median(assignment_costs)) + 1e-8

    drift_alert = bool(max_cost > delta_align)

    _LOG.info(
        "state_alignment_done",
        perm=perm.tolist(),
        assignment_costs=np.round(assignment_costs, 4).tolist(),
        max_cost=round(max_cost, 4),
        drift_alert=drift_alert,
    )
    return perm, drift_alert


def apply_permutation(model: "IOHMM", perm: np.ndarray) -> None:
    """
    Reorder model parameters in-place according to perm.
    perm[i] = new_index that should become old_index i.
    """
    model.pi = model.pi[perm]
    model.mu = model.mu[perm]
    model.sigma = model.sigma[perm]
    model.labels = [model.labels[p] for p in perm]

    # Transition: both row and column dimensions are state indices
    model.alpha = model.alpha[np.ix_(perm, perm)]
    model.beta = model.beta[np.ix_(perm, perm)]
