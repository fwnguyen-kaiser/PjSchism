"""
VIF + pairwise corr check. Observes and reports only — no auto-drop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from schism.utils.logger import ingestion_logger

_F8_COL = "f8_vol_shock"


def check_vif(
    feature_df: pd.DataFrame,
    vif_threshold: float = 5.0,
    rho_threshold: float = 0.85,
) -> tuple[bool, dict[str, float]]:
    """
    Compute VIF and pairwise |corr| for a post-Z-score feature snapshot.

    Per spec C3, F8 (f8_vol_shock) is the candidate for exclusion when its
    VIF >= vif_threshold. This function only observes and reports; it does NOT
    mutate the feature matrix or auto-drop any column. The caller decides.

    Args:
        feature_df:    DataFrame of Z-scored O_t columns (rows = bars).
                       NaN rows are dropped before computation.
        vif_threshold: Alert level; spec default = 5.
        rho_threshold: Pairwise |corr| alert level; spec default = 0.85.

    Returns:
        (f8_exceeds_threshold, vif_dict)
        f8_exceeds_threshold — True if F8 VIF >= vif_threshold (informational).
        vif_dict             — {col: vif_value} for every column.
    """
    clean = feature_df.dropna()
    n_required = max(clean.shape[1] + 1, 10)
    if len(clean) < n_required:
        ingestion_logger.warning(
            "vif_check_skipped",
            reason="insufficient_rows",
            rows=len(clean),
            required=n_required,
        )
        return False, {}

    arr = clean.values.astype(float)
    cols = list(clean.columns)

    vif_dict: dict[str, float] = {}
    for i, col in enumerate(cols):
        try:
            vif_dict[col] = float(variance_inflation_factor(arr, i))
        except Exception as exc:
            ingestion_logger.warning("vif_compute_error", col=col, error=str(exc))
            vif_dict[col] = float("nan")

    f8_exceeds = bool(
        _F8_COL in vif_dict
        and np.isfinite(vif_dict[_F8_COL])
        and vif_dict[_F8_COL] >= vif_threshold
    )

    # Pairwise correlation check — log pairs that breach rho_threshold
    corr = clean.corr().abs()
    high_corr_pairs = [
        (c1, c2, round(float(corr.loc[c1, c2]), 4))
        for i, c1 in enumerate(cols)
        for c2 in cols[i + 1 :]
        if np.isfinite(corr.loc[c1, c2]) and corr.loc[c1, c2] >= rho_threshold
    ]

    ingestion_logger.info(
        "vif_check_result",
        vifs={k: round(v, 4) for k, v in vif_dict.items()},
        f8_exceeds_threshold=f8_exceeds,
        high_corr_pairs=high_corr_pairs,
    )

    return f8_exceeds, vif_dict
