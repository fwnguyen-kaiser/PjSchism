"""
Custom exceptions and warnings for SCHISM.

Hierarchy:
  SchismError (base)
    ├── DataMissingError      — missing or incomplete source data
    ├── BanError              — exchange IP ban (HTTP 418)
    ├── RateLimitWarning      — exchange rate limit hit (HTTP 429)
    ├── RefitCooldownError    — refit requested during cooldown window
    ├── VIFViolationError     — feature collinearity exceeds threshold
    └── IdentifiabilityError  — insufficient training data for K, Dim

  SchismWarning (base)
    └── RegimeAlignmentWarning — Hungarian cost exceeds delta_align
"""

from __future__ import annotations

from typing import Optional


# ── Base classes ──────────────────────────────────────────────────────────────

class SchismError(Exception):
    """Base class for all SCHISM runtime errors."""


class SchismWarning(UserWarning):
    """Base class for all SCHISM warnings (non-fatal)."""


# ── Data / ingestion errors ───────────────────────────────────────────────────

class DataMissingError(SchismError):
    """
    Raised when required market data is absent or unretrievable.

    Attributes:
        source      — origin of the data (e.g. "binance_rest", "parquet")
        path        — endpoint path or file path that failed
        status_code — HTTP status code, if applicable
        reason      — free-form error string
    """

    def __init__(
        self,
        message: str,
        *,
        source: str = "",
        path: str = "",
        status_code: Optional[int] = None,
        reason: str = "",
    ) -> None:
        super().__init__(message)
        self.source = source
        self.path = path
        self.status_code = status_code
        self.reason = reason

    def __repr__(self) -> str:
        return (
            f"DataMissingError({self!s}, source={self.source!r}, "
            f"path={self.path!r}, status_code={self.status_code})"
        )


class BanError(SchismError):
    """
    Raised on HTTP 418 — Binance IP ban.

    Attributes:
        exchange        — exchange name (e.g. "binance")
        retry_after_ts  — UNIX ms timestamp after which requests may resume
        status_code     — always 418
    """

    def __init__(
        self,
        message: str,
        *,
        exchange: str = "binance",
        retry_after_ts: int = 0,
        status_code: int = 418,
    ) -> None:
        super().__init__(message)
        self.exchange = exchange
        self.retry_after_ts = retry_after_ts
        self.status_code = status_code


class RateLimitWarning(SchismError):
    """
    Raised when the exchange returns HTTP 429 and retries are exhausted.

    Attributes:
        exchange            — exchange name
        retry_after_seconds — seconds to wait before retrying
        used_weight         — weight consumed in the current 1-minute window
    """

    def __init__(
        self,
        message: str,
        *,
        exchange: str = "binance",
        retry_after_seconds: int = 60,
        used_weight: int = 0,
    ) -> None:
        super().__init__(message)
        self.exchange = exchange
        self.retry_after_seconds = retry_after_seconds
        self.used_weight = used_weight


# ── Training / model errors ───────────────────────────────────────────────────

class RefitCooldownError(SchismError):
    """
    Raised when a refit is requested but the cooldown window has not elapsed.

    Attributes:
        cooldown_bars_remaining — bars left before cooldown expires
        cooldown_end_bar_ts     — ISO timestamp when cooldown ends
    """

    def __init__(
        self,
        message: str,
        *,
        cooldown_bars_remaining: int = 0,
        cooldown_end_bar_ts: str = "",
    ) -> None:
        super().__init__(message)
        self.cooldown_bars_remaining = cooldown_bars_remaining
        self.cooldown_end_bar_ts = cooldown_end_bar_ts


class VIFViolationError(SchismError):
    """
    Raised when a feature's Variance Inflation Factor exceeds the threshold
    AND the corrective action (PCA / L2 reg) is not configured.

    Per spec §6: Dim(Ot) = 10 is permanently fixed; dropping features is
    NOT allowed. Callers must apply regularisation instead.

    Attributes:
        feature_index — 1-based feature index (matches Ot numbering in spec)
        vif_value     — computed VIF for the offending feature
        threshold     — configured vif_threshold (default 5)
    """

    def __init__(
        self,
        message: str,
        *,
        feature_index: int = 0,
        vif_value: float = 0.0,
        threshold: float = 5.0,
    ) -> None:
        super().__init__(message)
        self.feature_index = feature_index
        self.vif_value = vif_value
        self.threshold = threshold


class IdentifiabilityError(SchismError):
    """
    Raised when Ttrain < 10 × K × Dim² (asymptotic sufficiency condition, spec §7).

    At K=4, Dim=10 the minimum is 4,000 bars (≈667 days of 4h data).
    For training-only windows the minimum is 360 bars (60 days).

    Attributes:
        t_train        — actual number of training bars available
        t_required     — minimum bars required given K and Dim
        K              — number of states
        Dim            — observation dimension
    """

    def __init__(
        self,
        message: str,
        *,
        t_train: int = 0,
        t_required: int = 0,
        K: int = 4,
        Dim: int = 10,
    ) -> None:
        super().__init__(message)
        self.t_train = t_train
        self.t_required = t_required
        self.K = K
        self.Dim = Dim


# ── Post-refit warnings ───────────────────────────────────────────────────────

class RegimeAlignmentWarning(SchismWarning):
    """
    Issued when Hungarian alignment cost Ci,π*(i) > δ_align for any state i.

    Indicates a regime discontinuity — the new model's state i does not
    correspond well to the old model's state π*(i). Per spec §9, live
    deployment should be held for manual review.

    Attributes:
        state_index      — 0-based index of the drifted state (new model)
        old_state_index  — 0-based index of the matched old state
        cost             — Hungarian cost for this pair (L2² distance of means)
        delta_align      — configured threshold
        model_ver_old    — checkpoint version of the old model
        model_ver_new    — checkpoint version of the new model
    """

    def __init__(
        self,
        message: str,
        *,
        state_index: int = 0,
        old_state_index: int = 0,
        cost: float = 0.0,
        delta_align: float = 2.0,
        model_ver_old: str = "",
        model_ver_new: str = "",
    ) -> None:
        super().__init__(message)
        self.state_index = state_index
        self.old_state_index = old_state_index
        self.cost = cost
        self.delta_align = delta_align
        self.model_ver_old = model_ver_old
        self.model_ver_new = model_ver_new

    def __str__(self) -> str:
        return (
            f"RegimeAlignmentWarning: state {self.old_state_index} → {self.state_index} "
            f"cost={self.cost:.4f} > δ_align={self.delta_align:.4f} "
            f"(old={self.model_ver_old}, new={self.model_ver_new})"
        )