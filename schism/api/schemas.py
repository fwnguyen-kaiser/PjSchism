"""Pydantic v2 response models for all endpoints. Shared across routers."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RegimeSnapshot(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    bar_ts: datetime
    instrument_id: int
    timeframe_id: int
    state: int
    label: str
    confidence: float | None
    posterior: list[float] | None
    model_ver: str | None


class BarWithRegime(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    bar_ts: datetime
    close: float | None
    volume: float | None
    state: int
    label: str
    confidence: float | None


class RegimeStats(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    state: int
    label: str
    frequency_pct: float
    mean_sojourn_bars: float


class RefitLogEntry(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    refit_id: int
    refit_ts: datetime
    instrument_id: int
    timeframe_id: int
    trigger: str
    delta_bic: float | None
    alignment_ok: bool | None
    drift_alert: bool
    dim_used: int | None
    model_ver: str | None
    cooldown_end_ts: datetime | None
    notes: str | None
