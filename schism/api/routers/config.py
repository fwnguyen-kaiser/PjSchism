"""
GET  /config/model   — read model_config.yaml
PUT  /config/model   — write model_config.yaml
GET  /config/refit   — read refit_config.yaml
PUT  /config/refit   — write refit_config.yaml
POST /config/apply   — write both configs + signal engine via trigger flag
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from schism.utils.config_loader import load_yaml, _CONFIG_DIR

router = APIRouter()

_TRIGGER_FLAG = Path(os.environ.get("MODEL_PATH", "model_latest.pkl")).parent / "refit_trigger.flag"


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    K: int = Field(..., ge=2, le=8, description="Number of latent states")
    Dim: int = Field(..., ge=9, le=10, description="Observation dimension (9 or 10)")
    lambda_reg: float = Field(..., gt=0, description="L2 penalty on exogenous weights")
    tau_percentile: float = Field(..., ge=50, le=99, description="Covariance trace ceiling percentile")
    n_em_runs: int = Field(..., ge=1, le=20, description="EM multi-start restarts")
    max_iter: int = Field(..., ge=50, le=2000, description="Max EM iterations per run")
    tol: float = Field(..., gt=0, description="EM log-likelihood convergence threshold")
    covariance_floor: float = Field(..., gt=0, description="Diagonal floor added to Sigma_k after M-step")
    sticky_kappa: float = Field(..., ge=1, description="Self-transition bias (1=none)")


class RefitConfig(BaseModel):
    cooldown_bars: int = Field(..., ge=1, description="Minimum bars between refits")
    backstop_days: int = Field(..., ge=1, description="Force refit after this many days")
    rv_ratio_thresh: float = Field(..., gt=0, description="RV ratio alert threshold")
    rv_ratio_consecutive_bars: int = Field(..., ge=1, description="Consecutive bars above threshold to trigger")
    ll_rolling_window_days: int = Field(..., ge=1, description="Rolling window for LL degradation check")
    ll_degradation_sigma: float = Field(..., gt=0, description="Sigma threshold for LL degradation alert")
    delta_align: float = Field(..., gt=0, description="State alignment distance threshold")


class ApplyRequest(BaseModel):
    model_cfg: ModelConfig
    refit_cfg: RefitConfig

    model_config = {"protected_namespaces": ()}


class ApplyResponse(BaseModel):
    saved: bool
    triggered: bool
    trigger_path: str
    message: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_yaml(filename: str, data: dict) -> None:
    path = _CONFIG_DIR / filename
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/model", response_model=ModelConfig)
async def get_model_config() -> ModelConfig:
    try:
        return ModelConfig(**load_yaml("model_config.yaml"))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to read model_config.yaml: {exc}")


@router.put("/model", response_model=ModelConfig)
async def put_model_config(cfg: ModelConfig) -> ModelConfig:
    try:
        _write_yaml("model_config.yaml", cfg.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to write model_config.yaml: {exc}")
    return cfg


@router.get("/refit", response_model=RefitConfig)
async def get_refit_config() -> RefitConfig:
    try:
        return RefitConfig(**load_yaml("refit_config.yaml"))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to read refit_config.yaml: {exc}")


@router.put("/refit", response_model=RefitConfig)
async def put_refit_config(cfg: RefitConfig) -> RefitConfig:
    try:
        _write_yaml("refit_config.yaml", cfg.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to write refit_config.yaml: {exc}")
    return cfg


@router.post("/apply", response_model=ApplyResponse)
async def apply_config(body: ApplyRequest) -> ApplyResponse:
    try:
        _write_yaml("model_config.yaml", body.model_cfg.model_dump())
        _write_yaml("refit_config.yaml", body.refit_cfg.model_dump())  # noqa: E501
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to write configs: {exc}")

    triggered = False
    try:
        _TRIGGER_FLAG.parent.mkdir(parents=True, exist_ok=True)
        _TRIGGER_FLAG.touch()
        triggered = True
    except Exception:
        pass  # flag write failure is non-fatal; config is already saved

    return ApplyResponse(
        saved=True,
        triggered=triggered,
        trigger_path=str(_TRIGGER_FLAG),
        message=(
            "Config saved and refit triggered — engine will refit on next loop."
            if triggered
            else "Config saved. Could not write trigger flag; engine will pick up on next natural refit."
        ),
    )
