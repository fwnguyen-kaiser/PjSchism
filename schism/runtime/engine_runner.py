"""Runtime engine: initial fit, full-history decode, then live forward-filter loop."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from schism.models.alignment import align_states, apply_permutation
from schism.models.iohmm import IOHMM
from schism.persistence.db import create_engine, create_session_factory, session_scope
from schism.persistence.repositories.feature_repo import (
    FeatureRepository,
    _O_FEATURE_COLS,
    _U_FEATURE_COLS,
)
from schism.persistence.repositories.state_repo import (
    get_current,
    resolve_instrument_id,
    resolve_timeframe_id,
    upsert_states,
)
from schism.runtime.refit_monitor import RefitMonitor
from schism.runtime.refit_scheduler import RefitScheduler
from schism.utils.config_loader import load_yaml
from schism.utils.logger import ingestion_logger

_LOG = ingestion_logger

# ── Central config ────────────────────────────────────────────────────────────
_MODEL_CFG  = load_yaml("model_config.yaml")
_REFIT_CFG  = load_yaml("refit_config.yaml")

# ── Runtime config (env vars override yaml; yaml overrides inline defaults) ───
_EXCHANGE      = os.environ.get("SCHISM_EXCHANGE", "binance")
_SYMBOL        = os.environ.get("SCHISM_SYMBOL", "BTCUSDT")
_MARKET_TYPE   = os.environ.get("SCHISM_MARKET_TYPE", "perp")
_TIMEFRAME     = os.environ.get("SCHISM_TIMEFRAME", "4h")
_N_STATES      = int(os.environ.get("SCHISM_N_STATES",      str(_MODEL_CFG.get("K", 4))))
_TRAIN_WINDOW  = int(os.environ.get("SCHISM_TRAIN_WINDOW",  "1080"))
_MODEL_PATH    = Path(os.environ.get("MODEL_PATH",           "/data/models/iohmm.pkl"))
_BAR_SECONDS   = int(os.environ.get("BAR_INTERVAL_SECONDS", str(4 * 3600)))
_WARMUP_BARS   = int(os.environ.get("WARMUP_BARS",          "60"))

_LL_WINDOW     = int(os.environ.get("REFIT_LL_WINDOW",    str(_REFIT_CFG.get("ll_rolling_window_days", 30) * 6)))
_RV_THRESHOLD  = float(os.environ.get("REFIT_RV_THRESHOLD", str(_REFIT_CFG.get("rv_ratio_thresh", 1.8))))
_RV_CONSEC     = int(os.environ.get("REFIT_RV_CONSEC",    str(_REFIT_CFG.get("rv_ratio_consecutive_bars", 12))))
_BACKSTOP_BARS = int(os.environ.get("REFIT_BACKSTOP_BARS", str(_REFIT_CFG.get("backstop_days", 90) * 6)))
_COOLDOWN_BARS = int(os.environ.get("REFIT_COOLDOWN_BARS", str(_REFIT_CFG.get("cooldown_bars", 30))))

_HISTORY_START = datetime(2020, 1, 1, tzinfo=timezone.utc)

# f7_rv_ratio is index 6 in _O_FEATURE_COLS (0-based)
_RV_COL_IDX = _O_FEATURE_COLS.index("f7_rv_ratio")


# ── Array helpers ─────────────────────────────────────────────────────────────

def _df_to_arrays(
    df,
    zero_f8: bool = False,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Extract (O, U, timestamps) from a feature_vectors DataFrame.

    f5_spread NaN → 0  : historical bars have no bid/ask; keeps row valid.
    zero_f8=True       : set f8_vol_shock to 0 when VIF≥5 dominates training
                         data (§1.2 C3 — F8 dropped in favour of F6).
    """
    for col in _O_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        elif col == "f5_spread":
            df[col] = df[col].fillna(0.0)

    if zero_f8 and "f8_vol_shock" in df.columns:
        df["f8_vol_shock"] = 0.0

    O = df[_O_FEATURE_COLS].to_numpy(dtype=float)   # (T, 10)
    U = df[_U_FEATURE_COLS].to_numpy(dtype=float)   # (T, 4) — NaN kept; IOHMM._safe_U zeros them
    ts = df["bar_ts"].tolist()
    return O, U, ts


def _should_zero_f8(df) -> bool:
    """§1.2 C3: zero out f8 when the majority of rows have dim_used=9 (VIF≥5)."""
    if "dim_used" not in df.columns:
        return False
    dim_used = df["dim_used"].dropna()
    if dim_used.empty:
        return False
    return bool((dim_used == 9).mean() > 0.5)


def _gamma_to_state_rows(
    ts_list: list,
    gamma: np.ndarray,
    model: IOHMM,
) -> list[dict]:
    rows = []
    for ts, g in zip(ts_list, gamma):
        s = int(g.argmax())
        rows.append(
            {
                "bar_ts": ts,
                "state": s,
                "label": model.labels[s],
                "confidence": float(g.max()),
                "posterior": g.tolist(),
                "model_ver": model.model_ver,
            }
        )
    return rows


def _rebuild_log_alpha(model: IOHMM, O: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Run forward pass on warmup window; return last log_alpha (K,)."""
    log_b = model._log_emission(O)
    log_A = model._log_transition(model._safe_U(U))
    log_alpha_seq, _ = model._forward(log_b, log_A)
    return log_alpha_seq[-1]


# ── Initial fit + full-history decode ─────────────────────────────────────────

async def _initial_fit_and_decode(
    feature_repo: FeatureRepository,
    instrument_id: int,
    timeframe_id: int,
    session_factory,
    existing_model: IOHMM | None,
) -> tuple[IOHMM, np.ndarray, datetime]:
    """
    Fit (or reuse) IOHMM, decode full history into state_history.
    Returns (model, last_log_alpha, last_bar_ts).
    """
    now = datetime.now(tz=timezone.utc)
    train_from = now - timedelta(seconds=_TRAIN_WINDOW * _BAR_SECONDS)

    train_df = await feature_repo.fetch_features(instrument_id, timeframe_id, train_from, now)
    if train_df.empty:
        raise RuntimeError("No feature data found — run feature backfill first")

    zero_f8 = _should_zero_f8(train_df)
    O_train, U_train, _ = _df_to_arrays(train_df.copy(), zero_f8=zero_f8)

    if existing_model is None:
        _LOG.info("engine_fitting_model", train_bars=len(O_train), zero_f8=zero_f8)
        model = IOHMM.from_config(
            _MODEL_CFG,
            n_states=_N_STATES,
            n_obs=len(_O_FEATURE_COLS),
            n_exog=len(_U_FEATURE_COLS),
            rv_col=_RV_COL_IDX,
        )
        model.fit(O_train, U_train)
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save(_MODEL_PATH)
    else:
        model = existing_model
        _LOG.info("engine_reusing_model", model_ver=model.model_ver)

    # Decode full history with filter (causal posterior) and write to state_history
    full_df = await feature_repo.fetch_features(instrument_id, timeframe_id, _HISTORY_START, now)
    if not full_df.empty:
        O_full, U_full, ts_full = _df_to_arrays(full_df.copy(), zero_f8=zero_f8)
        gamma_full = model.filter(O_full, U_full)
        model.log_eval_criteria(gamma_full)
        state_rows = _gamma_to_state_rows(ts_full, gamma_full, model)

        async with session_scope(session_factory) as session:
            written = await upsert_states(session, instrument_id, timeframe_id, state_rows)
        _LOG.info("engine_state_backfill_done", rows_written=written)

        warmup_O = O_full[-_WARMUP_BARS:]
        warmup_U = U_full[-_WARMUP_BARS:]
        last_log_alpha = _rebuild_log_alpha(model, warmup_O, warmup_U)
        last_ts = ts_full[-1]
    else:
        last_log_alpha = np.log(model.pi + 1e-10)
        last_ts = train_df["bar_ts"].iloc[-1]

    return model, last_log_alpha, last_ts


# ── Refit ─────────────────────────────────────────────────────────────────────

async def _do_refit(
    model: IOHMM,
    feature_repo: FeatureRepository,
    instrument_id: int,
    timeframe_id: int,
    session_factory,
) -> IOHMM:
    """Refit on latest training window, align states, overwrite state_history window."""
    now = datetime.now(tz=timezone.utc)
    train_from = now - timedelta(seconds=_TRAIN_WINDOW * _BAR_SECONDS)

    train_df = await feature_repo.fetch_features(instrument_id, timeframe_id, train_from, now)
    if train_df.empty:
        _LOG.warning("refit_skipped_no_data")
        return model

    zero_f8 = _should_zero_f8(train_df)
    O, U, ts_list = _df_to_arrays(train_df.copy(), zero_f8=zero_f8)
    mu_old = model.mu.copy()

    new_model = IOHMM.from_config(
        _MODEL_CFG,
        n_states=model.K, n_obs=model.D, n_exog=model.M,
        rv_col=model.rv_col,
    )
    new_model.fit(O, U)

    perm, drift_alert = align_states(mu_old, new_model.mu)
    apply_permutation(new_model, perm)
    new_model.labels = list(model.labels)  # preserve human-assigned names

    new_model.save(_MODEL_PATH)
    _LOG.info("refit_complete", drift_alert=drift_alert, model_ver=new_model.model_ver)

    # Rewrite state_history for the training window with new model's assignments
    gamma = new_model.filter(O, U)
    new_model.log_eval_criteria(gamma)
    state_rows = _gamma_to_state_rows(ts_list, gamma, new_model)
    async with session_scope(session_factory) as session:
        await upsert_states(session, instrument_id, timeframe_id, state_rows)

    return new_model


# ── Main loop ─────────────────────────────────────────────────────────────────

async def run_forever() -> None:
    _LOG.info(
        "regime_engine_runner_started",
        exchange=_EXCHANGE, symbol=_SYMBOL,
        market_type=_MARKET_TYPE, timeframe=_TIMEFRAME,
    )

    db_engine = create_engine()
    if db_engine is None:
        raise RuntimeError("DATABASE_URL not configured")

    session_factory = create_session_factory(db_engine)
    feature_repo = FeatureRepository(session_factory)

    _RESOLVE_RETRIES = 30
    _RESOLVE_SLEEP   = 10  # seconds
    instrument_id: int | None = None
    timeframe_id:  int | None = None
    for attempt in range(_RESOLVE_RETRIES):
        async with session_scope(session_factory) as session:
            instrument_id = await resolve_instrument_id(session, _EXCHANGE, _SYMBOL, _MARKET_TYPE)
            timeframe_id  = await resolve_timeframe_id(session, _TIMEFRAME)
        if instrument_id is not None and timeframe_id is not None:
            break
        _LOG.warning(
            "engine_waiting_for_ingestion",
            attempt=attempt + 1,
            max_attempts=_RESOLVE_RETRIES,
            sleep_s=_RESOLVE_SLEEP,
        )
        await asyncio.sleep(_RESOLVE_SLEEP)
    else:
        raise RuntimeError(
            f"Instrument or timeframe not found in DB after {_RESOLVE_RETRIES} attempts: "
            f"{_EXCHANGE}/{_SYMBOL}/{_MARKET_TYPE}/{_TIMEFRAME}"
        )
    _LOG.info("engine_ids_resolved", instrument_id=instrument_id, timeframe_id=timeframe_id)

    # Load existing model if present
    existing_model: IOHMM | None = IOHMM.load(_MODEL_PATH) if _MODEL_PATH.exists() else None

    # Check for existing state_history
    async with session_scope(session_factory) as session:
        last_state = await get_current(session, instrument_id, timeframe_id)
    last_ts: datetime | None = last_state["bar_ts"] if last_state else None

    last_log_alpha: np.ndarray

    if existing_model is None or last_ts is None:
        model, last_log_alpha, last_ts = await _initial_fit_and_decode(
            feature_repo, instrument_id, timeframe_id, session_factory, existing_model
        )
    else:
        model = existing_model
        # Rebuild log_alpha from warmup window
        warmup_from = last_ts - timedelta(seconds=_WARMUP_BARS * _BAR_SECONDS)
        warmup_df = await feature_repo.fetch_features(
            instrument_id, timeframe_id, warmup_from, last_ts
        )
        if not warmup_df.empty:
            O_w, U_w, _ = _df_to_arrays(warmup_df)
            last_log_alpha = _rebuild_log_alpha(model, O_w, U_w)
        else:
            last_log_alpha = np.log(model.pi + 1e-10)

    monitor = RefitMonitor(
        ll_window=_LL_WINDOW,
        rv_threshold=_RV_THRESHOLD,
        rv_consec=_RV_CONSEC,
        backstop_bars=_BACKSTOP_BARS,
    )
    scheduler = RefitScheduler(cooldown_bars=_COOLDOWN_BARS)

    _LOG.info("engine_live_loop_start", last_ts=last_ts.isoformat() if last_ts else None)

    while True:
        await asyncio.sleep(_BAR_SECONDS)

        now = datetime.now(tz=timezone.utc)
        new_df = await feature_repo.fetch_features(
            instrument_id, timeframe_id,
            from_ts=last_ts + timedelta(seconds=1),
            to_ts=now,
        )

        if new_df.empty:
            continue

        O_new, U_new, ts_new = _df_to_arrays(new_df.copy(), zero_f8=_should_zero_f8(new_df))
        state_rows: list[dict] = []

        for idx in range(len(O_new)):
            o_t = O_new[idx]
            u_t = U_new[idx]

            prev_ll_norm = float(logsumexp(last_log_alpha))
            last_log_alpha, gamma_t = model.filter_step(last_log_alpha, o_t, u_t)
            delta_ll = float(logsumexp(last_log_alpha)) - prev_ll_norm

            rv_ratio = float(o_t[_RV_COL_IDX]) if not np.isnan(o_t[_RV_COL_IDX]) else 1.0

            s = int(gamma_t.argmax())
            state_rows.append(
                {
                    "bar_ts": ts_new[idx],
                    "state": s,
                    "label": model.labels[s],
                    "confidence": float(gamma_t.max()),
                    "posterior": gamma_t.tolist(),
                    "model_ver": model.model_ver,
                }
            )

            scheduler.tick()
            if monitor.update(delta_ll, rv_ratio, scheduler.bars_since_last):
                backstop = monitor.backstop_triggered(scheduler.bars_since_last)
                if scheduler.can_refit(backstop_override=backstop):
                    model = await _do_refit(
                        model, feature_repo, instrument_id, timeframe_id, session_factory
                    )
                    scheduler.record_refit()
                    monitor.reset()
                    # Rebuild log_alpha under the new model
                    cur_ts = ts_new[idx]
                    wf = cur_ts - timedelta(seconds=_WARMUP_BARS * _BAR_SECONDS)
                    wd = await feature_repo.fetch_features(instrument_id, timeframe_id, wf, cur_ts)
                    if not wd.empty:
                        O_w, U_w, _ = _df_to_arrays(wd)
                        last_log_alpha = _rebuild_log_alpha(model, O_w, U_w)

        async with session_scope(session_factory) as session:
            await upsert_states(session, instrument_id, timeframe_id, state_rows)

        last_ts = ts_new[-1]
        _LOG.info("engine_bars_processed", count=len(state_rows), last_ts=str(last_ts))
