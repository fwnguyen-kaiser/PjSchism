"""
Walk-forward validation for IOHMM regime model.

Fits on rolling train window, evaluates on non-overlapping OOS test window.
Checks: OOS LL generalisation, state character stability, OOS backtest PnL.

Layout (default):
  train_bars = 1080  (~6 months at 4h)
  test_bars  =  360  (~2 months)
  step_bars  =  360  (non-overlapping test windows)

  |─── train 1080 ───|─ test 360 ─|
                     |─── train 1080 ───|─ test 360 ─|
                                        |─── train 1080 ───|─ test 360 ─|

Per fold:
  - Fit IOHMM on train valid rows
  - Score OOS LL/bar
  - Viterbi decode + filter posteriors on test window
  - Compute fwd_ret by state (state character check)
  - Compute R_bal position and net PnL

Usage:
    python scripts/run_wf.py [--train 1080] [--test 360] [--step 360]
                             [--n-em-runs 1] [--cost-bps 4] [--smooth 0.05]
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from schism.data.preprocessing.zscore import RollingZScore
from schism.models.iohmm import IOHMM
from schism.utils.config_loader import load_yaml

_OHLCV_GLOB   = "schism/data/volumes/parquet/symbol=BTCUSDT/**/*.parquet"
_METRICS_GLOB = "schism/data/volumes/parquet/metrics/symbol=BTCUSDT/**/*.parquet"

_MODEL_CFG    = load_yaml("model_config.yaml")
_FEATURE_CFG  = load_yaml("feature_config.yaml")
_VAL_CFG      = load_yaml("validation_config.yaml")

_O_COLS = [
    "f1_cvd_vol", "f2_oi_chg", "f3_norm_ret", "f4_liq_sq",
    "f6_illiq", "f7_rv_ratio", "f8_vol_shock", "f9_flow_liq", "f10_flow_pos",
]
_U_COLS   = ["u1_ewma_fr", "u2_delta_fr", "u3_fr_spread"]
_RV_COL   = _O_COLS.index("f7_rv_ratio")
_W_BAL    = np.array([0.5, 1.0, -0.5])   # R_bal weights (one per state, K=3)
_STATE_NAMES = {0: "state_0", 1: "state_1", 2: "state_2"}
_BARS_PER_YEAR = 365 * 6


# ── Feature pipeline (same as run_label.py) ───────────────────────────────────

def _load_and_build() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (df_sorted, O_full, U_full, valid_mask)."""
    _alpha_vol  = _FEATURE_CFG.get("alpha_vol_ewma",    0.1)
    _alpha_fr   = _FEATURE_CFG.get("alpha_fr_ewma",     0.3)
    _rv24h      = _FEATURE_CFG.get("rv24h_warmup_bars", 6)
    _rv7d       = _FEATURE_CFG.get("rv7d_warmup_bars",  42)
    _eps        = _FEATURE_CFG.get("epsilon",            1e-8)
    _win_lo, _win_hi = (_FEATURE_CFG.get("winsorize_pct") or [1, 99])
    _zscore_win = _VAL_CFG.get("zscore_window_bars", 360)

    # Load
    ohlcv = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(_OHLCV_GLOB, recursive=True))],
        ignore_index=True,
    )
    metrics = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(_METRICS_GLOB, recursive=True))],
        ignore_index=True,
    )
    ohlcv["bar_ts"]   = pd.to_datetime(ohlcv["bar_ts"],   utc=True)
    metrics["bar_ts"] = pd.to_datetime(metrics["bar_ts"], utc=True)
    metrics = metrics.set_index("bar_ts")
    metrics = metrics[["sum_open_interest", "top_ls_ratio"]].resample("4h").last()
    metrics = metrics.rename(columns={"sum_open_interest": "oi", "top_ls_ratio": "lsr_top"}).reset_index()

    ohlcv = ohlcv.drop(columns=["oi", "lsr_top"], errors="ignore")
    df = ohlcv.merge(metrics, on="bar_ts", how="left").sort_values("bar_ts").reset_index(drop=True)

    # Features
    log_ret  = np.log(df["close"] / df["close"].shift(1))
    rv24h    = (log_ret**2).rolling(_rv24h, min_periods=1).sum()
    rv7d     = (log_ret**2).rolling(_rv7d,  min_periods=1).sum()
    delta_oi = df["oi"] - df["oi"].shift(1)
    oi_prev  = df["oi"].shift(1).replace(0.0, np.nan)
    vol_safe = df["volume"].replace(0.0, np.nan)

    feat = df[["bar_ts", "close"]].copy()
    feat["f1_cvd_vol"]   = df["cvd"] / vol_safe
    feat["f2_oi_chg"]    = delta_oi / oi_prev
    feat["f3_norm_ret"]  = np.arcsinh(log_ret / np.sqrt(rv24h.clip(lower=_eps)))
    feat["f4_liq_sq"]    = np.arcsinh(-np.minimum(0.0, delta_oi.fillna(0.0)) * np.sign(log_ret))
    feat["f6_illiq"]     = np.log(log_ret.abs() / (df["volume"] + _eps))
    feat["f7_rv_ratio"]  = rv24h / rv7d.clip(lower=_eps)
    ewma_vol = df["volume"].ewm(alpha=_alpha_vol, adjust=False).mean()
    feat["f8_vol_shock"] = np.log(df["volume"] / ewma_vol.clip(lower=_eps))
    f1_s = feat["f1_cvd_vol"]
    feat["f9_flow_liq"]  = f1_s * feat["f6_illiq"]
    feat["f10_flow_pos"] = f1_s * feat["f2_oi_chg"]

    fr = df["funding_rate"]
    feat["u1_ewma_fr"]   = fr.ewm(alpha=_alpha_fr, adjust=False).mean()
    feat["u2_delta_fr"]  = fr - fr.shift(1)
    feat["u3_fr_spread"] = fr - df["bybit_fr"]

    # Winsorize
    all_cols = _O_COLS + _U_COLS
    for col in all_cols:
        s  = feat[col].dropna()
        if s.empty:
            continue
        lo = float(np.percentile(s, _win_lo))
        hi = float(np.percentile(s, _win_hi))
        feat[col] = feat[col].clip(lo, hi)

    # Rolling z-score
    for col in all_cols:
        feat[col] = RollingZScore.batch_transform(feat[col].to_numpy(), window=_zscore_win)

    O = feat[_O_COLS].to_numpy(dtype=float)
    U = feat[_U_COLS].to_numpy(dtype=float)
    valid_mask = ~np.isnan(O).any(axis=1)

    # Forward return at each bar (for backtest PnL)
    lr = np.log(df["close"].to_numpy() / np.roll(df["close"].to_numpy(), 1))
    lr[0] = 0.0
    df["log_ret"] = lr

    print(f"Loaded: {len(df)} bars  valid: {valid_mask.sum()}  "
          f"({df['bar_ts'].iloc[0].date()} to {df['bar_ts'].iloc[-1].date()})")
    return df, O, U, valid_mask


def _apply_smoothing(raw: np.ndarray, thr: float) -> np.ndarray:
    if thr <= 0.0:
        return raw.copy()
    pos, cur = raw.copy(), raw[0]
    for t in range(1, len(raw)):
        if abs(raw[t] - cur) > thr:
            cur = raw[t]
        pos[t] = cur
    return pos


def _sharpe(pnl: np.ndarray) -> float:
    std = float(np.std(pnl))
    return float(np.mean(pnl)) / std * np.sqrt(_BARS_PER_YEAR) if std > 0 else 0.0


def _max_dd(pnl: np.ndarray) -> float:
    cum = np.cumsum(pnl)
    return float(np.min(cum - np.maximum.accumulate(cum)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",      type=int,   default=1080)
    ap.add_argument("--test",       type=int,   default=360)
    ap.add_argument("--step",       type=int,   default=None,
                    help="step size (default = test bars, non-overlapping)")
    ap.add_argument("--n-em-runs",  type=int,   default=1,
                    help="EM restarts per fold (1 = fast)")
    ap.add_argument("--cost-bps",   type=float, default=4.0)
    ap.add_argument("--smooth",     type=float, default=0.05)
    ap.add_argument("--out",        default="wf_results.csv")
    args = ap.parse_args()

    step      = args.step or args.test
    cost_pu   = args.cost_bps / 10_000
    n_states  = int(_MODEL_CFG.get("K", 4))

    sep  = "=" * 88
    sep2 = "-" * 88

    # ── Load data ─────────────────────────────────────────────────────────────
    df, O, U, valid_mask = _load_and_build()
    valid_idx   = np.where(valid_mask)[0]
    n_valid     = len(valid_idx)

    # Forward return aligned to valid rows (shift -1: earn next bar)
    log_ret_full = df["log_ret"].to_numpy()
    fwd_ret_full = np.roll(log_ret_full, -1)
    fwd_ret_full[-1] = 0.0

    n_folds = (n_valid - args.train) // step
    if n_folds < 1:
        print(f"ERROR: not enough data for even one fold "
              f"({n_valid} valid bars, need {args.train + step})")
        sys.exit(1)

    print(f"\n{sep}")
    print(f" WALK-FORWARD  |  train={args.train}  test={args.test}  "
          f"step={step}  n_em_runs={args.n_em_runs}  n_folds={n_folds}")
    print(f" Signal: R_bal {_W_BAL.tolist()}  smooth={args.smooth}  "
          f"cost={args.cost_bps}bps")
    print(sep)

    # ── Fold header ───────────────────────────────────────────────────────────
    hdr = (f"  {'fold':>4}  {'train_end':>10}  {'test_end':>10}  "
           f"{'LL_is':>8}  {'LL_oos':>8}  {'gap':>6}  "
           + "".join(f"  {f'S{k}%':>5}" for k in range(n_states))
           + f"  {'R_bal_sh':>9}  {'R_bal_dd':>9}")
    print(hdr)
    print(f"  {sep2[:86]}")

    # ── Per-fold storage ──────────────────────────────────────────────────────
    fold_records = []

    for fold in range(n_folds):
        tr_start = fold * step
        tr_end   = tr_start + args.train
        te_end   = tr_end   + args.test

        if te_end > n_valid:
            break

        vi_train = valid_idx[tr_start:tr_end]
        vi_test  = valid_idx[tr_end:te_end]

        O_tr, U_tr = O[vi_train], U[vi_train]
        O_te, U_te = O[vi_test],  U[vi_test]

        # ── Fit ───────────────────────────────────────────────────────────────
        model = IOHMM.from_config(
            _MODEL_CFG,
            n_states=n_states,
            n_obs=len(_O_COLS),
            n_exog=len(_U_COLS),
            rv_col=_RV_COL,
            n_em_runs=args.n_em_runs,
        )
        model.fit(O_tr, U_tr)
        ll_is  = float(model.ll_history[-1])
        ll_oos = float(model.score(O_te, U_te))
        gap    = ll_oos - ll_is

        # ── Decode test ───────────────────────────────────────────────────────
        states_te = model.decode(O_te, U_te)          # (n_test,) hard
        gamma_te  = model.filter(O_te, U_te)           # (n_test, K) soft

        state_freq = np.array([(states_te == k).mean() for k in range(n_states)])

        # ── OOS backtest ──────────────────────────────────────────────────────
        fwd_te  = fwd_ret_full[vi_test]
        raw_pos = gamma_te @ _W_BAL
        pos     = _apply_smoothing(raw_pos, args.smooth)
        delta   = np.abs(np.diff(pos, prepend=pos[0]))
        net_pnl = pos * fwd_te - delta * cost_pu
        sh_bal  = _sharpe(net_pnl)
        dd_bal  = _max_dd(net_pnl)

        # ── fwd_ret by state ──────────────────────────────────────────────────
        fwd_by_state = {}
        for k in range(n_states):
            m_k = (states_te == k)
            fwd_by_state[k] = float(fwd_te[m_k].mean()) if m_k.sum() > 0 else np.nan

        # ── Date labels ───────────────────────────────────────────────────────
        date_tr_end = df["bar_ts"].iloc[vi_train[-1]].strftime("%Y-%m-%d")
        date_te_end = df["bar_ts"].iloc[vi_test[-1]].strftime("%Y-%m-%d")

        row = (f"  {fold+1:>4}  {date_tr_end:>10}  {date_te_end:>10}  "
               f"{ll_is:>8.4f}  {ll_oos:>8.4f}  {gap:>+6.3f}  "
               + "".join(f"  {state_freq[k]*100:>4.0f}%" for k in range(n_states))
               + f"  {sh_bal:>+9.3f}  {dd_bal*100:>+9.2f}%")
        print(row)

        fold_records.append({
            "fold":        fold + 1,
            "date_tr_end": date_tr_end,
            "date_te_end": date_te_end,
            "ll_is":       ll_is,
            "ll_oos":      ll_oos,
            "ll_gap":      gap,
            **{f"s{k}_freq": state_freq[k] for k in range(n_states)},
            **{f"s{k}_fwd1": fwd_by_state[k] for k in range(n_states)},
            "rbal_sharpe": sh_bal,
            "rbal_dd":     dd_bal,
        })

    if not fold_records:
        print("No folds completed.")
        return

    rec = pd.DataFrame(fold_records)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f" SUMMARY  ({len(rec)} folds)")
    print(sep)

    # LL generalisation
    print(f"\n  LL/bar generalisation:")
    print(f"  {'':20}  {'mean':>8}  {'std':>7}  {'min':>8}  {'max':>8}")
    print(f"  {'-'*56}")
    for col, label in [("ll_is", "LL in-sample"), ("ll_oos", "LL OOS"), ("ll_gap", "LL gap (oos-is)")]:
        s = rec[col]
        print(f"  {label:20}  {s.mean():>8.4f}  {s.std():>7.4f}  {s.min():>8.4f}  {s.max():>8.4f}")

    # State frequency stability
    print(f"\n  State frequency across folds (should be stable):")
    print(f"  {'state':20}  {'mean%':>7}  {'std%':>7}  {'min%':>7}  {'max%':>7}")
    print(f"  {'-'*48}")
    for k in range(n_states):
        s = rec[f"s{k}_freq"] * 100
        print(f"  {_STATE_NAMES[k]:20}  {s.mean():>6.1f}%  {s.std():>6.1f}%  "
              f"{s.min():>6.1f}%  {s.max():>6.1f}%")

    # OOS fwd_ret by state vs in-sample
    # Load in-sample numbers from labels.csv if available
    labels_path = ROOT / "labels.csv"
    is_fwd: dict[int, float] = {}
    if labels_path.exists():
        ldf = pd.read_csv(labels_path)
        ldf = ldf[ldf["state"].notna()].copy()
        ldf["state"] = ldf["state"].astype(int)
        ldf_s = ldf.sort_values("bar_ts").reset_index(drop=True)
        lr_l = np.log(ldf_s["close"] / ldf_s["close"].shift(1)).to_numpy()
        fwd_l = np.roll(lr_l, -1); fwd_l[-1] = 0.0
        for k in range(n_states):
            m = (ldf_s["state"].to_numpy() == k)
            is_fwd[k] = float(fwd_l[m].mean()) if m.sum() > 0 else np.nan

    print(f"\n  OOS fwd_ret_1 by state  (mean across folds vs full in-sample):")
    hdr_f = f"  {'state':20}  {'OOS mean':>9}  {'OOS std':>8}  {'IS ref':>9}  {'drift':>8}"
    print(hdr_f)
    print(f"  {'-'*60}")
    for k in range(n_states):
        col   = f"s{k}_fwd1"
        oos_m = float(rec[col].mean())
        oos_s = float(rec[col].std())
        is_r  = is_fwd.get(k, np.nan)
        drift = oos_m - is_r if not np.isnan(is_r) else np.nan
        drift_str = f"{drift*100:>+8.3f}%" if not np.isnan(drift) else "     n/a"
        print(f"  {_STATE_NAMES[k]:20}  {oos_m*100:>+8.3f}%  {oos_s*100:>7.3f}%  "
              f"{is_r*100:>+8.3f}%  {drift_str}")

    # R_bal OOS backtest distribution
    print(f"\n  R_bal OOS Sharpe distribution across folds:")
    sh  = rec["rbal_sharpe"]
    dd  = rec["rbal_dd"] * 100
    pos_folds = int((sh > 0).sum())
    print(f"  mean={sh.mean():+.3f}  std={sh.std():.3f}  "
          f"min={sh.min():+.3f}  max={sh.max():+.3f}  "
          f"positive={pos_folds}/{len(rec)} ({pos_folds/len(rec)*100:.0f}%)")
    print(f"  mean DD={dd.mean():.2f}%  std={dd.std():.2f}%  worst={dd.min():.2f}%")

    # Per-fold Sharpe table
    print(f"\n  Per-fold R_bal Sharpe:")
    row_vals = "  " + "  ".join(f"{v:+.2f}" for v in sh.tolist())
    print(row_vals)

    # Walk-forward verdict
    print(f"\n  {sep2[:70]}")
    ll_gap_mean = float(rec["ll_gap"].mean())
    sh_mean     = float(sh.mean())
    freq_cv     = float(rec[[f"s{k}_freq" for k in range(n_states)]].std().mean() /
                        rec[[f"s{k}_freq" for k in range(n_states)]].mean().mean())

    verdict_ll   = "OK" if abs(ll_gap_mean) < 0.3 else "WARN (high LL gap)"
    verdict_sh   = "OK" if sh_mean > 0.2 else ("MARGINAL" if sh_mean > 0 else "FAIL")
    verdict_freq = "STABLE" if freq_cv < 0.2 else ("MODERATE" if freq_cv < 0.4 else "UNSTABLE")

    print(f"  LL generalisation : {verdict_ll}  (mean gap={ll_gap_mean:+.4f})")
    print(f"  OOS R_bal Sharpe  : {verdict_sh}  (mean={sh_mean:+.3f})")
    print(f"  State freq CV     : {verdict_freq}  (cv={freq_cv:.3f})")
    print(f"  {sep2[:70]}")

    # ── Save ──────────────────────────────────────────────────────────────────
    rec.to_csv(args.out, index=False)
    print(f"\n  Fold details saved -> {Path(args.out).resolve()}")
    print(sep)


if __name__ == "__main__":
    main()
