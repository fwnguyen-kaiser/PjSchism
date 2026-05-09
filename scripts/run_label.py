"""
Standalone: load parquet → compute features → fit IOHMM → output CSV for labeling.

Usage:
    python scripts/run_label.py [--train-bars 1080] [--n-states 4] [--out labels.csv]

Output columns:
    bar_ts, open, high, low, close, volume,
    state, label, confidence, p0, p1, p2, p3

IOHMM structure:
    Transition (U_t): u1_ewma_fr, u2_delta_fr, u3_fr_spread  [3 features]
    Observation (O_t): f1-f4, f6-f10                          [9 features]
    Note: f5_spread excluded — bid/ask unavailable in historical parquet (dead feature).
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from schism.data.preprocessing.zscore import RollingZScore
from schism.models.iohmm import IOHMM
from schism.utils.config_loader import load_yaml

# ── Central config ────────────────────────────────────────────────────────────

_OHLCV_GLOB   = "schism/data/volumes/parquet/symbol=BTCUSDT/**/*.parquet"
_METRICS_GLOB = "schism/data/volumes/parquet/metrics/symbol=BTCUSDT/**/*.parquet"

_MODEL_CFG      = load_yaml("model_config.yaml")
_FEATURE_CFG    = load_yaml("feature_config.yaml")
_VALIDATION_CFG = load_yaml("validation_config.yaml")

# Feature params
_ALPHA_VOL  = _FEATURE_CFG.get("alpha_vol_ewma",    0.1)
_ALPHA_FR   = _FEATURE_CFG.get("alpha_fr_ewma",     0.3)
_RV24H_BARS = _FEATURE_CFG.get("rv24h_warmup_bars", 6)
_RV7D_BARS  = _FEATURE_CFG.get("rv7d_warmup_bars",  42)
_EPS        = _FEATURE_CFG.get("epsilon",            1e-8)
_WIN_LO, _WIN_HI = (_FEATURE_CFG.get("winsorize_pct") or [1, 99])

# Validation params
_ZSCORE_WIN = _VALIDATION_CFG.get("zscore_window_bars", 360)

# O_t: 9 observation features (f5_spread dropped — dead in historical data)
_O_COLS = [
    "f1_cvd_vol", "f2_oi_chg", "f3_norm_ret", "f4_liq_sq",
    "f6_illiq", "f7_rv_ratio", "f8_vol_shock", "f9_flow_liq", "f10_flow_pos",
]
# Index of f7_rv_ratio in _O_COLS (used to order states by volatility regime)
_RV_COL = _O_COLS.index("f7_rv_ratio")   # = 5

# U_t: 3 transition features (funding-rate drivers)
_U_COLS = ["u1_ewma_fr", "u2_delta_fr", "u3_fr_spread"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ohlcv() -> pd.DataFrame:
    files = sorted(glob.glob(_OHLCV_GLOB, recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("bar_ts").reset_index(drop=True)
    print(f"OHLCV: {len(df)} bars  {df.bar_ts.iloc[0]} to {df.bar_ts.iloc[-1]}")
    return df


def load_metrics_4h() -> pd.DataFrame:
    """Load 5-min metrics parquets and resample to 4h (last OI, last LSR)."""
    files = sorted(glob.glob(_METRICS_GLOB, recursive=True))
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    raw = raw.sort_values("bar_ts")
    raw["bar_ts"] = pd.to_datetime(raw["bar_ts"], utc=True)
    raw = raw.set_index("bar_ts")
    resampled = raw[["sum_open_interest", "top_ls_ratio"]].resample("4h").last()
    resampled = resampled.rename(columns={
        "sum_open_interest": "oi",
        "top_ls_ratio":      "lsr_top",
    }).reset_index()
    print(f"Metrics (4h): {len(resampled)} bars")
    return resampled


# ── Feature computation ───────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("bar_ts").reset_index(drop=True)

    log_ret  = np.log(df["close"] / df["close"].shift(1))
    rv24h    = (log_ret ** 2).rolling(_RV24H_BARS, min_periods=1).sum()
    rv7d     = (log_ret ** 2).rolling(_RV7D_BARS,  min_periods=1).sum()

    delta_oi = df["oi"] - df["oi"].shift(1)
    oi_prev  = df["oi"].shift(1).replace(0.0, np.nan)
    vol_safe = df["volume"].replace(0.0, np.nan)

    f1  = df["cvd"] / vol_safe
    f2  = delta_oi / oi_prev
    f3  = np.arcsinh(log_ret / np.sqrt(rv24h.clip(lower=_EPS)))
    f4  = np.arcsinh(-np.minimum(0.0, delta_oi.fillna(0.0)) * np.sign(log_ret))
    # f5_spread omitted: no bid/ask in historical data
    f6  = np.log(log_ret.abs() / (df["volume"] + _EPS))
    f7  = rv24h / rv7d.clip(lower=_EPS)
    ewma_vol = df["volume"].ewm(alpha=_ALPHA_VOL, adjust=False).mean()
    f8  = np.log(df["volume"] / ewma_vol.clip(lower=_EPS))
    f9  = f1 * f6
    f10 = f1 * f2

    fr  = df["funding_rate"]
    u1  = fr.ewm(alpha=_ALPHA_FR, adjust=False).mean()
    u2  = fr - fr.shift(1)
    u3  = df["funding_rate"] - df["bybit_fr"]

    out = df[["bar_ts", "open", "high", "low", "close", "volume"]].copy()
    for col, s in [
        ("f1_cvd_vol", f1), ("f2_oi_chg", f2), ("f3_norm_ret", f3),
        ("f4_liq_sq", f4),  ("f6_illiq", f6),   ("f7_rv_ratio", f7),
        ("f8_vol_shock", f8), ("f9_flow_liq", f9), ("f10_flow_pos", f10),
        ("u1_ewma_fr", u1), ("u2_delta_fr", u2), ("u3_fr_spread", u3),
    ]:
        out[col] = s.values
    return out


def winsorize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in _O_COLS + _U_COLS:
        s = out[col].dropna()
        if s.empty:
            continue
        lo = float(np.percentile(s, _WIN_LO))
        hi = float(np.percentile(s, _WIN_HI))
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in _O_COLS + _U_COLS:
        out[col] = RollingZScore.batch_transform(out[col].to_numpy(), window=_ZSCORE_WIN)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-bars", type=int, default=1080)
    ap.add_argument("--n-states",   type=int, default=None)
    ap.add_argument("--out",        default="labels.csv")
    args = ap.parse_args()

    # ── 1. Load & merge ───────────────────────────────────────────────────────
    ohlcv   = load_ohlcv()
    metrics = load_metrics_4h()

    ohlcv["bar_ts"]   = pd.to_datetime(ohlcv["bar_ts"],   utc=True)
    metrics["bar_ts"] = pd.to_datetime(metrics["bar_ts"], utc=True)
    ohlcv = ohlcv.drop(columns=["oi", "lsr_top"], errors="ignore")
    df = ohlcv.merge(metrics, on="bar_ts", how="left")

    # ── 2. Compute features ───────────────────────────────────────────────────
    feats = compute_features(df)
    feats = winsorize(feats)
    feats = zscore(feats)

    O = feats[_O_COLS].to_numpy(dtype=float)
    U = feats[_U_COLS].to_numpy(dtype=float)

    valid_mask = ~np.isnan(O).any(axis=1)
    n_valid = valid_mask.sum()
    print(f"Valid rows for model: {n_valid} / {len(O)}")

    if n_valid < args.train_bars:
        print(f"WARNING: fewer valid rows ({n_valid}) than train_bars ({args.train_bars}); using all valid rows.")
        train_start = 0
    else:
        train_start = n_valid - args.train_bars

    valid_idx  = np.where(valid_mask)[0]
    train_idx  = valid_idx[train_start:]
    O_train    = O[train_idx]
    U_train    = U[train_idx]

    # ── 3. Fit ────────────────────────────────────────────────────────────────
    n_states = args.n_states or _MODEL_CFG.get("K", 4)
    print(f"\nFitting IOHMM: K={n_states}  O_t={len(_O_COLS)}  U_t={len(_U_COLS)}")
    print(f"  T={len(O_train)} train bars  n_em_runs={_MODEL_CFG.get('n_em_runs', 1)}")
    model = IOHMM.from_config(
        _MODEL_CFG,
        n_states=n_states,
        n_obs=len(_O_COLS),
        n_exog=len(_U_COLS),
        rv_col=_RV_COL,
    )
    model.fit(O_train, U_train)
    print(f"Model version: {model.model_ver}")
    print(f"LL/bar (final): {model.ll_history[-1]:.6f}")
    model_path = ROOT / "model_latest.pkl"
    model.save(model_path)
    print(f"Model saved   -> {model_path}")

    # ── 4. Decode full history (Viterbi) ──────────────────────────────────────
    print("\nDecoding full history via Viterbi...")
    O_valid      = O[valid_mask]
    U_valid      = U[valid_mask]
    states_valid = model.decode(O_valid, U_valid)   # (n_valid,) hard assignment
    gamma_valid  = model.filter(O_valid, U_valid)   # (n_valid, K) soft posteriors

    # ── 5. Eval (logger) ──────────────────────────────────────────────────────
    model.log_eval_criteria(gamma_valid)

    # ── 6. Build output ───────────────────────────────────────────────────────
    result = feats[["bar_ts", "open", "high", "low", "close", "volume"]].copy()
    result["state"]      = np.nan
    result["label"]      = ""
    result["confidence"] = np.nan
    for k in range(n_states):
        result[f"p{k}"] = np.nan

    valid_positions = np.where(valid_mask)[0]
    for pos, state, g in zip(valid_positions, states_valid, gamma_valid):
        result.at[pos, "state"]      = int(state)
        result.at[pos, "label"]      = model.labels[state]
        result.at[pos, "confidence"] = round(float(g[state]), 4)
        for k in range(n_states):
            result.at[pos, f"p{k}"] = round(float(g[k]), 4)

    # ── 7. Per-state diagnostics ──────────────────────────────────────────────
    sep = "=" * 72

    # Sojourn stats from Viterbi sequence
    sojourn: dict[int, list[int]] = {k: [] for k in range(n_states)}
    cur_s, cur_run = int(states_valid[0]), 1
    for sv in states_valid[1:]:
        sv = int(sv)
        if sv == cur_s:
            cur_run += 1
        else:
            sojourn[cur_s].append(cur_run)
            cur_s, cur_run = sv, 1
    sojourn[cur_s].append(cur_run)

    print(f"\n{sep}")
    print(f"IOHMM  K={n_states}  |  O_t: {len(_O_COLS)} observation features  |  U_t: {len(_U_COLS)} transition features")
    print(f"Decode: Viterbi (hard)  |  Confidence / p0..p{n_states-1}: forward filter (soft)")
    print(sep)

    for k in range(n_states):
        mask_k    = (states_valid == k)
        count_k   = int(mask_k.sum())
        freq_k    = count_k / n_valid
        runs_k    = sojourn[k]
        mean_soj  = float(np.mean(runs_k))  if runs_k else 0.0
        med_soj   = float(np.median(runs_k)) if runs_k else 0.0
        max_soj   = int(max(runs_k))         if runs_k else 0
        n_runs    = len(runs_k)

        print(f"\n{'-'*72}")
        print(f" State {k}  label={model.labels[k]:<12}  "
              f"{count_k} bars ({freq_k*100:.1f}%)  "
              f"n_runs={n_runs}  sojourn: mean={mean_soj:.1f}  median={med_soj:.1f}  max={max_soj}")
        print(f"{'-'*72}")

        # Observation emission: mu and sqrt(diag(Sigma))
        print(f"\n  O_t emission  [{len(_O_COLS)} features]")
        print(f"  {'feature':<16}  {'mu':>10}  {'sigma_diag':>10}")
        print(f"  {'-'*40}")
        for d, col in enumerate(_O_COLS):
            mu_kd  = float(model.mu[k, d])
            sig_kd = float(np.sqrt(max(model.sigma[k, d, d], 0.0)))
            print(f"  {col:<16}  {mu_kd:>10.4f}  {sig_kd:>10.4f}")

        # Transition from state k: alpha and per-U_t beta coefficients
        print(f"\n  Transition from state_{k}  [{len(_U_COLS)} U_t features]")
        hdr_u = "  {:>10}  {:>8}" + "  {:>13}" * len(_U_COLS)
        print(hdr_u.format("to", "alpha", *_U_COLS))
        print(f"  {'-'*60}")
        for j in range(n_states):
            row = f"  {'state_'+str(j):>10}  {float(model.alpha[k, j]):>8.4f}"
            for m in range(model.M):
                row += f"  {float(model.beta[k, j, m]):>13.6f}"
            print(row)

        # Marginal transition at U=0, p10, p90
        U_p10 = np.nanpercentile(U_valid, 10, axis=0)
        U_p90 = np.nanpercentile(U_valid, 90, axis=0)

        def _row_trans(u_vec: np.ndarray) -> np.ndarray:
            u_safe  = np.where(np.isnan(u_vec), 0.0, u_vec)
            logits  = model.alpha[k] + model.beta[k] @ u_safe
            return np.exp(logits - logsumexp(logits))

        A_0   = _row_trans(np.zeros(model.M))
        A_p10 = _row_trans(U_p10)
        A_p90 = _row_trans(U_p90)

        print(f"\n  P(next | from state_{k}):")
        hdr_p = "  {:>10}" + "  {:>10}" * n_states
        print(hdr_p.format("U", *[f"->state_{j}" for j in range(n_states)]))
        for label_u, A_row in [("U=0", A_0), ("U=p10", A_p10), ("U=p90", A_p90)]:
            row = f"  {label_u:>10}" + "".join(f"  {A_row[j]:>10.4f}" for j in range(n_states))
            print(row)

    # ── 8. Save ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    out_path = Path(args.out)
    result.to_csv(out_path, index=False)
    print(f"Saved -> {out_path.resolve()}  ({len(result)} rows)")

    # Last 20 bars
    print("\n=== Last 20 bars ===")
    cols_show = ["bar_ts", "close", "state", "label", "confidence"] + [f"p{k}" for k in range(n_states)]
    print(result[cols_show].tail(20).to_string(index=False))


if __name__ == "__main__":
    main()
