"""
Vectorized backtest: regime-conditioned signals from soft posteriors.

Position at bar t is computed from (p0..p3) at bar t, then held for 1 bar.
PnL = position[t] * log_ret[t+1] - cost * |position[t] - position[t-1]|

Regime-based signals  (production-compatible via filter_step posteriors):
  position = p0*w_calm + p1*w_bull + p2*w_dist + p3*w_chop

  R_aggr   : [0.3,  1.0, -1.0,  0.0]  full short in DIST
  R_mod    : [0.3,  1.0, -0.5,  0.0]  half short in DIST
  R_bal    : [0.5,  1.0, -0.5,  0.2]  higher calm base, small chop long
  R_long   : [0.3,  1.0,  0.0,  0.1]  no shorting

Model-derived signal (no human label needed):
  R_var    : w_k = sign(mu_k[f3_norm_ret]) / sqrt(Sigma_k[f7_rv_ratio, f7_rv_ratio])
             normalized so max |w_k| = 1.  Uses emission params directly.

Reference signals (for comparison):
  HODL     : position = 1.0 always
  SOFT     : p1 - p2  (original signal, no baseline)

Usage:
    python scripts/run_backtest.py [--labels labels.csv] [--cost-bps 4] [--conf 0.6]
                                   [--model model_latest.pkl]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Indices of f3_norm_ret and f7_rv_ratio within the 9-feature O_t vector
# O_t = [f1_cvd_vol, f2_oi_chg, f3_norm_ret, f4_liq_sq,
#         f6_illiq, f7_rv_ratio, f8_vol_shock, f9_flow_liq, f10_flow_pos]
_F3_IDX = 2   # f3_norm_ret  — return direction signal
_F7_IDX = 5   # f7_rv_ratio  — realized-vol proxy

_STATE_NAMES = {0: "CALM_DRIFT", 1: "BULL_MOMENTUM", 2: "DISTRIBUTION", 3: "VOLATILE_CHOP"}
_BARS_PER_YEAR = 365 * 6   # 4h bars


# ── Regime weight sets ────────────────────────────────────────────────────────
#   Each row: [w_calm, w_bull, w_dist, w_chop]
#   position_t = p0*w0 + p1*w1 + p2*w2 + p3*w3

_REGIME_WEIGHTS: dict[str, list[float]] = {
    #              calm  bull   dist  chop
    "R_aggr":  [  0.3,  1.0,  -1.0,  0.0],   # full short in DIST
    "R_mod":   [  0.3,  1.0,  -0.5,  0.0],   # half short in DIST
    "R_bal":   [  0.5,  1.0,  -0.5,  0.2],   # higher calm base, small chop
    "R_long":  [  0.3,  1.0,   0.0,  0.1],   # no shorting
}


# ── Signal functions ──────────────────────────────────────────────────────────

def compute_signals(df: pd.DataFrame, conf_threshold: float,
                    model=None,
                    kelly_scale: float = 0.5) -> tuple[pd.DataFrame, dict | None]:
    p    = df[["p0", "p1", "p2", "p3"]].to_numpy(dtype=float)
    hodl = np.ones(len(df))
    soft = p[:, 1] - p[:, 2]   # reference: no baseline

    sigs: dict[str, np.ndarray] = {"HODL": hodl, "SOFT": soft}

    for name, w in _REGIME_WEIGHTS.items():
        w_arr      = np.array(w, dtype=float)
        sigs[name] = p @ w_arr   # (T, 4) @ (4,) -> (T,)

    model_info: dict | None = None
    if model is not None:
        # ── R_var: sign(mu_f3) / sqrt(var_f7), normalized ──────────────────
        mu_ret  = model.mu[:, _F3_IDX]
        var_rv  = model.sigma[:, _F7_IDX, _F7_IDX]
        w_raw   = np.sign(mu_ret) / np.sqrt(np.maximum(var_rv, 1e-8))
        w_var   = w_raw / max(np.abs(w_raw).max(), 1e-8)
        sigs["R_var"] = p @ w_var

        # ── R_kelly: half-Kelly from f3 emission parameters ─────────────────
        # E[r | p]   = p @ mu_f3          (law of total expectation)
        # Var[r | p] = p @ (var_f3 + mu_f3^2) - E[r|p]^2  (law of total variance)
        # position   = clip(kelly_scale * E[r|p] / Var[r|p], -1, 1)
        mu_f3  = model.mu[:, _F3_IDX]
        var_f3 = model.sigma[:, _F3_IDX, _F3_IDX]
        e_r    = p @ mu_f3                              # (T,)
        e_r2   = p @ (var_f3 + mu_f3 ** 2)             # E[r^2 | p_t]
        var_r  = np.maximum(e_r2 - e_r ** 2, 1e-8)    # Var[r | p_t]
        sigs["R_kelly"] = np.clip(kelly_scale * e_r / var_r, -1.0, 1.0)

        model_info = {
            # R_var
            "mu_ret": mu_ret, "var_rv": var_rv, "w_var": w_var,
            # R_kelly
            "mu_f3": mu_f3, "var_f3": var_f3,
            "kelly_scale": kelly_scale,
            "kelly_pure": kelly_scale * mu_f3 / np.maximum(var_f3, 1e-8),
            "e_r": e_r, "var_r": var_r,
        }

    return pd.DataFrame(sigs, index=df.index), model_info


# ── Position smoothing ───────────────────────────────────────────────────────

def apply_smoothing(raw_pos: np.ndarray, threshold: float) -> np.ndarray:
    """Only rebalance when |new_signal - current_position| > threshold."""
    if threshold <= 0.0:
        return raw_pos.copy()
    pos = raw_pos.copy()
    cur = raw_pos[0]
    for t in range(1, len(raw_pos)):
        if abs(raw_pos[t] - cur) > threshold:
            cur = raw_pos[t]
        pos[t] = cur
    return pos


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pnl: pd.Series, position: pd.Series) -> dict:
    cum   = pnl.cumsum()
    total = float(cum.iloc[-1])
    n     = len(pnl)
    ann_r = total * (_BARS_PER_YEAR / n)

    roll_max = cum.cummax()
    dd       = cum - roll_max
    max_dd   = float(dd.min())

    std_bar = float(pnl.std())
    sharpe  = (float(pnl.mean()) / std_bar * np.sqrt(_BARS_PER_YEAR)) if std_bar > 0 else 0.0
    calmar  = (ann_r / abs(max_dd)) if max_dd < 0 else np.nan

    # trade count = number of bar-to-bar position changes (non-zero delta)
    pos_arr   = position.to_numpy()
    pos_delta = np.abs(np.diff(pos_arr, prepend=pos_arr[0]))
    n_trades  = int((pos_delta > 1e-6).sum())
    win_rate  = float((pnl > 0).mean())

    return {
        "total_ret":  total,
        "ann_ret":    ann_r,
        "sharpe":     sharpe,
        "max_dd":     max_dd,
        "calmar":     calmar,
        "win_rate":   win_rate,
        "n_trades":   n_trades,
    }


# ── Regime attribution ────────────────────────────────────────────────────────

def regime_attribution(pnl: pd.Series, states: pd.Series, n_states: int) -> dict:
    attr = {}
    total = float(pnl.sum())
    for s in range(n_states):
        mask    = (states == s)
        pnl_s   = float(pnl[mask].sum())
        n_s     = int(mask.sum())
        pct_pnl = pnl_s / total * 100 if total != 0 else 0.0
        attr[s] = {"pnl": pnl_s, "n_bars": n_s, "pct_pnl": pct_pnl}
    return attr


# ── ASCII sparkline ───────────────────────────────────────────────────────────

def sparkline(series: np.ndarray, width: int = 60, height: int = 8,
              label: str = "") -> str:
    xs = np.linspace(0, len(series) - 1, width).astype(int)
    vals = series[xs]
    lo, hi = vals.min(), vals.max()
    span = hi - lo if hi > lo else 1.0

    grid = [[" "] * width for _ in range(height)]
    for col, v in enumerate(vals):
        row = int((v - lo) / span * (height - 1))
        row = max(0, min(height - 1, height - 1 - row))
        grid[row][col] = "*"

    # zero line
    zero_row = int((0 - lo) / span * (height - 1))
    zero_row = max(0, min(height - 1, height - 1 - zero_row))
    for col in range(width):
        if grid[zero_row][col] == " ":
            grid[zero_row][col] = "-"

    lines = ["  " + "".join(row) for row in grid]
    lines.append(f"  {lo:+.3f}" + " " * (width - 12) + f"{hi:+.3f}")
    if label:
        lines.insert(0, f"  {label}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels",   default="labels.csv")
    ap.add_argument("--cost-bps", type=float, default=4.0,
                    help="one-way cost in bps (default 4 = 0.04%%)")
    ap.add_argument("--conf",     type=float, default=0.6,
                    help="confidence threshold for SOFT_CONF signal")
    ap.add_argument("--smooth",   type=float, default=0.05,
                    help="min position delta to rebalance (0 = off)")
    ap.add_argument("--model",       default="model_latest.pkl",
                    help="path to saved IOHMM pickle for R_var / R_kelly signals")
    ap.add_argument("--kelly-scale", type=float, default=0.5,
                    help="Kelly fraction (0.5 = half-Kelly, default)")
    ap.add_argument("--out",         default="backtest_results.csv")
    args = ap.parse_args()

    cost_per_unit = args.cost_bps / 10_000

    # ── Load model (optional — for R_var) ────────────────────────────────────
    model = None
    model_path = Path(args.model)
    if model_path.exists():
        from schism.models.iohmm import IOHMM
        model = IOHMM.load(model_path)
    else:
        print(f"  [R_var] model not found at {model_path} — skipping variance signal")

    # ── Load ─────────────────────────────────────────────────────────────────
    path = Path(args.labels)
    if not path.exists():
        print(f"ERROR: {path} not found. Run run_label.py first.")
        sys.exit(1)

    df = pd.read_csv(path, parse_dates=["bar_ts"])
    df["bar_ts"] = pd.to_datetime(df["bar_ts"], utc=True)
    df = df.sort_values("bar_ts").reset_index(drop=True)

    # Drop bars without state (no valid features — not tradeable)
    df = df[df["state"].notna()].copy()
    df["state"] = df["state"].astype(int)
    n = len(df)

    ts_min = df["bar_ts"].iloc[0].strftime("%Y-%m-%d")
    ts_max = df["bar_ts"].iloc[-1].strftime("%Y-%m-%d")

    # ── Returns (shift -1: pnl realised on NEXT bar) ──────────────────────────
    log_ret = np.log(df["close"] / df["close"].shift(1)).to_numpy()
    # position[t] set at close of bar t, earns ret at bar t+1
    # so pnl[t] = position[t-1] * log_ret[t]
    # shift position forward by 1
    fwd_ret = np.roll(log_ret, -1)
    fwd_ret[-1] = 0.0   # last bar has no forward return

    # ── Signals ───────────────────────────────────────────────────────────────
    sigs, var_info = compute_signals(df, args.conf, model=model,
                                     kelly_scale=args.kelly_scale)

    sep  = "=" * 80
    sep2 = "-" * 80

    # ── Model-derived signal diagnostics ─────────────────────────────────────
    if var_info is not None:
        print(f"\n{sep}")
        print(f" R_var WEIGHTS  (no human label — emission params only)")
        print(f"   w_k = sign(mu_k[f3_norm_ret]) / sqrt(Sigma_k[f7_rv_ratio,f7_rv_ratio])")
        print(sep)
        print(f"  {'state':<6}  {'label':<16}  {'mu_f3':>10}  {'var_f7':>10}  {'w_norm':>8}")
        print(f"  {'-'*58}")
        for k in range(model.K):
            label = _STATE_NAMES.get(k, f"state_{k}")
            print(f"  {k:<6}  {label:<16}  {var_info['mu_ret'][k]:>+10.4f}"
                  f"  {var_info['var_rv'][k]:>10.4f}"
                  f"  {var_info['w_var'][k]:>+8.4f}")

        print(f"\n{sep}")
        print(f" R_kelly WEIGHTS  ({args.kelly_scale}-Kelly from f3 emission params)")
        print(f"   E[r|p] = p @ mu_f3;  Var[r|p] via law of total variance")
        print(f"   position = clip({args.kelly_scale} * E[r|p] / Var[r|p], -1, 1)")
        print(sep)
        print(f"  {'state':<6}  {'label':<16}  {'mu_f3':>10}  {'var_f3':>10}  {'pure_kelly':>12}")
        print(f"  {'-'*62}")
        for k in range(model.K):
            label = _STATE_NAMES.get(k, f"state_{k}")
            print(f"  {k:<6}  {label:<16}  {var_info['mu_f3'][k]:>+10.4f}"
                  f"  {var_info['var_f3'][k]:>10.4f}"
                  f"  {var_info['kelly_pure'][k]:>+12.4f}")
        e_r_series  = var_info["e_r"]
        var_r_series = var_info["var_r"]
        print(f"\n  Signal distribution across {n} bars:")
        print(f"  E[r|p]:    min={e_r_series.min():+.4f}  mean={e_r_series.mean():+.4f}"
              f"  max={e_r_series.max():+.4f}  pct_pos={100*(e_r_series>0).mean():.1f}%")
        raw_kelly = args.kelly_scale * e_r_series / var_r_series
        print(f"  raw Kelly: min={raw_kelly.min():+.4f}  mean={raw_kelly.mean():+.4f}"
              f"  max={raw_kelly.max():+.4f}  clipped={(np.abs(raw_kelly)>1).mean()*100:.1f}%")

    print(f"\n{sep}")
    print(f" BACKTEST  |  BTCUSDT 4h  |  {ts_min} to {ts_max}  |  {n} bars")
    print(f" Cost: {args.cost_bps} bps one-way  |  smooth threshold: {args.smooth}")
    print(sep)

    results = {}
    equity_curves = {}

    for sig_name, pos_raw in sigs.items():
        # HODL never smoothed (always 1.0); SOFT reference unsmoothed for comparison
        threshold = 0.0 if sig_name in ("HODL", "SOFT") else args.smooth
        pos = apply_smoothing(pos_raw.to_numpy(dtype=float), threshold)

        # transaction cost on position change
        pos_delta = np.abs(np.diff(pos, prepend=pos[0]))
        cost      = pos_delta * cost_per_unit

        # gross pnl then net
        gross = pos * fwd_ret
        net   = gross - cost

        pnl_series = pd.Series(net, index=df.index)
        pos_series = pd.Series(pos, index=df.index)

        metrics    = compute_metrics(pnl_series, pos_series)
        attr       = regime_attribution(pnl_series, df["state"], 4)

        results[sig_name]       = metrics
        equity_curves[sig_name] = pnl_series.cumsum().to_numpy()

        results[sig_name]["_attr"] = attr

    # ── Metrics table ─────────────────────────────────────────────────────────
    sig_names = list(sigs.keys())
    col_w = 13

    print(f"\n{'metric':<16}" + "".join(f"  {s:>{col_w}}" for s in sig_names))
    print(sep2)

    rows = [
        ("total_ret",  "total_ret",   lambda v: f"{v*100:>+.2f}%"),
        ("ann_ret",    "ann_ret",     lambda v: f"{v*100:>+.2f}%"),
        ("sharpe",     "sharpe",      lambda v: f"{v:>+.3f}"),
        ("max_dd",     "max_dd",      lambda v: f"{v*100:>.2f}%"),
        ("calmar",     "calmar",      lambda v: f"{v:>.3f}" if not np.isnan(v) else "  n/a"),
        ("win_rate",   "win_rate",    lambda v: f"{v*100:>.1f}%"),
        ("n_trades",   "n_trades",    lambda v: f"{int(v):>d}"),
    ]

    for row_label, key, fmt in rows:
        line = f"{row_label:<16}"
        for s in sig_names:
            v    = results[s][key]
            line += f"  {fmt(v):>{col_w}}"
        print(line)

    # ── Alpha vs HODL ─────────────────────────────────────────────────────────
    print(f"\n  Alpha vs HODL (Sharpe diff / ann_ret diff):")
    hodl_sharpe = results["HODL"]["sharpe"]
    hodl_annret = results["HODL"]["ann_ret"]
    line_sh = f"  {'sharpe_alpha':<16}"
    line_ar = f"  {'ann_ret_alpha':<16}"
    for s in sig_names:
        d_sh = results[s]["sharpe"]  - hodl_sharpe
        d_ar = results[s]["ann_ret"] - hodl_annret
        line_sh += f"  {d_sh:>+{col_w}.3f}"
        line_ar += f"  {d_ar*100:>+{col_w}.2f}%"
    print(line_sh)
    print(line_ar)

    # ── Regime attribution ────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f" REGIME ATTRIBUTION  (% of total net PnL per state)")
    print(sep)

    hdr = f"  {'signal':<14}" + "".join(
        f"  {_STATE_NAMES.get(s, f'S{s}')[:13]:>13}" for s in range(4)
    )
    print(hdr)
    print(f"  {sep2[:76]}")

    for sig_name in sig_names:
        attr = results[sig_name]["_attr"]
        row  = f"  {sig_name:<14}"
        for s in range(4):
            row += f"  {attr[s]['pct_pnl']:>12.1f}%"
        print(row)

    # ── Sparklines ────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f" EQUITY CURVES  (cumulative log-return, net of costs)")
    print(sep)

    spark_names = ["HODL", "SOFT", "R_aggr", "R_mod", "R_bal", "R_long"]
    for extra in ("R_var", "R_kelly"):
        if extra in equity_curves:
            spark_names.append(extra)
    for sig_name in spark_names:
        eq = equity_curves[sig_name]
        final = eq[-1]
        print(sparkline(eq, width=72, height=7,
                        label=f"{sig_name}  final={final*100:+.2f}%"))
        print()

    # ── Threshold sweep on R_bal ──────────────────────────────────────────────
    print(f"\n{sep}")
    print(f" SMOOTHING SWEEP  (R_bal  [0.5, 1.0, -0.5, 0.2]  —  effect of threshold)")
    print(sep)

    w_bal    = np.array(_REGIME_WEIGHTS["R_bal"])
    p_mat    = df[["p0", "p1", "p2", "p3"]].to_numpy(dtype=float)
    raw_rbal = p_mat @ w_bal

    thresholds = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    sw_hdr = f"  {'threshold':>10}  {'sharpe':>8}  {'ann_ret':>9}  {'max_dd':>9}  {'calmar':>8}  {'n_trades':>9}  {'cost_drag':>10}"
    print(sw_hdr)
    print(f"  {'-'*72}")

    for thr in thresholds:
        pos_sw = apply_smoothing(raw_rbal, thr)
        delta  = np.abs(np.diff(pos_sw, prepend=pos_sw[0]))
        cost   = delta * cost_per_unit
        net_sw = pd.Series(pos_sw * fwd_ret - cost, index=df.index)
        m      = compute_metrics(net_sw, pd.Series(pos_sw, index=df.index))
        n_tr   = m["n_trades"]
        # cost drag = total cost as % of gross pnl
        gross_total = float((pos_sw * fwd_ret).sum())
        cost_total  = float(cost.sum())
        drag        = cost_total / abs(gross_total) * 100 if gross_total != 0 else 0.0
        print(f"  {thr:>10.2f}  {m['sharpe']:>+8.3f}  {m['ann_ret']*100:>+8.2f}%"
              f"  {m['max_dd']*100:>+8.2f}%  {m['calmar']:>8.3f}"
              f"  {n_tr:>9d}  {drag:>9.1f}%")

    # ── Save equity CSV ───────────────────────────────────────────────────────
    eq_df = pd.DataFrame(
        {s: equity_curves[s] for s in sig_names},
        index=df["bar_ts"].values,
    )
    eq_df.index.name = "bar_ts"
    eq_df.to_csv(args.out)
    print(f"{sep}")
    print(f" Equity curves saved -> {Path(args.out).resolve()}")
    print(f" Load in notebook/Excel for full chart.")
    print(sep)


if __name__ == "__main__":
    main()
