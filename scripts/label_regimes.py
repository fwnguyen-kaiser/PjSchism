"""
Assign human labels to regimes, print a visual regime map + emission profile,
and update labels.csv with the new names.

Usage:
    python scripts/label_regimes.py [--labels labels.csv] [--model model_latest.pkl]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Human labels (assign after seeing data) ───────────────────────────────────
_STATE_NAMES = {
    0: "state_0",
    1: "state_1",
    2: "state_2",
}

_BARS_PER_YEAR = 365 * 6   # 4h timeframe

_O_COLS = [
    "f1_cvd_vol", "f2_oi_chg", "f3_norm_ret", "f4_liq_sq",
    "f6_illiq",   "f7_rv_ratio", "f8_vol_shock", "f9_flow_liq", "f10_flow_pos",
]

_O_DESC = {
    "f1_cvd_vol":   "CVD / volume         (buy/sell pressure)",
    "f2_oi_chg":    "OI change            (position accumulation)",
    "f3_norm_ret":  "Normalized return    (momentum)",
    "f4_liq_sq":    "Liquidation^2        (forced unwind impulse)",
    "f6_illiq":     "Amihud illiquidity   (price-impact ratio)",
    "f7_rv_ratio":  "Realized vol ratio   [state ordering key]",
    "f8_vol_shock": "Volume shock         (vs MA)",
    "f9_flow_liq":  "Flow x liq           (interaction term)",
    "f10_flow_pos": "Flow positional      (directional carry)",
}


def _sojourn(states: np.ndarray) -> dict[int, list[int]]:
    runs: dict[int, list[int]] = {k: [] for k in range(states.max() + 1)}
    cur, run = int(states[0]), 1
    for s in states[1:]:
        s = int(s)
        if s == cur:
            run += 1
        else:
            runs[cur].append(run)
            cur, run = s, 1
    runs[cur].append(run)
    return runs


def _bar(freq: float, width: int = 40) -> str:
    filled = round(freq * width)
    return "[" + "#" * filled + " " * (width - filled) + "]"


def _mini_bar(z: float, width: int = 7) -> str:
    """Horizontal bar centered at 0 for a z-score."""
    half = width // 2
    pos = min(half, max(-half, round(z * half / 2.0)))
    bar = [" "] * width
    mid = half
    if pos >= 0:
        for i in range(mid, mid + pos + 1):
            if 0 <= i < width:
                bar[i] = "#"
    else:
        for i in range(mid + pos, mid + 1):
            if 0 <= i < width:
                bar[i] = "#"
    bar[mid] = "|"
    return "".join(bar)


def _load_model(path: Path) -> dict | None:
    if not path.exists():
        return None
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def _emission_table(model: dict, state_names: list[str]) -> None:
    K = model["K"]
    mu    = np.array(model["mu"])     # shape (K, D)
    sigma = np.array(model["sigma"])  # shape (K, D, D)
    labels = model["labels"]          # internal-idx -> state-name

    # Build mapping: state_name -> internal index
    name_to_idx = {name: i for i, name in enumerate(labels)}

    # Ordered list of (state_name, internal_idx) by _STATE_NAMES order
    ordered = [(state_names[s], name_to_idx.get(state_names[s])) for s in range(K)]

    sep = "=" * 88
    print(f"\n{sep}")
    print(" Emission Profile  (all features z-scored at training time)")
    print(sep)

    # Header
    hdr_names = [f"{n:>12}" for n, _ in ordered]
    print(f"  {'Feature':<16}" + "  " + "  ".join(hdr_names))
    std_hdr   = [f"{'mu   std':>12}" for _ in ordered]
    print(f"  {'':16}  " + "  ".join([f"{'mu':>6}{'std':>6}" for _ in ordered]))
    print(f"  {'-'*85}")

    for d, feat in enumerate(_O_COLS):
        row = f"  {feat:<16}  "
        for name, idx in ordered:
            if idx is None:
                row += f"  {'N/A':>12}"
                continue
            mu_d  = float(mu[idx, d])
            std_d = float(np.sqrt(sigma[idx, d, d]))
            sign  = "+" if mu_d >= 0 else ""
            row  += f"  {sign}{mu_d:>5.3f} {std_d:>5.3f}"
        # mini bar for first state as visual anchor
        row += f"   {_mini_bar(float(mu[ordered[0][1] if ordered[0][1] is not None else 0, d]))}"
        print(row)

    print(f"  {'-'*85}")

    # Vol regime summary row
    rv_idx = _O_COLS.index("f7_rv_ratio")
    row = f"  {'RV regime':<16}  "
    for name, idx in ordered:
        if idx is None:
            row += f"  {'N/A':>12}"
            continue
        rv_mu = float(mu[idx, rv_idx])
        tag   = "LOW " if rv_mu < -0.3 else ("HIGH" if rv_mu > 0.3 else "MED ")
        row  += f"  {tag:>12}"
    print(row)

    # Momentum summary row
    f3_idx = _O_COLS.index("f3_norm_ret")
    row = f"  {'Momentum':<16}  "
    for name, idx in ordered:
        if idx is None:
            row += f"  {'N/A':>12}"
            continue
        f3_mu = float(mu[idx, f3_idx])
        tag   = "STRONG+" if f3_mu > 0.4 else ("STRONG-" if f3_mu < -0.4 else ("WEAK+" if f3_mu > 0 else "WEAK-"))
        row  += f"  {tag:>12}"
    print(row)

    print(sep)
    print(" Feature descriptions:")
    for feat in _O_COLS:
        print(f"   {feat:<16}  {_O_DESC[feat]}")
    print(sep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--model",  default="model_latest.pkl")
    args = ap.parse_args()

    path = Path(args.labels)
    if not path.exists():
        print(f"ERROR: {path} not found. Run run_label.py first.")
        sys.exit(1)

    df = pd.read_csv(path, parse_dates=["bar_ts"])
    df["bar_ts"] = pd.to_datetime(df["bar_ts"], utc=True)

    valid = df[df["state"].notna()].copy()
    valid["state"] = valid["state"].astype(int)
    n_valid = len(valid)
    n_states = int(valid["state"].max()) + 1

    # ── Forward metrics ───────────────────────────────────────────────────────
    df_s = df.sort_values("bar_ts").reset_index(drop=True)
    for h in [1, 6, 12]:
        df_s[f"fwd_ret_{h}"] = np.log(df_s["close"].shift(-h) / df_s["close"])
    log_ret = np.log(df_s["close"] / df_s["close"].shift(1))
    df_s["fwd_rv6"] = sum(log_ret.shift(-k) ** 2 for k in range(1, 7))

    valid2 = df_s[df_s["state"].notna()].copy()
    valid2["state"] = valid2["state"].astype(int)

    overall_rv = float(valid2["fwd_rv6"].dropna().mean())

    # ── Sojourn ───────────────────────────────────────────────────────────────
    states_seq = valid.sort_values("bar_ts")["state"].to_numpy()
    sojourn = _sojourn(states_seq)

    # ── Date range ────────────────────────────────────────────────────────────
    ts_min = df["bar_ts"].min().strftime("%Y-%m-%d")
    ts_max = df["bar_ts"].max().strftime("%Y-%m-%d")

    sep = "=" * 80

    # ── Print regime map ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f" SCHISM Regime Map  |  BTCUSDT 4h  |  {ts_min} to {ts_max}")
    print(f" {n_valid} labeled bars  |  Viterbi decode  |  IOHMM K={n_states}")
    print(sep)

    state_names_ordered = []
    for s in range(n_states):
        name  = _STATE_NAMES.get(s, f"state_{s}")
        state_names_ordered.append(name)
        grp   = valid2[valid2["state"] == s]
        n_s   = len(grp)
        freq  = n_s / n_valid
        runs  = sojourn.get(s, [1])
        mean_soj = float(np.mean(runs))
        med_soj  = float(np.median(runs))
        hours    = mean_soj * 4

        stats = {}
        for h, col in [(1, "fwd_ret_1"), (6, "fwd_ret_6"), (12, "fwd_ret_12")]:
            s_ret = grp[col].dropna()
            if len(s_ret) > 5:
                mean_r = float(s_ret.mean())
                std_r  = float(s_ret.std())
                hit    = float((s_ret > 0).mean())
                sharpe = (mean_r / std_r * np.sqrt(_BARS_PER_YEAR / h)) if std_r > 0 else 0.0
                stats[h] = (mean_r, hit, sharpe)

        rv_s   = float(grp["fwd_rv6"].dropna().mean())
        rv_rat = rv_s / overall_rv if overall_rv > 0 else 1.0

        ret1_str  = f"{stats[1][0]*100:+.3f}%" if 1 in stats else "n/a"
        ret6_str  = f"{stats[6][0]*100:+.3f}%" if 6 in stats else "n/a"
        ret12_str = f"{stats[12][0]*100:+.3f}%" if 12 in stats else "n/a"
        hit_str   = f"{stats[1][1]*100:.1f}%" if 1 in stats else "n/a"
        sh_str    = f"{stats[1][2]:+.2f}" if 1 in stats else "n/a"

        print(f"\n  [{s}] {name:<16} {freq*100:5.1f}%  {_bar(freq)}  "
              f"sojourn {mean_soj:.1f} bars (med {med_soj:.0f})  ~{hours:.0f}h")
        print(f"       fwd_ret:  {ret1_str}/4h  {ret6_str}/24h  {ret12_str}/48h"
              f"  |  hit {hit_str}  sharpe(4h) {sh_str}")
        print(f"       fwd_vol:  {rv_rat:.3f}x avg"
              + ("  [LOWEST VOL]"  if rv_rat < 0.9  else
                 "  [HIGHEST VOL]" if rv_rat > 1.30 else
                 "  [elevated]"    if rv_rat > 1.15 else ""))

    # ── Quarterly regime heatmap ──────────────────────────────────────────────
    print(f"\n{sep}")
    print(f" Quarterly regime occupancy  (% of bars per state)")
    print(sep)

    df_q = valid.copy()
    df_q["quarter"] = df_q["bar_ts"].dt.to_period("Q")
    quarters = sorted(df_q["quarter"].unique())

    name_cols = [_STATE_NAMES.get(s, f"S{s}") for s in range(n_states)]
    hdr = f"  {'Quarter':<10}" + "".join(f"  {n[:14]:>14}" for n in name_cols) + "  dominant"
    print(hdr)
    print(f"  {'-'*75}")

    for q in quarters:
        qdf = df_q[df_q["quarter"] == q]
        n_q = len(qdf)
        if n_q == 0:
            continue
        freqs = [float((qdf["state"] == s).sum()) / n_q for s in range(n_states)]
        dom   = int(np.argmax(freqs))
        row   = f"  {str(q):<10}"
        for f_s in freqs:
            row += f"  {f_s*100:>13.1f}%"
        row += f"  {_STATE_NAMES.get(dom, f'S{dom}')}"
        print(row)

    # ── Emission profile table ────────────────────────────────────────────────
    model = _load_model(Path(args.model))
    if model is not None:
        _emission_table(model, state_names_ordered)
    else:
        print(f"\n[emission table skipped: {args.model} not found]")

    # ── Apply and save ────────────────────────────────────────────────────────
    label_map = {s: _STATE_NAMES.get(s, f"state_{s}") for s in range(n_states)}
    df["label"] = df["state"].map(label_map).fillna(df["label"])
    df.to_csv(path, index=False)

    print(f"\n{sep}")
    print(f" Labels updated and saved -> {path.resolve()}")
    print(f" Mapping: { {str(k): v for k, v in label_map.items()} }")
    print(sep)


if __name__ == "__main__":
    main()
