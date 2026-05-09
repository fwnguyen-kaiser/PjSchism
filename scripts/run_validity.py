"""
Predictive validity: forward outcomes grouped by Viterbi state.

Loads labels.csv + raw OHLCV parquet, computes forward returns and
realized vol at multiple horizons, prints per-state stats to check
whether each state has distinct and consistent economic character.

Usage:
    python scripts/run_validity.py [--labels labels.csv]

Metrics per state:
    fwd_ret_1/6/12  : log return over next 1 / 6 / 12 bars (4h / 24h / 48h)
    fwd_rv6         : realized vol (sum sq log-ret) over next 6 bars
    fwd_fr1         : funding rate at t+1
    hit_rate        : % bars with fwd_ret > 0
    t_stat          : two-sided t-test of mean vs 0 (returns)
    sharpe_ann      : annualized Sharpe at each horizon
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from schism.utils.config_loader import load_yaml

_OHLCV_GLOB = "schism/data/volumes/parquet/symbol=BTCUSDT/**/*.parquet"

# bars per year at 4h timeframe
_BARS_PER_YEAR = 365 * 6


def load_ohlcv_raw() -> pd.DataFrame:
    files = sorted(glob.glob(_OHLCV_GLOB, recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["bar_ts"] = pd.to_datetime(df["bar_ts"], utc=True)
    df = df.sort_values("bar_ts").reset_index(drop=True)
    return df[["bar_ts", "close", "funding_rate", "bybit_fr"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels.csv")
    args = ap.parse_args()

    # ── Load labels ───────────────────────────────────────────────────────────
    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found. Run scripts/run_label.py first.")
        sys.exit(1)

    ldf = pd.read_csv(labels_path, parse_dates=["bar_ts"])
    ldf["bar_ts"] = pd.to_datetime(ldf["bar_ts"], utc=True)
    print(f"Labels: {len(ldf)} rows  |  valid (state not NaN): {ldf['state'].notna().sum()}")

    # ── Load raw OHLCV for funding rate (not in labels.csv) ──────────────────
    raw = load_ohlcv_raw()
    df = ldf.merge(raw[["bar_ts", "funding_rate", "bybit_fr"]], on="bar_ts", how="left")
    df = df.sort_values("bar_ts").reset_index(drop=True)

    # ── Compute forward metrics ───────────────────────────────────────────────
    log_ret = np.log(df["close"] / df["close"].shift(1))

    for h in [1, 6, 12]:
        df[f"fwd_ret_{h}"] = np.log(df["close"].shift(-h) / df["close"])

    # Realized vol over next 6 bars: sum of squared log returns at t+1..t+6
    rv_next = sum(log_ret.shift(-k) ** 2 for k in range(1, 7))
    df["fwd_rv6"] = rv_next

    df["fwd_fr1"] = df["funding_rate"].shift(-1)

    # ── Filter to valid (Viterbi-assigned) rows ───────────────────────────────
    valid = df[df["state"].notna()].copy()
    valid["state"] = valid["state"].astype(int)
    n_valid = len(valid)

    state_ids   = sorted(valid["state"].unique())
    state_labels = {}
    for s in state_ids:
        lbl = valid.loc[valid["state"] == s, "label"].iloc[0]
        state_labels[s] = lbl

    # ── Overall baselines ─────────────────────────────────────────────────────
    overall = {
        "fwd_ret_1":  valid["fwd_ret_1"].dropna(),
        "fwd_ret_6":  valid["fwd_ret_6"].dropna(),
        "fwd_ret_12": valid["fwd_ret_12"].dropna(),
        "fwd_rv6":    valid["fwd_rv6"].dropna(),
        "fwd_fr1":    valid["fwd_fr1"].dropna(),
    }

    sep  = "=" * 80
    sep2 = "-" * 80

    print(f"\n{sep}")
    print(f"PREDICTIVE VALIDITY  |  {n_valid} valid bars  |  horizons: 1-bar=4h, 6-bar=24h, 12-bar=48h")
    print(sep)

    # ── Per-state blocks ──────────────────────────────────────────────────────
    for s in state_ids:
        mask  = valid["state"] == s
        grp   = valid[mask]
        n_s   = len(grp)
        freq  = n_s / n_valid

        print(f"\n{sep2}")
        print(f" State {s}  ({state_labels[s]})  |  {n_s} bars  ({freq*100:.1f}%)")
        print(sep2)

        # -- Returns
        print(f"\n  Returns:")
        print(f"  {'horizon':<14}  {'mean':>10}  {'median':>10}  {'std':>10}  "
              f"{'hit_rate':>9}  {'t_stat':>8}  {'sharpe_ann':>11}")
        print(f"  {'-'*78}")

        for h, col in [(1, "fwd_ret_1"), (6, "fwd_ret_6"), (12, "fwd_ret_12")]:
            s_ret = grp[col].dropna()
            if len(s_ret) < 5:
                continue
            mean_r  = float(s_ret.mean())
            med_r   = float(s_ret.median())
            std_r   = float(s_ret.std())
            hit     = float((s_ret > 0).mean())
            t_stat  = float(stats.ttest_1samp(s_ret, 0.0).statistic)
            sharpe  = (mean_r / std_r * np.sqrt(_BARS_PER_YEAR / h)) if std_r > 0 else 0.0
            horizon = f"fwd_ret_{h} ({h*4}h)"
            print(f"  {horizon:<14}  {mean_r:>+10.5f}  {med_r:>+10.5f}  {std_r:>10.5f}  "
                  f"  {hit:>7.1%}  {t_stat:>8.2f}  {sharpe:>+11.3f}")

        # -- Realized vol vs overall
        print(f"\n  Realized vol (next 24h):")
        rv_s   = grp["fwd_rv6"].dropna()
        rv_all = overall["fwd_rv6"]
        if len(rv_s) > 5 and rv_all.mean() > 0:
            mean_rv    = float(rv_s.mean())
            overall_rv = float(rv_all.mean())
            ratio      = mean_rv / overall_rv
            t_rv       = float(stats.ttest_ind(rv_s, rv_all).statistic)
            print(f"  mean fwd_rv6 = {mean_rv:.5f}  (overall {overall_rv:.5f})  "
                  f"ratio = {ratio:.3f}x  t_vs_overall = {t_rv:.2f}")

        # -- Funding rate
        fr_s = grp["fwd_fr1"].dropna()
        if len(fr_s) > 5:
            mean_fr  = float(fr_s.mean())
            t_fr     = float(stats.ttest_1samp(fr_s, float(overall["fwd_fr1"].mean())).statistic)
            print(f"\n  Funding rate (t+1):")
            print(f"  mean = {mean_fr:+.6f}  (overall {float(overall['fwd_fr1'].mean()):+.6f})  "
                  f"t_vs_overall = {t_fr:.2f}")

    # ── Cross-state return comparison table ───────────────────────────────────
    print(f"\n{sep}")
    print(f"CROSS-STATE SUMMARY  (mean fwd_ret | t-stat vs state_0 as baseline)")
    print(sep)

    base_state = state_ids[0]

    for col, label in [("fwd_ret_1", "fwd_ret_1 (4h)"),
                       ("fwd_ret_6", "fwd_ret_6 (24h)"),
                       ("fwd_ret_12", "fwd_ret_12 (48h)")]:
        hdr = f"  {label:<20}" + "".join(f"  {'state_'+str(s):>14}" for s in state_ids)
        print(hdr)

        means_row = f"  {'mean':>20}"
        tstat_row = f"  {'t_vs_state_0':>20}"

        base_ret = valid[valid["state"] == base_state][col].dropna()

        for s in state_ids:
            s_ret = valid[valid["state"] == s][col].dropna()
            mean_s = float(s_ret.mean())
            if s == base_state:
                t_s = float(stats.ttest_1samp(s_ret, 0.0).statistic)
            else:
                t_s = float(stats.ttest_ind(s_ret, base_ret).statistic)
            means_row += f"  {mean_s:>+14.5f}"
            tstat_row += f"  {t_s:>+14.2f}"

        print(means_row)
        print(tstat_row)
        print()

    # ── Volatility regime check ───────────────────────────────────────────────
    print(sep)
    print(f"VOLATILITY PROFILE  (fwd_rv6 mean per state vs overall)")
    print(sep)
    hdr_rv = f"  {'':>14}" + "".join(f"  {'state_'+str(s):>10}" for s in state_ids) + f"  {'overall':>10}"
    print(hdr_rv)
    rv_row  = f"  {'fwd_rv6 mean':>14}"
    rat_row = f"  {'ratio':>14}"
    for s in state_ids:
        rv_s = float(valid[valid["state"] == s]["fwd_rv6"].dropna().mean())
        rv_row  += f"  {rv_s:>10.5f}"
        rat_row += f"  {rv_s / float(overall['fwd_rv6'].mean()):>10.3f}x"
    rv_row  += f"  {float(overall['fwd_rv6'].mean()):>10.5f}"
    rat_row += f"  {'1.000x':>10}"
    print(rv_row)
    print(rat_row)
    print()


if __name__ == "__main__":
    main()
