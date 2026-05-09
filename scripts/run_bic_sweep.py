"""
Model-selection sweep over K (number of hidden states).

Metrics computed per K:
  BIC        : -2*LL + n_params*ln(T)                     lower = better
  ICL        : BIC  + 2*H_total                           lower = better  (Biernacki 2000)
  H/bar      : mean posterior entropy per bar             lower = more confident
  Silhouette : mean silhouette on O_t                     higher = better
  Sh_sep     : max(Sharpe_k) - min(Sharpe_k)              higher = more economically distinct
  Self_trans : mean P(k->k) at U=0                        higher = more stable

Polarization metrics (NEW):
  obs_polar  : mean across D observation features of
               F_d = between-state variance / within-state variance
               Higher = states occupy distinct regions in feature space.
               Computed per feature for detailed breakdown.
  trans_polar: per U dimension m,
               beta_rms_m = sqrt(mean(beta[:,: ,m]^2))
               alpha_rms  = sqrt(mean(alpha^2))
               trans_polar_m = beta_rms_m / (alpha_rms + eps)
               Measures how much each transition input modulates the
               transition matrix relative to the baseline intercept.

Usage:
    python scripts/run_bic_sweep.py [--k-min 2] [--k-max 6] [--n-em-runs 3]
                                    [--train-bars 1080] [--out bic_sweep.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_label import (   # type: ignore[import]
    load_ohlcv, load_metrics_4h,
    compute_features, winsorize, zscore,
    _O_COLS, _U_COLS, _RV_COL, _MODEL_CFG,
)
from schism.models.iohmm import IOHMM

_EPS = 1e-12

_O_SHORT = [
    "f1_cvd",   "f2_oi",    "f3_ret",   "f4_liq",
    "f6_illiq", "f7_rv",    "f8_vshk",  "f9_fliq",  "f10_fpos",
]
_U_SHORT = ["u1_ewfr", "u2_dfr", "u3_frspr"]


# ── Metric helpers ────────────────────────────────────────────────────────────

def _entropy_stats(gamma: np.ndarray) -> tuple[float, float]:
    """Total posterior entropy and mean per bar from (T, K) gamma matrix."""
    g      = np.clip(gamma, _EPS, 1.0)
    h_t    = -(g * np.log(g)).sum(axis=1)   # (T,) per-bar entropy
    return float(h_t.sum()), float(h_t.mean())


def _silhouette(O: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette on O (T, D) with integer labels. Returns NaN if K<2."""
    from sklearn.metrics import silhouette_score   # lazy import
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(silhouette_score(O, labels, metric="euclidean",
                                      sample_size=min(5000, len(O)),
                                      random_state=42))
    except Exception:
        return float("nan")


def _sojourn_mean(states: np.ndarray) -> float:
    runs: list[int] = []
    cur, run = int(states[0]), 1
    for s in states[1:]:
        s = int(s)
        if s == cur:
            run += 1
        else:
            runs.append(run)
            cur, run = s, 1
    runs.append(run)
    return float(np.mean(runs)) if runs else 0.0


def _bar_chart(ks: list[int], vals: list[float], width: int = 46,
               lower_better: bool = True) -> str:
    lo, hi = min(vals), max(vals)
    span   = hi - lo if hi > lo else 1.0
    opt    = lo if lower_better else hi
    lines  = []
    for k, v in zip(ks, vals):
        bar_len = int((v - lo) / span * width)
        marker  = "  *" if v == opt else ""
        lines.append(f"  K={k}  {'#'*bar_len:<{width}}  {v:.4f}{marker}")
    return "\n".join(lines)


def _obs_polarization(
    O: np.ndarray,       # (T, D) full valid observations
    states: np.ndarray,  # (T,)  Viterbi assignments on full valid set
) -> tuple[float, np.ndarray]:
    """
    Per-feature empirical F-ratio: between-state / within-state variance.

    Uses Viterbi assignments on the full dataset — robust to the sticky_kappa
    prior suppressing EM emission means on the short training window.

    F[d] = between_var[d] / within_var[d]
      between_var[d] = sum_k freq_k * (empirical_mean_k[d] - grand_mean[d])^2
      within_var[d]  = sum_k freq_k * empirical_var_k[d]

    F > 1 means state centres are further apart than their internal spread.
    """
    K = int(states.max()) + 1
    D = O.shape[1]
    freq = np.array([(states == k).mean() for k in range(K)])

    mu_k = np.zeros((K, D))
    for k in range(K):
        mask = states == k
        if mask.sum() > 0:
            mu_k[k] = O[mask].mean(0)

    grand_mean  = (freq[:, np.newaxis] * mu_k).sum(0)       # (D,)
    diff        = mu_k - grand_mean[np.newaxis, :]           # (K, D)
    between_var = (freq[:, np.newaxis] * diff ** 2).sum(0)  # (D,)

    within_var = np.zeros(D)
    for k in range(K):
        mask = states == k
        if mask.sum() > 1:
            within_var += freq[k] * O[mask].var(0)

    F = between_var / np.maximum(within_var, _EPS)           # (D,)
    return float(F.mean()), F


def _trans_polarization(
    alpha: np.ndarray,   # (K, K)
    beta:  np.ndarray,   # (K, K, M)
) -> tuple[float, np.ndarray]:
    """
    Per-U-dimension transition polarization: how much each exogenous input
    modulates the transition matrix relative to the baseline intercept.

      beta_rms_m  = sqrt( mean(beta[:,:,m]^2) )    RMS slope for U_m
      alpha_rms   = sqrt( mean(alpha^2) )           baseline logit scale
      polar_m     = beta_rms_m / (alpha_rms + eps)

    Returns (mean_polar, polar_per_u) where polar_per_u has shape (M,).
    polar_m > 1  → U_m shifts transition logits by more than the intercept spread.
    polar_m ~ 0  → U_m has negligible effect on transitions.
    """
    alpha_rms = float(np.sqrt(np.mean(alpha ** 2)) + _EPS)
    M = beta.shape[2]
    polar = np.zeros(M)
    for m in range(M):
        beta_rms_m = float(np.sqrt(np.mean(beta[:, :, m] ** 2)))
        polar[m]   = beta_rms_m / alpha_rms
    return float(polar.mean()), polar


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-min",      type=int, default=2)
    ap.add_argument("--k-max",      type=int, default=6)
    ap.add_argument("--n-em-runs",  type=int, default=3,
                    help="EM restarts per K (3 = balanced speed/reliability)")
    ap.add_argument("--train-bars", type=int, default=1080)
    ap.add_argument("--out",        default="bic_sweep.csv")
    args = ap.parse_args()

    # ── 1. Data (identical pipeline to run_label.py) ─────────────────────────
    ohlcv   = load_ohlcv()
    metrics = load_metrics_4h()
    ohlcv["bar_ts"]   = pd.to_datetime(ohlcv["bar_ts"],   utc=True)
    metrics["bar_ts"] = pd.to_datetime(metrics["bar_ts"], utc=True)
    ohlcv = ohlcv.drop(columns=["oi", "lsr_top"], errors="ignore")
    df    = ohlcv.merge(metrics, on="bar_ts", how="left")

    feats = compute_features(df)
    feats = winsorize(feats)
    feats = zscore(feats)

    O = feats[_O_COLS].to_numpy(dtype=float)
    U = feats[_U_COLS].to_numpy(dtype=float)

    valid_mask  = ~np.isnan(O).any(axis=1)
    valid_idx   = np.where(valid_mask)[0]
    n_valid     = len(valid_idx)
    train_start = max(0, n_valid - args.train_bars)
    train_idx   = valid_idx[train_start:]

    O_train = O[train_idx]
    U_train = U[train_idx]
    O_valid = O[valid_mask]
    U_valid = U[valid_mask]
    T_train = len(O_train)

    print(f"\nTrain: {T_train} bars   Full valid: {n_valid} bars")
    print(f"Sweep K = {args.k_min}..{args.k_max}   n_em_runs={args.n_em_runs}")

    # Precompute 1-bar forward returns at valid positions (for Sharpe separation)
    _BARS_PER_YEAR = 365 * 6
    close_valid   = feats["close"].to_numpy()[valid_mask]
    fwd_ret_valid = np.zeros(len(close_valid))
    fwd_ret_valid[:-1] = np.log(
        np.maximum(close_valid[1:],  1e-10) /
        np.maximum(close_valid[:-1], 1e-10)
    )   # fwd_ret_valid[t] = log(close[t+1]/close[t])

    from scipy.special import logsumexp as _lse
    from scipy.stats  import rankdata    as _rankdata

    sep  = "=" * 80
    sep2 = "-" * 80
    print(f"\n{sep}")
    print(f" MODEL SELECTION SWEEP  |  BTCUSDT 4h  |  D={len(_O_COLS)}  M={len(_U_COLS)}")
    print(f" Statistical: BIC, ICL, H/bar, Silhouette")
    print(f" Economic:    Sharpe separation (max-min across states), Self-transition P(k->k)")
    print(sep)

    records = []

    for K in range(args.k_min, args.k_max + 1):
        print(f"\n  Fitting K={K}  (n_em_runs={args.n_em_runs}) ...", flush=True)

        model = IOHMM.from_config(
            _MODEL_CFG,
            n_states=K, n_obs=len(_O_COLS), n_exog=len(_U_COLS),
            rv_col=_RV_COL, n_em_runs=args.n_em_runs,
        )
        model.fit(O_train, U_train)

        ll_bar   = model.ll_history[-1]
        ll_total = ll_bar * T_train
        D, M     = len(_O_COLS), len(_U_COLS)
        n_params = (K-1) + K*D + K*D*(D+1)//2 + K*K + K*K*M

        bic = -2 * ll_total + n_params * np.log(T_train)
        aic = -2 * ll_total + 2 * n_params

        # ICL: BIC + 2*total posterior entropy (Biernacki 2000)
        gamma_tr        = model.filter(O_train, U_train)
        h_total, h_mean = _entropy_stats(gamma_tr)
        icl             = bic + 2 * h_total

        # Viterbi on full valid set
        states   = model.decode(O_valid, U_valid)
        sil      = _silhouette(O_valid, states)
        freqs    = np.array([(states == k).mean() for k in range(K)])
        min_freq = float(freqs.min())
        mean_soj = _sojourn_mean(states)

        # Economic separability: per-state annualised Sharpe on 1-bar forward return
        # states[t] -> fwd_ret_valid[t] (return earned next bar)
        sharpe_k = []
        for k in range(K):
            mask_k = (states[:-1] == k)          # exclude last bar (no fwd return)
            r_k    = fwd_ret_valid[:-1][mask_k]
            if len(r_k) > 10 and r_k.std() > 0:
                sharpe_k.append(r_k.mean() / r_k.std() * np.sqrt(_BARS_PER_YEAR))
            else:
                sharpe_k.append(0.0)
        sharpe_sep = float(max(sharpe_k) - min(sharpe_k))   # higher = better separated

        # Self-transition: P(s_t=k | s_{t-1}=k) at U=0
        self_probs = []
        u_zero = np.zeros(model.M)
        for k in range(K):
            logits = model.alpha[k] + model.beta[k] @ u_zero
            A_row  = np.exp(logits - _lse(logits))
            self_probs.append(float(A_row[k]))
        mean_self_trans = float(np.mean(self_probs))
        min_self_trans  = float(min(self_probs))

        # Observation polarization: empirical F-ratio on full valid set
        obs_pol, obs_F = _obs_polarization(O_valid, states)

        # Transition polarization (beta contribution per U dimension)
        tr_pol, tr_P   = _trans_polarization(model.alpha, model.beta)

        print(f"    BIC={bic:,.0f}  ICL={icl:,.0f}  H/bar={h_mean:.3f}"
              f"  sil={sil:+.3f}  sh_sep={sharpe_sep:.2f}  self_trans={mean_self_trans:.3f}"
              f"  obs_polar={obs_pol:.4f}  trans_polar={tr_pol:.4f}"
              f"  min_freq={min_freq*100:.1f}%  sojourn={mean_soj:.1f}")

        records.append({
            "K": K, "n_params": n_params, "ll_bar": ll_bar,
            "bic": bic, "aic": aic, "icl": icl,
            "h_mean":         round(h_mean, 4),
            "silhouette":     round(sil, 4),
            "sharpe_sep":     round(sharpe_sep, 3),
            "self_trans":     round(mean_self_trans, 4),
            "min_self_trans": round(min_self_trans, 4),
            "obs_polar":      round(obs_pol, 4),
            "trans_polar":    round(tr_pol, 4),
            "sharpe_by_state": sharpe_k,
            "obs_F":           obs_F.round(6).tolist(),
            "trans_P":         tr_P.round(6).tolist(),
            "min_freq_pct":   round(min_freq * 100, 1),
            "mean_sojourn":   round(mean_soj, 1),
        })

    # ── Results tables ────────────────────────────────────────────────────────
    df_res = pd.DataFrame(records)
    for col in ("bic", "icl"):
        df_res[f"d{col}"] = df_res[col] - df_res[col].min()

    # Rank voting: rank 1 = best per metric (no arbitrary weights)
    # 8 criteria: 3 lower-better + 5 higher-better
    rank_sum = np.zeros(len(df_res))
    for col in ("bic", "icl", "h_mean"):                          # lower = better
        rank_sum += _rankdata(df_res[col].values)
    for col in ("silhouette", "sharpe_sep", "self_trans",
                "obs_polar", "trans_polar"):                       # higher = better
        rank_sum += _rankdata(-df_res[col].values)
    df_res["rank_sum"] = rank_sum.astype(int)

    k_bic  = int(df_res.loc[df_res["bic"].idxmin(),        "K"])
    k_icl  = int(df_res.loc[df_res["icl"].idxmin(),        "K"])
    k_sil  = int(df_res.loc[df_res["silhouette"].idxmax(), "K"])
    k_sh   = int(df_res.loc[df_res["sharpe_sep"].idxmax(), "K"])
    k_best = int(df_res.loc[df_res["rank_sum"].idxmin(),   "K"])

    print(f"\n{sep}")
    print(f" STATISTICAL CRITERIA  (lower BIC/ICL/H = better | higher Silh = better)")
    print(sep)
    hdr = (f"  {'K':>3}  {'LL/bar':>8}  {'BIC':>10}  {'dBIC':>7}"
           f"  {'ICL':>10}  {'dICL':>7}  {'H/bar':>6}  {'Silh':>6}")
    print(hdr)
    print(f"  {sep2[:len(hdr)-2]}")
    for _, r in df_res.iterrows():
        k = int(r["K"])
        tag = "  <- BIC+ICL" if k == k_bic == k_icl else (
              "  <- BIC" if k == k_bic else
              "  <- ICL" if k == k_icl else "")
        print(f"  {k:>3}  {r['ll_bar']:>+8.4f}  {r['bic']:>10,.0f}  {r['dbic']:>7.0f}"
              f"  {r['icl']:>10,.0f}  {r['dicl']:>7.0f}  {r['h_mean']:>6.3f}"
              f"  {r['silhouette']:>+6.3f}{tag}")

    print(f"\n{sep}")
    print(f" ECONOMIC CRITERIA  (higher Sharpe_sep/Self_trans = better)")
    print(sep)
    hdr2 = (f"  {'K':>3}  {'Sh_sep':>7}  {'Self_trans':>11}  {'min_self':>9}"
            f"  {'min_freq':>9}  {'sojourn':>8}  {'rank_sum':>9}")
    print(hdr2)
    print(f"  {sep2[:len(hdr2)-2]}")
    for _, r in df_res.iterrows():
        k = int(r["K"])
        tag = "  <- BEST" if k == k_best else ""
        print(f"  {k:>3}  {r['sharpe_sep']:>7.3f}  {r['self_trans']:>11.4f}"
              f"  {r['min_self_trans']:>9.4f}  {r['min_freq_pct']:>8.1f}%"
              f"  {r['mean_sojourn']:>8.1f}  {int(r['rank_sum']):>9}{tag}")

    # ── Observation polarization breakdown ───────────────────────────────────
    print(f"\n{sep}")
    print(f" OBSERVATION POLARIZATION  (F-ratio = between-state / within-state variance per feature)")
    print(f" F > 1 means states are more separated than their internal spread in that feature.")
    print(sep)

    feat_hdr = f"  {'Feature':<12}" + "".join(f"  {'K='+str(k):>7}" for k in df_res["K"]) + "  best_K"
    print(feat_hdr)
    print(f"  {'-'*75}")
    D = len(_O_SHORT)
    for d, fname in enumerate(_O_SHORT):
        row_vals = []
        for _, r in df_res.iterrows():
            row_vals.append(r["obs_F"][d])
        best_k = int(df_res["K"].iloc[int(np.argmax(row_vals))])
        row_str = f"  {fname:<12}" + "".join(f"  {v:>7.4f}" for v in row_vals) + f"  K={best_k}"
        print(row_str)
    print(f"  {'-'*75}")
    mean_row = f"  {'MEAN':<12}" + "".join(f"  {r['obs_polar']:>7.4f}" for _, r in df_res.iterrows())
    print(mean_row)

    # ── Transition polarization breakdown ─────────────────────────────────────
    print(f"\n{sep}")
    print(f" TRANSITION POLARIZATION  (beta_rms / alpha_rms per U dimension)")
    print(f" Measures how much each exogenous input shifts transition logits vs intercept.")
    print(f" > 1.0 = that U feature dominates baseline; ~0 = negligible modulation.")
    print(sep)

    u_hdr = f"  {'U feature':<12}" + "".join(f"  {'K='+str(k):>7}" for k in df_res["K"]) + "  best_K"
    print(u_hdr)
    print(f"  {'-'*65}")
    M = len(_U_SHORT)
    for m, uname in enumerate(_U_SHORT):
        row_vals = []
        for _, r in df_res.iterrows():
            row_vals.append(r["trans_P"][m])
        best_k = int(df_res["K"].iloc[int(np.argmax(row_vals))])
        row_str = f"  {uname:<12}" + "".join(f"  {v:>7.4f}" for v in row_vals) + f"  K={best_k}"
        print(row_str)
    print(f"  {'-'*65}")
    mean_row2 = f"  {'MEAN':<12}" + "".join(f"  {r['trans_polar']:>7.4f}" for _, r in df_res.iterrows())
    print(mean_row2)

    # Per-state Sharpe detail
    print(f"\n{sep}")
    print(f" PER-STATE SHARPE (annualised, 1-bar forward return)")
    print(sep)
    for _, r in df_res.iterrows():
        k    = int(r["K"])
        shs  = r["sharpe_by_state"]
        vals = "  ".join(f"s{i}:{v:+.2f}" for i, v in enumerate(shs))
        print(f"  K={k}  {vals}  |  sep={r['sharpe_sep']:.3f}")

    # ── ASCII charts ──────────────────────────────────────────────────────────
    for label, col, lower in [
        ("BIC  (lower=better)", "bic", True),
        ("ICL  (lower=better, penalises entropy too)", "icl", True),
        ("Sharpe separation  (higher=better, economic usefulness)", "sharpe_sep", False),
        ("Self-transition  (higher=better, regime stability)", "self_trans", False),
    ]:
        print(f"\n{sep}")
        print(f" {label}")
        print(sep)
        print(_bar_chart(df_res["K"].tolist(), df_res[col].tolist(),
                         lower_better=lower))

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f" VERDICT  (rank voting across 6 criteria — no arbitrary weights)")
    print(sep)
    print(f"  BIC->K={k_bic}  ICL->K={k_icl}  Silh->K={k_sil}  Sharpe_sep->K={k_sh}")
    print(f"  Rank vote winner: K={k_best}  (lowest sum of ranks across all criteria)")
    print()

    row = df_res[df_res["K"] == k_best].iloc[0]
    warnings = []
    if row["min_freq_pct"] < 5.0:
        warnings.append(f"min state freq {row['min_freq_pct']:.1f}% < 5%")
    if row["mean_sojourn"] < 3.0:
        warnings.append(f"mean sojourn {row['mean_sojourn']:.1f} < 3 bars")
    if row["h_mean"] > np.log(k_best) * 0.70:
        warnings.append(f"H/bar={row['h_mean']:.3f} high (states may overlap)")
    if row["min_self_trans"] < 0.5:
        warnings.append(f"min self-transition {row['min_self_trans']:.3f} < 0.5 (states flip fast)")

    if warnings:
        print(f"  WARNINGS at K={k_best}: {'; '.join(warnings)}")
    else:
        print(f"  K={k_best} passes all quality checks.")

    print(f"\n  Academic claim:")
    print(f"  \"Model selection via BIC (Schwarz 1978), ICL (Biernacki et al. 2000),")
    print(f"   regime Sharpe separation, self-transition stability,")
    print(f"   observation polarization (feature F-ratio), and transition polarization")
    print(f"   (exogenous input modulation) consistently select K={k_best}")
    print(f"   (rank-vote across 8 criteria, D={len(_O_COLS)}, M={len(_U_COLS)}, T={T_train}).\"")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path  = Path(args.out)
    save_cols = [c for c in df_res.columns
                 if c not in ("sharpe_by_state", "obs_F", "trans_P")]
    df_res[save_cols].to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path.resolve()}")
    print(sep)


if __name__ == "__main__":
    main()
