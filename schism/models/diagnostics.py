"""
State inspection and manual labelling tool.

Usage:
    python -m schism.models.diagnostics --model /data/models/iohmm.pkl
    python -m schism.models.diagnostics --model /data/models/iohmm.pkl --out labels.csv

Output: rich terminal table (one row per state) + optional CSV.
State ordering follows §2: ascending μ_s[f7_rv_ratio], so state 0 = calmest.

Reading the table to assign labels:
  • f7_rv_ratio   — regime volatility level (low = calm, high = stressed)
  • f1_cvd_vol    — aggressive order flow (positive = buy pressure)
  • f2_oi_chg     — OI growth (positive = leveraging up)
  • f3_norm_ret   — vol-normalised return (positive = up-move)
  • f4_liq_sq     — liquidation squeeze proxy (positive = forced selling)
  • f5_spread     — bid-ask spread (high = illiquid)
  • f6_illiq      — Amihud illiquidity (high = thin book)
  • f8_vol_shock  — volume vs EWMA (positive = surge)
  • f9_flow_liq   — flow × illiq interaction
  • f10_flow_pos  — flow × OI interaction
  u1–u4           — exogenous funding/LSR (not in emission, shown separately)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

_O_COLS = [
    "f1_cvd_vol", "f2_oi_chg", "f3_norm_ret", "f4_liq_sq", "f5_spread",
    "f6_illiq", "f7_rv_ratio", "f8_vol_shock", "f9_flow_liq", "f10_flow_pos",
]

# Columns shown prominently in the terminal table
_DISPLAY_COLS = ["f7_rv_ratio", "f1_cvd_vol", "f2_oi_chg", "f3_norm_ret", "f4_liq_sq", "f6_illiq"]

_MECH = {
    "f1_cvd_vol":   "Mflow",
    "f2_oi_chg":    "Mpos",
    "f3_norm_ret":  "Minfo",
    "f4_liq_sq":    "Mpos",
    "f5_spread":    "Mliq",
    "f6_illiq":     "Mliq",
    "f7_rv_ratio":  "Mreg",
    "f8_vol_shock": "Mreg",
    "f9_flow_liq":  "Fx",
    "f10_flow_pos": "Fx",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    """Format a Z-scored mean: sign + two decimals."""
    return f"{v:+.2f}"


def _arrow(v: float, threshold: float = 0.3) -> str:
    """Return ↑/↓/→ based on Z-scored mean magnitude."""
    if v > threshold:
        return "↑"
    if v < -threshold:
        return "↓"
    return "→"


def _dominant_features(mu_row: np.ndarray, n: int = 3) -> str:
    """Return the n features with the largest absolute Z-score means."""
    idx = np.argsort(np.abs(mu_row))[::-1][:n]
    return "  ".join(f"{_O_COLS[i]}={_fmt(mu_row[i])}" for i in idx)


def state_summary_df(model) -> list[dict]:
    """
    Build a list of dicts (one per state) summarising emission means,
    sorted by ascending μ[f7_rv_ratio] (§2 ordering, already applied by fit).
    """
    rows = []
    for k in range(model.K):
        mu = model.mu[k]
        row = {
            "state": k,
            "label": model.labels[k],
            "model_ver": model.model_ver,
        }
        for i, col in enumerate(_O_COLS):
            row[col] = round(float(mu[i]), 4)
        row["dominant"] = _dominant_features(mu)
        rows.append(row)
    return rows


# ── Rich terminal output ──────────────────────────────────────────────────────

def print_summary(model, extra_gamma: np.ndarray | None = None) -> None:
    """
    Print a rich terminal table.

    extra_gamma : (T, K) posterior from filter() if available — used to compute
                  empirical frequency and mean sojourn for the provided window.
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        from rich.text import Text
    except ImportError:
        _print_plain(model, extra_gamma)
        return

    console = Console()
    table = Table(
        title=f"[bold]SCHISM State Summary[/bold]  model_ver={model.model_ver}",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
    )

    table.add_column("State", justify="center", style="bold", width=8)
    table.add_column("Label\n(edit me)", width=18)
    for col in _DISPLAY_COLS:
        table.add_column(f"{col}\n({_MECH[col]})", justify="right", width=12)
    table.add_column("Dominant features", width=40)
    if extra_gamma is not None:
        table.add_column("Freq%", justify="right", width=7)
        table.add_column("Sojourn\n(bars)", justify="right", width=9)

    # Freq / sojourn from gamma if provided
    freq_map: dict[int, float] = {}
    sojourn_map: dict[int, float] = {}
    if extra_gamma is not None:
        states = extra_gamma.argmax(axis=1)
        T = len(states)
        for k in range(model.K):
            freq_map[k] = float((states == k).mean()) * 100
        runs: dict[int, list[int]] = {k: [] for k in range(model.K)}
        cur, run = int(states[0]), 1
        for s in states[1:]:
            s = int(s)
            if s == cur:
                run += 1
            else:
                runs[cur].append(run)
                cur, run = s, 1
        runs[cur].append(run)
        for k in range(model.K):
            sojourn_map[k] = float(np.mean(runs[k])) if runs[k] else 0.0

    for k in range(model.K):
        mu = model.mu[k]
        rv = float(mu[_O_COLS.index("f7_rv_ratio")])

        # Colour: low RV = green, high RV = red
        if rv < -0.3:
            state_style = "bold green"
        elif rv > 0.5:
            state_style = "bold red"
        else:
            state_style = "bold yellow"

        cells = [
            Text(f"S{k}  {_arrow(rv)}", style=state_style),
            Text(model.labels[k], style="dim"),
        ]
        for col in _DISPLAY_COLS:
            v = float(mu[_O_COLS.index(col)])
            style = "green" if v > 0.2 else ("red" if v < -0.2 else "white")
            cells.append(Text(f"{_fmt(v)} {_arrow(v)}", style=style))
        cells.append(Text(_dominant_features(mu), style="dim"))

        if extra_gamma is not None:
            freq_ok = freq_map.get(k, 0) >= 5.0
            soj = sojourn_map.get(k, 0)
            soj_ok = 3 <= soj <= 100
            cells.append(Text(
                f"{freq_map.get(k, 0):.1f}%",
                style="green" if freq_ok else "bold red"
            ))
            cells.append(Text(
                f"{soj:.1f}",
                style="green" if soj_ok else "bold red"
            ))

        table.add_row(*cells)

    console.print()
    console.print(table)
    console.print(
        "\n[bold]How to label:[/bold] edit [cyan]model.labels[/cyan] "
        "then call [cyan]model.save(path)[/cyan].\n"
        "  e.g.  model.labels = ['LowVol', 'Trending', 'Volatile', 'Crisis']\n"
    )
    _print_label_snippet(model)


def _print_label_snippet(model) -> None:
    """Print a ready-to-paste Python snippet for setting labels."""
    try:
        from rich.console import Console
        from rich.syntax import Syntax
        console = Console()
        code = (
            f"from schism.models.iohmm import IOHMM\n"
            f"model = IOHMM.load('{os.environ.get('MODEL_PATH', '/data/models/iohmm.pkl')}')\n"
            f"model.labels = {model.labels!r}  # ← edit these\n"
            f"model.save('{os.environ.get('MODEL_PATH', '/data/models/iohmm.pkl')}')\n"
        )
        console.print(Syntax(code, "python", theme="monokai", line_numbers=False))
    except Exception:
        pass


def _print_plain(model, extra_gamma=None) -> None:
    """Fallback plain-text output when rich is not installed."""
    print(f"\n=== SCHISM State Summary  [{model.model_ver}] ===")
    header = "State | RV_ratio | CVD_vol | OI_chg | norm_ret | liq_sq | illiq | Dominant"
    print(header)
    print("-" * len(header))
    for k in range(model.K):
        mu = model.mu[k]
        vals = "  ".join(_fmt(mu[_O_COLS.index(c)]) for c in _DISPLAY_COLS)
        print(f"  S{k} ({model.labels[k]:<12}) | {vals} | {_dominant_features(mu)}")
    print()


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(model, out_path: str | Path) -> None:
    """Export full emission means to CSV for spreadsheet labelling."""
    rows = state_summary_df(model)
    fieldnames = ["state", "label", "model_ver"] + _O_COLS + ["dominant"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(description="Inspect fitted IOHMM states for manual labelling")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", "/data/models/iohmm.pkl"),
        help="Path to pickled IOHMM (default: $MODEL_PATH)",
    )
    parser.add_argument("--out", default=None, help="Optional CSV output path")
    parser.add_argument(
        "--db-features",
        action="store_true",
        help="Connect to DATABASE_URL, fetch recent features, and show empirical freq/sojourn",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    from schism.models.iohmm import IOHMM
    model = IOHMM.load(model_path)

    gamma = None
    if args.db_features:
        gamma = _fetch_gamma_from_db(model)

    print_summary(model, extra_gamma=gamma)

    if args.out:
        export_csv(model, args.out)


def _fetch_gamma_from_db(model) -> np.ndarray | None:
    """
    Fetch the last TRAIN_WINDOW bars of feature_vectors from DB,
    run filter() to get empirical gamma for freq/sojourn stats.
    Returns None on any error.
    """
    import asyncio
    from datetime import datetime, timedelta, timezone

    async def _run():
        from schism.persistence.db import create_engine, create_session_factory, session_scope
        from schism.persistence.repositories.feature_repo import (
            FeatureRepository, _O_FEATURE_COLS, _U_FEATURE_COLS,
        )
        from schism.persistence.repositories.state_repo import (
            resolve_instrument_id, resolve_timeframe_id,
        )

        db_engine = create_engine()
        if db_engine is None:
            return None
        session_factory = create_session_factory(db_engine)
        repo = FeatureRepository(session_factory)

        async with session_scope(session_factory) as session:
            iid = await resolve_instrument_id(session, "binance", "BTCUSDT", "futures")
            tid = await resolve_timeframe_id(session, "4h")
        if iid is None or tid is None:
            return None

        now = datetime.now(tz=timezone.utc)
        window = int(os.environ.get("SCHISM_TRAIN_WINDOW", "1080"))
        from_ts = now - timedelta(hours=4 * window)
        df = await repo.fetch_features(iid, tid, from_ts, now)
        if df.empty:
            return None

        if "f5_spread" in df.columns:
            df["f5_spread"] = df["f5_spread"].fillna(0.0)

        O = df[_O_FEATURE_COLS].to_numpy(dtype=float)
        U = df[_U_FEATURE_COLS].to_numpy(dtype=float)
        return model.filter(O, U)

    try:
        return asyncio.run(_run())
    except Exception as exc:
        print(f"[db-features] could not fetch: {exc}", file=sys.stderr)
        return None


if __name__ == "__main__":
    _main()
