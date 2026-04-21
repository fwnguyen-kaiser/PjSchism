#!/usr/bin/env python3
"""
SCHISM — project scaffold
Usage:  python scaffold.py [target_dir]
        python scaffold.py            -> creates structure in current dir
        python scaffold.py ~/projects -> creates ~/projects/PjSchism/

Generates standard SE folder structure + empty files (Separation of Concerns).
Does not overwrite existing files.
Python 3.11+ required.
"""

import sys
from pathlib import Path

# ─── APP STRUCTURE (Inside schism/) ─────────────────────────────────────────
DIRS = [
    "config",
    "data/ingestion",
    "data/preprocessing",
    "model",
    "training",
    "runtime",
    "backtesting",
    "api/routers",
    "persistence/repositories",
    "utils",
    "tests",
    "db/migrations",
]

# ─── INFRASTRUCTURE SHELL (Root directory) ──────────────────────────────────
ROOT_FILES = [
    ("pyproject.toml", ""),
    (".env.example",
     "BINANCE_API_KEY=\nBINANCE_API_SECRET=\n"
     "DATABASE_URL=postgresql+asyncpg://schism:112200@localhost:5432/schism\n"
     "REDIS_URL=redis://localhost:6379\n"
     "ENV=dev\n"
     "DB_PASSWORD=112200\n"),
    (".gitignore",
     ".env\n__pycache__/\n*.pyc\n*.pyo\n.pytest_cache/\n"
     ".venv/\ndist/\nbuild/\n*.egg-info/\n"
     "# data volumes — never commit\n"
     "schism/data/volumes/\n"),
    ("docker-compose.yml", ""),
    ("docker-compose.prod.yml", ""),
    ("Dockerfile.ingestion", ""),
    ("Dockerfile.model", ""),
    ("Dockerfile.api", ""),
    ("docker-compose.volumes.note", ""),
]

# ─── APP CORE (Inside schism/) ──────────────────────────────────────────────
APP_FILES = [
    # ── root of schism ──
    ("main.py", "FastAPI app entry point — create app, include routers, startup event."),

    # ── config ──
    ("config/__init__.py", "Load and expose model_config, feature_config, refit_config."),
    ("config/model_config.yaml",
     "K: 4\nDim: 10\nlambda_reg: 0.01\ntau_percentile: 95\nn_em_runs: 5\n"
     "max_iter: 200\ntol: 1.0e-5\ncovariance_floor: 1.0e-4\n"),
    ("config/feature_config.yaml",
     "vif_threshold: 5\nrho: 0.85\nwinsorize_pct: [1, 99]\n"
     "interaction_features: [9, 10]\n"),
    ("config/refit_config.yaml",
     "cooldown_bars: 30\nbackstop_days: 90\nrv_ratio_thresh: 1.8\n"
     "rv_ratio_consecutive_bars: 12\nll_rolling_window_days: 30\n"
     "ll_degradation_sigma: 2.0\ndelta_align: 2.0\n"),

    # ── data / ingestion ──
    ("data/__init__.py", ""),
    ("data/ingestion/__init__.py", ""),
    ("data/ingestion/binance_client.py",
     "Binance REST historical klines + WebSocket live 4h bar feed."),
    ("data/ingestion/bar_builder.py",
     "Aggregate raw trades / 1m bars -> 4h OHLCV + CVD computation."),
    ("data/ingestion/vision_crawler.py",
     "Crawl data.binance.vision daily metrics zips for OI/LSR history > 30 days."),
    ("data/ingestion/data_store.py",
     "Read/write parquet. Merge ohlcv + metrics on timestamp (inner join)."),

    # ── data / preprocessing ──
    ("data/preprocessing/__init__.py", ""),
    ("data/preprocessing/feature_engine.py",
     "Build Ot (dim 10, Eq.5) and Ut (dim 4, Eq.2). Interaction terms formed from raw inputs."),
    ("data/preprocessing/vif_checker.py",
     "VIF + pairwise corr check. Returns valid feature mask. Drops F8 if VIF >= 5."),
    ("data/preprocessing/zscore.py",
     "RollingZScore: fit on train window, transform online. Interaction-last order per V1.4."),

    # ── model ──
    ("model/__init__.py", ""),
    ("model/iohmm.py",
     "IOHMMBase: subclass hmmlearn GaussianHMM. Override transition with softmax(Ut). "
     "Enforce Tr(Sigma) <= tau + epsilon*I floor post M-step."),
    ("model/transition.py",
     "Softmax P(St=j | St-1=i, Ut) per Eq.6. Custom M-step override for beta_ij with L2 reg."),
    ("model/emission.py",
     "Gaussian emission N(mu_s, Sigma_s). compute_tau() bootstrap 95th-percentile. "
     "project_covariance() enforces Tr constraint."),
    ("model/inference.py",
     "Filtered posterior P(St | O1:t, U1:t) + Viterbi. Causal-only — no look-ahead."),
    ("model/alignment.py",
     "Hungarian label alignment across refits. Cost matrix Cij = ||mu_old_i - mu_new_j||^2. "
     "Raises RegimeAlignmentWarning if cost > delta_align."),

    # ── training ──
    ("training/__init__.py", ""),
    ("training/initializer.py",
     "K-means init for mu_s. Sigma_s = cov(Ot) + eps*I. Deterministic seed."),
    ("training/trainer.py",
     "Orchestrate EM: init -> fit -> validate -> pick best run (highest logL). "
     "Warns if any state freq < 5% — suggests K-1."),
    ("training/walk_forward.py",
     "Generate (train_idx, val_idx) folds with 10-bar embargo. Min 360 bars train."),
    ("training/evaluator.py",
     "Compute sojourn, state freq, BIC, CV of feature contributions across folds."),

    # ── runtime ──
    ("runtime/__init__.py", ""),
    ("runtime/regime_engine.py",
     "Main loop: load checkpoint -> infer filtered posterior -> emit regime event to Redis."),
    ("runtime/refit_monitor.py",
     "Track delta_LL rolling stats, RVratio consecutive bars, backstop countdown."),
    ("runtime/refit_scheduler.py",
     "Enforce 5-day cooldown. Trigger trainer.fit_best() when conditions met. "
     "Backstop overrides cooldown."),

    # ── backtesting ──
    ("backtesting/__init__.py", ""),
    ("backtesting/engine.py",
     "Replay historical bars in causal inference mode (filtered posterior only, no smoothing)."),
    ("backtesting/refit_simulator.py",
     "Replay full refit schedule: triggers + cooldown + alignment as they would fire live."),
    ("backtesting/position_simulator.py",
     "Simulate fills, slippage, funding cost per regime label."),
    ("backtesting/metrics.py",
     "Sharpe, max drawdown, regime-conditional hit rate, sojourn distribution."),

    # ── api ──
    ("api/__init__.py", ""),
    ("api/routers/__init__.py", ""),
    ("api/routers/regime.py",
     "GET /regime/current — latest state + posterior + features snapshot. "
     "GET /regime/history?from=&to= — bars with regime overlay. "
     "GET /regime/stats — per-state freq, sojourn, health."),
    ("api/routers/refit.py",
     "GET /refit/log — refit history with trigger, delta_bic, alignment result."),
    ("api/routers/backtest.py",
     "GET /backtest/results?run_id= — equity curve + regime bars + metrics."),
    ("api/schemas.py",
     "Pydantic v2 response models for all endpoints. Shared across routers."),
    ("api/dependencies.py",
     "FastAPI dependency injection: get_db(), get_redis(), get_regime_engine()."),

    # ── persistence ──
    ("persistence/__init__.py", ""),
    ("persistence/db.py",
     "SQLAlchemy async engine + AsyncSessionLocal factory. DATABASE_URL from env."),
    ("persistence/repositories/__init__.py", ""),
    ("persistence/repositories/bar_repo.py",
     "Read/write ohlcv_bars hypertable. Bulk insert from vision_crawler output."),
    ("persistence/repositories/state_repo.py",
     "Append state_history rows. Query by time window for /regime/history."),
    ("persistence/repositories/refit_repo.py",
     "Insert refit_log entries. Query for /refit/log endpoint."),
    ("persistence/checkpoint.py",
     "Save/load IOHMM params + metadata to volume mount (not DB). "
     "Versioned by timestamp: checkpoint_<iso_ts>.pkl"),

    # ── db migrations ──
    ("db/__init__.py", ""),
    ("db/migrations/001_create_hypertables.sql",
     "-- ohlcv_bars, feature_vectors, state_history as TimescaleDB hypertables."),
    ("db/migrations/002_create_refit_log.sql",
     "-- refit_log table (regular, not hypertable — low cardinality)."),
    ("db/init.sql",
     "-- Run once on container startup: CREATE EXTENSION timescaledb; "
     "then execute migrations in order."),

    # ── utils ──
    ("utils/__init__.py", ""),
    ("utils/logger.py",
     "Structured JSON logging via structlog. "
     "Two channels: regime_logger (state transitions) + refit_logger (refit events). "
     "Includes bar_ts (exchange time) not just wall clock."),
    ("utils/date_helpers.py",
     "normalize_ts(ts, source) — handle Binance ms vs s inconsistency. "
     "bar_index_to_utc(idx, start_ts, interval) — map model bar index to wall time. "
     "ms_to_datetime(), datetime_to_bar_ts()."),
    ("utils/exceptions.py",
     "DataMissingError, RefitCooldownError, VIFViolationError, "
     "IdentifiabilityError, RegimeAlignmentWarning."),

    # ── tests ──
    ("tests/__init__.py", ""),
    ("tests/conftest.py",
     "Fixtures: synthetic_ohlcv(), synthetic_features(), mock_iohmm(), "
     "async_db_session(), mock_redis()."),
    ("tests/test_feature_engine.py",
     "Test zscore interaction order, VIF drop logic, winsorization bounds."),
    ("tests/test_iohmm.py",
     "Test EM convergence on synthetic data, covariance floor applied, "
     "Tr(Sigma) <= tau after fit."),
    ("tests/test_alignment.py",
     "Test Hungarian assignment correctness, RegimeAlignmentWarning on drift > delta_align."),
    ("tests/test_refit_triggers.py",
     "Test cooldown enforcement, backstop override during cooldown edge case."),
    ("tests/test_backtesting.py",
     "Test causal inference mode only — assert smoothed posterior never used. "
     "No look-ahead leakage."),
]

# ─── volume structure note ──────────────────────────────────────────────────
VOLUME_README = """\
# data/volumes/

This directory contains Docker bind-mount volumes.
DO NOT commit to git (already excluded via .gitignore).

Structure:
  data/volumes/pgdata/        <- PostgreSQL data (bind-mount instead of named volume)
  data/volumes/checkpoints/   <- Model checkpoint .pkl files
  data/volumes/parquet/       <- Raw parquet files from vision_crawler

Why bind-mount instead of named volume:
  Named volumes are deleted when running `docker compose down -v`.
  Bind-mounts exist on the host file system -> safe from docker teardowns.

Recommended backup strategy:
  pgdata/      -> periodic pg_dump or rsync
  checkpoints/ -> git-lfs or cloud storage
  parquet/     -> critical (crawled OI history), backup to S3/GDrive
"""

COMPOSE_VOLUME_SNIPPET = """\
# ── docker-compose.yml volume section (bind-mount) ──────────────────────────
# ... (Content remains unchanged) ...
"""

# ─── scaffold logic ─────────────────────────────────────────────────────────

def write_python_stub(path: Path, hint: str) -> None:
    """Create .py stub with docstring from hint."""
    if not hint:
        content = '"""TODO"""\n'
    else:
        content = f'"""\n{hint}\n"""\n'
    path.write_text(content, encoding="utf-8")

def write_file(path: Path, hint: str) -> None:
    suffix = path.suffix
    if suffix == ".py":
        write_python_stub(path, hint)
    elif suffix in (".yaml", ".yml"):
        path.write_text(hint if hint else "# TODO\n", encoding="utf-8")
    elif suffix == ".sql":
        path.write_text(hint if hint else "-- TODO\n", encoding="utf-8")
    elif suffix == ".toml":
        path.write_text(
            '[tool.poetry]\nname = "schism"\nversion = "0.1.0"\n'
            'description = "IOHMM crypto regime detection"\n\n'
            '[tool.poetry.dependencies]\npython = "^3.11"\n',
            encoding="utf-8",
        )
    else:
        path.write_text(hint if hint else "", encoding="utf-8")

def scaffold(target: Path) -> None:
    project_root = target.resolve()
    app_root = project_root / "schism"
    
    print(f"Scaffolding Project Root into: {project_root}")
    print(f"Scaffolding App Root into: {app_root}")

    project_root.mkdir(parents=True, exist_ok=True)
    app_root.mkdir(parents=True, exist_ok=True)

    for d in DIRS:
        (app_root / d).mkdir(parents=True, exist_ok=True)

    created = skipped = 0

    # 1. Write infrastructure files to Root
    for rel, hint in ROOT_FILES:
        p = project_root / rel
        if p.exists():
            print(f"  SKIP (exists)  {rel} (Root)")
            skipped += 1
        else:
            write_file(p, hint)
            print(f"  CREATE         {rel} (Root)")
            created += 1

    # 2. Write app code files to schism/
    for rel, hint in APP_FILES:
        p = app_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            print(f"  SKIP (exists)  schism/{rel}")
            skipped += 1
        else:
            write_file(p, hint)
            print(f"  CREATE         schism/{rel}")
            created += 1

    # 3. Handle Volume directories inside schism/
    for vdir in ["data/volumes/pgdata", "data/volumes/checkpoints", "data/volumes/parquet"]:
        vp = app_root / vdir
        vp.mkdir(parents=True, exist_ok=True)
        gitkeep = vp / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    vol_readme = app_root / "data" / "volumes" / "README.md"
    if not vol_readme.exists():
        vol_readme.write_text(VOLUME_README, encoding="utf-8")

    print(f"\nDone. {created} files created, {skipped} skipped.")
    print("\nNext steps:")
    print("  1. cp .env.example .env  && fill in credentials")
    print("  2. pip install -r requirements.txt")
    print("  3. docker compose up db redis -d")
    print("  4. python -m schism.data.ingestion.vision_crawler --symbol BTCUSDT --start 2023-01-01")

if __name__ == "__main__":
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    scaffold(target_dir)