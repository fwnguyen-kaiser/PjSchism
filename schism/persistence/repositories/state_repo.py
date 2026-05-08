"""state_history queries: current regime snapshot, history window, per-state stats."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_RESOLVE_INSTRUMENT_SQL = text("""
    SELECT instrument_id FROM instruments
    WHERE exchange = :exchange AND symbol = :symbol AND market_type = :market_type
""")

_RESOLVE_TIMEFRAME_SQL = text("""
    SELECT timeframe_id FROM timeframes_metadata WHERE label = :label
""")

_CURRENT_SQL = text("""
    SELECT bar_ts, instrument_id, timeframe_id, state, label, confidence, posterior, model_ver
    FROM state_history
    WHERE instrument_id = :instrument_id AND timeframe_id = :timeframe_id
    ORDER BY bar_ts DESC
    LIMIT 1
""")

_HISTORY_SQL = text("""
    SELECT sh.bar_ts, ob.close, ob.volume, sh.state, sh.label, sh.confidence
    FROM state_history sh
    LEFT JOIN ohlcv_bars ob
        ON ob.instrument_id = sh.instrument_id
       AND ob.timeframe_id  = sh.timeframe_id
       AND ob.bar_ts        = sh.bar_ts
    WHERE sh.instrument_id = :instrument_id
      AND sh.timeframe_id  = :timeframe_id
      AND sh.bar_ts >= :from_ts
      AND sh.bar_ts <  :to_ts
    ORDER BY sh.bar_ts DESC
""")

_ALL_STATES_SQL = text("""
    SELECT state, label, bar_ts
    FROM state_history
    WHERE instrument_id = :instrument_id AND timeframe_id = :timeframe_id
    ORDER BY bar_ts
""")


async def resolve_instrument_id(
    session: AsyncSession,
    exchange: str,
    symbol: str,
    market_type: str,
) -> int | None:
    result = await session.execute(
        _RESOLVE_INSTRUMENT_SQL,
        {"exchange": exchange.lower(), "symbol": symbol.upper(), "market_type": market_type.lower()},
    )
    row = result.fetchone()
    return int(row[0]) if row else None


async def resolve_timeframe_id(session: AsyncSession, label: str) -> int | None:
    result = await session.execute(_RESOLVE_TIMEFRAME_SQL, {"label": label.lower()})
    row = result.fetchone()
    return int(row[0]) if row else None


async def get_current(
    session: AsyncSession, instrument_id: int, timeframe_id: int
) -> dict | None:
    result = await session.execute(
        _CURRENT_SQL, {"instrument_id": instrument_id, "timeframe_id": timeframe_id}
    )
    row = result.fetchone()
    if row is None:
        return None
    return {
        "bar_ts": row.bar_ts,
        "instrument_id": row.instrument_id,
        "timeframe_id": row.timeframe_id,
        "state": row.state,
        "label": row.label,
        "confidence": row.confidence,
        "posterior": row.posterior,
        "model_ver": row.model_ver,
    }


async def get_history(
    session: AsyncSession,
    instrument_id: int,
    timeframe_id: int,
    from_ts: datetime,
    to_ts: datetime,
) -> list[dict]:
    result = await session.execute(
        _HISTORY_SQL,
        {
            "instrument_id": instrument_id,
            "timeframe_id": timeframe_id,
            "from_ts": from_ts,
            "to_ts": to_ts,
        },
    )
    return [
        {
            "bar_ts": row.bar_ts,
            "close": row.close,
            "volume": row.volume,
            "state": row.state,
            "label": row.label,
            "confidence": row.confidence,
        }
        for row in result.fetchall()
    ]


async def get_stats(
    session: AsyncSession, instrument_id: int, timeframe_id: int
) -> list[dict]:
    result = await session.execute(
        _ALL_STATES_SQL, {"instrument_id": instrument_id, "timeframe_id": timeframe_id}
    )
    rows = result.fetchall()
    if not rows:
        return []

    counts: dict[tuple[int, str], int] = defaultdict(int)
    # Track run lengths for sojourn computation
    runs: dict[tuple[int, str], list[int]] = defaultdict(list)
    prev_state: int | None = None
    run_len = 0
    run_key: tuple[int, str] | None = None

    for row in rows:
        key = (row.state, row.label)
        counts[key] += 1
        if row.state == prev_state:
            run_len += 1
        else:
            if run_key is not None:
                runs[run_key].append(run_len)
            run_len = 1
            run_key = key
            prev_state = row.state
    if run_key is not None:
        runs[run_key].append(run_len)

    total = len(rows)
    return sorted(
        [
            {
                "state": state,
                "label": label,
                "frequency_pct": round(100.0 * count / total, 2),
                "mean_sojourn_bars": round(
                    sum(runs.get((state, label), [1]))
                    / max(len(runs.get((state, label), [1])), 1),
                    2,
                ),
            }
            for (state, label), count in counts.items()
        ],
        key=lambda d: d["state"],
    )
