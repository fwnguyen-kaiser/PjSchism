"""
test_ingestion_integrity.py - Schema and contract integrity between ingestion and API layers.

Unit tests (no DB): verify that Bar.to_dict() keys align with BarRepository SQL columns,
and that state_repo/refit_repo SQL column names match the DB schema definition.

Integration tests (requires running DB): marked with pytest.mark.integration.
Run with: pytest -m integration
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from schism.tests.conftest import make_bar, _bar_ts
from schism.data.ingestion.bar_builder import Bar


# ── Unit: Bar.to_dict() contract ─────────────────────────────────────────────

class TestBarContract:
    def test_bar_to_dict_has_all_upsert_columns(self):
        """bar_repo._UPSERT_BARS_SQL binds these keys — all must be in Bar.to_dict()."""
        expected_keys = {
            "bar_ts", "open", "high", "low", "close",
            "volume", "cvd", "oi", "lsr_top", "funding_rate",
            "num_trades", "taker_buy_base", "quote_volume",
        }
        bar = make_bar(_bar_ts(0))
        d = bar.to_dict()
        missing = expected_keys - d.keys()
        assert not missing, f"Bar.to_dict() missing keys needed by BarRepository: {missing}"

    def test_bar_to_dict_no_instrument_id_or_timeframe_id(self):
        """bar_repo adds instrument_id/timeframe_id separately; Bar must not pre-populate them."""
        bar = make_bar(_bar_ts(0))
        d = bar.to_dict()
        assert "instrument_id" not in d
        assert "timeframe_id" not in d

    def test_bar_timestamps_are_utc_aware(self):
        bar = make_bar(_bar_ts(0))
        assert bar.bar_ts.tzinfo is not None
        assert bar.to_dict()["bar_ts"].tzinfo is not None


# ── Unit: state_repo SQL column alignment ────────────────────────────────────

class TestStateRepoContract:
    def test_current_sql_selects_all_snapshot_fields(self):
        """RegimeSnapshot fields must all appear in _CURRENT_SQL SELECT list."""
        from schism.persistence.repositories.state_repo import _CURRENT_SQL
        from schism.api.schemas import RegimeSnapshot
        sql = str(_CURRENT_SQL)
        for field in RegimeSnapshot.model_fields:
            assert field in sql, f"_CURRENT_SQL missing field: {field}"

    def test_history_sql_selects_all_bar_regime_fields(self):
        from schism.persistence.repositories.state_repo import _HISTORY_SQL
        from schism.api.schemas import BarWithRegime
        sql = str(_HISTORY_SQL)
        for field in BarWithRegime.model_fields:
            assert field in sql, f"_HISTORY_SQL missing field: {field}"


# ── Unit: refit_repo SQL column alignment ────────────────────────────────────

class TestRefitRepoContract:
    def test_log_sql_selects_all_entry_fields(self):
        from schism.persistence.repositories.refit_repo import _LOG_SQL
        from schism.api.schemas import RefitLogEntry
        sql = str(_LOG_SQL)
        for field in RefitLogEntry.model_fields:
            assert field in sql, f"_LOG_SQL missing field: {field}"

    def test_insert_sql_does_not_include_refit_id(self):
        """refit_id is BIGSERIAL — must not be in the INSERT column list."""
        from schism.persistence.repositories.refit_repo import _INSERT_SQL
        sql = str(_INSERT_SQL)
        # refit_id must only appear in RETURNING, not in VALUES
        assert "RETURNING refit_id" in sql
        lines_before_returning = sql.split("RETURNING")[0]
        assert "refit_id" not in lines_before_returning.split("INSERT")[1]


# ── Integration: live DB schema vs repo expectations ─────────────────────────

@pytest_asyncio.fixture
async def db_engine():
    """Live DB engine for integration tests. Skipped if DATABASE_URL not set."""
    from schism.persistence.db import create_engine, ping_database
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set — skipping integration tests")
    e = create_engine(url)
    await ping_database(e)
    yield e
    await e.dispose()


@pytest.mark.asyncio
@pytest.mark.integration
class TestDbSchemaIntegrity:
    """Requires DATABASE_URL pointing to the running schism_db container."""

    async def test_all_expected_tables_exist(self, db_engine):
        from sqlalchemy import text
        expected = {"instruments", "timeframes_metadata", "ohlcv_bars",
                    "feature_vectors", "state_history", "refit_log"}
        async with db_engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            ))
            found = {row[0] for row in result}
        missing = expected - found
        assert not missing, f"Missing DB tables: {missing}"

    async def test_refit_log_pk_is_bigserial(self, db_engine):
        from sqlalchemy import text
        async with db_engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT column_name, data_type, column_default "
                "FROM information_schema.columns "
                "WHERE table_name='refit_log' AND column_name='refit_id'"
            ))
            row = result.fetchone()
        assert row is not None, "refit_id column missing from refit_log"
        assert row.data_type == "bigint"
        assert "nextval" in (row.column_default or ""), \
            "refit_id must be BIGSERIAL (nextval sequence)"

    async def test_refit_log_has_ts_index(self, db_engine):
        from sqlalchemy import text
        async with db_engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename='refit_log' AND indexname='idx_refit_log_ts'"
            ))
            row = result.fetchone()
        assert row is not None, "idx_refit_log_ts index missing from refit_log"

    async def test_secondary_indexes_exist(self, db_engine):
        from sqlalchemy import text
        expected_indexes = {
            "idx_ohlcv_instrument",
            "idx_feature_instrument",
            "idx_state_instrument",
        }
        async with db_engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT indexname FROM pg_indexes WHERE schemaname='public'"
            ))
            found = {row[0] for row in result}
        missing = expected_indexes - found
        assert not missing, f"Missing secondary indexes: {missing}"

    async def test_feature_vectors_f_columns_not_null(self, db_engine):
        from sqlalchemy import text
        async with db_engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT column_name, is_nullable FROM information_schema.columns "
                "WHERE table_name='feature_vectors' AND column_name LIKE 'f%' "
                "ORDER BY ordinal_position"
            ))
            rows = result.fetchall()
        nullable_f_cols = [r.column_name for r in rows if r.is_nullable == "YES"]
        assert not nullable_f_cols, \
            f"Observation vector columns must be NOT NULL, but these are nullable: {nullable_f_cols}"

    async def test_binance_perp_btcusdt_seeded(self, db_engine):
        from sqlalchemy import text
        async with db_engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT instrument_id FROM instruments "
                "WHERE exchange='binance' AND symbol='BTCUSDT' AND market_type='perp'"
            ))
            row = result.fetchone()
        assert row is not None, "Binance BTCUSDT perp instrument must be seeded by 001 migration"

    async def test_4h_timeframe_seeded(self, db_engine):
        from sqlalchemy import text
        async with db_engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT timeframe_id FROM timeframes_metadata WHERE label='4h'"
            ))
            row = result.fetchone()
        assert row is not None, "4h timeframe must be seeded by 001 migration"
        assert row.timeframe_id == 3
