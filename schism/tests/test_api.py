"""
test_api.py - Unit tests for the FastAPI application.

All DB and Redis interactions are mocked. No real network or container needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from schism.api.application import create_app
from schism.api.dependencies import get_session

pytestmark = pytest.mark.asyncio

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client():
    """App without DB or Redis — lifespan skips connections, state.session_factory=None."""
    app = create_app()
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock()
    mock_redis.aclose = AsyncMock()
    with (
        patch("schism.api.application.create_engine", return_value=None),
        patch("schism.api.application.aioredis.from_url", return_value=mock_redis),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


@pytest_asyncio.fixture
async def client_with_db():
    """App with get_session overridden via dependency_overrides — no lifespan DB connection."""
    app = create_app()
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock()
    mock_redis.aclose = AsyncMock()
    mock_session = AsyncMock()

    async def _override_get_session():
        yield mock_session

    app.dependency_overrides[get_session] = _override_get_session

    with (
        patch("schism.api.application.create_engine", return_value=None),
        patch("schism.api.application.aioredis.from_url", return_value=mock_redis),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, mock_session

    app.dependency_overrides.clear()


# ── Health ─────────────────────────────────────────────────────────────────────

class TestHealth:
    async def test_health_returns_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    async def test_health_ok_when_db_unavailable(self, client):
        # Health must return 200 even when DB is None (no DB dependency injected)
        resp = await client.get("/health")
        assert resp.status_code == 200


# ── Regime endpoints return 503 when DB unavailable ───────────────────────────

class TestRegimeNoDb:
    async def test_current_503_when_no_db(self, client):
        resp = await client.get("/regime/current")
        assert resp.status_code == 503

    async def test_history_503_when_no_db(self, client):
        resp = await client.get("/regime/history")
        assert resp.status_code == 503

    async def test_stats_503_when_no_db(self, client):
        resp = await client.get("/regime/stats")
        assert resp.status_code == 503


# ── Refit endpoint returns 503 when DB unavailable ───────────────────────────

class TestRefitNoDb:
    async def test_log_503_when_no_db(self, client):
        resp = await client.get("/refit/log")
        assert resp.status_code == 503


# ── Regime endpoints with mocked DB ──────────────────────────────────────────

class TestRegimeWithDb:
    async def test_current_404_when_instrument_not_found(self, client_with_db):
        client, mock_session = client_with_db
        mock_session.execute = AsyncMock(return_value=_mock_result(None))

        resp = await client.get("/regime/current?symbol=UNKNOWN")
        assert resp.status_code == 404

    async def test_current_404_when_no_state_yet(self, client_with_db):
        client, mock_session = client_with_db
        mock_session.execute = AsyncMock(side_effect=[
            _mock_result((1,)),   # resolve_instrument_id
            _mock_result((3,)),   # resolve_timeframe_id
            _mock_result(None),   # get_current → no row
        ])

        resp = await client.get("/regime/current")
        assert resp.status_code == 404
        assert "no regime state" in resp.json()["detail"]

    async def test_current_returns_snapshot(self, client_with_db):
        client, mock_session = client_with_db
        ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        state_row = _make_state_row(ts, instrument_id=1, timeframe_id=3,
                                    state=2, label="regime_2", confidence=0.91,
                                    posterior=[0.03, 0.03, 0.91, 0.03], model_ver="v1")
        mock_session.execute = AsyncMock(side_effect=[
            _mock_result((1,)),
            _mock_result((3,)),
            _mock_result(state_row),
        ])

        resp = await client.get("/regime/current")
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"] == 2
        assert body["label"] == "regime_2"
        assert body["confidence"] == pytest.approx(0.91)

    async def test_stats_returns_empty_list_when_no_data(self, client_with_db):
        client, mock_session = client_with_db
        mock_session.execute = AsyncMock(side_effect=[
            _mock_result((1,)),
            _mock_result((3,)),
            _mock_result([]),
        ])

        resp = await client.get("/regime/stats")
        assert resp.status_code == 200
        assert resp.json() == []


# ── Refit /log with mocked DB ─────────────────────────────────────────────────

class TestRefitWithDb:
    async def test_log_returns_entries(self, client_with_db):
        client, mock_session = client_with_db
        ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        log_row = _make_refit_row(refit_id=1, refit_ts=ts, instrument_id=1,
                                  timeframe_id=3, trigger="ll_degradation")
        mock_session.execute = AsyncMock(side_effect=[
            _mock_result((1,)),
            _mock_result((3,)),
            _mock_result(log_row),
        ])

        resp = await client.get("/refit/log")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["trigger"] == "ll_degradation"
        assert body[0]["refit_id"] == 1


# ── Backtest stub ─────────────────────────────────────────────────────────────

class TestBacktest:
    async def test_results_stub(self, client):
        resp = await client.get("/backtest/results")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_implemented"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_result(row):
    mock = MagicMock()
    if row is None:
        mock.fetchone.return_value = None
        mock.fetchall.return_value = []
    elif isinstance(row, list):
        mock.fetchone.return_value = row[0] if row else None
        mock.fetchall.return_value = row
    else:
        mock.fetchone.return_value = row
        mock.fetchall.return_value = [row]
    return mock


def _make_state_row(bar_ts, *, instrument_id, timeframe_id, state, label,
                   confidence, posterior, model_ver):
    row = MagicMock()
    row.bar_ts = bar_ts
    row.instrument_id = instrument_id
    row.timeframe_id = timeframe_id
    row.state = state
    row.label = label
    row.confidence = confidence
    row.posterior = posterior
    row.model_ver = model_ver
    return row


def _make_refit_row(*, refit_id, refit_ts, instrument_id, timeframe_id, trigger):
    row = MagicMock()
    row.refit_id = refit_id
    row.refit_ts = refit_ts
    row.instrument_id = instrument_id
    row.timeframe_id = timeframe_id
    row.trigger = trigger
    row.delta_bic = None
    row.alignment_ok = None
    row.drift_alert = False
    row.dim_used = None
    row.model_ver = None
    row.cooldown_end_ts = None
    row.notes = None
    return row
