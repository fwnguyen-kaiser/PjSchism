-- Metadata bootstrap for instrument/timeframe identity.

CREATE TABLE IF NOT EXISTS instruments (
    instrument_id BIGSERIAL PRIMARY KEY,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    market_type TEXT NOT NULL,
    base_asset TEXT,
    quote_asset TEXT,
    launch_ts TIMESTAMPTZ,
    active BOOLEAN DEFAULT TRUE,
    UNIQUE (exchange, symbol, market_type)
);

CREATE TABLE IF NOT EXISTS timeframes_metadata (
    timeframe_id SMALLINT PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL UNIQUE,
    duration_seconds INTEGER NOT NULL,
    is_primary BOOLEAN DEFAULT FALSE
);

INSERT INTO timeframes_metadata (timeframe_id, code, label, duration_seconds, is_primary)
VALUES
    (1, 'PT15M', '15m', 900, FALSE),
    (2, 'PT1H', '1h', 3600, FALSE),
    (3, 'PT4H', '4h', 14400, TRUE),
    (4, 'P1D', '1d', 86400, FALSE)
ON CONFLICT (timeframe_id) DO UPDATE SET
    code = EXCLUDED.code,
    label = EXCLUDED.label,
    duration_seconds = EXCLUDED.duration_seconds,
    is_primary = EXCLUDED.is_primary;

INSERT INTO instruments (exchange, symbol, market_type, base_asset, quote_asset)
VALUES
    ('binance', 'BTCUSDT', 'perp', 'BTC', 'USDT'),
    ('binance', 'BTCUSDT', 'spot', 'BTC', 'USDT')
ON CONFLICT (exchange, symbol, market_type) DO NOTHING;
