-- ohlcv_bars, feature_vectors, state_history as TimescaleDB hypertables.
-- 001_create_hypertables.sql
-- TimescaleDB hypertables cho time-series data

-- OHLCV 4h bars
CREATE TABLE IF NOT EXISTS ohlcv_bars (
    bar_ts          TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION,
    volume          DOUBLE PRECISION,
    oi              DOUBLE PRECISION,
    lsr_top         DOUBLE PRECISION,
    funding_rate    DOUBLE PRECISION,
    PRIMARY KEY (bar_ts, symbol)
);
SELECT create_hypertable('ohlcv_bars', 'bar_ts', if_not_exists => TRUE);

-- Feature vectors (Ot dim 10, Ut dim 4) — computed once, stored to avoid recompute
CREATE TABLE IF NOT EXISTS feature_vectors (
    bar_ts          TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    f1_cvd_vol      DOUBLE PRECISION,
    f2_oi_chg       DOUBLE PRECISION,
    f3_norm_ret     DOUBLE PRECISION,
    f4_liq_sq       DOUBLE PRECISION,
    f5_spread       DOUBLE PRECISION,
    f6_illiq        DOUBLE PRECISION,
    f7_rv_ratio     DOUBLE PRECISION,
    f8_vol_shock    DOUBLE PRECISION,   -- NULL nếu VIF drop
    f9_flow_liq     DOUBLE PRECISION,
    f10_flow_pos    DOUBLE PRECISION,
    -- Ut exogenous
    u1_ewma_fr      DOUBLE PRECISION,
    u2_delta_fr     DOUBLE PRECISION,
    u3_fr_spread    DOUBLE PRECISION,
    u4_delta_lsr    DOUBLE PRECISION,
    dim_used        SMALLINT,           -- 9 hoặc 10
    PRIMARY KEY (bar_ts, symbol)
);
SELECT create_hypertable('feature_vectors', 'bar_ts', if_not_exists => TRUE);

-- State history — append-only, một row per bar
CREATE TABLE IF NOT EXISTS state_history (
    bar_ts          TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    state           SMALLINT        NOT NULL,   -- 1..K
    label           TEXT            NOT NULL,   -- accumulation|mean_reversion|distribution|liquidation
    confidence      DOUBLE PRECISION,           -- max(posterior)
    posterior       DOUBLE PRECISION[],         -- array[K]
    model_ver       TEXT,
    PRIMARY KEY (bar_ts, symbol)
);
SELECT create_hypertable('state_history', 'bar_ts', if_not_exists => TRUE);

-- Compression policy: bars cũ hơn 7 ngày compress lại
-- SELECT add_compression_policy('ohlcv_bars',    INTERVAL '7 days');
-- SELECT add_compression_policy('feature_vectors', INTERVAL '7 days');
-- SELECT add_compression_policy('state_history', INTERVAL '7 days');