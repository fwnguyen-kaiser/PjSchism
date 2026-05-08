-- Canonical time-series tables keyed by instrument/timeframe/timestamp.

CREATE TABLE IF NOT EXISTS ohlcv_bars (
    bar_ts          TIMESTAMPTZ     NOT NULL,
    instrument_id   BIGINT          NOT NULL REFERENCES instruments(instrument_id),
    timeframe_id    SMALLINT        NOT NULL REFERENCES timeframes_metadata(timeframe_id),
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION,
    volume          DOUBLE PRECISION,
    cvd             DOUBLE PRECISION,
    oi              DOUBLE PRECISION,
    lsr_top         DOUBLE PRECISION,
    funding_rate    DOUBLE PRECISION,
    num_trades      BIGINT,
    taker_buy_base  DOUBLE PRECISION,
    quote_volume    DOUBLE PRECISION,
    ingested_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    source          TEXT,
    PRIMARY KEY (instrument_id, timeframe_id, bar_ts)
);
SELECT create_hypertable('ohlcv_bars', 'bar_ts', if_not_exists => TRUE);
-- Secondary index: hypertable chunk indexes only cover bar_ts; this covers instrument-filtered queries.
CREATE INDEX IF NOT EXISTS idx_ohlcv_instrument ON ohlcv_bars (instrument_id, timeframe_id, bar_ts DESC);

CREATE TABLE IF NOT EXISTS feature_vectors (
    bar_ts          TIMESTAMPTZ     NOT NULL,
    instrument_id   BIGINT          NOT NULL REFERENCES instruments(instrument_id),
    timeframe_id    SMALLINT        NOT NULL REFERENCES timeframes_metadata(timeframe_id),
    -- Observation vector O_t (f1-f10): all required for IOHMM training; NULL means pipeline bug.
    f1_cvd_vol      DOUBLE PRECISION NOT NULL,
    f2_oi_chg       DOUBLE PRECISION NOT NULL,
    f3_norm_ret     DOUBLE PRECISION NOT NULL,
    f4_liq_sq       DOUBLE PRECISION NOT NULL,
    f5_spread       DOUBLE PRECISION NOT NULL,
    f6_illiq        DOUBLE PRECISION NOT NULL,
    f7_rv_ratio     DOUBLE PRECISION NOT NULL,
    f8_vol_shock    DOUBLE PRECISION NOT NULL,
    f9_flow_liq     DOUBLE PRECISION NOT NULL,
    f10_flow_pos    DOUBLE PRECISION NOT NULL,
    -- Exogenous vector U_t (u1-u4): perp-only fields; NULL for spot instruments is intentional.
    u1_ewma_fr      DOUBLE PRECISION,
    u2_delta_fr     DOUBLE PRECISION,
    u3_fr_spread    DOUBLE PRECISION,
    u4_delta_lsr    DOUBLE PRECISION,
    dim_used        SMALLINT,
    PRIMARY KEY (instrument_id, timeframe_id, bar_ts)
);
SELECT create_hypertable('feature_vectors', 'bar_ts', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_feature_instrument ON feature_vectors (instrument_id, timeframe_id, bar_ts DESC);

CREATE TABLE IF NOT EXISTS state_history (
    bar_ts          TIMESTAMPTZ     NOT NULL,
    instrument_id   BIGINT          NOT NULL REFERENCES instruments(instrument_id),
    timeframe_id    SMALLINT        NOT NULL REFERENCES timeframes_metadata(timeframe_id),
    state           SMALLINT        NOT NULL,
    label           TEXT            NOT NULL,
    confidence      DOUBLE PRECISION,
    posterior       DOUBLE PRECISION[],
    model_ver       TEXT,
    PRIMARY KEY (instrument_id, timeframe_id, bar_ts)
);
SELECT create_hypertable('state_history', 'bar_ts', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_state_instrument ON state_history (instrument_id, timeframe_id, bar_ts DESC);
