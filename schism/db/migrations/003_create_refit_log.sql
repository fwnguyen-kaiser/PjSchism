-- Regular audit table for model refits (non-hypertable).

CREATE TABLE IF NOT EXISTS refit_log (
    -- BIGSERIAL PK: refit_ts cannot be PK because concurrent test runs or burst scenarios
    -- can produce the same microsecond timestamp, causing silent insert failures on an audit table.
    refit_id        BIGSERIAL       PRIMARY KEY,
    refit_ts        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    instrument_id   BIGINT          NOT NULL REFERENCES instruments(instrument_id),
    timeframe_id    SMALLINT        NOT NULL REFERENCES timeframes_metadata(timeframe_id),
    trigger         TEXT            NOT NULL,
    delta_bic       DOUBLE PRECISION,
    alignment_ok    BOOLEAN,
    drift_alert     BOOLEAN         DEFAULT FALSE,
    -- Source of truth for valid dim_used values. If valid dims change, update here and in 004.
    dim_used        SMALLINT        CHECK (dim_used IN (9, 10)),
    model_ver       TEXT,
    cooldown_end_ts TIMESTAMPTZ,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_refit_log_ts ON refit_log (refit_ts DESC);
