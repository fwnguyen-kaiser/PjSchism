-- refit_log table (regular, not hypertable — low cardinality).
-- 002_create_refit_log.sql
-- Regular table (không phải hypertable) — low cardinality, vài chục rows

CREATE TABLE IF NOT EXISTS refit_log (
    refit_ts        TIMESTAMPTZ     PRIMARY KEY DEFAULT NOW(),
    symbol          TEXT            NOT NULL,
    trigger         TEXT            NOT NULL,   -- ll_degradation|rv_ratio|backstop
    delta_bic       DOUBLE PRECISION,
    alignment_ok    BOOLEAN,
    drift_alert     BOOLEAN         DEFAULT FALSE,
    dim_used        SMALLINT,                   -- 9 hoặc 10
    model_ver       TEXT,
    cooldown_end_ts TIMESTAMPTZ,
    notes           TEXT                        -- RegimeAlignmentWarning message nếu có
);