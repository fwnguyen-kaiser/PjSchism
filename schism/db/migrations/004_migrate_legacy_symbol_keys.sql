-- Legacy upgrade path: migrate symbol-based tables to metadata identity.
-- Legacy assumption: existing symbol rows are Binance perpetual 4h data.
-- If any spot data existed under the same symbol it will be mislabeled as 'perp' after this migration.

-- Removed redundant seed of timeframe_id=3 here; 001_bootstrap_metadata.sql is the source of truth.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'ohlcv_bars' AND column_name = 'symbol'
    ) THEN
        ALTER TABLE ohlcv_bars ADD COLUMN IF NOT EXISTS instrument_id BIGINT;
        ALTER TABLE ohlcv_bars ADD COLUMN IF NOT EXISTS timeframe_id SMALLINT;
        ALTER TABLE ohlcv_bars ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
        ALTER TABLE ohlcv_bars ADD COLUMN IF NOT EXISTS source TEXT;

        INSERT INTO instruments (exchange, symbol, market_type)
        SELECT DISTINCT 'binance', symbol, 'perp'
        FROM ohlcv_bars
        WHERE symbol IS NOT NULL
        ON CONFLICT (exchange, symbol, market_type) DO NOTHING;

        UPDATE ohlcv_bars o
        SET instrument_id = i.instrument_id, timeframe_id = 3
        FROM instruments i
        WHERE i.exchange = 'binance'
          AND i.market_type = 'perp'
          AND i.symbol = o.symbol
          AND (o.instrument_id IS NULL OR o.timeframe_id IS NULL);

        ALTER TABLE ohlcv_bars ALTER COLUMN instrument_id SET NOT NULL;
        ALTER TABLE ohlcv_bars ALTER COLUMN timeframe_id SET NOT NULL;
        ALTER TABLE ohlcv_bars DROP CONSTRAINT IF EXISTS ohlcv_bars_pkey;
        ALTER TABLE ohlcv_bars ADD CONSTRAINT ohlcv_bars_pkey PRIMARY KEY (instrument_id, timeframe_id, bar_ts);
        ALTER TABLE ohlcv_bars DROP CONSTRAINT IF EXISTS ohlcv_bars_instrument_id_fkey;
        ALTER TABLE ohlcv_bars ADD CONSTRAINT ohlcv_bars_instrument_id_fkey FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id);
        ALTER TABLE ohlcv_bars DROP CONSTRAINT IF EXISTS ohlcv_bars_timeframe_id_fkey;
        ALTER TABLE ohlcv_bars ADD CONSTRAINT ohlcv_bars_timeframe_id_fkey FOREIGN KEY (timeframe_id) REFERENCES timeframes_metadata(timeframe_id);
        ALTER TABLE ohlcv_bars DROP COLUMN IF EXISTS symbol;

        CREATE INDEX IF NOT EXISTS idx_ohlcv_instrument ON ohlcv_bars (instrument_id, timeframe_id, bar_ts DESC);
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'feature_vectors' AND column_name = 'symbol'
    ) THEN
        ALTER TABLE feature_vectors ADD COLUMN IF NOT EXISTS instrument_id BIGINT;
        ALTER TABLE feature_vectors ADD COLUMN IF NOT EXISTS timeframe_id SMALLINT;

        INSERT INTO instruments (exchange, symbol, market_type)
        SELECT DISTINCT 'binance', symbol, 'perp'
        FROM feature_vectors
        WHERE symbol IS NOT NULL
        ON CONFLICT (exchange, symbol, market_type) DO NOTHING;

        UPDATE feature_vectors fv
        SET instrument_id = i.instrument_id, timeframe_id = 3
        FROM instruments i
        WHERE i.exchange = 'binance'
          AND i.market_type = 'perp'
          AND i.symbol = fv.symbol
          AND (fv.instrument_id IS NULL OR fv.timeframe_id IS NULL);

        ALTER TABLE feature_vectors ALTER COLUMN instrument_id SET NOT NULL;
        ALTER TABLE feature_vectors ALTER COLUMN timeframe_id SET NOT NULL;
        ALTER TABLE feature_vectors DROP CONSTRAINT IF EXISTS feature_vectors_pkey;
        ALTER TABLE feature_vectors ADD CONSTRAINT feature_vectors_pkey PRIMARY KEY (instrument_id, timeframe_id, bar_ts);
        ALTER TABLE feature_vectors DROP CONSTRAINT IF EXISTS feature_vectors_instrument_id_fkey;
        ALTER TABLE feature_vectors ADD CONSTRAINT feature_vectors_instrument_id_fkey FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id);
        ALTER TABLE feature_vectors DROP CONSTRAINT IF EXISTS feature_vectors_timeframe_id_fkey;
        ALTER TABLE feature_vectors ADD CONSTRAINT feature_vectors_timeframe_id_fkey FOREIGN KEY (timeframe_id) REFERENCES timeframes_metadata(timeframe_id);
        ALTER TABLE feature_vectors DROP COLUMN IF EXISTS symbol;

        CREATE INDEX IF NOT EXISTS idx_feature_instrument ON feature_vectors (instrument_id, timeframe_id, bar_ts DESC);
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'state_history' AND column_name = 'symbol'
    ) THEN
        ALTER TABLE state_history ADD COLUMN IF NOT EXISTS instrument_id BIGINT;
        ALTER TABLE state_history ADD COLUMN IF NOT EXISTS timeframe_id SMALLINT;

        INSERT INTO instruments (exchange, symbol, market_type)
        SELECT DISTINCT 'binance', symbol, 'perp'
        FROM state_history
        WHERE symbol IS NOT NULL
        ON CONFLICT (exchange, symbol, market_type) DO NOTHING;

        UPDATE state_history sh
        SET instrument_id = i.instrument_id, timeframe_id = 3
        FROM instruments i
        WHERE i.exchange = 'binance'
          AND i.market_type = 'perp'
          AND i.symbol = sh.symbol
          AND (sh.instrument_id IS NULL OR sh.timeframe_id IS NULL);

        ALTER TABLE state_history ALTER COLUMN instrument_id SET NOT NULL;
        ALTER TABLE state_history ALTER COLUMN timeframe_id SET NOT NULL;
        ALTER TABLE state_history DROP CONSTRAINT IF EXISTS state_history_pkey;
        ALTER TABLE state_history ADD CONSTRAINT state_history_pkey PRIMARY KEY (instrument_id, timeframe_id, bar_ts);
        ALTER TABLE state_history DROP CONSTRAINT IF EXISTS state_history_instrument_id_fkey;
        ALTER TABLE state_history ADD CONSTRAINT state_history_instrument_id_fkey FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id);
        ALTER TABLE state_history DROP CONSTRAINT IF EXISTS state_history_timeframe_id_fkey;
        ALTER TABLE state_history ADD CONSTRAINT state_history_timeframe_id_fkey FOREIGN KEY (timeframe_id) REFERENCES timeframes_metadata(timeframe_id);
        ALTER TABLE state_history DROP COLUMN IF EXISTS symbol;

        CREATE INDEX IF NOT EXISTS idx_state_instrument ON state_history (instrument_id, timeframe_id, bar_ts DESC);
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'refit_log' AND column_name = 'symbol'
    ) THEN
        ALTER TABLE refit_log ADD COLUMN IF NOT EXISTS instrument_id BIGINT;
        ALTER TABLE refit_log ADD COLUMN IF NOT EXISTS timeframe_id SMALLINT;

        INSERT INTO instruments (exchange, symbol, market_type)
        SELECT DISTINCT 'binance', symbol, 'perp'
        FROM refit_log
        WHERE symbol IS NOT NULL
        ON CONFLICT (exchange, symbol, market_type) DO NOTHING;

        UPDATE refit_log rl
        SET instrument_id = i.instrument_id, timeframe_id = 3
        FROM instruments i
        WHERE i.exchange = 'binance'
          AND i.market_type = 'perp'
          AND i.symbol = rl.symbol
          AND (rl.instrument_id IS NULL OR rl.timeframe_id IS NULL);

        ALTER TABLE refit_log ALTER COLUMN instrument_id SET NOT NULL;
        ALTER TABLE refit_log ALTER COLUMN timeframe_id SET NOT NULL;
        ALTER TABLE refit_log DROP CONSTRAINT IF EXISTS refit_log_instrument_id_fkey;
        ALTER TABLE refit_log ADD CONSTRAINT refit_log_instrument_id_fkey FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id);
        ALTER TABLE refit_log DROP CONSTRAINT IF EXISTS refit_log_timeframe_id_fkey;
        ALTER TABLE refit_log ADD CONSTRAINT refit_log_timeframe_id_fkey FOREIGN KEY (timeframe_id) REFERENCES timeframes_metadata(timeframe_id);
        ALTER TABLE refit_log DROP COLUMN IF EXISTS symbol;
        -- Re-applying constraint defined in 003. If valid dims change, update 003 first.
        ALTER TABLE refit_log DROP CONSTRAINT IF EXISTS refit_log_dim_used_check;
        ALTER TABLE refit_log ADD CONSTRAINT refit_log_dim_used_check CHECK (dim_used IN (9, 10));
    END IF;
END $$;
