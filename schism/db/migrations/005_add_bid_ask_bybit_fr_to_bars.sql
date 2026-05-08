-- Add bid/ask snapshot columns (from bookTicker stream at bar close) and
-- Bybit funding rate (for cross-exchange FR spread U3 = bnb_fr - bybit_fr).
-- Historical bars will have NULL for all three; live ingestion fills them going forward.

ALTER TABLE ohlcv_bars
    ADD COLUMN IF NOT EXISTS best_bid  DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS best_ask  DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS bybit_fr  DOUBLE PRECISION;

-- f5_spread (bid-ask) was incorrectly defined NOT NULL.
-- Historical bars have no bid/ask data; feature engine must handle NULL.
ALTER TABLE feature_vectors
    ALTER COLUMN f5_spread DROP NOT NULL;
