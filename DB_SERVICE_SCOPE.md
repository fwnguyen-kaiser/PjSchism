# DB Service — Scope Disclosure

## Những gì đã được build (thực sự có trong code)

### Migrations (4 file, chạy theo thứ tự)

| File | Nội dung |
|------|----------|
| `001_bootstrap_metadata.sql` | Tạo `instruments`, `timeframes_metadata`. Seed sẵn: Binance BTCUSDT perp+spot, 4 timeframe (15m/1h/4h/1d). |
| `002_create_timeseries_tables.sql` | Tạo 3 hypertable: `ohlcv_bars`, `feature_vectors`, `state_history`. Thêm secondary index `(instrument_id, timeframe_id, bar_ts DESC)` cho mỗi table. |
| `003_create_refit_log.sql` | Tạo `refit_log` — audit table thường (không phải hypertable), PK là `BIGSERIAL`. Index trên `refit_ts DESC`. |
| `004_migrate_legacy_symbol_keys.sql` | Path migrate data cũ từ `symbol TEXT` sang `(instrument_id, timeframe_id)`. Chỉ chạy khi detect column `symbol` tồn tại — idempotent. |

### Schema thực tế

**`ohlcv_bars`** — hypertable, partition theo `bar_ts`
```
PRIMARY KEY (instrument_id, timeframe_id, bar_ts)
Columns: open, high, low, close, volume, cvd, oi, lsr_top,
         funding_rate, num_trades, taker_buy_base, quote_volume,
         ingested_at, source
```

**`feature_vectors`** — hypertable
```
PRIMARY KEY (instrument_id, timeframe_id, bar_ts)
f1-f10: DOUBLE PRECISION NOT NULL  (observation vector O_t cho IOHMM)
u1-u4:  DOUBLE PRECISION nullable  (exogenous U_t, perp-only)
dim_used: SMALLINT
```

**`state_history`** — hypertable
```
PRIMARY KEY (instrument_id, timeframe_id, bar_ts)
Columns: state, label, confidence, posterior DOUBLE PRECISION[], model_ver
```

**`refit_log`** — plain table
```
PK: refit_id BIGSERIAL
Columns: refit_ts, instrument_id, timeframe_id, trigger, delta_bic,
         alignment_ok, drift_alert, dim_used CHECK(9,10), model_ver,
         cooldown_end_ts, notes
```

### Repositories

| File | Chức năng |
|------|-----------|
| `bar_repo.py` | `BarRepository.upsert_bars(bars)` — upsert batch bars vào `ohlcv_bars`. Cache instrument_id/timeframe_id trong memory. |
| `state_repo.py` | `get_current`, `get_history`, `get_stats` — query `state_history`. Stats tính frequency% + mean sojourn trong Python. |
| `refit_repo.py` | `get_log`, `insert` — query và insert `refit_log`. INSERT không đưa `refit_id` vào column list, chỉ dùng `RETURNING`. |

### Docker / init
- `init.sql`: `CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE`
- `run_migrations.sh`: mounted as `01_run_migrations.sh`, iterate và `psql -f` từng file `.sql` trong thư mục `migrations/`
- `docker-entrypoint-initdb.d` không execute file trong subdirectory — `run_migrations.sh` là workaround cần thiết

---

## Trả lời thẳng: mình có dùng trick để pass test không?

**Có một phần, không có phần khác.**

### Không trick (test thật)
- **Unit tests cho repositories** (`TestStateRepoContract`, `TestRefitRepoContract`): check trực tiếp nội dung SQL string — nếu field thiếu trong SQL thì fail thật. Không mock SQL.
- **Integration tests** (`TestDbSchemaIntegrity`): query thẳng `information_schema` và `pg_indexes` trên DB thật. Chạy được là DB schema thật đúng.
- **`TestBarContract`**: gọi `Bar.to_dict()` và so sánh key set với column list trong SQL — contract thật.

### Có dùng mock (và đây là trick hợp lệ trong unit test)
- **`test_api.py`**: toàn bộ test dùng `AsyncMock` cho session và `MagicMock` cho DB row. Nghĩa là các test này verify **routing logic và HTTP contract**, không verify query SQL thật.
- Cụ thể: `_mock_result()` trả về mock object, các test như `test_current_returns_snapshot` không thực sự chạy SQL — chúng test rằng khi repository trả về data thì endpoint serialize đúng sang JSON với đúng status code.
- Đây là mock **có chủ đích** (unit test API layer, tách khỏi DB), nhưng mình nói thẳng để bạn biết: nếu SQL query trong repo bị sai, unit test API sẽ không phát hiện. Integration tests mới cover điều đó.

### Tóm lại
```
test_api.py              → mock DB hoàn toàn, test HTTP contract
test_ingestion_integrity → unit: test SQL string content, integration: test live DB schema
```
Không có test nào được viết để "làm cho pass" mà bỏ qua logic thật.

---

## Vấn đề bạn đang nhắc: Continuous Aggregates

Bạn đúng — **TimescaleDB Continuous Aggregates** là tính năng liên quan trực tiếp đến use case của project.

### Schema hiện tại làm gì

Ingestion hiện tại ghi thẳng bars theo `timeframe_id` vào `ohlcv_bars`. Nghĩa là:
- H4 bars từ Binance → ghi với `timeframe_id = 3`
- H1 bars từ Binance → ghi với `timeframe_id = 2`
- Đây là **raw bars từ exchange**, không phải được tính toán từ bars nhỏ hơn

### Continuous Aggregates giải quyết gì

TimescaleDB Continuous Aggregate (`CREATE MATERIALIZED VIEW ... WITH (timescaledb.continuous)`) cho phép:
- Lưu 1 timeframe gốc (ví dụ: **1m** hoặc **5m** raw tick) vào `ohlcv_bars`
- Tự động **roll up** thành H1, H4, 1D mà không cần code Python
- Incremental refresh: chỉ recompute chunk mới, không full scan
- `time_bucket('4 hours', bar_ts)` → OHLCV aggregation tự động

### Tình trạng hiện tại của schema

Schema hiện tại **không dùng continuous aggregates**. Các vấn đề cụ thể:

1. `ohlcv_bars` lưu H4 trực tiếp từ Binance REST/WS — đúng nhưng **bỏ qua khả năng rebuild nến**. Nếu có dữ liệu 1m thì H4 có thể build từ đó.

2. Không có `source_timeframe` concept: không biết H4 bar này là từ Binance trực tiếp hay aggregate từ 1m.

3. **Không có migration nào tạo continuous aggregate view** — đây là gap thật.

### Câu hỏi cần bạn xác nhận

Trước khi code, mình cần biết:

**A. Timeframe gốc là gì?**
- Option 1: Ingest thẳng H4 (và H1 riêng) từ Binance kline — đơn giản nhất, đang làm
- Option 2: Ingest 1m hoặc 5m raw, dùng continuous aggregate build H1/H4 — phức tạp hơn nhưng cho phép rebuild và consistency

**B. "Vài phần H1" nghĩa là gì?**
- H1 phục vụ mục đích gì trong pipeline? Intra-day context cho IOHMM? Hay chỉ để backfill display?

**C. Continuous aggregate apply ở đâu?**
- Nếu ingest 1m: CA view cho H1 và H4 (2 view)
- Nếu ingest H1: CA view chỉ cho H4 (1 view)
- `ohlcv_bars` giữ nguyên là base table hay split thành `ohlcv_raw_1m` + CA views?

---

**Mình không code gì thêm cho đến khi bạn xác nhận approach.**
