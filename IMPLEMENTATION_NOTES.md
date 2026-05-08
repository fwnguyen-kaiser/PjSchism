# SCHISM — Implementation Notes

Maps FinancialFramework v1.4 spec decisions to what is actually built.
Structure per section: **Spec** → **Built** → **Deviations & Assumptions**.

---

## 0. Honest Preamble on Proxy Selection

The MVM framework provides grounding for *which mechanism* a proxy targets,
not *proof* that the proxy captures it sufficiently. Every proxy in Ot and Ut
is a theoretical bet:

- CVD as Mflow proxy: grounded in market microstructure (aggressive order
  imbalance → price pressure), but whether it survives as a discriminating
  feature across K=4 regimes in a specific IOHMM is an empirical question.
- OI as Mpos proxy: widely used, but OI conflates new leverage with rolled
  positions and does not separate direction.
- ILLIQ as Mliq proxy: Amihud (2002) was calibrated on equities; crypto perps
  have structurally different volume profiles. The log-transform variant used
  here reduces but does not eliminate the mismatch.

The framework provides a structured reason to include each proxy, not a
guarantee it is sufficient. Walk-forward BIC and the A2/A3 criteria in §6.2
of the spec are the actual test.

---

## 1. Ingestion Pipeline

### 1.1 CVD — Ot feature 1 (Mflow)

**Spec:** ΔCVDt = Σ qty_i × sign(taker side) over all aggTrades in bar window.

**Built:** Two paths:

| Path | Source | Method |
|------|--------|--------|
| Live WS | `stream_kline_close` | Bar delta proxy: `2 × taker_buy_base − volume` |
| Historical backfill (REST) | `get_klines` pagination | Bar delta proxy (same formula) |
| Historical backfill (REST, premium) | `get_agg_trades` per bar | Full aggTrades sum — not used by default |

**Bar delta proxy derivation:**
```
taker_buy  = taker_buy_base
taker_sell = volume − taker_buy_base
CVD = taker_buy − taker_sell = 2 × taker_buy_base − volume
```
Kline rows include `taker_buy_base` natively; no extra API call needed.

**Why not aggTrades:** Binance aggTrades for a 4h window on an active symbol
can exceed 10,000 records. Fetching them for every bar in a multi-year backfill
would exhaust the weight budget (20 weight/request × bars × symbols). The bar
delta proxy is exact for the net quantity and costs 0 extra weight.

**Known approximation:** The proxy gives the correct *net* CVD but loses
intra-bar trade-size distribution. This matters if the feature engine uses
higher moments of CVD, but the current spec uses only ΔCVDt / Volt.

**WS field fix (historical bug):** Binance WS kline messages use `k.V` for
taker buy base asset volume and `k.Q` for taker buy *quote* volume. The original
`stream_kline_close` used `k["Q"]` — wrong field. Fixed in this codebase:
`taker_buy_base = k["V"]`.

---

### 1.2 Open Interest & LSR — Ot features 2, 4 (Mpos)

**Spec:** ΔOIt/OIt-1 (proportional OI change), liquidation squeeze proxy
requires sign(Rt) and min(0, ΔOIt).

**Built two data sources:**

| Source | Coverage | Method |
|--------|----------|--------|
| Binance REST (`openInterestHist`) | ≤30 days | Native 4h period endpoint |
| binance.vision parquet (VisionCrawler) | Full history | 5-min CSV files, merged to 4h |

**OI/LSR sync gap:** `run_klines()` in BackfillService writes bars to DB with
`oi = NULL` for history beyond 30 days (REST limit). `sync_vision_to_db()` closes
this gap: reads the merged vision parquet, aligns to bar timestamps via
`merge_asof(direction="backward", tolerance=4h)`, and batch-UPDATEs `ohlcv_bars`.

**`merge_asof` direction choice:** Vision OI/LSR records are 5-min granularity.
For a 4h bar at T, the closest 5-min record is at or before T (backward).
Forward or nearest would introduce look-ahead bias (using OI from within the
next bar).

---

### 1.3 Bid-Ask Spread — Ot feature 5 (Mliq)

**Spec:** `b^f_t − b^s_t` — best ask minus best bid at bar close.

**Built:** REST snapshot via `/fapi/v1/ticker/bookTicker` (weight=2) called
at each bar close inside `LiveService._on_bar_close`. Fields stored:
`best_bid`, `best_ask` in `ohlcv_bars`.

**Historical bars: NULL.** Binance provides no historical L1 order book via
REST. All backfilled bars will have `best_bid = NULL`, `best_ask = NULL`.

**Implication for f5:** `feature_vectors.f5_spread` was changed from `NOT NULL`
to nullable (migration 005). The feature engine must decide how to handle NULL
f5 for historical bars — options are imputation from ILLIQ (spec §4.2 note:
"ILLIQ subsumes the volume-shock signal"), or exclusion of f5 from Ot for
historical-only training windows.

**Rate cost:** 2 weight/bar × ~6 bars/day × symbols — negligible vs 2400/min budget.

**Failure mode:** snapshot call is non-fatal; `best_bid`/`best_ask` stay NULL
if the REST call times out at bar close.

---

### 1.4 Funding Rate — Ut components 1, 2 (Mpos) and Cross-Exchange Spread — Ut component 3 (Minfo)

**Spec:**
- U1: EWMA(FRt) — smoothed cost-of-leverage
- U2: ΔFRt — acceleration in positioning pressure
- U3: FR_t^Bnb − FR_t^Bybit — cross-exchange arbitrage pressure

**Built (Binance FR):** `FundingCache` refreshes via `get_funding_rate()` at
each bar close. Stored in `ohlcv_bars.funding_rate`. Feature engine computes
U1 and U2 from this column.

**Built (Bybit FR — U3):**
- `BybitClient`: single public REST endpoint, `/v5/market/funding/history`,
  no auth required. Linear perp symbols match Binance naming (BTCUSDT → BTCUSDT).
- `CrossExchangeFRCache`: same atomic-snapshot pattern as `FundingCache`.
  Stores latest Bybit FR per symbol. Refreshed alongside `OICache` at bar close.
- Stored in `ohlcv_bars.bybit_fr`.
- Backfill: `BackfillService.run_klines()` fetches Bybit FR history and aligns
  using the same sliding-index pattern as Binance FR.

**U3 computation:** deferred to feature engine. Formula: `u3 = funding_rate − bybit_fr`.
Both raw values stored in the bar so the feature engine can compute it without
additional queries.

**Bybit symbol note:** Bybit linear perps use the same ticker as Binance
(BTCUSDT, ETHUSDT). If a symbol does not exist on Bybit, the REST call returns
an error; `CrossExchangeFRCache.refresh()` catches it and leaves the cache
entry as None. The feature engine must handle `bybit_fr = NULL` (drop u3 or
treat as 0).

---

### 1.5 LSR Top — Ut component 4 (Mpos)

**Spec:** Δln(LSR_top_t) — change in top-trader L/S ratio.

**Built:** `lsr_top` stored in `ohlcv_bars`. Feature engine computes
`u4 = ln(lsr_top_t) − ln(lsr_top_{t-1})` from sequential bar reads.
No extra storage needed.

---

## 2. Database Schema

### 2.1 `ohlcv_bars` — raw bar storage

TimescaleDB hypertable. All raw inputs for Ot and Ut are stored here:

| Column | Spec target | Nullable | Notes |
|--------|-------------|----------|-------|
| `cvd` | Ot f1 input | No | Bar delta proxy |
| `oi` | Ot f2,f4 input | Yes | NULL for >30d history before vision sync |
| `lsr_top` | Ut u4 input | Yes | NULL before vision sync |
| `funding_rate` | Ut u1,u2 input | Yes | NULL for very old history |
| `best_bid` | Ot f5 input | Yes | NULL for all historical bars |
| `best_ask` | Ot f5 input | Yes | NULL for all historical bars |
| `bybit_fr` | Ut u3 input | Yes | NULL when BybitClient not configured |
| `volume` | Ot f1,f6,f8 input | No | Base asset volume |
| `close` | Ot f3 input | No | Bar close price |
| `taker_buy_base` | CVD proxy input | No | From kline row |

### 2.2 `feature_vectors` — computed Ot + Ut

Pre-defined columns `f1`–`f10`, `u1`–`u4`. Feature engine writes here after
computing and Z-scoring. `f5_spread` is nullable (see §1.3).

`f1`–`f10` are `NOT NULL` except `f5_spread` — a NULL in any other feature
column indicates a pipeline bug, not a data gap.

### 2.3 `state_history` — IOHMM output

Written by the model engine after inference. Columns: `state`, `label`,
`confidence`, `posterior[]`, `model_ver`. Label is currently stored as a
string; initial labeling convention (ascending RV ratio ordering) is the
spec's identifiability anchor pending semantic research (spec §2 note).

---

## 3. Feature Engine (Not Yet Implemented)

Stub at `schism/data/preprocessing/feature_engine.py`.

**What the data layer provides (all stored in `ohlcv_bars`):**

| Feature | Required raw inputs | Available? |
|---------|---------------------|------------|
| f1: ΔCVDt/Volt | `cvd`, `volume` | Yes |
| f2: ΔOIt/OIt-1 | `oi` (sequential) | Yes (nullable) |
| f3: asinh(Rt/RV24h) | `close`, rolling window | Yes — RV24h must be computed |
| f4: liq squeeze proxy | `oi`, `close` (sequential) | Yes (nullable) |
| f5: best_ask − best_bid | `best_bid`, `best_ask` | Live bars only |
| f6: ILLIQ | `close`, `volume` | Yes |
| f7: RV24h/RV7d | rolling window | Yes — both RVs must be computed |
| f8: log(Volt/EWMA(Volt)) | `volume`, rolling EWMA | Yes — EWMA must be computed |
| f9: (f1) × (f6) raw | `cvd`, `volume`, `close` | Yes |
| f10: (f1) × (f2) raw | `cvd`, `oi`, `volume` | Yes (nullable) |
| u1: EWMA(FRt) | `funding_rate`, rolling | Yes |
| u2: ΔFRt | `funding_rate` (sequential) | Yes |
| u3: FRt^Bnb − FRt^Bybit | `funding_rate`, `bybit_fr` | Yes (bybit_fr nullable) |
| u4: Δln(LSR_top_t) | `lsr_top` (sequential) | Yes (nullable) |

**Key implementation notes for feature engine:**

- Z-scoring order: features 1–8 Z-scored individually; features 9–10 formed
  from raw (un-Z-scored) inputs, then Z-scored (spec §4.2).
- `zscore.py` must be a rolling scaler: fitted on training window, applied
  online — no look-ahead into validation.
- VIF check (`vif_checker.py`): if VIF(f8) ≥ 5, drop f8; ILLIQ (f6) subsumes
  it (spec §4.2 implementation note).
- f3, f7 require RV24h = rolling 24h realized variance over `close` returns.
  RV7d = rolling 7d version. These are bar-level rolling computations, not stored.
- NULL handling: rows with NULL `oi` → f2, f4, f10 are NULL → those rows
  cannot be used for training. Recommend excluding pre-vision-sync bars from
  training window or imputing with 0 for f2/f4 with a flag.

---

## 4. Model Layer (Not Yet Implemented)

All stubs at `schism/model/`. Key implementation decisions not yet made:

**IOHMM base:** spec overrides softmax transition on top of Gaussian HMM.
`hmmlearn.GaussianHMM` is the likely base — custom M-step for βij with L2
regularization (spec §5.3). The covariance trace constraint Tr(Σs) ≤ τ
requires a post-M-step projection step.

**τ bootstrap:** spec defines τ as the 95th percentile of Tr(Σ̂_pooled) on the
initial training window. Must be recomputed at each refit (not carried over).

**Identifiability:** at K=4, Dim=10, need Ttrain ≥ 4000 bars (~667 days at 4h).
Binance perp history starts ~Sep 2019 → ~6.5 years available. Production
training windows should use 180-day minimum (spec §6.1).

**Label alignment:** Hungarian algorithm on cost matrix Cij = ‖µ^old_i − µ^new_j‖²
after each refit. Semantic labels (Accumulation/Distribution/etc.) are
intentionally omitted from this spec version; ordering by ascending RV ratio
is the only anchoring convention (spec §2, v1.4 rev note).

---

## 5. CI Coverage

| Workflow | Trigger | Scope |
|----------|---------|-------|
| `ci-unit.yml` | push/PR to main | All unit tests, no DB |
| `ci-db.yml` | push/PR to main | DB integration tests with TimescaleDB service |
| `ci-ingestion.yml` | push/PR to main | bar_builder, data_store, ingestion_service, ingestion_integrity |

DB tests use `postgresql+asyncpg://...?ssl=disable` — required for asyncpg
on localhost (no TLS handshake).

---

## 6. Known Gaps and Intentional Deferments

| Gap | Impact | Status |
|-----|--------|--------|
| f5 NULL for historical bars | M_liq coverage incomplete for backtest window | Acknowledged, deferred to feature engine |
| No bid-ask WS stream cache | Snapshot timing jitter (~ms) at bar close | Acceptable for 4h bars |
| Bybit FR NULL when client not configured | u3 unavailable | Feature engine must handle |
| Continuous aggregates (H1 rollup) | No native H1 from DB | Not implemented — feature engine can compute |
| OI/LSR NULL before vision sync | f2, f4, f10 unavailable for oldest bars | Documented in §1.2; post-sync resolves for funded window |
| Semantic state labels | State interpretability | Intentionally deferred (spec §2 note) |
