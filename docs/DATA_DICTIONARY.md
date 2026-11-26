# Data Dictionary

## Identity

| Field | Value |
| --- | --- |
| Instrument | `XAU/USD` spot gold |
| Provider | Twelve Data |
| Exchange label | `Commodity Aggregate` |
| Timezone | UTC |
| Base interval | `5min` |
| Parent interval | `15min` |
| Context intervals | `1h`, `4h` |
| History intervals ingested | `1min`, `5min`, `15min`, `1h`, `4h` |

Do not substitute another gold instrument or feed. `XAU/USD` on the provider's `Commodity Aggregate` venue is the only identity this project validates. If an account or plan serves a different venue, the ingestion probe records the returned `meta.exchange`; a mismatch must be resolved before training.

## Canonical OHLC frame

Every raw, staged and derived frame uses the same canonical columns:

| Column | Type | Nullable | Meaning |
| --- | --- | --- | --- |
| `timestamp` | UTC datetime | no | **Bar open time**, not close time |
| `open` | float | no | First traded price in the bar |
| `high` | float | no | Highest price in the bar |
| `low` | float | no | Lowest price in the bar |
| `close` | float | no | Last traded price in the bar |
| `symbol` | string | raw/staged only | `XAU/USD` |
| `interval` | string | raw/staged only | `5min`, `15min`, `1h`, `4h` |
| `source` | string | raw/staged only | Provider/venue label, or `synthetic` |

Invariants for every row: prices finite and strictly positive; `high >= max(open, close)`; `low <= min(open, close)`; timestamps unique and sorted ascending; spacing equals the interval except for genuine market closures.

## Raw and staged layout

```
<data-root>/
  raw/
    XAUUSD/5min/20250101T0000_20250201T0000.parquet   # immutable request windows
    XAUUSD/15min/...
    XAUUSD_synthetic_5min.parquet                    # only when source=synthetic
  staged/
    XAUUSD_5min.parquet                              # deduplicated union of raw partitions
  manifests/
    ingest_XAU_USD_5min.json                         # completed windows + provider metadata
    prepared_XAU_USD.json                            # tensor build + hashes
```

Raw partitions are written atomically and never edited. `rebuild_staged` concatenates partitions, sorts by timestamp, keeps the last duplicate and writes atomically. The prepared manifest records source, row counts, split sizes, temporal width, normalizer statistics and array hashes.

Synthetic fixtures use `XAUUSD_synthetic_*` paths and `source='synthetic'`; they can never be confused with `XAUUSD_*` provider files.

## Encoded coordinates

| Coordinate | Definition | Model-space transform |
| --- | --- | --- |
| `g` | `ln(open / previous_close)` | standardized |
| `b` | `ln(close / open)` | standardized |
| `a_u` | inverse-softplus of `ln(high / max(open, close))` | standardized |
| `a_d` | inverse-softplus of `ln(min(open, close) / low)` | standardized |

`previous_close` is the close of the immediately preceding bar at the same resolution, never the previous target. For the first row of a frame there is no predecessor, so it is excluded during preparation.

## Temporal features (14 dimensions)

With the default `configs/temporal.yaml` (`include_sessions=true`, `include_events=false`):

| Index | Feature |
| --- | --- |
| 0 | `hour_sin` |
| 1 | `hour_cos` |
| 2 | `dow_sin` |
| 3 | `dow_cos` |
| 4 | `session_asia` |
| 5 | `session_london` |
| 6 | `session_new_york` |
| 7 | `session_overlap` |
| 8 | `asia_since_open` |
| 9 | `asia_until_close` |
| 10 | `london_since_open` |
| 11 | `london_until_close` |
| 12 | `new_york_since_open` |
| 13 | `new_york_until_close` |

Session clocks use `zoneinfo` (Tokyo, London, New York), so DST is handled by the IANA database. Event features add `2 * len(CATEGORIES)` columns when `include_events=true`; the checkpoint records the temporal width and inference refuses a mismatch.

## Sample indexing

One sample is one target M15 parent:

| Key | Meaning |
| --- | --- |
| `i` | index of the last observed M5 bar before the target children |
| `p15` | index of the target M15 parent |
| `j1` / `j4` | last completed H1 / H4 indices at `i` |
| `target_ts` | timestamp of the first target M5 child |
| `parent_prices` | actual M15 OHLC |
| `child_prices` | actual `(3,4)` M5 OHLC |
| `parent_enc` | normalized model coordinates `(4,)` |
| `children_enc` | normalized model coordinates `(3,4)` |
| `m15_prev_close` | previous M15 close used to encode/decode the parent |
| `children_prev_close` | `(3,)` previous closes: parent open, C1 close, C2 close |

Context windows end before `target_ts`. The M15 context stops at `p15-1`; H1/H4 use the last bar whose close (`open + interval`) is `<= target_ts`, so no context bar contains future target prices.

## Split manifest

`prepared_XAU_USD.json` contains:

```json
{
  "source": "synthetic",
  "provider": "synthetic-fixture",
  "licensed": false,
  "rows": {"1min": 129600, "5min": 25920, "15min": 8640, "1h": 2160, "4h": 540},
  "splits": {"train": 0, "val": 0, "test": 0},
  "boundaries": {"train_end": "...", "val_end": "...", "embargo_minutes": 60},
  "normalizer": {"mean": [0, 0, 0, 0], "std": [1, 1, 1, 1]},
  "hashes": {"normalizer": "..."}
}
```

Exact counts depend on history length and rejected samples. The manifest is versioned by array hashes, not by wall-clock time alone.

## Forecast log

`artifacts/predictions/predictions.jsonl` is append-only. Record types:

- `forecast`: `forecast_id`, `as_of`, `model_version`, full forecast payload, UTC timestamp, content hash.
- `actual`: the three realized M5 children.
- `evaluation`: scenario-ensemble mean MAE per coordinate, scenario count, and an explicit note that it is a point score.

API keys and provider secrets are redacted before any log write. Forecast ids are 24 hex characters; the redactor intentionally masks 32-hex strings, so a 32-hex id would be corrupted.
