# OLM-Gold

OLM-Gold is a CPU-trainable, autoregressive probabilistic foundation model for **XAU/USD** OHLC candles. It learns the joint distribution of a 15-minute parent candle and its exactly three 5-minute children, conditioned on multi-resolution history, the clock and optional event features.

> **Research software, not a trading system.** It does not place orders, size positions, or claim an economic edge. Probabilistic forecasts are research output, not investment advice. The bundled demonstration dataset is **synthetic** and is never presented as market history.

## The hierarchy contract

Every parent is one aligned 15-minute window and exactly three 5-minute children. For every generated or observed parent:

```
parent.open  = child[0].open
parent.high  = max(child.high)
parent.low   = min(child.low)
parent.close = child[2].close
```

Dataset construction rejects any candidate whose M5 children do not reconstruct the M15 parent bit-for-bit (`np.allclose`, 1e-10). Generation samples a parent and then three children; a documented final **coherence projection** clamps children into the sampled parent range and aligns open/close/extremes, so returned scenarios satisfy the contract exactly. The raw (pre-projection) gap is logged as a diagnostic so the projection cannot hide model error.

## What is implemented

- Twelve Data ingestion for `XAU/USD` (`Commodity Aggregate`, UTC) with seven-key discovery, per-account budgets, redacted logging, resumable raw partitions, staged rebuilds and usage reports.
- Four authorization gates (pooling, training, retention, commercial use), all closed by default. A network-free `download-history --dry-run` plan needs no gate and makes no request.
- Invertible candle encoding with a wick softplus, K-component 4-D Gaussian mixtures (Cholesky parameterised), and a Jacobian-corrected candle-space likelihood.
- Causal pre-LN Transformer over M5 history, shared causal context encoders for M15/H1/H4, learned timeframe and sinusoidal position embeddings, 14 clock/session features.
- M15 parent head, sequential M5 child decoder trained with teacher forcing, and three multi-task losses: parent NLL, child NLL, child-aggregate reconstruction and parent/child consistency.
- Chronological train/validation/test splits with an embargo, training-only normalizer, complete checkpoints, resume, early stopping, JSONL metrics.
- Point, probabilistic and calibration metrics (MAE, model-space NLL, CRPS, 50/90% coverage, aggregation error), free-running multi-step evaluation, sequential time-block stability reports, baselines.
- FastAPI service (`/health`, `/v1/model`, `/v1/forecast`, `/v1/actual`, `/v1/predictions/{id}`), append-only forecast log, PSI/KS drift checks, experiment tracker and champion/challenger registry.
- 55 offline tests covering encoding, aggregation, validation, provider parsing, credentials, splits, point-in-time context, model shapes, losses, training, inference, API and monitoring.

## Quick start (Windows, CPU)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

.\.venv\Scripts\python.exe -m olm verify-environment --json
```

Generate a labeled synthetic fixture, prepare tensors, train and evaluate:

```powershell
.\.venv\Scripts\python.exe -m olm prepare-datasets --source synthetic --synthetic-days 90 --data-root artifacts/synth90 --json
.\.venv\Scripts\python.exe -m olm train --source synthetic --data-root artifacts/synth90 --epochs 12 --batch-size 64 --run-name synth90 --json
.\.venv\Scripts\python.exe -m olm evaluate --source synthetic --data-root artifacts/synth90 --checkpoint artifacts/checkpoints/<run>_best.pt --split test --samples 64 --folds 4 --report artifacts/reports/eval_synth90.md --json
.\.venv\Scripts\python.exe -m olm forecast --source synthetic --data-root artifacts/synth90 --checkpoint artifacts/checkpoints/<run>_best.pt --scenarios 32 --seed 1 --out artifacts/reports/forecast_synth90.json
```

`artifacts/synth90` contains generated data. It is not a price history and must not be used to justify a market claim.

## Real data path

1. Copy `.env.example` to `.env` and add **your own rotated** keys as `TWELVE_DATA_API_KEY_1..7`.
2. Never commit `.env`; it is git-ignored. Keys are never printed by the CLI, and log redaction masks provider secrets.
3. Inspect the plan without spending API credits:

```powershell
.\.venv\Scripts\python.exe -m olm download-history --dry-run --start 2024-01-01 --end 2026-12-31 --intervals 1min 5min 15min 1h 4h --concurrency 4 --json
```

The planner splits each interval into <=5,000-record windows (234 requests for 1-minute data over 2024-2026, 47 for M5) and reports the exact count before any network call.

4. Enable `TD_DATA_RETENTION_APPROVED=true` only after written permission to retain the data. Then run the same command without `--dry-run`.
5. Enable `TD_MODEL_TRAINING_APPROVED=true` only after written permission to train on the data. The training command refuses provider-sourced data otherwise.
6. `TD_MULTI_ACCOUNT_POOL_AUTHORIZED` and `TD_COMMERCIAL_USE_APPROVED` remain separate, explicit decisions.

See [docs/DATA_LICENSE_AND_PROVIDER_NOTES.md](docs/DATA_LICENSE_AND_PROVIDER_NOTES.md) before ingesting anything.

## Data preparation and leakage controls

- Splits are chronological: 70% train, 15% validation, 15% test, with a 60-minute embargo. Timestamps never cross splits.
- The normalizer is fit **only** on M5 rows before the training cutoff, then applied to every resolution.
- Samples need full M5/M15/H1/H4 context before the target, so the first usable part of the sample is discarded rather than padded.
- H1/H4 context is selected from the last bar that has **closed at or before the target start**. The earlier implementation selected the bar containing the target, which leaked future prices; `leakage_check` now asserts zero leaking context bars in every split.
- Market closures stay as timestamp gaps. Aggregation never fills missing candles; incomplete buckets are dropped.
- Teacher-forced one-step metrics and free-running three-child scenarios are reported separately; falling training loss alone is not evidence of forecast skill.

## CLI

| Command | Purpose |
| --- | --- |
| `verify-environment` | Python, package, CPU/RAM and device report. |
| `verify-provider` | Masked credential discovery plus a cheap symbol/instrument/sample probe. |
| `inspect-data` / `validate-data` | Schema, gaps, OHLC validity and hierarchy checks. |
| `download-history` | Gated, resumable Twelve Data retrieval; `--dry-run` is network-free. |
| `prepare-datasets` | Build aligned arrays, splits, normalizer and manifest. |
| `train` | Teacher-forced training with checkpoints and early stopping. |
| `evaluate` | Test/validation metrics, free-running quality, optional `--folds` stability blocks. |
| `benchmark` | CPU parameter count and samples/second. |
| `forecast` | Sample parent + three children, log the result, optionally write JSON. |
| `serve` | Run the FastAPI service. |
| `usage` | Per-account request/error/cooldown report. |
| `experiment` | List runs, show a run, promote a challenger under fixed criteria. |

All commands accept repeatable `--config`, `--data-root` and most support `--json`.

## API

```powershell
.\.venv\Scripts\python.exe -m olm serve --checkpoint artifacts/checkpoints/<run>_best.pt --host 127.0.0.1 --port 8000
```

`POST /v1/forecast` takes at least 1000 aligned M5 candles (the default model needs 192 M5 bars plus enough history for 16 H4 bars) and returns scenarios plus a `forecast_id`. `POST /v1/actual` accepts exactly the three realized M5 children and scores the scenario-ensemble mean against their aggregate parent. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for payloads and failure modes.

## Layout

```
olm/
  data/          ingestion, credentials, encoding, validation, aggregation, datasets
  models/        candle/timeframe/temporal encoders, Transformer, mixture heads, decoder
  training/      losses, scheduler, checkpoints, trainer
  evaluation/    metrics, free-running, blocks, baselines, reports
  inference/     forecaster, FastAPI
  monitoring/    prediction log, drift
  experiments/   run tracker, champion registry
configs/         data, model, training, temporal, inference
tests/           offline test suite
docs/            design, research decisions, data dictionary, guides, notes
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Research decisions](docs/RESEARCH_DECISIONS.md)
- [Data dictionary](docs/DATA_DICTIONARY.md)
- [Training guide](docs/TRAINING_GUIDE.md)
- [CPU optimization](docs/CPU_OPTIMIZATION.md)
- [Data license and provider notes](docs/DATA_LICENSE_AND_PROVIDER_NOTES.md)
- [Experiment log](docs/EXPERIMENT_LOG.md)
- [Audit report](docs/AUDIT_REPORT.md)

## Known limitations

- The current demonstration checkpoint is trained on **synthetic** data for one epoch; its metrics show an underfit model, not that it forecasts gold. The earlier 12-epoch synthetic checkpoint is invalid for evaluation because it was trained with H1/H4 lookahead.
- The live 2024-2026 backfill has not been run in this repository: no API keys are present and the retention/training gates remain closed. See [the audit](docs/AUDIT_REPORT.md) for exact status.
- No live brokerage integration, execution, position sizing or P&L claims.
- Twelve Data plans, symbol coverage and rate limits vary by account; verify entitlements before relying on the ingestion path.
- The coherence projection enforces the OHLC hierarchy after sampling. It improves internal consistency but can mask raw parent/child disagreement; the raw gap is always reported.
- CPU training is practical for research-scale runs but not for large-scale pretraining. See the measured throughput in the experiment log.
