# Training Guide

## 1. Environment

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m olm verify-environment --json
```

Python 3.13, CPU-only PyTorch, numpy, pandas, pyarrow, duckdb, FastAPI, uvicorn, pydantic, PyYAML, requests, python-dotenv, pytest. `requirements-optional.txt` adds MLflow, Optuna and psutil but nothing in the core path depends on them.

## 2. Choose a data source

### Synthetic (offline, labeled)

```powershell
.\.venv\Scripts\python.exe -m olm prepare-datasets --source synthetic --synthetic-days 90 --data-root artifacts/synth90 --json
```

This writes `XAUUSD_synthetic_{1min,5min,15min,1h,4h}.parquet` plus a prepared manifest. The source is recorded as `synthetic`; it is not market history.

### Provider data

1. Create `.env` from `.env.example` and add rotated `TWELVE_DATA_API_KEY_1..7`.
2. Probe credentials and the instrument (cheap, no bulk retention):

```powershell
.\.venv\Scripts\python.exe -m olm verify-provider --symbol XAU/USD --interval 5min --json
```

3. Inspect a network-free download plan. `end` is exclusive; the planner subtracts one interval from the provider request so it never stores the still-forming candle:

```powershell
.\.venv\Scripts\python.exe -m olm download-history --dry-run --start 2024-01-01 --end 2026-12-31 --intervals 1min 5min 15min 1h 4h --concurrency 4 --json
```

4. After written retention permission, set `TD_DATA_RETENTION_APPROVED=true` and run without `--dry-run`. The download is resumable; completed windows are tracked in `manifests/ingest_*.json`.
5. Validate and inspect:

```powershell
.\.venv\Scripts\python.exe -m olm validate-data --source twelvedata --json
.\.venv\Scripts\python.exe -m olm inspect-data --source twelvedata --json
```

6. After written training permission, set `TD_MODEL_TRAINING_APPROVED=true`.

Bulk retention and provider-data training are blocked by default. Do not enable a gate without actual permission. In this checkout the backfill is complete and `TD_DATA_RETENTION_APPROVED=true`, but `TD_MODEL_TRAINING_APPROVED=false`; no real-data optimization has been run.

## 3. Prepare tensors

```powershell
.\.venv\Scripts\python.exe -m olm prepare-datasets --source twelvedata --json
```

Outputs split sizes, boundaries, temporal feature names, normalizer statistics, a normalizer content hash and a leakage check. The corrected 90-day synthetic fixture yields **5,857 train / 1,251 validation / 1,252 test** samples (the counts are lower than the pre-fix 5,868/1,253/1,254 because the H1/H4 close-before-target rule rejects a few boundary samples). Exact counts depend on gaps, alignment and history.

The staged real 2024-2026 Twelve Data dataset prepares as **56,709 train / 12,153 validation / 12,154 test**. Its boundaries are frozen at `2026-03-18 01:45Z` (train end) and `2026-07-22 18:45Z` (validation end), with a 60-minute embargo. The split boundary uses the candidate timestamp universe before the exact-child-timestamp filter, so removing malformed windows cannot move the test period. The final test partition is unchanged at 12,154 samples and has not been used for development decisions.

## 4. Train

```powershell
.\.venv\Scripts\python.exe -m olm train --source synthetic --data-root artifacts/synth90 --epochs 12 --batch-size 64 --threads 4 --num-workers 0 --run-name synth90 --json
```

Useful flags:

| Flag | Effect |
| --- | --- |
| `--epochs` | Override configured epochs |
| `--max-steps` | Stop after N optimizer steps (smoke tests) |
| `--batch-size` | Override batch size |
| `--lr` | Override learning rate |
| `--no-context` | M5-only ablation (changes model input width) |
| `--resume PATH` | Resume model, optimizer, scheduler, epoch and best validation |
| `--device cpu` | Explicit device; `auto` resolves to CPU here |
| `--threads N` | PyTorch CPU threads; benchmark before choosing |
| `--num-workers N` | DataLoader processes; 0 is usually best on Windows for this small dataset |
| `--parent-teacher-forcing P` | Probability of feeding the actual parent to the child decoder; 1.0 is full teacher forcing, lower values expose predicted-parent errors |
| `--no-day-balance` | Disable inverse-weekday sampling and use uniform shuffled windows |
| `--skip-validation` | Skip per-frame validation (not recommended for provider data) |

Run outputs:

```
artifacts/runs/<run-name>-<utc>-<id>/run.json
artifacts/runs/<run-name>-<utc>-<id>/metrics.jsonl       # one row per epoch
artifacts/runs/<run-name>-<utc>-<id>/batches.jsonl       # one row per optimizer step
artifacts/runs/<run-name>-<utc>-<id>/status.json         # atomic latest-state snapshot
artifacts/runs/<run-name>-<utc>-<id>/learning_curves.svg # 2x2 training/validation curves
artifacts/checkpoints/<run-name>-<utc>-<id>_best.pt
artifacts/checkpoints/<run-name>-<utc>-<id>_last.pt
```

Every batch row records total/parent/child/aggregate/consistency/KL losses, learning rate, gradient norm, batch size and samples/second. Every epoch row records the same losses plus validation price, calibration and baseline metrics. `status.json` is replaced atomically after each batch, so a monitor never observes a half-written file. The SVG is regenerated at each epoch and never aborts training if rendering fails.

```powershell
.\.venv\Scripts\python.exe -m olm monitor --json
.\.venv\Scripts\python.exe -m olm monitor --run-dir artifacts/runs/<run-name>-<utc>-<id>
```

Validation runs after every epoch; early stopping uses `patience` and `min_delta`. Checkpoints are complete: model, optimizer, scheduler, epoch, global step, config, normalizer, dataset metadata and best validation. The old 12-epoch synthetic checkpoint is invalid because it was trained with H1/H4 lookahead; do not resume it. Start any new run from fresh initialization.

Resume is supported only for a checkpoint that is known-good under the current leakage and child-alignment rules:

```powershell
.\.venv\Scripts\python.exe -m olm train --source synthetic --data-root artifacts/synth90 --resume artifacts/checkpoints/<run>_last.pt --epochs 12 --run-name synth90_resume --json
```

## 5. Evaluate

```powershell
.\.venv\Scripts\python.exe -m olm evaluate ^
  --source synthetic --data-root artifacts/synth90 ^
  --checkpoint artifacts/checkpoints/<run>_best.pt ^
  --split test --samples 64 --folds 4 ^
  --report artifacts/reports/eval_synth90.md --json
```

The evaluator refuses a checkpoint whose normalizer does not match the rebuilt dataset. It reports:

- Parent and child MAE, normalized MAE, body/wick/range error.
- Model-space parent/child NLL (training objective coordinates).
- CRPS and 50/90% empirical coverage from scenario samples.
- Aggregation log error between predicted child means and the actual parent.
- Persistence baselines and training-only AR(1) baselines. Parent AR(1) is one-step. Child AR(1) is a recursive M5 baseline: each predicted candle becomes the next state, so it is deliberately not teacher-forced.
- Free-running three-child metrics, clearly separated from teacher-forced metrics.
- Optional sequential block metrics (`--folds`), with an explicit note that the model is not retrained per block.

Teacher-forced child metrics condition on the actual parent and actual earlier children. They are not scenario quality. Free-running metrics condition only on the predicted parent and start from the last observed M5 close.

## 6. Baseline and experiment hygiene

```powershell
.\.venv\Scripts\python.exe -m olm benchmark --source synthetic --data-root artifacts/synth90 --json
.\.venv\Scripts\python.exe -m olm experiment list --json
.\.venv\Scripts\python.exe -m olm experiment show <run-id> --json
```

`evaluate` reports persistence and AR(1) MAE beside model MAE. AR(1) parameters are fitted only on contiguous training-split pairs and are never learned by the network. `benchmark --no-update` measures forward/backward throughput without an optimizer step; it is a hardware benchmark, not a training run. `--no-context` trains a genuine M5-only ablation, but it must be trained separately; wrapping a context model would be invalid.

Compare runs only on the same split, normalizer, context lengths and temporal configuration. The champion registry requires configured metrics to be present and finite, a minimum test sample count and a minimum improvement over the primary metric.

### Weekend sessions and day balancing

The real provider feed contains full Saturday/Sunday M5 sessions from **2025-04-26 00:00Z** onward; earlier weekends are sparse. Weekend candles are genuinely thinner (mean M5 range 0.7242 versus 3.0862 on weekdays) and are retained as observed, not deleted or filled. Because the chronological split leaves weekends at **8.2% of train** but **about 28.4% of validation/test**, `day_balance: true` samples training weekdays with inverse frequency over the UTC weekday. Missing weekdays get zero weight; validation and test loaders are not resampled. Disable it with `--no-day-balance` for an ablation, and report both results if you compare them.

## 7. Forecast

```powershell
.\.venv\Scripts\python.exe -m olm forecast ^
  --source synthetic --data-root artifacts/synth90 ^
  --checkpoint artifacts/checkpoints/<run>_best.pt ^
  --scenarios 32 --seed 1 ^
  --out artifacts/reports/forecast_synth90.json --json
```

Forecasts are logged to `artifacts/predictions/predictions.jsonl`. The diagnostics include raw and projected aggregation gaps, seed, scenario count and model version. Verify the raw gap before trusting scenarios; a large gap means the projection is doing heavy lifting.

## 8. Troubleshooting

| Symptom | Likely cause / action |
| --- | --- |
| `no market data found` | No staged provider files and no synthetic files; prepare or download first. |
| `authorization gate ... is false` | Retention/training gate not enabled; enable only with permission. |
| `history is too short or not aligned` | Need full M5/M15/H1/H4 context; supply more history or check alignment. |
| `checkpoint normalizer does not match` | Rebuilt data or config differs from training; rebuild with the training config. |
| `temporal feature width ... does not match` | Temporal config at inference differs; reproduce training config. |
| Validation loss rises while training falls | Overfitting; use early stopping, more data, or a small dropout after re-measuring CPU cost. |
| Very slow steps | Check `docs/CPU_OPTIMIZATION.md`; verify dropout is 0 in the CPU profile. |

## 9. What not to infer

A lower training loss does not prove forecast skill. Synthetic metrics show the implementation learns the synthetic generator; they say nothing about gold. Real-data results require licensed data, a clean chronological evaluation and a documented backtest that includes costs.
