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

Bulk retention and provider-data training are blocked by default. Do not enable a gate without actual permission.

## 3. Prepare tensors

```powershell
.\.venv\Scripts\python.exe -m olm prepare-datasets --source twelvedata --json
```

Outputs split sizes, boundaries, temporal feature names, normalizer statistics, array hashes and a leakage check. The corrected 90-day synthetic fixture yields **5,857 train / 1,251 validation / 1,252 test** samples (the counts are lower than the pre-fix 5,868/1,253/1,254 because the H1/H4 close-before-target rule rejects a few boundary samples). Exact counts depend on gaps, alignment and history.

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
| `--skip-validation` | Skip per-frame validation (not recommended for provider data) |

Run outputs:

```
artifacts/runs/<run-name>-<utc>-<id>/run.json
artifacts/runs/<run-name>-<utc>-<id>/metrics.jsonl
artifacts/checkpoints/<run-name>-<utc>-<id>_best.pt
artifacts/checkpoints/<run-name>-<utc>-<id>_last.pt
```

Training logs every `log_every` steps and at each epoch end. Validation runs after every epoch; early stopping uses `patience` and `min_delta`. Checkpoints are complete: model, optimizer, scheduler, epoch, global step, config, normalizer, dataset metadata and best validation.

Resume:

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
- Free-running three-child metrics, clearly separated from teacher-forced metrics.
- Optional sequential block metrics (`--folds`), with an explicit note that the model is not retrained per block.

Teacher-forced child metrics condition on the actual parent and actual earlier children. They are not scenario quality.

## 6. Baseline and experiment hygiene

```powershell
.\.venv\Scripts\python.exe -m olm benchmark --source synthetic --data-root artifacts/synth90 --json
.\.venv\Scripts\python.exe -m olm experiment list --json
.\.venv\Scripts\python.exe -m olm experiment show <run-id> --json
```

`evaluate` reports persistence MAE beside model MAE. AR1 exists in `olm/evaluation/baselines.py` but is not yet wired into the report; do not describe the report as a complete baseline study. `--no-context` trains a genuine M5-only ablation, but it must be trained separately; wrapping a context model would be invalid.

Compare runs only on the same split, normalizer, context lengths and temporal configuration. The champion registry requires configured metrics to be present and finite, a minimum test sample count and a minimum improvement over the primary metric.

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
