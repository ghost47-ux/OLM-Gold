# Independent audit report

Date: 2026-11-26 (updated after the real backfill and dataset audit). Scope: `C:\Users\Monarch Group\Contacts\OLM-Gold`. The audit was run on CPU against the code in this repository, not against marketing claims.

## Verdict in one line

**YES**, a genuine, independently trainable PyTorch network remains when the data, API and dashboard layers are removed. **NO**, the repository is not yet a validated forecaster: the original 12-epoch checkpoint is invalid because of H1/H4 lookahead, and the corrected one-epoch smoke model is much worse than persistence.

## Status table

| # | Audit question | Status after remediation |
| --- | --- | --- |
| 1 | Is there a real trainable PyTorch model? | **PASS** |
| 2 | Does backpropagation and the optimizer actually change parameters? | **PASS** |
| 3 | Are forecasts genuinely probabilistic and calibrated? | **PARTIAL** (real mixtures; calibration fails) |
| 4 | Is the M15 -> exactly 3xM5 hierarchy enforced? | **PASS** |
| 5 | Is future information excluded from every context? | **PASS** after fixing a critical leak |
| 6 | Is ingestion parallel, resumable, account-aware and gated? | **PASS** for the completed backfill; credentials still need rotation |
| 7 | Are splits chronological and checkpoints portable/resumable? | **PASS** |
| 8 | Are baseline and walk-forward comparisons implemented? | **PARTIAL**: persistence and AR(1) wired; no retrained walk-forward |
| 9 | Do inference, scenarios, API and projection work? | **PASS** mechanically; quality **FAIL** |
| 10 | Is it ready to start a real-data training run? | **PARTIAL** (data staged; training authorization still false) |

## Evidence by question

### 1. Genuine model - PASS

`olm/models/olm.py` defines `OLMModel`, `CandleEncoder`, `TemporalEncoder`, `TimeframeEncoder`, `CausalTransformer`, `ParentHead`, `ChildDecoder` and an optional X-Factor latent. `olm/models/transformer.py` contains a real causal masked pre-LN transformer block. The default model has **1,493,752 parameters** and 136 state tensors. `tests/test_model.py` checks the forward path, the causal mask, the mixture density, the candle Jacobian, gradient flow and teacher-forcing dependency.

The architecture does not call an external model or an LLM. Removing `olm/data/providers`, `olm/inference/api.py` and `olm/monitoring` leaves a network that accepts tensors and trains from `olm/training/trainer.py`.

### 2. Optimization - PASS

`olm/training/trainer.py` calls `zero_grad`, `backward` and `optimizer.step` on every batch. On a fixed batch, 15 AdamW steps reduced total loss from **13.1657** to **2.3794**; the maximum parameter delta was **0.1471**; **76/76** trainable gradient tensors were non-zero. This proves the graph is connected and trainable. It does not prove generalization.

A 12-step CLI run on `artifacts/audit_smoke` produced a checkpoint and trained from loss 15.1735 to validation 10.6205; resuming at epoch 1/step 12 completed epoch 2/step 16 with validation 9.5853.

### 3. Probabilistic outputs - PARTIAL

`olm/models/probability_head.py` implements a K=4, 4-D Gaussian mixture with Cholesky `scale_tril`, softplus scales, `logsumexp` mixture NLL, `sample`, `sample_many`, `mean`, `dominant_mean` and `log_prob`. The candle likelihood includes the wick softplus Jacobian (`log p(u) = log p(a) - log sigmoid(a)`). This is a real density model, not a point regressor with fake error bars.

Calibration is not acceptable yet. The corrected one-epoch model reports parent NLL 2.5963, child NLL 2.6413, CRPS 34.7597, coverage@50 0.9073 and coverage@90 **1.0000** on the synthetic test split. Intervals are over-wide and the model is underfit. The original 12-epoch coverage was 0.7923/0.9926, but those numbers are invalid for the leakage reason below.

### 4. Hierarchy - PASS

`olm/data/datasets.py::_candidate_samples` accepts an M15 bucket only when its three M5 children have exactly the timestamps `target + 0/5/10 minutes` and reconstruct the parent under `np.allclose(..., atol=1e-10)`. `olm/models/decoding.py::aggregate_children_torch` recomputes the parent. `project_children_to_parent` is the documented final coherence projection. In an inference smoke run all children were valid and the projected maximum gap was **0.0**; the raw gap (0.006217 in that run) is logged separately so projection cannot hide disagreement.

Generation samples a parent first, then exactly three children; `tests/test_forecast.py` asserts the four OHLC identities for every scenario.

### 5. Leakage - PASS after a critical fix

**Initial finding (FAIL):** H1 and H4 context was chosen by bar-open timestamp. For the example target `2024-03-17T22:25:00Z`, the selected H1 bar was 22:00-23:00 and the H4 bar was 20:00-00:00, both closing after the target started. **100%** of train, validation and test samples leaked this way. All metrics from that checkpoint are invalid.

**Fix:** `_completed_index` in `olm/data/datasets.py` selects the last bar whose open plus interval is `<= target_start`; inference uses the same rule in `Forecaster._latest_context`. `leakage_check` now returns per-resolution counts and asserts zero leaking context bars. The rebuilt 90-day synthetic dataset has 5,857/1,251/1,252 samples; the real prepared dataset has 56,709/12,153/12,154. Both report `ok: true` and `leaking: 0` for H1 and H4. `tests/test_datasets.py` and `tests/test_forecast.py` cover both training and inference.

Other controls: chronological 70/15/15 splits, 60-minute embargo, normalizer fit only on pre-cutoff training M5 rows, and no random split. H1/H4 bars are context only; they never enter the target window.

### 6. Ingestion - PASS for the completed backfill

`olm/data/providers/twelvedata.py` implements retries, redaction and thread-local `requests.Session` objects. `olm/data/ingestion.py` implements per-account scheduling, resumable partitions, staged rebuilds and `download-history`. `olm/data/credentials.py` enforces the four gates.

The real 2024-2026 backfill has now run: **439 windows**, **0 failures**, distributed 88/87/87/89/88 across five provider accounts. Raw rows: 1,213,129 1-minute, 245,258 5-minute, 82,110 15-minute, 20,594 1-hour and 5,321 provider 4-hour candles. The planner uses <=5,000-record windows and treats `end` as exclusive. Retention and multi-account pooling are authorized; `TD_MODEL_TRAINING_APPROVED` remains false. The provider-native 4-hour series failed UTC alignment, so the staged 4-hour series is derived from the 1-hour series (4,853 complete bars); raw partitions are untouched.

### 7. Splits, checkpoints and resume - PASS

`olm/training/checkpointing.py` stores model, optimizer, scheduler, epoch, global step, config, normalizer, dataset metadata and best validation. Reloading a saved model produced `max_abs_logit_diff_reloaded: 0.0` and child loc difference `0.0`. A real resume run continued from epoch 1/step 12 to epoch 2/step 16 without resetting weights. `tests/test_training.py` covers save/load and resume.

Caveat: portability was tested in the same Python/PyTorch environment, not across PyTorch major versions or operating systems.

### 8. Baselines and walk-forward - PARTIAL

Persistence is computed inside `evaluate_predictions` (`parent_mae_persistence`, `child_mae_persistence`). AR(1) is now wired: parameters are fitted only on contiguous training-split pairs; parent AR(1) is one-step and child AR(1) is a recursive M5 baseline (`child_mae_ar1_recursive`). `build_m5_only_model` builds a genuine no-context ablation, but no separately trained M5-only checkpoint exists.

`olm/evaluation/blocks.py::evaluate_sequential_blocks` evaluates one fixed model on sequential time blocks. It is explicitly labelled as **not** a retrained walk-forward. A true walk-forward that refits the model per fold is still missing.

### 9. Inference, scenarios and API - PASS mechanically, FAIL on quality

`olm/inference/forecast.py` samples a parent and three children, keeps previous-child closes, projects to the parent and logs raw/projected gaps. On a 16-sample smoke run it built 32 scenarios in 0.206 s; all child OHLC checks passed and the projected gap was 0.0. The FastAPI service exposes `/health`, `/v1/model`, `/v1/forecast`, `/v1/actual` and `/v1/predictions/{id}`; forecast IDs are shortened to 24 hex. `tests/test_api.py` covers the round trip.

Quality fails. On the corrected synthetic test set the model parent MAE is **20.7253** versus persistence **0.7034**; child MAE is **96.3164** versus persistence **0.4204**. The predictive mixture includes low-weight far components, so both the mixture mean and sampled tails are too wide. This is a model/data/training problem, not hidden by the coherence projection. The API is also unauthenticated and binds localhost by default; do not expose it publicly as-is.

### 10. Real-data readiness - PARTIAL

The data path is now exercised: gated `download-history`, staged rebuild, `prepare-datasets --source twelvedata`, exact child alignment, one shared model, overlapping windows and day-balanced batches. The real prepared dataset is 56,709/12,153/12,154 with an unchanged test partition. What is missing is a real-data training run (gate false), rotated credentials, a retrained walk-forward and a champion comparison against persistence.

## Prioritized remediation

**P0 - before any real training**

1. Rotate every Twelve Data key pasted into chat. Put only rotated keys in `.env`; never print or commit them.
2. Keep the retained data under the existing retention permission. Obtain written **model-training** authorization, then set `TD_MODEL_TRAINING_APPROVED=true` for that run only. Keep pooling/commercial gates separate.
3. The 2024-2026 backfill is complete and validated; retain the frozen manifest with SHA-256. Do not silently rewrite it.
4. Retrain from scratch. Do not resume the leaky checkpoint and do not promote it.
5. Evaluate on the untouched chronological test split against persistence and AR1. Require the model to beat persistence on MAE **and** show usable calibration before any promotion.

**P1 - before calling it a foundation model**

6. Implement retrained walk-forward folds; the current sequential-block report is not a walk-forward.
7. Train a separate M5-only Transformer and compare it on the same frozen splits; AR(1) is already wired.
8. Diagnose mixture collapse/over-wide scales (component weights, scale floors, temperature/calibration, likelihood weighting) and tune the scheduled predicted-parent probability; the hook exists as `parent_teacher_forcing_prob`.
9. Add a real provider integration test with recorded fixtures and a concurrency test that proves per-account request budgets.

**P2 - engineering**

10. Remove temporary audit scripts, untrack generated `artifacts/`, and add API authentication/rate limiting before any non-local deployment.

## What was actually changed during remediation

- Point-in-time H1/H4 selection in training and inference, plus leakage assertions and tests.
- Gap-aware resolution aggregation that drops incomplete windows instead of fabricating candles.
- Thread-local HTTP sessions and a configurable concurrency cap.
- Exclusive `end` boundary handling in download windows.
- Dominant-component diagnostics while retaining the mathematically correct mixture mean as `expected`.
- Persistence and training-only AR(1) metrics, plus a genuine M5-only model builder.
- Scheduled parent sampling for child training (`parent_teacher_forcing_prob`), defaulting to full teacher forcing.
- Batch-level and epoch-level JSONL logs, atomic `status.json`, SVG learning curves and a `monitor` command.
- Exact child-timestamp validation with a frozen split-boundary universe, and inverse-weekday training sampling.
- `--threads`, `--num-workers`, `--parent-teacher-forcing` and `--no-day-balance` CLI options.

## Final answer to the central question

If the data, API and dashboard code are removed, **a genuine independently trainable PyTorch network remains**. It has a real causal transformer, real mixture densities, a parent head, a sequential child decoder, multi-task losses and a working optimizer/checkpoint path.

It is **not yet a validated XAU/USD foundation model**. The only long run is invalid due to lookahead; the only leakage-free run is a one-epoch synthetic smoke model that loses badly to persistence. The real 2024-2026 data and leakage-free tensors are staged, but no real-data optimization has been authorized. Starting one requires rotated credentials, explicit training authorization and a fresh initialization; do not resume the invalid checkpoint.