# Research Decisions

This file records choices that materially affect what the model can claim. Numbers are measured on the development machine unless marked otherwise.

## D1. Model the hierarchy directly, not a single candle

**Decision:** one parent distribution plus three conditional child distributions, with exact OHLC aggregation.

**Why:** independent M5 predictions can violate `parent.high >= each child.high`, and an M15 model cannot answer questions about the path inside the bar. The hierarchy is the smallest unit that preserves both the parent view and intra-bar structure.

**Rejected:** four independent per-resolution models; a single M5 model with no explicit parent; generating children independently and fixing them only at post-processing.

**Cost:** generation is autoregressive over four steps, not one; training needs aligned triplets and rejects misaligned candidates.

## D2. Invertible candle coordinates with softplus wicks

**Decision:** `(g, b, a_u, a_d)` with `a = log(exp(u+eps)-1)`, decoded by softplus minus epsilon, plus the log-sigmoid Jacobian.

**Why:** raw OHLC has hard ordering constraints (`high >= max(open,close)`, etc.). A Gaussian or mixture directly on prices assigns mass to invalid candles. The transform maps the valid region to unconstrained coordinates and is analytically invertible.

**Rejected:** clipping raw Gaussian samples (biases the distribution and hides invalid mass); Dirichlet on normalized ranges (cannot represent absolute price/body jointly); quantile regression only (no joint density across the four coordinates).

**Cost:** wick coordinates can produce extreme values when the sampled latent is large; decoding clamps only at zero, so evaluation checks the resulting OHLC validity.

## D3. Mixtures of 4-D Gaussians with Cholesky covariance

**Decision:** `K=4` components, each with a full 4x4 Cholesky factor, diagonal softplus plus `min_scale=1e-4`.

**Why:** a full covariance captures body/wick correlations and cross-coordinate dependence. A diagonal Gaussian would factorize the candle and lose the joint structure the task asks for. Cholesky guarantees positive definiteness and gives a stable `log_prob`.

**Rejected:** independent per-coordinate heads; normalizing flows (more parameters and tuning, no clear gain at this data scale); fixed-variance Gaussian.

**Cost:** `K * 14` output numbers per candle; mixture collapse is possible and is monitored through component usage in future work.

## D4. Condition each child on the parent and earlier children

**Decision:** `p(P|H) p(C1|H,P) p(C2|H,P,C1) p(C3|H,P,C1,C2)`.

**Why:** this is the exact chain rule for the joint, not an independence approximation. It also gives generation a natural causal order matching time.

**Rejected:** predicting all three children in parallel from the parent (conditional independence, wrong joint); a single 12-coordinate head (loses sequential structure and makes free running impossible).

## D5. Multi-resolution history with one shared context trunk

**Decision:** 192 M5 / 64 M15 / 32 H1 / 16 H4 tokens, separate learned timeframe embeddings, two shared causal context layers, four-layer M5 trunk, 4-way fusion.

**Why:** gold has session and multi-day structure that a 16-hour M5 window alone cannot see. A shared context trunk keeps parameters and CPU cost down.

**Rejected:** concatenating all resolutions into one long sequence (quadratic cost, alignment complexity); one model per timeframe (no joint state); no context (the ablation is available via `--no-context`).

## D6. Teacher forcing plus a separate free-running evaluation

**Decision:** train on `[state, actual parent, actual C1, actual C2]`; evaluate both one-step teacher-forced metrics and free-running three-child generation.

**Why:** teacher forcing gives stable, parallel training. It can hide exposure bias, so free-running metrics are reported separately. Training loss alone is never treated as forecast skill.

**Rejected:** scheduled sampling early in training (extra complexity; free-running evaluation catches the gap); only free-running training (slow, unstable at this scale).

## D7. Multi-task loss with aggregate and consistency terms

**Decision:** parent NLL + child NLL + 0.5 aggregate reconstruction + 0.25 parent/child consistency, SmoothL1 in log-price space.

**Why:** pure likelihood trains each head but does not force the aggregate of child means to look like a valid parent. The aggregate term grounds children in the observed parent; the consistency term couples the two heads.

**Rejected:** a hard aggregation layer inside the decoder (non-differentiable extrema; would require approximations); post-hoc projection during training (hides disagreement instead of penalizing it).

## D8. Chronological splits, training-only normalizer, embargo

**Decision:** sort by target timestamp, 70/15/15, 60-minute embargo, normalizer fit only on M5 rows before the training boundary.

**Why:** random splits leak future volatility regimes; fitting normalization on all data leaks future scale. The embargo prevents adjacent windows from straddling a boundary.

**Rejected:** random or stratified random splits; per-split normalizers (incomparable model coordinates); walk-forward CV in the first release (cost; see D12).

## D9. Coherence projection after sampling

**Decision:** clamp children into the sampled parent range, force first open/last close and extreme wicks, then assert exact aggregation. Report the raw pre-projection gap.

**Why:** a probabilistic model will occasionally produce a child outside the sampled parent range. Returning an incoherent scenario would violate the product contract; discarding it would bias the sample toward easy cases.

**Rejected:** rejection sampling until coherent (unknown acceptance rate, biases the distribution, unbounded latency); no projection (broken hierarchy).

**Risk:** projection can mask disagreement. The raw gap is logged and the docs call this a projection, not a probabilistic correction.

## D10. CPU-first model size and no dropout in the default profile

**Decision:** `d_model=128`, 4 layers, 4 heads, `d_ff=384`, 1,493,752 parameters, dropout 0.0 in the default CPU profile.

**Why:** the original 3.1M profile measured 2.7-5.1 s per batch-64 step on the development CPU; most of the excess was PyTorch CPU dropout (1.6 s/step at p=0.05). The smaller profile measured ~1.1 s/step with dropout 0 and ~2.7 s/step at p=0.05. At this data scale, weight decay and early stopping are the practical regularizers.

**Rejected:** keeping 3.1M parameters and 10% dropout (a 12-epoch run would take hours and still be synthetic-only); GPU assumptions (the target environment is CPU-only).

**Risk:** no dropout can overfit a small real dataset. Re-enable a small rate on larger data and monitor validation; the cost is documented in `docs/CPU_OPTIMIZATION.md`.

## D11. Local experiment tracking, no external service

**Decision:** JSON/JSONL run directories plus a champion registry with explicit required metrics, minimum test samples and a minimum improvement margin.

**Why:** reproducibility with no network dependency; champion changes are auditable and cannot happen on a single noisy metric.

**Rejected:** MLflow as a hard dependency (optional extra only); automatic promotion on training loss (not a forecast metric).

## D12. Sequential blocks, not a retrained walk-forward backtest

**Decision:** `evaluate --folds N` scores consecutive blocks of the held-out split with one fixed model. The output explicitly says this measures temporal stability, not a retrained walk-forward backtest.

**Why:** a true walk-forward backtest requires refitting the normalizer and model at each origin, which is expensive and easy to implement incorrectly. Shipping a mislabeled backtest would be worse than shipping an honest stability diagnostic.

**Still missing:** expanding-window retraining with per-fold normalization and per-fold model selection. This is the next evaluation milestone before any economic claim.

## D13. No external LLM and no trading layer

**Decision:** the model is a from-scratch PyTorch network. It emits distributions over candles, not trade instructions.

**Why:** the specification asks for a foundation model, and an LLM wrapper would add no calibrated density over OHLC. Position sizing and execution introduce risks outside the scope.

## D14. Authorization gates and credential hygiene

**Decision:** four gates default to false; bulk retention and provider-data training are blocked in code; keys are read from `.env` only, masked in reports and redacted in logs.

**Why:** provider terms and account ownership are external constraints, not configuration preferences. A convenience default that enables bulk retention would be unsafe.

**Risk:** users can set the gates to true without actually having permission. The software cannot verify a contract; the documentation and error messages state the requirement.
