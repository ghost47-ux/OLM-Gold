# Architecture

## 1. Problem decomposition

OLM-Gold models the joint distribution of one aligned M15 candle and its three M5 children:

```
p(P, C1, C2, C3 | H) = p(P | H) * p(C1 | H, P) * p(C2 | H, P, C1) * p(C3 | H, P, C1, C2)
```

`H` is multi-resolution history: 192 M5 bars, 64 M15 bars, 32 H1 bars and 16 H4 bars, plus the calendar features for every timestamp. The three children are ordered chronologically inside the parent and are exactly 5 minutes each.

The decomposition is not an independence assumption. Each child factor is conditioned on the sampled parent and every earlier sampled child, so the three children are dependent through the decoder state.

## 2. Candle encoding

For previous close `PC` and candle `(O, H, L, C)`:

```
g = ln(O / PC)
b = ln(C / O)
u = ln(H / max(O, C))
d = ln(min(O, C) / L)
```

`g` and `b` are unconstrained; `u` and `d` are non-negative. Model coordinates use a softplus inverse:

```
a_u = log(exp(u + eps) - 1),  a_d = log(exp(d + eps) - 1),  eps = 1e-8
```

Decoding is exact for the positive part: `u = softplus(a_u) - eps`, clamped at zero. The change of variables contributes

```
log p(u) = log p(a_u) - log sigmoid(a_u)
```

per wick. The Jacobian is applied in `MixtureDensityHead.candle_space_log_prob`; training NLL is in standardized model coordinates, and candle-space NLL is an evaluation-only diagnostic.

All four coordinates are standardized per axis with mean/std fit on the training M5 rows only. Every encoded resolution uses that same normalizer.

## 3. Tensors

Default research profile: `d_model=128`, 4 Transformer layers, 4 heads, `d_ff=384`, `K=4` mixture components, **1,493,752 trainable parameters**.

| Name | Shape | Meaning |
| --- | --- | --- |
| `m5` | `(B, 192, 4)` | encoded M5 history ending at the last observed bar |
| `m5_temporal` | `(B, 192, 14)` | clock/session features for those bars |
| `m15` | `(B, 64, 4)` | M15 history strictly before the target parent |
| `h1` | `(B, 32, 4)` | H1 history up to the last completed hour |
| `h4` | `(B, 16, 4)` | H4 history up to the last completed 4-hour block |
| `parent_enc` | `(B, 4)` | target M15 candle in model coordinates |
| `children_enc` | `(B, 3, 4)` | target M5 children in model coordinates |
| `state` | `(B, 128)` | fused last-token representation |
| `parent` | `logits (B,K)`, `loc (B,K,4)`, `scale_tril (B,K,4,4)` | parent mixture |
| `children[i]` | same mixture shapes, `i=0..2` | child step `i+1` mixture |

## 4. Encoders and backbone

1. **Candle encoder:** `4 -> 128 -> GELU -> 128 -> LayerNorm`, shared by all resolutions and by the child decoder.
2. **Temporal encoder:** `14 -> 128 -> GELU -> 128` followed by LayerNorm, added to candle embeddings.
3. **Timeframe embedding:** learned vector per timeframe id `(M5=0, M15=1, H1=2, H4=3)`; added to every token.
4. **Positions:** fixed sinusoidal encoding of position inside each context window.
5. **M5 trunk:** 4 pre-LayerNorm causal Transformer blocks. Only the last token is used.
6. **Context trunk:** 2 causal blocks shared by M15, H1 and H4; the last token of each is taken.
7. **Fusion:** concatenate the four state vectors `(B, 4*d_model)`, LayerNorm, linear back to `d_model`, GELU.

Each block is `x + Dropout(Attention(LN(x)))`, then `x + Dropout(FFN(LN(x)))`, with `need_weights=False` and an additive causal mask.

The optional X-Factor latent is disabled by default. When enabled, a diagonal Gaussian `q(z|state)` is sampled during training and the KL term is added to the loss.

## 5. Heads

### Parent head

`LayerNorm -> Linear(128,128) -> GELU -> MixtureDensityHead(K=4)` over four coordinates. The mixture uses a softmax over component logits and a Cholesky factor per component:

```
scale_tril = [[softplus(a11)+min_scale, 0, 0, 0],
              [a21, softplus(a22)+min_scale, 0, 0],
              [a31, a32, softplus(a33)+min_scale, 0],
              [a41, a42, a43, softplus(a44)+min_scale]]
```

### Child decoder

The decoder receives token sequence `[state, parent, C1, C2]`; token `C3` is unnecessary because no fourth child is predicted. A learned step embedding distinguishes the four positions. Causal Transformer output at positions 1, 2 and 3 produces the mixtures for C1, C2 and C3.

- Teacher forcing uses the actual parent and actual C1/C2 as prefix tokens.
- Free running starts with `[state, parent]`, samples C1, appends it, samples C2, appends it, then samples C3.
- The decoder is never conditioned on future children in either mode.

## 6. Training objective

```
L = 1.0*L_parent + 1.0*L_child + 0.5*L_aggregate + 0.25*L_consistency [+ beta*KL]
```

- `L_parent`: mean negative log-likelihood of the actual M15 candle under the parent mixture.
- `L_child`: mean over the three steps of the child mixture NLL.
- `L_aggregate`: decode each child **mixture mean** with the actual preceding close, aggregate with real OHLC rules, and compare `log` prices to the actual parent with SmoothL1 (`beta=0.01`).
- `L_consistency`: compare the same child aggregate to the **predicted** parent mean, so parent and child heads cannot drift apart. This is the path by which the parent prediction influences the child loss.
- `KL`: only when X-Factor is enabled.

Gradients from the aggregate and consistency terms flow through the decoded mixture means into both heads and the shared backbone.

## 7. Sampling and coherence projection

1. Encode the latest aligned context and produce the parent mixture.
2. Sample `S` parent coordinates (`S=32` by default), inverse-normalize, decode to prices with the last actual M15 close.
3. Expand the state to `S` rows. For step 1..3: build the token prefix, get the mixture, sample one coordinate vector per row, inverse-normalize, decode with the previous child's close, append.
4. Aggregate raw children and record the mean absolute log gap to the sampled parent.
5. Apply `project_children_to_parent`: clamp every price into `[parent.low, parent.high]`, force child 1 open to parent open, child 3 close to parent close, repair wick ordering, and place the maximum high/minimum low on the parent extremes.
6. Assert the projected scenarios satisfy the hierarchy and expose `raw_aggregation_gap`, `projected_max_gap`, `coherence_projection` and `seed` in diagnostics.

The projection guarantees consistency. It is **not** a probabilistic correction; the raw gap measures how often it was needed and how large the distortion was.

## 8. Splits and leakage

Candidate samples are indexed by the first child's M5 bar. The target timestamp is the first child timestamp. Boundaries are computed from target timestamps and enforced with a 60-minute embargo. Context windows must lie entirely before the target. M15 context stops before the parent; H1/H4 use only the last bar whose close (`open + interval`) is at or before the target start. The normalizer is fit on M5 rows earlier than the training boundary and then frozen for all splits and resolutions. `leakage_check` asserts zero H1/H4 bars with a close after the target in every split.

## 9. Inference payloads

`POST /v1/forecast`:

```json
{
  "m5": [{"timestamp": "2025-01-01T00:00:00Z", "open": 2650.1, "high": 2651.0, "low": 2649.5, "close": 2650.6}],
  "scenarios": 32,
  "seed": 7
}
```

Response: `as_of`, `parent_distribution` (mixture-mean `expected` candle, `dominant_component` diagnostic, and q05/q50/q95), `scenarios` (each with parent, three children and `coherent`), `diagnostics`, `forecast_id`, `model_version`, `disclaimer`. The mixture mean is the mathematical expectation; when low-weight components are far from the data it can be a poor point estimate, so both views are returned.

`POST /v1/actual`:

```json
{"forecast_id": "...", "candles": [{"timestamp": "...", "open": 1, "high": 2, "low": 0.5, "close": 1.5}]}
```

Exactly three realized M5 children are required. They are aggregated into the actual parent, the scenario-ensemble mean is scored with MAE per coordinate, and the evaluation is appended to the forecast log. This is a point error of the ensemble mean, not a probabilistic score.

Failure modes: 422 for malformed candles, invalid OHLC, non-5-minute intervals, insufficient or unaligned history; 404 for unknown forecast ids; 503 when the checkpoint is missing.

## 10. Service and monitoring

FastAPI loads the checkpoint lazily. Every forecast is stored as an append-only JSONL record with a content hash. `PredictionLog.evaluate_forecast` refuses to overwrite a prior evaluation, writes a separate evaluation record, and `monitoring/drift.py` computes PSI/KS diagnostics against a reference window. Drift signals are descriptive; they never retrain or promote a model automatically.
