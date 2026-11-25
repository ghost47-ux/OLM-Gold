# CPU Optimization Notes

All timings below are from the development machine, not a portable benchmark:

| Component | Value |
| --- | --- |
| CPU | AMD Ryzen 7 5825U, 8 cores / 16 threads |
| RAM | 15.37 GB |
| OS | Windows 11 |
| Python | 3.13.15 |
| PyTorch | 2.9.1+cpu |
| Threads | 8 (`torch.get_num_threads()`) |

## Measured profile

Batch 64, default synthetic split (5,868 train samples), one optimizer step including forward, loss, backward and AdamW:

| Configuration | ms/step | Samples/s |
| --- | ---: | ---: |
| 3.1M params, dropout 0.10 | ~5,150 | ~12 |
| 1.49M params, dropout 0.05 | ~2,695 | ~24 |
| 1.49M params, dropout 0.00 | ~1,111 | ~58 |
| 1.49M params, dropout 0.00, 4 threads | ~1,352 | ~47 |
| 1.49M params, dropout 0.00, 2 threads | ~1,797 | ~36 |

Forward-only batch time at the smaller profile was ~256 ms. A full 12-epoch run is ~1,104 steps; at ~1.1 s/step that is roughly 20 minutes plus validation and data preparation.

## What dominated

A profile of one backward pass showed:

- `torch.dropout`: ~1.4 s of a ~4.4 s step at p=0.05, spread over 48 calls. PyTorch 2.9 CPU dropout was memory-bound and unexpectedly expensive on this host. Setting `dropout: 0.0` was the single biggest win.
- Causal scaled dot-product attention: ~1.46 s over 12 calls.
- Backward: ~1.15 s.

The default profile therefore uses `dropout: 0.0`, `d_model=128`, 4 layers, 4 heads and `d_ff=384`. The model remains a real Transformer with 1,493,752 parameters.

## Practical settings

```yaml
training:
  batch_size: 64
  eval_batch_size: 128
  num_workers: 0
model:
  d_model: 128
  n_layers: 4
  n_heads: 4
  d_ff: 384
  dropout: 0.0
  m5_context: 192
```

- Keep `num_workers: 0` on Windows. Worker startup and shared-memory overhead outweighed loading time for this dataset.
- Use the default 8 threads. 4 and 2 threads were 22% and 62% slower respectively.
- `eval_batch_size=128` amortizes validation overhead; evaluation uses `torch.no_grad`.
- Early stopping (patience 6) avoids burning epochs after validation stops improving.
- `--max-steps` is useful for smoke tests; the scheduler still uses the configured total.

## Commands

```powershell
.\.venv\Scripts\python.exe -m olm benchmark --source synthetic --data-root artifacts/synth90 --batch-size 64 --steps 10 --json
.\.venv\Scripts\python.exe -m olm train --source synthetic --data-root artifacts/synth90 --epochs 12 --batch-size 64 --run-name synth90 --json
```

## Not used (and why)

- `torch.compile`: compile time and Windows CPU inductor risk outweigh a 20-minute run.
- Quantization: training is FP32; inference latency is not the bottleneck for research use.
- Mixed precision: CPU autocast gains are workload-dependent and were not validated here.
- Multi-process data loading: dataset items are small, in-memory NumPy slices.

## If you need more speed

1. Reduce `m5_context` from 192 to 128 (fewer attention positions) and document the shorter history.
2. Reduce `n_layers` to 3 or `d_ff` to 256; retrain and compare on the same split.
3. Increase batch size only if memory allows; the CPU is compute-bound, not allocator-bound.
4. Re-measure dropout before enabling it; on this PyTorch build it is disproportionately expensive.

Never compare runs trained with different context lengths, temporal features, normalizers or splits as if they were the same experiment.
