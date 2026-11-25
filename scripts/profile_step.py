import cProfile
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from olm.config import load_config
from olm.data.preparation import prepare_datasets
from olm.models.olm import build_model
from olm.training.losses import compute_losses


def main():
    config = load_config()
    config.data.data_root = 'artifacts/synth90'
    config.training.batch_size = 64
    bundle, _ = prepare_datasets(config, source='synthetic', validate=False)
    model = build_model(config.model, bundle.meta['temporal_dim'])
    item = bundle.train[0]
    batch = {key: value.unsqueeze(0).repeat(64, *([1] * value.dim())) for key, value in item.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_losses(model(batch), batch, config, bundle.normalizer).total
        loss.backward()
        optimizer.step()
    start = time.perf_counter()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_losses(model(batch), batch, config, bundle.normalizer).total
        loss.backward()
        optimizer.step()
    print('steps/s', 3 / (time.perf_counter() - start))
    cProfile.runctx('compute_losses(model(batch), batch, config, bundle.normalizer).total.backward()', globals(), locals(), 'profile.out')
    pstats.Stats('profile.out').sort_stats('cumtime').print_stats(20)

if __name__ == '__main__':
    main()
