import sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch
from olm.config import load_config
from olm.data.preparation import prepare_datasets
from olm.models.olm import build_model
from olm.training.losses import compute_losses

config = load_config()
config.data.data_root = 'artifacts/synth90'
bundle, _ = prepare_datasets(config, source='synthetic', validate=False)
item = bundle.train[0]
batch = {key: value.unsqueeze(0).repeat(64, *([1] * value.dim())) for key, value in item.items()}
for threads in (8, 4, 2):
    torch.set_num_threads(threads)
    for dropout in (0.05, 0.0):
        config.model.dropout = dropout
        model = build_model(config.model, bundle.meta['temporal_dim'])
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            compute_losses(model(batch), batch, config, bundle.normalizer).total.backward()
            opt.step()
        start = time.perf_counter()
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            compute_losses(model(batch), batch, config, bundle.normalizer).total.backward()
            opt.step()
        print('threads', threads, 'dropout', dropout, 'ms/step', round(1000*(time.perf_counter()-start)/3, 1), flush=True)
