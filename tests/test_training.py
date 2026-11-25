"""Checkpoint round-trip, resume and a short optimisation smoke test."""

from pathlib import Path

import torch

from olm.data.calendar import TemporalFeatures
from olm.data.datasets import build_datasets
from olm.training.checkpointing import load_checkpoint, model_from_checkpoint, save_checkpoint
from olm.training.scheduler import build_scheduler
from olm.training.trainer import Trainer


def _bundle(config, frames):
    return build_datasets(config, frames, TemporalFeatures(config.temporal))


def test_checkpoint_round_trip(tiny_config, synthetic_frames, tmp_path):
    from olm.models.olm import build_model

    bundle = _bundle(tiny_config, synthetic_frames)
    model = build_model(tiny_config.model, bundle.meta['temporal_dim'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, total_steps=10, warmup_steps=1)
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, epoch=2, global_step=7,
        config=tiny_config, normalizer=bundle.normalizer, dataset_meta={'k': 'v'}, best_val=1.5,
    )
    checkpoint = load_checkpoint(path)
    assert checkpoint['epoch'] == 2 and checkpoint['global_step'] == 7
    restored = model_from_checkpoint(checkpoint)
    for left, right in zip(model.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(left, right)


def test_trainer_writes_checkpoints_and_resumes(tiny_config, synthetic_frames, tmp_path):
    tiny_config.training.run_dir = str(tmp_path / 'runs')
    tiny_config.training.checkpoint_dir = str(tmp_path / 'checkpoints')
    tiny_config.training.epochs = 3
    tiny_config.training.max_steps = 2
    tiny_config.data.embargo_minutes = 60
    bundle = _bundle(tiny_config, synthetic_frames)
    trainer = Trainer(config=tiny_config, bundle=bundle, run_name='resume')
    summary = trainer.fit(max_steps=2)
    assert summary['global_step'] == 2
    last = Path(tiny_config.training.checkpoint_dir) / 'resume_last.pt'
    assert last.exists()
    resumed = Trainer(config=tiny_config, bundle=bundle, run_name='resume')
    summary2 = resumed.fit(resume_from=str(last), max_steps=4)
    assert summary2['global_step'] == 4
    metrics = Path(tiny_config.training.run_dir) / 'resume' / 'metrics.jsonl'
    assert metrics.exists() and len(metrics.read_text().strip().splitlines()) >= 1


def test_training_decreases_loss_on_a_fixed_batch(tiny_config, synthetic_frames):
    from olm.models.olm import build_model
    from olm.training.losses import compute_losses

    bundle = _bundle(tiny_config, synthetic_frames)
    model = build_model(tiny_config.model, bundle.meta['temporal_dim'])
    item = bundle.train[0]
    batch = {key: value.unsqueeze(0) for key, value in item.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = None
    model.train()
    for step in range(6):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_losses(model(batch), batch, tiny_config, bundle.normalizer).total
        if first is None:
            first = float(loss)
        loss.backward()
        optimizer.step()
    assert float(loss) < first
