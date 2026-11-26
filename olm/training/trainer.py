"""Training loop with resume, early stopping, checkpointing and JSONL logging."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from olm.config import Config
from olm.data.datasets import DatasetBundle, leakage_check
from olm.hardware import resolve_device
from olm.logging_utils import get_logger
from olm.evaluation.baselines import fit_ar1_bundle
from olm.models.olm import OLMModel, build_model
from olm.training.checkpointing import load_checkpoint, restore_rng_state, save_checkpoint
from olm.training.losses import compute_losses
from olm.training.plots import render_learning_curves
from olm.training.scheduler import build_scheduler
from olm.training.validation import evaluate_validation

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


@dataclass
class Trainer:
    config: Config
    bundle: DatasetBundle
    model: OLMModel | None = None
    device: str | None = None
    run_name: str = 'run'
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        set_seed(self.config.training.seed)
        self.device = self.device or resolve_device(self.config.training.device)
        self.temporal_dim = self.bundle.meta['temporal_dim']
        if self.model is None:
            self.model = build_model(self.config.model, self.temporal_dim)
        self.model.to(self.device)
        self.run_dir = Path(self.config.training.run_dir) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(self.config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / 'metrics.jsonl'
        self.batch_metrics_path = self.run_dir / 'batches.jsonl'
        self.status_path = self.run_dir / 'status.json'
        self.curves_path = self.run_dir / 'learning_curves.svg'
        self.ar1_params = fit_ar1_bundle(self.bundle)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        self.steps_per_epoch = max(1, math.ceil(len(self.bundle.train) / self.config.training.batch_size))
        self.total_steps = self.config.training.max_steps or self.steps_per_epoch * self.config.training.epochs
        self.scheduler = build_scheduler(
            self.optimizer, self.total_steps, self.config.training.warmup_steps, self.config.training.min_lr_ratio
        )
        self.epoch = 0
        self.global_step = 0
        self.best_val = float('inf')
        self.bad_epochs = 0

    def _day_balance_weights(self) -> torch.Tensor:
        """Inverse-frequency weights over UTC weekdays for the training split."""
        target_ts = self.bundle.train.targets['target_ts'].astype('datetime64[s]')
        dow = (target_ts.astype('datetime64[D]').astype(np.int64) + 3) % 7
        counts = np.bincount(dow, minlength=7).astype(np.float64)
        weights = np.zeros(len(dow), dtype=np.float64)
        present = counts[dow] > 0
        weights[present] = 1.0 / counts[dow][present]
        weights /= weights.sum()
        return torch.as_tensor(weights, dtype=torch.double)

    def _loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_cfg = self.config.training
        generator = torch.Generator().manual_seed(train_cfg.seed)
        common = dict(num_workers=train_cfg.num_workers, pin_memory=False)
        if train_cfg.num_workers > 0:
            common['persistent_workers'] = True
        if train_cfg.day_balance:
            weights = self._day_balance_weights()
            sampler = WeightedRandomSampler(
                weights, num_samples=len(weights), replacement=True, generator=generator
            )
            train_loader = DataLoader(
                self.bundle.train, batch_size=train_cfg.batch_size, sampler=sampler, drop_last=False, **common
            )
        else:
            train_loader = DataLoader(
                self.bundle.train, batch_size=train_cfg.batch_size, shuffle=True, generator=generator,
                drop_last=False, **common
            )
        val_loader = DataLoader(self.bundle.val, batch_size=train_cfg.eval_batch_size, shuffle=False, **common)
        test_loader = DataLoader(self.bundle.test, batch_size=train_cfg.eval_batch_size, shuffle=False, **common)
        return train_loader, val_loader, test_loader

    def _log(self, record: dict) -> None:
        self.history.append(record)
        with open(self.metrics_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')

    def _log_batch(self, record: dict) -> None:
        with open(self.batch_metrics_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')

    def _write_status(self, state: str, latest: dict | None = None) -> None:
        status = {
            'run_name': self.run_name,
            'state': state,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'total_steps': getattr(self, 'limit', self.total_steps),
            'best_val': self.best_val,
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'latest': latest or {},
            'metrics_jsonl': str(self.metrics_path),
            'batches_jsonl': str(self.batch_metrics_path),
            'curves_svg': str(self.curves_path),
        }
        tmp = self.status_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(status, indent=2), encoding='utf-8')
        os.replace(tmp, self.status_path)

    def _render_curves(self) -> None:
        try:
            render_learning_curves(self.batch_metrics_path, self.metrics_path, self.curves_path)
        except Exception as exc:  # plotting must never abort training
            logger.warning('learning-curve render failed: %s', exc)

    def _dataset_meta(self) -> dict:
        return {
            'boundaries': self.bundle.boundaries,
            'meta': self.bundle.meta,
            'normalizer': self.bundle.normalizer.to_dict(),
        }

    def _save(self, name: str, val_loss: float) -> Path:
        return save_checkpoint(
            self.checkpoint_dir / name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            global_step=self.global_step,
            config=self.config,
            normalizer=self.bundle.normalizer,
            dataset_meta=self.dataset_meta,
            best_val=val_loss,
            extra={'run_name': self.run_name, 'device': self.device, 'leakage': leakage_check(self.bundle)},
        )

    def fit(self, resume_from: str | None = None, max_steps: int | None = None) -> dict:
        train_loader, val_loader, _ = self._loaders()
        self.dataset_meta = self._dataset_meta()
        if resume_from:
            checkpoint = load_checkpoint(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if checkpoint.get('scheduler_state'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.epoch = int(checkpoint['epoch'])
            self.global_step = int(checkpoint['global_step'])
            self.best_val = float(checkpoint.get('best_val', float('inf')))
            restore_rng_state(checkpoint['rng_state'])
            logger.info('resumed %s at epoch %d step %d', resume_from, self.epoch, self.global_step)
        limit = max_steps or self.config.training.max_steps or self.total_steps
        self.limit = limit
        train_cfg = self.config.training
        started = time.time()
        last_batch_time = time.perf_counter()
        samples_seen = 0
        stop = False
        self._write_status('running')
        while self.epoch < train_cfg.epochs and self.global_step < limit and not stop:
            self.model.train()
            last_batch_time = time.perf_counter()
            epoch_losses = []
            epoch_start = time.time()
            epoch_samples = 0
            epoch_step = 0
            for batch in train_loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = self.model(batch)
                losses = compute_losses(outputs, batch, self.config, self.bundle.normalizer)
                self.optimizer.zero_grad(set_to_none=True)
                losses.total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.global_step += 1
                batch_size = int(batch['m5'].shape[0])
                epoch_samples += batch_size
                samples_seen += batch_size
                epoch_step += 1
                values = losses.detached()
                values['lr'] = self.scheduler.get_last_lr()[0]
                values['grad_norm'] = float(grad_norm)
                epoch_losses.append(values)
                now = time.perf_counter()
                throughput = batch_size / max(now - last_batch_time, 1e-9)
                last_batch_time = now
                batch_record = {
                    'type': 'batch',
                    'epoch': self.epoch + 1,
                    'step': self.global_step,
                    'step_in_epoch': epoch_step,
                    'batch_size': batch_size,
                    'samples_per_second': throughput,
                    **values,
                }
                self._log_batch(batch_record)
                self._write_status('running', batch_record)
                if self.global_step % train_cfg.log_every == 0 or self.global_step == 1:
                    logger.info(
                        'step %d/%d | loss %.4f | parent %.4f | child %.4f | agg %.4f | cons %.4f | lr %.2e | |g| %.2f | %.1f samples/s',
                        self.global_step, limit, values['total'], values['parent'], values['child'],
                        values['aggregate'], values['consistency'], values['lr'], values['grad_norm'], throughput,
                    )
                if self.global_step >= limit:
                    stop = True
                    break
            self.epoch += 1
            val = evaluate_validation(
                self.model, val_loader, self.config, self.device, self.bundle.normalizer,
                n_samples=train_cfg.eval_samples, ar1_params=self.ar1_params,
            )
            if epoch_losses:
                train_mean = {key: float(np.mean([row[key] for row in epoch_losses])) for key in epoch_losses[0]}
            else:
                train_mean = {key: float('nan') for key in ('total', 'parent', 'child', 'aggregate', 'consistency', 'kl')}
            elapsed = time.time() - epoch_start
            improved = bool(np.isfinite(val['total']) and val['total'] < self.best_val - train_cfg.min_delta)
            if improved:
                self.best_val = val['total']
                self.bad_epochs = 0
                self._save(f'{self.run_name}_best.pt', self.best_val)
            else:
                self.bad_epochs += 1
            self._save(f'{self.run_name}_last.pt', self.best_val)
            overall = time.time() - started
            steps_done = max(1, self.global_step)
            record = {
                'epoch': self.epoch,
                'global_step': self.global_step,
                'train': train_mean,
                'val': val,
                'elapsed_s': round(overall, 2),
                'seconds_per_step': round(overall / steps_done, 4),
                'samples_processed': samples_seen,
                'epoch_samples': epoch_samples,
                'samples_per_second': round(samples_seen / max(overall, 1e-9), 2),
                'improved': improved,
            }
            self._log(record)
            self._render_curves()
            self._write_status('running', {'type': 'epoch', **record})
            logger.info(
                'epoch %d | train %.4f | val %.4f | best %.4f | parent_mae %.4f | child_mae %.4f | %.1fs | bad epochs %d',
                self.epoch, train_mean['total'], val['total'], self.best_val,
                val.get('parent_mae', float('nan')), val.get('child_mae', float('nan')), elapsed, self.bad_epochs,
            )
            if self.bad_epochs >= train_cfg.patience:
                logger.info('early stopping after %d epochs without improvement', self.bad_epochs)
                stop = True
        self._save(f'{self.run_name}_last.pt', self.best_val)
        self._render_curves()
        finite_history = [
            row for row in self.history if np.isfinite(row['val'].get('total', np.nan))
        ]
        best_record = min(finite_history, key=lambda row: row['val']['total']) if finite_history else None
        summary = {
            'run_name': self.run_name,
            'epochs': self.epoch,
            'global_step': self.global_step,
            'best_val': self.best_val,
            'best_validation': best_record['val'] if best_record else None,
            'elapsed_s': round(time.time() - started, 2),
            'device': self.device,
            'parameters': self.model.parameter_count(),
            'history': self.history,
        }
        (self.run_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        self._write_status('completed', {'type': 'summary', **summary})
        return summary


def train_model(config: Config, bundle: DatasetBundle, run_name: str = 'run', resume_from: str | None = None) -> dict:
    trainer = Trainer(config=config, bundle=bundle, run_name=run_name)
    return trainer.fit(resume_from=resume_from)
