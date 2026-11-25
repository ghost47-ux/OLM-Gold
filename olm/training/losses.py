"""Joint parent/child training objective (spec section 6.4).

L_total = lp*L_parent + lc*L_child + la*L_aggregate + ls*L_consistency [+ beta*KL]

L_parent and L_child are joint negative log-likelihoods in model coordinates.
L_aggregate decodes the predicted child means with the actual previous closes,
aggregates them with real OHLC semantics and compares the result to the actual
M15 parent in log-price space. L_consistency compares that same aggregate with
the predicted parent, so the two heads cannot drift apart. Gradients flow
through both heads into the shared backbone; the consistency term is the only
path where a parent prediction influences a child loss.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

from olm.config import Config
from olm.models.decoding import aggregate_children_torch, decode_model_torch


@dataclass
class LossBreakdown:
    total: torch.Tensor
    parent: torch.Tensor
    child: torch.Tensor
    aggregate: torch.Tensor
    consistency: torch.Tensor
    kl: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            'total': float(self.total.detach()),
            'parent': float(self.parent.detach()),
            'child': float(self.child.detach()),
            'aggregate': float(self.aggregate.detach()),
            'consistency': float(self.consistency.detach()),
            'kl': float(self.kl.detach()),
        }


def _log_prices(prices: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(prices, min=1e-8))


def to_model_coords(coords: torch.Tensor, normalizer) -> torch.Tensor:
    """Undo the per-coordinate standardization before decoding prices."""
    mean = torch.as_tensor(normalizer.mean, dtype=coords.dtype, device=coords.device)
    std = torch.as_tensor(normalizer.std, dtype=coords.dtype, device=coords.device)
    return coords * std + mean


def decode_children_means(children: list, children_prev_close: torch.Tensor, normalizer) -> torch.Tensor:
    means = []
    for step, mixture in enumerate(children):
        means.append(
            decode_model_torch(
                to_model_coords(mixture.mean(), normalizer), children_prev_close[:, step]
            )
        )
    return torch.stack(means, dim=1)


def compute_losses(outputs: dict, batch: dict, config: Config, normalizer) -> LossBreakdown:
    train_cfg = config.training
    parent_nll = -outputs['parent'].log_prob(batch['parent_enc'])
    child_terms = [
        -mixture.log_prob(batch['children_enc'][:, step])
        for step, mixture in enumerate(outputs['children'])
    ]
    child_nll = torch.stack(child_terms, dim=1)
    parent_term = parent_nll.mean()
    child_term = child_nll.mean()

    child_mean_prices = decode_children_means(outputs['children'], batch['children_prev_close'], normalizer)
    aggregated = aggregate_children_torch(child_mean_prices)
    parent_mean_prices = decode_model_torch(
        to_model_coords(outputs['parent'].mean(), normalizer), batch['m15_prev_close']
    )
    actual_parent = batch['parent_prices']

    aggregate_term = F.smooth_l1_loss(
        _log_prices(aggregated), _log_prices(actual_parent), beta=0.01
    )
    consistency_term = F.smooth_l1_loss(
        _log_prices(aggregated), _log_prices(parent_mean_prices), beta=0.01
    )
    kl = outputs.get('kl', torch.zeros((), device=parent_term.device))
    total = (
        train_cfg.lambda_parent * parent_term
        + train_cfg.lambda_child * child_term
        + train_cfg.lambda_aggregate * aggregate_term
        + train_cfg.lambda_consistency * consistency_term
        + config.model.xfactor_beta * kl
    )
    return LossBreakdown(
        total=total,
        parent=parent_term,
        child=child_term,
        aggregate=aggregate_term,
        consistency=consistency_term,
        kl=kl.detach(),
    )


@torch.no_grad()
def aggregate_consistency_error(predicted_children: torch.Tensor, parent_prices: torch.Tensor) -> float:
    """Mean absolute log-price error of child aggregation vs the actual parent."""
    aggregated = aggregate_children_torch(predicted_children)
    return float((_log_prices(aggregated) - _log_prices(parent_prices)).abs().mean())
