"""M15 parent prediction head: P(parent candle | history)."""

from __future__ import annotations

from torch import nn

from olm.models.probability_head import MixtureDensityHead, MixtureOutput


class ParentHead(nn.Module):
    def __init__(self, d_model: int, n_components: int, min_scale: float, dropout: float) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        self.mixture = MixtureDensityHead(d_model, n_components, min_scale, dropout)

    def forward(self, state) -> MixtureOutput:
        return self.mixture(self.project(state))
