"""Joint four-dimensional Gaussian mixture over encoded candle coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

DIM = 4
N_TRIL = DIM * (DIM + 1) // 2


@dataclass
class MixtureOutput:
    logits: torch.Tensor          # (B, K)
    loc: torch.Tensor             # (B, K, 4)
    scale_tril: torch.Tensor      # (B, K, 4, 4)

    def component_probs(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Joint log p(x) under the mixture, shape (B,)."""
        if x.shape != self.loc.shape[:1] + (DIM,):
            raise ValueError(f'expected target shape {(x.shape[0], DIM)}, got {tuple(x.shape)}')
        component = torch.distributions.MultivariateNormal(
            loc=self.loc, scale_tril=self.scale_tril, validate_args=False
        )
        log_component = component.log_prob(x.unsqueeze(1))
        log_weights = torch.log_softmax(self.logits, dim=-1)
        return torch.logsumexp(log_weights + log_component, dim=-1)

    def mean(self) -> torch.Tensor:
        weights = self.component_probs().unsqueeze(-1)
        return (weights * self.loc).sum(dim=1)

    def dominant_index(self) -> torch.Tensor:
        """Index of the highest-weight mixture component, shape (B,)."""
        return self.component_probs().argmax(dim=-1)

    def dominant_mean(self) -> torch.Tensor:
        """Mean of the highest-weight component, shape (B, 4).

        This is a robust point estimate, not the mixture expectation: a tiny
        far-away component can otherwise dominate the mixture mean in price space.
        """
        return self.loc[torch.arange(self.loc.shape[0], device=self.loc.device), self.dominant_index()]

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw one sample per row, shape (B, 4)."""
        probs = self.component_probs()
        index = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        loc = self.loc[torch.arange(len(index)), index]
        tril = self.scale_tril[torch.arange(len(index)), index]
        noise = torch.randn(loc.shape, generator=generator, dtype=loc.dtype, device=loc.device)
        return loc + torch.einsum('bij,bj->bi', tril, noise)

    def sample_many(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw n samples per row, shape (B, n, 4)."""
        return torch.stack([self.sample(generator) for _ in range(n)], dim=1)


class MixtureDensityHead(nn.Module):
    def __init__(self, d_model: int, n_components: int = 4, min_scale: float = 1e-4, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_components = n_components
        self.min_scale = min_scale
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_components * (1 + DIM + N_TRIL)),
        )
        tril = torch.tril_indices(DIM, DIM)
        self.register_buffer('tril_index', tril, persistent=False)
        diagonal = torch.tensor([0, 2, 5, 9], dtype=torch.long)
        self.register_buffer('diagonal_positions', diagonal, persistent=False)
        self.register_buffer('diagonal_index', torch.arange(DIM), persistent=False)

    def forward(self, hidden: torch.Tensor) -> MixtureOutput:
        leading = hidden.shape[:-1]
        raw = self.net(hidden).view(*leading, self.n_components, 1 + DIM + N_TRIL)
        logits = raw[..., 0]
        loc = raw[..., 1 : 1 + DIM]
        tril_raw = raw[..., 1 + DIM :]
        tril = torch.zeros(
            *leading, self.n_components, DIM, DIM, dtype=hidden.dtype, device=hidden.device
        )
        rows, cols = self.tril_index
        tril[..., rows, cols] = tril_raw
        diag_raw = tril_raw[..., self.diagonal_positions]
        tril[..., self.diagonal_index, self.diagonal_index] = F.softplus(diag_raw) + self.min_scale
        return MixtureOutput(logits=logits, loc=loc, scale_tril=tril)

    @staticmethod
    def candle_space_log_prob(model_log_prob: torch.Tensor, model_coords: torch.Tensor) -> torch.Tensor:
        """Convert log p(a_u, a_d) to log p(u, d) with the Jacobian correction."""
        log_scale = F.logsigmoid(model_coords[..., 2:4])
        return model_log_prob - log_scale.sum(dim=-1)
