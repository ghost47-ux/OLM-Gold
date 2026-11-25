"""Sequential M5 child decoder: P(C1 | H,P) P(C2 | H,P,C1) P(C3 | H,P,C1,C2)."""

from __future__ import annotations

import torch
from torch import nn

from olm.models.probability_head import MixtureDensityHead, MixtureOutput
from olm.models.transformer import CausalTransformer


class ChildDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        n_components: int,
        min_scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.candle_proj = nn.Linear(4, d_model)
        self.step_embedding = nn.Embedding(4, d_model)
        self.trunk = CausalTransformer(d_model, n_heads, d_ff, n_layers, dropout)
        self.head = MixtureDensityHead(d_model, n_components, min_scale, dropout)

    def _tokens(self, state: torch.Tensor, parent: torch.Tensor, children: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]
        tokens = [state.unsqueeze(1), self.candle_proj(parent).unsqueeze(1)]
        if children.shape[1]:
            tokens.append(self.candle_proj(children))
        x = torch.cat(tokens, dim=1)
        steps = torch.arange(x.shape[1], device=x.device)
        return x + self.step_embedding(steps).unsqueeze(0)

    @staticmethod
    def _slice(output: MixtureOutput, position: int) -> MixtureOutput:
        return MixtureOutput(
            logits=output.logits[:, position],
            loc=output.loc[:, position],
            scale_tril=output.scale_tril[:, position],
        )

    def forward_teacher(
        self, state: torch.Tensor, parent: torch.Tensor, children: torch.Tensor
    ) -> list[MixtureOutput]:
        """Teacher-forced pass. `children` holds the three actual child encodings.

        Tokens are [state, parent, C1, C2]; the heads at positions 1..3 predict
        C1, C2 and C3 respectively. C3 itself never needs to be fed because no
        fourth child is predicted.
        """
        if children.shape[1] != 3:
            raise ValueError('teacher forcing requires exactly three child encodings')
        tokens = self._tokens(state, parent, children[:, :2])
        output = self.head(self.trunk(tokens))
        return [self._slice(output, position) for position in (1, 2, 3)]

    def forward_step(
        self, state: torch.Tensor, parent: torch.Tensor, previous_children: torch.Tensor | None = None
    ) -> MixtureOutput:
        """Free-running step: predict the next child from the current prefix."""
        if previous_children is None:
            previous_children = state.new_zeros((state.shape[0], 0, 4))
        tokens = self._tokens(state, parent, previous_children)
        output = self.head(self.trunk(tokens))
        return self._slice(output, tokens.shape[1] - 1)
