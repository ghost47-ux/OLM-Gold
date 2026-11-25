"""OLM-Gold: hierarchical M15 parent / M5 child generative Transformer."""

from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from olm.config import ModelConfig
from olm.models.candle_encoder import CandleEncoder
from olm.models.child_decoder import ChildDecoder
from olm.models.parent_head import ParentHead
from olm.models.temporal_encoder import TemporalEncoder
from olm.models.timeframe_encoder import TimeframeEncoder
from olm.models.transformer import CausalTransformer, sinusoidal_positions


class OLMModel(nn.Module):
    """Shared causal backbone with parent and sequential child heads.

    Tensor shapes (B = batch):
      m5            (B, T5, 4)
      m5_temporal   (B, T5, F)
      m15/h1/h4     (B, Tr, 4) and matching temporal tensors
      state         (B, d_model)
      parent        K-component mixture over (B, 4)
      children      list of three mixtures over (B, 4)
    """

    def __init__(self, config: ModelConfig, temporal_dim: int) -> None:
        super().__init__()
        self.config = config
        self.temporal_dim = temporal_dim
        d = config.d_model
        self.candle_encoder = CandleEncoder(d, config.dropout)
        self.temporal_encoder = TemporalEncoder(temporal_dim, d, config.dropout)
        self.timeframe_encoder = TimeframeEncoder(d)
        self.trunk = CausalTransformer(d, config.n_heads, config.d_ff, config.n_layers, config.dropout)
        self.context_trunk = CausalTransformer(
            d, config.n_heads, config.d_ff, config.context_encoder_layers, config.dropout
        )
        fusion_width = 4 * d if config.use_context else d
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_width),
            nn.Linear(fusion_width, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.parent_head = ParentHead(d, config.n_components, config.min_scale, config.dropout)
        self.child_decoder = ChildDecoder(
            d, config.n_heads, config.d_ff, config.context_encoder_layers,
            config.n_components, config.min_scale, config.dropout,
        )
        if config.use_xfactor:
            self.latent_mu = nn.Linear(d, config.latent_dim)
            self.latent_logvar = nn.Linear(d, config.latent_dim)
            self.latent_proj = nn.Linear(config.latent_dim, d)

    def _encode(
        self,
        candles: torch.Tensor,
        temporal: torch.Tensor,
        timeframe_id: int,
        trunk: CausalTransformer,
    ) -> torch.Tensor:
        length = candles.shape[1]
        x = self.candle_encoder(candles)
        x = x + self.temporal_encoder(temporal)
        x = x + self.timeframe_encoder(timeframe_id, length).to(x.device)
        x = x + sinusoidal_positions(length, x.shape[-1], device=x.device).unsqueeze(0)
        return trunk(x)

    def _sample_latent(self, state: torch.Tensor, sample: bool) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.latent_mu(state)
        logvar = torch.clamp(self.latent_logvar(state), -8.0, 4.0)
        if sample:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return state + self.latent_proj(z), kl

    def encode_state(self, batch: dict[str, torch.Tensor], sample_latent: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        m5 = self._encode(batch['m5'], batch['m5_temporal'], 0, self.trunk)[:, -1]
        if self.config.use_context:
            m15 = self._encode(batch['m15'], batch['m15_temporal'], 1, self.context_trunk)[:, -1]
            h1 = self._encode(batch['h1'], batch['h1_temporal'], 2, self.context_trunk)[:, -1]
            h4 = self._encode(batch['h4'], batch['h4_temporal'], 3, self.context_trunk)[:, -1]
            features = torch.cat([m5, m15, h1, h4], dim=-1)
        else:
            features = m5
        state = self.fusion(features)
        kl = torch.zeros((), device=state.device)
        if self.config.use_xfactor:
            state, kl = self._sample_latent(state, sample_latent or self.training)
        return state, kl

    def forward(self, batch: dict[str, torch.Tensor], sample_latent: bool = False) -> dict:
        state, kl = self.encode_state(batch, sample_latent=sample_latent)
        parent = self.parent_head(state)
        result = {'state': state, 'parent': parent, 'kl': kl}
        if 'parent_enc' in batch:
            result['children'] = self.child_decoder.forward_teacher(
                state, batch['parent_enc'], batch['children_enc']
            )
        return result

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict:
        payload = asdict(self.config)
        payload['temporal_dim'] = self.temporal_dim
        return payload


def build_model(config: ModelConfig, temporal_dim: int) -> OLMModel:
    model = OLMModel(config, temporal_dim)
    if not any(True for _ in model.parameters()):
        raise RuntimeError('model has no trainable parameters')
    return model
