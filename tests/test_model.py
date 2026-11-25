"""Model shape, causality, density and gradient tests."""

import math

import torch

from olm.data.encoding import Normalizer
from olm.models.olm import build_model
from olm.models.probability_head import MixtureDensityHead, MixtureOutput
from olm.models.transformer import causal_mask
from olm.training.losses import compute_losses


def test_forward_shapes(tiny_model_config, tiny_batch):
    model = build_model(tiny_model_config, 14)
    out = model(tiny_batch)
    assert out['state'].shape == (2, 32)
    assert out['parent'].logits.shape == (2, 2)
    assert out['parent'].loc.shape == (2, 2, 4)
    assert len(out['children']) == 3
    assert all(child.loc.shape == (2, 2, 4) for child in out['children'])


def test_causal_mask_is_lower_triangular():
    mask = causal_mask(4)
    assert mask.shape == (4, 4)
    assert torch.isinf(mask[0, 1]) and mask[0, 1] < 0
    assert torch.equal(torch.diag(mask), torch.zeros(4))


def test_mixture_density_matches_multivariate_normal():
    loc = torch.zeros(1, 1, 4)
    scale = torch.eye(4).reshape(1, 1, 4, 4)
    mixture = MixtureOutput(torch.zeros(1, 1), loc, scale)
    log_prob = mixture.log_prob(torch.zeros(1, 4))
    expected = -0.5 * 4 * math.log(2 * math.pi)
    assert abs(float(log_prob) - expected) < 1e-5


def test_candle_space_jacobian():
    coords = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    correction = MixtureDensityHead.candle_space_log_prob(torch.tensor([1.0]), coords)
    assert abs(float(correction) - (1.0 + 2 * math.log(2.0))) < 1e-5


def test_loss_backpropagates(tiny_model_config, tiny_batch):
    model = build_model(tiny_model_config, 14)
    normalizer = Normalizer(mean=[0.0, 0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0, 1.0])
    losses = compute_losses(model(tiny_batch), tiny_batch, type('C', (), {'training': type('T', (), {
        'lambda_parent': 1.0, 'lambda_child': 1.0, 'lambda_aggregate': 0.5, 'lambda_consistency': 0.25,
        'batch_size': 2})(), 'model': tiny_model_config})(), normalizer)
    losses.total.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(gradient).all() for gradient in grads)


def test_context_ablation_still_runs(tiny_model_config, tiny_batch):
    tiny_model_config.use_context = False
    model = build_model(tiny_model_config, 14)
    out = model(tiny_batch)
    assert out['state'].shape == (2, 32)


def test_parameter_count_is_positive(tiny_model_config):
    model = build_model(tiny_model_config, 14)
    assert model.parameter_count() > 1000
