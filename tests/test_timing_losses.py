import torch

from timing.losses import dirichlet_nll, lognormal_nll


def test_lognormal_nll_minimized_at_true_params():
    log_d = torch.tensor([0.0, 0.1, -0.2])
    mu = torch.zeros_like(log_d)
    log_sigma = torch.zeros_like(log_d)
    loss = lognormal_nll(log_d, mu, log_sigma)
    assert torch.all(loss >= 0)


def test_dirichlet_nll_lower_for_matching_template():
    target = torch.tensor([[0.6, 0.4, 0.0]], dtype=torch.float32)
    template_good = torch.tensor([[0.6, 0.4, 0.0]], dtype=torch.float32)
    template_bad = torch.tensor([[0.4, 0.6, 0.0]], dtype=torch.float32)
    concentration = torch.tensor([10.0], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    loss_good = dirichlet_nll(
        target,
        template_good,
        concentration,
        mask,
        smoothing=0.0,
    )
    loss_bad = dirichlet_nll(
        target,
        template_bad,
        concentration,
        mask,
        smoothing=0.0,
    )
    assert loss_good < loss_bad
