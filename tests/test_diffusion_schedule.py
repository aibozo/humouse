import torch

from diffusion.noise.schedule import (
    DiffusionScheduleConfig,
    build_schedule,
    compute_v,
    eps_from_v,
    q_sample,
    v_from_eps,
    x0_from_v,
)


def test_cosine_schedule_is_monotonic():
    cfg = DiffusionScheduleConfig(timesteps=128, schedule="cosine")
    schedule = build_schedule(cfg)
    alpha_bar = schedule.alpha_bar
    assert alpha_bar.shape[0] == cfg.timesteps
    diff = alpha_bar[:-1] - alpha_bar[1:]
    assert torch.all(diff >= -1e-6)
    assert alpha_bar[0] <= 1.0
    assert alpha_bar[-1] < alpha_bar[0]


def test_roundtrip_x0_eps_v():
    torch.manual_seed(42)
    cfg = DiffusionScheduleConfig(timesteps=50, schedule="cosine")
    schedule = build_schedule(cfg)
    batch, seq_len, channels = 4, 32, 2
    x0 = torch.randn(batch, seq_len, channels)
    t = torch.randint(0, cfg.timesteps, (batch,))
    xt, eps = q_sample(schedule, x0, t)
    alpha, sigma = schedule.coefficients(t, device=x0.device)
    v = compute_v(x0, eps, alpha, sigma)
    x0_hat = x0_from_v(xt, v, alpha, sigma)
    eps_hat = eps_from_v(v, x0_hat, alpha, sigma)
    v_hat = v_from_eps(x0_hat, eps_hat, alpha, sigma)

    assert torch.allclose(x0_hat, x0, atol=1e-5, rtol=1e-4)
    assert torch.allclose(eps_hat, eps, atol=1e-5, rtol=1e-4)
    assert torch.allclose(v_hat, v, atol=1e-5, rtol=1e-4)
