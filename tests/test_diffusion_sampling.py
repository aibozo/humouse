from dataclasses import asdict

import torch

from diffusion.models import UNet1D, UNet1DConfig
from diffusion.noise import DiffusionScheduleConfig, build_schedule
from diffusion.sample import DiffusionSampler, generate_diffusion_samples, load_sampler_from_checkpoint
from diffusion.utils import EMAModel


def test_diffusion_sampler_generates_expected_shape():
    torch.manual_seed(0)
    model_cfg = UNet1DConfig(
        in_channels=2,
        out_channels=2,
        base_channels=16,
        channel_mults=(1.0, 2.0),
        num_res_blocks=1,
        self_condition=False,
        cond_dim=0,
        use_attention=False,
    )
    model = UNet1D(model_cfg)
    schedule = build_schedule(DiffusionScheduleConfig(timesteps=10))
    sampler = DiffusionSampler(model, schedule, device="cpu", cond_dim=model_cfg.cond_dim, in_channels=model_cfg.in_channels)
    samples = sampler.sample(4, 32, steps=5)
    assert samples.shape == (4, 32, model_cfg.in_channels)
    assert torch.isfinite(samples).all()


def test_generate_samples_from_checkpoint(tmp_path):
    torch.manual_seed(123)
    model_cfg = UNet1DConfig(
        in_channels=2,
        out_channels=2,
        base_channels=8,
        channel_mults=(1.0,),
        num_res_blocks=1,
        self_condition=True,
        cond_dim=0,
        use_attention=False,
    )
    model = UNet1D(model_cfg)
    ema = EMAModel(model, decay=0.9)
    diffusion_cfg = DiffusionScheduleConfig(timesteps=8)
    schedule = build_schedule(diffusion_cfg)

    ckpt_path = tmp_path / "diffusion_ckpt.pt"
    config_payload = {"model": asdict(model_cfg), "diffusion": asdict(diffusion_cfg)}
    torch.save(
        {
            "epoch": 1,
            "global_step": 1,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "config": config_payload,
        },
        ckpt_path,
    )

    sampler = load_sampler_from_checkpoint(ckpt_path, device="cpu")
    samples = sampler.sample(2, 16, steps=4)
    assert samples.shape == (2, 16, model_cfg.in_channels)

    samples_via_helper = generate_diffusion_samples(2, 16, checkpoint_path=ckpt_path, steps=4)
    assert samples_via_helper.shape == (2, 16, model_cfg.in_channels)
