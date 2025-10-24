import torch

from diffusion.models import UNet1D, UNet1DConfig


def test_unet_forward_shape():
    config = UNet1DConfig(
        in_channels=2,
        out_channels=2,
        base_channels=32,
        channel_mults=(1.0, 2.0),
        num_res_blocks=1,
        self_condition=False,
        cond_dim=0,
        use_attention=False,
    )
    model = UNet1D(config)
    x = torch.randn(3, 2, 64)
    t = torch.randint(0, 100, (3,))
    output = model(x, t)
    assert output.shape == (3, 2, 64)


def test_unet_supports_self_condition_and_mask():
    config = UNet1DConfig(
        in_channels=2,
        out_channels=2,
        base_channels=32,
        channel_mults=(1.0, 2.0),
        num_res_blocks=1,
        self_condition=True,
        cond_dim=4,
        use_attention=True,
        attn_heads=2,
    )
    model = UNet1D(config)
    x = torch.randn(2, 2, 32)
    t = torch.randint(0, 50, (2,))
    cond = torch.randn(2, 4)
    self_cond = torch.randn(2, 2, 32)
    mask = torch.zeros(2, 32)
    output = model(x, t, cond=cond, mask=mask, self_cond=self_cond)
    assert output.shape == (2, 2, 32)
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)
