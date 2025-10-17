from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from models.generator import ConditionalGenerator, GeneratorConfig  # noqa: E402
from models.discriminator import GestureDiscriminator, DiscriminatorConfig  # noqa: E402
from models.detector import GestureDetector, DetectorConfig  # noqa: E402


def test_generator_output_shape():
    config = GeneratorConfig(latent_dim=8, condition_dim=16, hidden_dim=32, target_len=20, num_layers=2)
    model = ConditionalGenerator(config)
    z = torch.randn(4, config.latent_dim)
    cond = torch.randn(4, config.condition_dim)
    out = model(z, cond)
    assert out.shape == (4, config.target_len, 3)


def test_discriminator_output_shape():
    config = DiscriminatorConfig(input_dim=3, condition_dim=16, hidden_dim=32, num_layers=2)
    model = GestureDiscriminator(config)
    sequence = torch.randn(4, 20, 3)
    cond = torch.randn(4, config.condition_dim)
    critic, aux = model(sequence, cond)
    assert critic.shape == (4,)
    assert aux.shape == (4, 4)


def test_detector_output_shape():
    config = DetectorConfig(feature_dim=16, sequence_dim=3, hidden_dim=32, tcn_layers=2)
    model = GestureDetector(config)
    sequence = torch.randn(4, 20, 3)
    features = torch.randn(4, config.feature_dim)
    logits = model(sequence, features)
    assert logits.shape == (4,)

