from pathlib import Path
import sys
from itertools import islice

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data import load_balabit, segment_event_stream  # noqa: E402
from features import FEATURE_SPECS, compute_feature_matrix, compute_features  # noqa: E402


def collect_gestures(limit: int = 5):
    events = list(islice(load_balabit("train"), 1000))
    gestures = list(
        islice(
            segment_event_stream(events, gap_threshold_ms=250.0, target_len=32),
            limit,
        )
    )
    return gestures


def test_compute_features_vector_length():
    gestures = collect_gestures(limit=1)
    assert gestures, "Need at least one gesture"
    vec = compute_features(gestures[0])
    assert vec.shape == (len(FEATURE_SPECS),)
    assert np.isfinite(vec).all()


def test_compute_feature_matrix_shape():
    gestures = collect_gestures(limit=3)
    matrix = compute_feature_matrix(gestures)
    assert matrix.shape == (len(gestures), len(FEATURE_SPECS))

