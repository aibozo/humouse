"""Feature extraction package."""
from .neuromotor import (
    FEATURE_SPECS,
    FeatureSpec,
    compute_feature_matrix,
    compute_features,
    compute_features_from_sequence,
    tensor_to_gesture,
)
from .sigma_lognormal import (
    StrokeParams,
    decompose_sigma_lognormal,
    sigma_lognormal_features_from_sequence,
    sigma_lognormal_features,
)

__all__ = [
    "FeatureSpec",
    "FEATURE_SPECS",
    "compute_features",
    "compute_feature_matrix",
    "compute_features_from_sequence",
    "tensor_to_gesture",
    "sigma_lognormal_features",
    "sigma_lognormal_features_from_sequence",
    "decompose_sigma_lognormal",
    "StrokeParams",
]
