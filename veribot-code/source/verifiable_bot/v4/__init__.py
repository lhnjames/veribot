"""VeriRoute v4: 33-signal rendering, frozen train-fitted buckets, and the E1 feature vector.

Import surface is deliberately small; see the module docstrings for the design rationale.
"""
from .features_v4 import FEATURE_NAMES, N_FEATURES, extract_features, feature_matrix
from .render_v4 import build_units_v4, record_payload, render_units, unit_map, unit_metadata
from .thresholds_v4 import fit_thresholds, load_thresholds, save_thresholds, tier_of

__all__ = [
    "FEATURE_NAMES",
    "N_FEATURES",
    "extract_features",
    "feature_matrix",
    "build_units_v4",
    "record_payload",
    "unit_metadata",
    "render_units",
    "unit_map",
    "fit_thresholds",
    "load_thresholds",
    "save_thresholds",
    "tier_of",
]
