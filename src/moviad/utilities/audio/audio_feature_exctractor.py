"""Deprecated alias for :mod:`moviad.utilities.audio.audio_feature_extractor`.

The module used to live under this misspelled name. It is kept so that scripts
outside this repository keep working; new code must import
``moviad.utilities.audio.audio_feature_extractor``.
"""

import warnings

from moviad.utilities.audio.audio_feature_extractor import (  # noqa: F401
    SUPPORTED_BACKBONES,
    AudioFeatureExtractor,
    _checkpoint_state_dict,
)

warnings.warn(
    "moviad.utilities.audio.audio_feature_exctractor is deprecated, "
    "import moviad.utilities.audio.audio_feature_extractor instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["AudioFeatureExtractor", "SUPPORTED_BACKBONES", "_checkpoint_state_dict"]
