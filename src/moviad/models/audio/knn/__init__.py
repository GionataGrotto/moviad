"""Embedding + k-nearest-neighbour anomaly detection for audio."""

from moviad.models.audio.knn.encoders import BEATsEncoder
from moviad.models.audio.knn.knn import AudioKNN

__all__ = ["AudioKNN", "BEATsEncoder"]
