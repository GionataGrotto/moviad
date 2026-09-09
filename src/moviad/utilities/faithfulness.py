from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from tqdm import tqdm


def _feature_extractors(model) -> list:
    extractors = []
    for attribute in ("feature_extractor", "backbone_model", "teacher", "student"):
        candidate = getattr(model, attribute, None)
        if hasattr(candidate, "spectrogram_transform_enabled") and candidate not in extractors:
            extractors.append(candidate)
    return extractors


def _predictions(model, spectrograms: torch.Tensor, device: torch.device):
    output = model(spectrograms.to(device))
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise ValueError("The model must return an anomaly map and an anomaly score.")
    anomaly_map = output[0]
    anomaly_score = output[1]
    if not isinstance(anomaly_map, torch.Tensor):
        anomaly_map = torch.as_tensor(anomaly_map, device=device, dtype=torch.float32)
    else:
        anomaly_map = anomaly_map.to(device=device, dtype=torch.float32)
    if not isinstance(anomaly_score, torch.Tensor):
        anomaly_score = torch.as_tensor(anomaly_score, device=device, dtype=torch.float32)
    else:
        anomaly_score = anomaly_score.to(device=device, dtype=torch.float32)
    return anomaly_map, anomaly_score.flatten()


def _minmax_per_sample(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(start_dim=1)
    minimum = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maximum = flat.max(dim=1).values.view(-1, 1, 1, 1)
    denominator = (maximum - minimum).clamp_min(torch.finfo(value.dtype).eps)
    return (value - minimum) / denominator


def _normalise_scores(scores: Iterable[float], modified: Iterable[float]):
    scores = np.asarray(list(scores), dtype=np.float64)
    modified = np.asarray(list(modified), dtype=np.float64)
    if scores.size == 0:
        return scores, modified
    minimum = min(scores.min(), modified.min())
    maximum = max(scores.max(), modified.max())
    denominator = maximum - minimum
    if denominator == 0:
        return np.zeros_like(scores), np.zeros_like(modified)
    return (scores - minimum) / denominator, (modified - minimum) / denominator


def audio_spectro_transform(model):
    """Return the spectrogram transform belonging to an audio AD model."""
    extractors = _feature_extractors(model)
    if extractors:
        return extractors[0].spectro_transform

    # Spectrogram-only adapters (AudioDinomaly, AudioDRAEM) own their frontend
    # directly instead of wrapping it in an AudioFeatureExtractor. During
    # faithfulness the model receives the already-computed 4D spectrogram,
    # which those audio adapters support.
    transform = getattr(model, "spectrogram_transform", None)
    if callable(transform):
        return transform

    raise ValueError("Could not find an audio spectrogram transform on the model.")


def _spectrogram_mode(model):
    extractors = _feature_extractors(model)
    previous = [extractor.spectrogram_transform_enabled for extractor in extractors]
    for extractor in extractors:
        extractor.disable_wavs_to_spectros()
    return extractors, previous


def _restore_spectrogram_mode(extractors, previous):
    for extractor, enabled in zip(extractors, previous):
        extractor.spectrogram_transform_enabled = enabled


def compute_faithfulness(
    model,
    dataloader,
    snr_db: float | None,
    wav_to_spectro,
    device: torch.device,
) -> np.ndarray:
    """FF v1: apply the original moviad-audio SNR-aware masking rule."""
    wav_to_spectro = wav_to_spectro.to(device)
    extractors, previous = _spectrogram_mode(model)
    scores, modified_scores = [], []
    snr_scale = None if snr_db is None else 1 + 10 ** (float(snr_db) / 20)
    model.eval()
    try:
        for audio_clips, _, _ in tqdm(dataloader, desc="Faithfulness v1"):
            spectrograms = wav_to_spectro(audio_clips.to(device))
            with torch.no_grad():
                anomaly_map, score = _predictions(model, spectrograms, device)
                anomaly_map = _minmax_per_sample(anomaly_map)
                # Match the original moviad-audio implementation. For a
                # specified SNR, attenuate the predicted anomaly map by the
                # corresponding linear amplitude factor.
                if snr_scale is None:
                    modified = spectrograms * (1 - anomaly_map)
                else:
                    modified = spectrograms * (1 - anomaly_map / snr_scale)
                _, modified_score = _predictions(model, modified, device)
            scores.extend(score.cpu().numpy())
            modified_scores.extend(modified_score.cpu().numpy())
    finally:
        _restore_spectrogram_mode(extractors, previous)

    scores, modified_scores = _normalise_scores(scores, modified_scores)
    return scores - modified_scores


def compute_faithfulness_v2(
    model,
    dataloader,
    threshold: float,
    wav_to_spectro,
    device: torch.device,
) -> np.ndarray:
    """FF v2: replace predicted anomaly regions with the clean background."""
    wav_to_spectro = wav_to_spectro.to(device)
    extractors, previous = _spectrogram_mode(model)
    scores, modified_scores = [], []
    model.eval()
    try:
        for audio_clips, _, background_clips in tqdm(dataloader, desc="Faithfulness v2"):
            spectrograms = wav_to_spectro(audio_clips.to(device))
            background = wav_to_spectro(background_clips.to(device))
            with torch.no_grad():
                anomaly_map, score = _predictions(model, spectrograms, device)
                anomaly_map = _minmax_per_sample(anomaly_map)
                binary_map = (anomaly_map > float(threshold)).to(spectrograms.dtype)
                modified = spectrograms * (1 - binary_map) + background * binary_map
                _, modified_score = _predictions(model, modified, device)
            scores.extend(score.cpu().numpy())
            modified_scores.extend(modified_score.cpu().numpy())
    finally:
        _restore_spectrogram_mode(extractors, previous)

    scores, modified_scores = _normalise_scores(scores, modified_scores)
    return scores - modified_scores
