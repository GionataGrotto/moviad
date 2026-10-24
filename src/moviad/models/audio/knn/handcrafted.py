"""Handcrafted sub-band statistics scored with a shrunk Mahalanobis distance.

A training-free baseline in the spirit of the signal-processing systems of the DCASE 2026 Task 2
challenge: the power spectrogram is split into equal-width bands and every clip is described by,
per band, the mean log energy, its spread over time and the spread of its frame-to-frame change.
Normal clips define a Gaussian in that space (covariance shrunk towards its diagonal, since there
are only about a hundred clips); the score is the Mahalanobis distance to it.
"""

from __future__ import annotations

import torch

from moviad.models.audio.knn.knn import AudioKNN


class SubbandStatsEncoder:
    """``[B, T]`` 16 kHz waveforms -> one ``[B, 3 * n_bands]`` statistics vector (as a one-element list)."""

    sample_rate = 16000

    def __init__(self, n_bands: int = 32, n_fft: int = 1024, hop_length: int = 512, device="cpu"):
        self.n_bands, self.n_fft, self.hop_length = n_bands, n_fft, hop_length
        self.device = torch.device(device)
        self.window = torch.hann_window(n_fft)

    def to(self, device):
        self.device = torch.device(device)
        self.window = self.window.to(self.device)
        return self

    def __call__(self, waveforms: torch.Tensor) -> list[torch.Tensor]:
        waveforms = waveforms.to(self.device).float()
        window = self.window.to(self.device)
        power = torch.stft(waveforms, self.n_fft, self.hop_length, window=window, return_complex=True).abs().pow(2)
        bins = power.shape[1] - 1  # drop the Nyquist bin so that the bands have the same width
        per_band = bins // self.n_bands
        bands = power[:, : per_band * self.n_bands].reshape(power.shape[0], self.n_bands, per_band, -1).mean(dim=2)
        log_energy = torch.log(bands + 1e-10)  # [B, bands, frames]
        change = log_energy.diff(dim=-1)
        return [torch.cat([log_energy.mean(-1), log_energy.std(-1), change.std(-1)], dim=1)]


class SubbandMahalanobis(AudioKNN):
    """Gaussian model of the handcrafted statistics of the normal training clips."""

    def __init__(self, encoder, device, shrinkage: float = 0.1):
        super().__init__(encoder, device, top_k=1, metric="euclidean", ignore_exact_matches=False)
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        self.shrinkage = shrinkage
        self.mean = None
        self.precision = None

    def fit(self, train_dataloader) -> None:
        super().fit(train_dataloader)
        features = self.memory[0].double()
        self.mean = features.mean(dim=0)
        centred = features - self.mean
        covariance = centred.T @ centred / max(features.shape[0] - 1, 1)
        target = torch.diag(torch.diag(covariance))
        covariance = (1 - self.shrinkage) * covariance + self.shrinkage * target
        covariance += 1e-6 * torch.eye(covariance.shape[0], dtype=covariance.dtype) * covariance.diagonal().mean().clamp_min(1e-12)
        self.precision = torch.linalg.pinv(covariance)

    def forward(self, waveforms: torch.Tensor):
        embeddings = self.embed(waveforms)
        if self.training:
            return embeddings
        if self.precision is None:
            raise RuntimeError("Fit the model before evaluating it")
        delta = embeddings[0].double().cpu() - self.mean
        scores = torch.einsum("bi,ij,bj->b", delta, self.precision, delta).clamp_min(0).sqrt().float()
        return scores.view(-1, 1, 1, 1), scores, scores.view(-1, 1)

    def reset_model(self):
        super().reset_model()
        self.mean = self.precision = None
