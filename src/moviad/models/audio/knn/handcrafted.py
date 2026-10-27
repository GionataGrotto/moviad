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


STATS = ("mean", "std", "diffstd", "kurtosis", "q10", "q90", "corr")
DEFAULT_STATS = ("mean", "std", "diffstd")


class SubbandStatsEncoder:
    """``[B, T]`` 16 kHz waveforms -> one statistics vector (as a one-element list).

    Per band: ``mean`` / ``std`` of the log energy over time, ``diffstd`` (std of its frame-to-frame change),
    ``kurtosis`` (peakedness over time), ``q10`` / ``q90`` (quantiles over time); ``corr`` adds the correlation
    over time between neighbouring bands (``n_bands - 1`` values).
    """

    sample_rate = 16000

    def __init__(self, n_bands: int = 32, n_fft: int = 1024, hop_length: int = 512, device="cpu", stats=DEFAULT_STATS):
        unknown = set(stats) - set(STATS)
        if unknown or not stats:
            raise ValueError(f"stats must be a non-empty subset of {STATS}, got {tuple(stats)}")
        self.stats = tuple(stats)
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
        features = []
        for name in self.stats:
            if name == "mean":
                features.append(log_energy.mean(-1))
            elif name == "std":
                features.append(log_energy.std(-1))
            elif name == "diffstd":
                features.append(log_energy.diff(dim=-1).std(-1))
            elif name == "kurtosis":
                centred = log_energy - log_energy.mean(-1, keepdim=True)
                features.append(centred.pow(4).mean(-1) / (centred.pow(2).mean(-1).pow(2) + 1e-8))
            elif name in ("q10", "q90"):
                features.append(torch.quantile(log_energy, 0.1 if name == "q10" else 0.9, dim=-1))
            else:  # corr
                centred = log_energy - log_energy.mean(-1, keepdim=True)
                centred = centred / (centred.norm(dim=-1, keepdim=True) + 1e-8)
                features.append((centred[:, :-1] * centred[:, 1:]).sum(-1))
        return [torch.cat(features, dim=1)]


SCORERS = ("mahalanobis", "knn", "iforest")


class SubbandMahalanobis(AudioKNN):
    """Model of the handcrafted statistics of the normal training clips.

    ``scorer``: ``mahalanobis`` (Gaussian, covariance shrunk towards its diagonal), ``knn`` (mean distance to the
    ``top_k`` nearest normal clips after standardising every feature) or ``iforest`` (isolation forest).
    """

    def __init__(self, encoder, device, shrinkage: float = 0.1, scorer: str = "mahalanobis", top_k: int = 3, seed: int = 0):
        super().__init__(encoder, device, top_k=top_k, metric="euclidean", ignore_exact_matches=False)
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        if scorer not in SCORERS:
            raise ValueError(f"scorer must be one of {SCORERS}, got {scorer!r}")
        self.shrinkage, self.scorer, self.seed = shrinkage, scorer, seed
        self.mean = None
        self.precision = None
        self.scale = None
        self.forest = None

    def fit(self, train_dataloader) -> None:
        super().fit(train_dataloader)
        features = self.memory[0].double()
        self.mean = features.mean(dim=0)
        if self.scorer == "knn":
            self.scale = features.std(dim=0).clamp_min(1e-8)
        elif self.scorer == "iforest":
            from sklearn.ensemble import IsolationForest

            self.forest = IsolationForest(n_estimators=300, random_state=self.seed).fit(features.numpy())
        else:
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
        if self.mean is None:
            raise RuntimeError("Fit the model before evaluating it")
        query = embeddings[0].double().cpu()
        if self.scorer == "knn":
            memory = (self.memory[0].double() - self.mean) / self.scale
            distances = torch.cdist((query - self.mean) / self.scale, memory)
            scores = distances.topk(min(self.top_k, memory.shape[0]), dim=1, largest=False).values.mean(dim=1).float()
        elif self.scorer == "iforest":
            scores = torch.from_numpy(-self.forest.score_samples(query.numpy())).float()
        else:
            delta = query - self.mean
            scores = torch.einsum("bi,ij,bj->b", delta, self.precision, delta).clamp_min(0).sqrt().float()
        return scores.view(-1, 1, 1, 1), scores, scores.view(-1, 1)

    def reset_model(self):
        super().reset_model()
        self.mean = self.precision = self.scale = self.forest = None
