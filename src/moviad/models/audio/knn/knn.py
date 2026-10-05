"""k-NN anomaly detector on clip-level embeddings of a pretrained audio encoder.

This is the "pretrained embeddings + outlier detector" family used by the top
DCASE Task 2 systems: the encoder is frozen, every normal training clip becomes one
embedding per selected layer, and a test clip is scored by its distance to the
nearest training embeddings.
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from moviad.models.audio.audio_vad_model import AudioVADModel
from moviad.models.training_args import TrainingArgs


class AudioKNN(AudioVADModel):
    """
    Args:
        encoder: callable ``waveform [B, T] -> list of [B, D] tensors`` (one per layer), with ``.to(device)``.
        device: device for the encoder and the distance computation.
        top_k: number of neighbours; the score is the mean distance to the ``top_k`` nearest ones.
        metric: ``"cosine"`` (L2-normalised embeddings, euclidean distance) or ``"euclidean"``.
        ignore_exact_matches: skip neighbours at distance ~0, so that scoring a clip that is in
            the memory (e.g. the training clips) does not simply return its own copy.
    """

    EXACT_MATCH_EPS = 1e-6

    def __init__(self, encoder, device, top_k: int = 1, metric: str = "cosine", ignore_exact_matches: bool = True):
        super().__init__(feature_extractor=encoder, device=device)
        if metric not in ("cosine", "euclidean"):
            raise ValueError(f"Unsupported metric {metric!r}, use 'cosine' or 'euclidean'")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.encoder = encoder
        self.top_k = top_k
        self.metric = metric
        self.ignore_exact_matches = ignore_exact_matches
        self.memory: list[torch.Tensor] = []

    def embed(self, waveforms: torch.Tensor) -> list[torch.Tensor]:
        with torch.no_grad():
            embeddings = self.encoder(waveforms.to(self.device))
        embeddings = [e.float() for e in embeddings]
        if self.metric == "cosine":
            embeddings = [torch.nn.functional.normalize(e, dim=1) for e in embeddings]
        return embeddings

    def fit(self, train_dataloader) -> None:
        per_layer: list[list[torch.Tensor]] = []
        for batch in tqdm(train_dataloader, desc="kNN embedding extraction"):
            embeddings = self.embed(self.batch_input(batch))
            if not per_layer:
                per_layer = [[] for _ in embeddings]
            for store, embedding in zip(per_layer, embeddings):
                store.append(embedding.cpu())
        if not per_layer:
            raise RuntimeError("Cannot fit the kNN detector on an empty dataset")
        self.memory = [torch.cat(store) for store in per_layer]

    def train_epoch(self, epoch, train_dataloader, training_args: TrainingArgs):
        self.fit(train_dataloader)
        return 0.0

    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs):
        raise NotImplementedError("The kNN detector is fitted with fit/train_epoch, not train_step")

    def _layer_scores(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(queries, memory.to(queries.device), compute_mode="donot_use_mm_for_euclid_dist")
        if self.ignore_exact_matches:
            distances = distances.masked_fill(distances < self.EXACT_MATCH_EPS, float("inf"))
        k = min(self.top_k, memory.shape[0])
        nearest = distances.topk(k, dim=1, largest=False).values
        return nearest.mean(dim=1)

    def forward(self, waveforms: torch.Tensor):
        embeddings = self.embed(waveforms)
        if self.training:
            return embeddings
        if not self.memory:
            raise RuntimeError("The kNN memory is empty: fit the model before evaluating it")
        scores = torch.stack([self._layer_scores(q, m) for q, m in zip(embeddings, self.memory)]).mean(dim=0)
        # Clip-level detector: no time-frequency map. A 1x1 map keeps the
        # (maps, scores, temporal_scores) contract, so every DCASE score
        # aggregation ("max", "mean", "temporal_topk_mean") returns the score itself.
        return scores.view(-1, 1, 1, 1), scores, scores.view(-1, 1)

    def reset_model(self):
        self.memory = []
