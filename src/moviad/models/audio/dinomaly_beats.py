"""Dinomaly with BEATs as the frozen encoder.

Dinomaly reconstructs the features of a frozen ViT encoder with a bottleneck (an MLP with dropout) and a decoder of
linear-attention transformer blocks; at test time the disagreement between encoder and decoder features is the anomaly map.
The audio adapter of the visual implementation feeds the encoder an image of the spectrogram. Here the encoder is BEATs:
the hooked transformer blocks give ``(B, 768, time_patches, 8)`` feature maps (the same ones PatchCore consumes), their
mean is what the decoder has to reproduce, and the maps come out at the filterbank resolution like those of the other
BEATs detectors. A BEATs adapted with ``beats_ssl`` / ``beats_adapter`` is used as it is (the encoder stays frozen).
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_

from moviad.models.dinomaly.components import Block as VitBlock
from moviad.models.dinomaly.components import LinearAttention2, bMlp
from moviad.models.dinomaly.dinomaly import DinomalyTrainArgs
from moviad.models.vad_model import VADModel


class BEATsDinomaly(VADModel):
    """``extractor``: a frozen ``BEATsFeatureExtractor`` whose hooked blocks (Dinomaly uses 8 consecutive ones) are the encoder features."""

    def __init__(
        self,
        extractor,
        device: torch.device | str,
        dim: int | None = None,
        num_heads: int | None = None,
        decoder_blocks: int = 8,
        distance: str = "cosine",
    ) -> None:
        super().__init__()
        if distance not in ("cosine", "l2"):
            raise ValueError(f"distance must be 'cosine' or 'l2', got {distance!r}")
        self.encoder = extractor
        self.device = torch.device(device)
        self.distance = distance
        cfg = getattr(extractor.model, "cfg", None)
        dim = dim or getattr(cfg, "encoder_embed_dim", None)
        if dim is None:
            raise ValueError("The embedding size is not known: pass dim=")
        num_heads = num_heads or getattr(cfg, "encoder_attention_heads", 12)
        self.bottleneck = nn.ModuleList([bMlp(dim, dim * 4, dim, drop=0.2)])
        self.decoder = nn.ModuleList([
            VitBlock(dim=dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.0, attn=LinearAttention2)
            for _ in range(decoder_blocks)
        ])
        self.trainable = nn.ModuleList([self.bottleneck, self.decoder])
        for module in self.trainable.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        self.train_steps = 0

    def to(self, device: torch.device | str):
        device = torch.device(device)
        super().to(device)
        self.device = device
        self.encoder.to(device)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # the encoder (BEATs, adapted or not) is frozen: no dropout, no layerdrop
        self.bottleneck.train(mode)
        self.decoder.train(mode)
        return self

    def _maps(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder map ``(B, D, T, F)`` and decoder map of the same shape."""
        with torch.no_grad():
            encoder_maps = self.encoder(waveforms.to(self.device))
            encoder = torch.stack(encoder_maps, dim=0).mean(dim=0)  # Dinomaly fuses all its encoder layers by their mean
        batch, dim, time, freq = encoder.shape
        x = encoder.flatten(2).transpose(1, 2)  # (B, T * F, D), time-major
        for block in self.bottleneck:
            x = block(x)
        outputs = []
        for block in self.decoder:
            x = block(x, attn_mask=None)
            outputs.append(x)
        decoder = torch.stack(outputs, dim=0).mean(dim=0)
        return encoder, decoder.transpose(1, 2).reshape(batch, dim, time, freq)

    def forward(self, waveforms: torch.Tensor):
        encoder, decoder = self._maps(waveforms)
        if self.training:
            return [encoder], [decoder]
        if self.distance == "cosine":
            distance = 1 - F.cosine_similarity(encoder, decoder, dim=1)  # (B, T, F)
        else:
            distance = torch.linalg.vector_norm(encoder - decoder, dim=1)
        frames, bins = self.encoder.spec_shape[2], self.encoder.spec_shape[3]
        anomaly_maps = F.interpolate(distance.unsqueeze(1), size=(int(frames), int(bins)), mode="bilinear", align_corners=False)
        scores = anomaly_maps.flatten(1).amax(dim=1)
        top_k = min(5, anomaly_maps.shape[-1])
        temporal = anomaly_maps.squeeze(1).topk(top_k, dim=2).values.mean(dim=2)
        return anomaly_maps, scores, temporal

    def train_step(self, batch, training_args: DinomalyTrainArgs):
        if training_args.optimizer is None or training_args.lr_scheduler is None:
            training_args.init_train(self)
        waveforms = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(self.device)
        encoder_features, decoder_features = self(waveforms)
        p = min(0.9 * self.train_steps / 1000, 0.9)
        loss = training_args.loss_function(encoder_features, decoder_features, p=p, factor=0.1)
        training_args.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.trainable.parameters(), max_norm=0.1)
        training_args.optimizer.step()
        training_args.lr_scheduler.step()
        self.train_steps += 1
        return float(loss.detach().item())

    def train_chunk(self, train_dataloader, training_args):  # the abstract hook of VADModel; the benchmark uses TrainerDinomaly
        raise NotImplementedError("Train with TrainerDinomaly")


__all__ = ["BEATsDinomaly", "DinomalyTrainArgs"]
