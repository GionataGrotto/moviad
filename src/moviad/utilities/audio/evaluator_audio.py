from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from ..metrics import cal_f1_img, cal_img_roc, cal_pr_auc_img


def min_max_norm(x):
    d = x.max() - x.min()
    return (x - x.min()) / d if d != 0 else x


class AudioEvaluator:
    """Small evaluator used by the legacy audio utilities."""

    def __init__(self, test_dataloader, device):
        self.test_dataloader = test_dataloader
        self.device = device

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 5:
            return batch[0], batch[1], batch[2]
        if len(batch) == 4:
            return batch[0], batch[1], batch[2]
        if len(batch) == 3:
            return batch
        raise ValueError(f"Unsupported batch structure with {len(batch)} elements")

    def evaluate(self, model, output_path: bool = False):
        model.eval()

        true_labels, pred_labels = [], []

        for batch in tqdm(self.test_dataloader, desc="Eval"):
            audio, label, _ = self._unpack_batch(batch)
            with torch.no_grad():
                outputs = model(audio.to(self.device))
                anomaly_maps, anomaly_scores = outputs[:2]

            true_labels.extend(np.atleast_1d(label.cpu().numpy()))
            pred_labels.extend(np.atleast_1d(anomaly_scores.cpu().numpy()))

        true_labels = np.asarray(true_labels)
        pred_labels = np.asarray(pred_labels)

        img_roc_auc = cal_img_roc(pred_labels, true_labels)[2]
        f1_img = cal_f1_img(pred_labels, true_labels)
        pr_auc_img = cal_pr_auc_img(pred_labels, true_labels)

        return img_roc_auc, f1_img, pr_auc_img
