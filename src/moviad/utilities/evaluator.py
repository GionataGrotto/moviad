from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from moviad.datasets.audio_dataset import SpectrogramBinarizer
from moviad.utilities.metrics import (
    cal_f1_img,
    cal_f1_pxl,
    cal_img_roc,
    cal_mse_pxl,
    cal_pr_auc_img,
    cal_pr_auc_pxl,
    cal_pro_auc_pxl,
    cal_pxl_roc,
)


def min_max_norm(x):
    d = x.max() - x.min()
    return (x - x.min()) / d if d != 0 else x


class Evaluator:
    """Compatibility evaluator for the audio benchmark scripts.

    The newer codebase already has a generic evaluator under
    ``moviad.utilities.evaluation.evaluator``. Audio benchmarking code in this
    repository still imports ``moviad.utilities.evaluator`` and expects the old
    string-based metric interface, so this module keeps that contract alive.
    """

    def __init__(self, test_dataloader, device):
        self.test_dataloader = test_dataloader
        self.device = device

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 5:
            return batch
        if len(batch) == 4:
            images, labels, gt_masks, paths = batch
            return images, labels, gt_masks, None, paths
        raise ValueError(f"Unsupported batch structure with {len(batch)} elements")

    @staticmethod
    def _unpack_predictions(preds):
        if isinstance(preds, (tuple, list)):
            if len(preds) == 2:
                return preds[0], preds[1], None
            if len(preds) >= 3:
                return preds[0], preds[1], preds[2]
        return preds, None, None

    def inference(self, model, return_inputs: bool = False):
        model.eval()

        gt_masks_list, gt_masks_tmp_list = [], []
        true_img_scores = []
        pred_masks, pred_img_scores, pred_tmp_scores = [], [], []
        inputs_list = [] if return_inputs else None

        for batch in tqdm(self.test_dataloader, desc="Eval"):
            images, labels, gt_masks, gt_masks_tmp, _ = self._unpack_batch(batch)

            with torch.no_grad():
                anomaly_maps, anomaly_scores, tmp_scores = self._unpack_predictions(
                    model(images.to(self.device))
                )

            if isinstance(anomaly_maps, torch.Tensor):
                anomaly_maps = anomaly_maps.cpu().numpy()
            if isinstance(anomaly_scores, torch.Tensor):
                anomaly_scores = anomaly_scores.cpu().numpy()
            if isinstance(tmp_scores, torch.Tensor):
                tmp_scores = tmp_scores.cpu().numpy()

            gt_masks_list.extend(np.atleast_1d(gt_masks.cpu().numpy()))
            true_img_scores.extend(np.atleast_1d(labels.cpu().numpy()))
            pred_masks.extend(np.atleast_1d(anomaly_maps))
            pred_img_scores.extend(np.atleast_1d(anomaly_scores))

            if gt_masks_tmp is not None:
                gt_masks_tmp_list.extend(np.atleast_1d(gt_masks_tmp.cpu().numpy()))
            if tmp_scores is not None:
                pred_tmp_scores.extend(np.atleast_1d(tmp_scores))

            if inputs_list is not None:
                inputs_list.extend(images.cpu().numpy())

        return (
            np.asarray(gt_masks_list),
            np.asarray(gt_masks_tmp_list) if gt_masks_tmp_list else None,
            np.asarray(true_img_scores),
            np.asarray(pred_masks),
            np.asarray(pred_img_scores),
            np.asarray(pred_tmp_scores) if pred_tmp_scores else None,
            np.asarray(inputs_list) if inputs_list is not None else None,
        )

    def compute_metrics(
        self,
        gt_masks_list,
        gt_masks_tmp_list,
        true_img_scores,
        pred_masks,
        pred_img_scores,
        pred_tmp_scores,
        metrics_to_compute,
        metrics_to_dict,
        binarize_masks,
        binarize_thresh_tmp,
    ):
        pred_masks = min_max_norm(pred_masks)
        binary_gt_masks = binarize_masks(gt_masks_list)

        metrics = {
            "img_roc_auc": lambda: cal_img_roc(pred_img_scores, true_img_scores)[2],
            "per_pixel_rocauc": lambda: cal_pxl_roc(binary_gt_masks, pred_masks)[2],
            "f1_img": lambda: cal_f1_img(pred_img_scores, true_img_scores),
            "f1_pxl": lambda: cal_f1_pxl(pred_masks, binary_gt_masks),
            "pr_auc_img": lambda: cal_pr_auc_img(pred_img_scores, true_img_scores),
            "pr_auc_pxl": lambda: cal_pr_auc_pxl(pred_masks, binary_gt_masks),
            "au_pro_pxl": lambda: cal_pro_auc_pxl(
                np.squeeze(pred_masks, axis=1), binary_gt_masks
            ),
            "mse_pxl": lambda: cal_mse_pxl(pred_masks, min_max_norm(gt_masks_list)),
        }

        if pred_tmp_scores is not None and gt_masks_tmp_list is not None:
            metrics.update(
                {
                    "f1_tmp": lambda: cal_f1_pxl(pred_tmp_scores, gt_masks_tmp_list),
                    "tmp_auc_roc": lambda: cal_pxl_roc(
                        gt_masks_tmp_list, pred_tmp_scores
                    )[2],
                    "pr_auc_tmp": lambda: cal_pr_auc_pxl(
                        pred_tmp_scores, gt_masks_tmp_list
                    ),
                    "f1_tmp_thresh": lambda: cal_f1_pxl(
                        pred_tmp_scores, binarize_thresh_tmp(gt_masks_list)
                    ),
                    "tmp_auc_roc_thresh": lambda: cal_pxl_roc(
                        binarize_thresh_tmp(gt_masks_list), pred_tmp_scores
                    )[2],
                    "pr_auc_tmp_thresh": lambda: cal_pr_auc_pxl(
                        pred_tmp_scores, binarize_thresh_tmp(gt_masks_list)
                    ),
                }
            )

        if metrics_to_compute is None:
            metrics_to_compute = list(metrics.keys())

        available_metrics = [metric for metric in metrics_to_compute if metric in metrics]

        if metrics_to_dict:
            return {metric: metrics[metric]() for metric in available_metrics}
        return [metrics[metric]() for metric in available_metrics]

    def evaluate(
        self,
        model,
        output_path: Optional[str] = None,
        metrics_to_compute: Optional[Sequence[str]] = None,
        metrics_to_dict: bool = False,
        binarize_masks=lambda x: x,
        binarize_thresh_tmp=lambda x: SpectrogramBinarizer.temporal_top_k_threshold(
            0.5, x
        ),
    ):
        (
            gt_masks_list,
            gt_masks_tmp_list,
            true_img_scores,
            pred_masks,
            pred_img_scores,
            pred_tmp_scores,
            _,
        ) = self.inference(model)

        return self.compute_metrics(
            gt_masks_list,
            gt_masks_tmp_list,
            true_img_scores,
            pred_masks,
            pred_img_scores,
            pred_tmp_scores,
            metrics_to_compute,
            metrics_to_dict,
            binarize_masks,
            binarize_thresh_tmp,
        )
