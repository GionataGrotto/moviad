from __future__ import annotations
import os
from typing import Union, Optional, Tuple

import pandas as pd
from tqdm import tqdm

import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import cv2 as cv

from ..metrics import *


def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min())


class AudioEvaluator:
    """
    This class will evaluate the trained model on the test set
    and it will produce the evaluation metrics needed

    Args:
        test_dataloader (Dataloader): test dataloader
        device (torch.device): device where to run the model
    """

    def __init__(self, test_dataloader, device):
        """
        Args:
            test_dataloader (Dataloader): test dataloader, the images should already be normalized
            device (torch.device): device where to run the model
        """
        self.test_dataloader = test_dataloader
        self.device = device

    def evaluate(self, model, output_path = False):
        """
        Args:
            model: a model object on which you can call model.predict(batched_audio)
                and returns a tuple of anomaly_maps and anomaly_scores
            output_path (str): path where to store the output masks
        """

        model.eval()

        # Initialize results.
        true_labels, pred_labels = (list(), list())

        for audio, label, _  in tqdm(self.test_dataloader, desc="Eval"):
            # get anomaly map and score
            with torch.no_grad():
                anomaly_maps, anomaly_scores = model(audio.to(self.device))

            true_labels.extend(label.cpu().numpy())
            pred_labels.extend(anomaly_scores.cpu().numpy())


        true_labels = np.asarray(true_labels)
        pred_labels = np.asarray(pred_labels)

        # image level ROC AUC
        fpr, tpr, img_roc_auc = cal_img_roc(pred_labels, true_labels)

        # image level F1
        f1_img = cal_f1_img(pred_labels, true_labels)

        # image level PR AUC
        pr_auc_img = cal_pr_auc_img(pred_labels, true_labels)

        return (
            img_roc_auc,
            f1_img,
            pr_auc_img,
        )