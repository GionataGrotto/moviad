from __future__ import annotations
from torch.nn import functional as F
from sklearn.metrics import *
import numpy as np
from skimage.measure import label, regionprops
from tqdm import trange


def cal_img_roc(img_scores: np.ndarray, gt_list: list) -> tuple[float, float, float]:
    """
    Calculate image-level roc auc score

    Args:
        scores (np.array) : numpy array of shape (b 1 h w) with the pixel level anomaly scores
        gt_list (list)    : list of ground truth labels

    Returns:
        fpr (float)     : false positive rate
        tpr (float)     : true positive rate
        img_roc (float) : img roc auc score
    """

    # for every image in the batch take the max pixel anomaly score

    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)

    return fpr, tpr, img_roc_auc


def cal_pxl_roc(gt_mask: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    """
    Calculate pixel-level roc auc score

    Args:
        gt_mask (np.array) : numpy array of ground truth masks
        scores (np.array)  : numpy array of predicted masks

    Returns:
        fpr (float)     : false positive rate
        tpr (float)     : true positive rate
        img_roc (float) : pixel roc auc score
    """

    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())

    return fpr, tpr, per_pixel_rocauc


def cal_f1_img(img_scores: np.ndarray, gt_list: list) -> float:
    """
    Calculate image-level f1 score

    Args:
        scores (np.array) : numpy array of shape (b 1 h w) with the pixel level anomaly scores
        gt_list (list)    : list of ground truth labels

    Returns:
        f1 (float)     : f1 score image level
    """

    gt_list = np.asarray(gt_list)

    precision, recall, _ = precision_recall_curve(gt_list, img_scores)
    a = 2 * precision * recall
    b = precision + recall

    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    return np.max(f1)


def cal_f1_pxl(scores: np.ndarray, gt_masks: np.ndarray) -> float:
    """
    Calculate image-level f1 score

    Args:
        scores (np.array) : numpy array of shape (b 1 h w) with the pixel level anomaly scores
        gt_masks (list)   : list of ground truth masks

    Returns:
        f1 (float)     : f1 score pixel level
    """
    gt_masks, scores = np.asarray(gt_masks), np.asarray(scores)

    if np.any(gt_masks == 1):
        precision, recall, _ = precision_recall_curve(gt_masks.flatten(), scores.flatten())
    else:
        # No positive class found, mimic sklearn's behavior
        precision, recall = np.array([0.0]), np.array([1.0])

    a = 2 * precision * recall
    b = precision + recall

    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    return np.max(f1)


def cal_pr_auc_img(scores: np.ndarray, gt_list: list) -> float:
    """
    Calculate image-level pr auc score

    Args:
        scores (np.array) : numpy array of shape (b 1 h w) with the pixel level anomaly scores
        gt_list (list)    : list of ground truth labels

    Returns:
        pr_auc_img (float)     : pr auc score image level
    """

    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)

    return average_precision_score(gt_list, img_scores)


def cal_pr_auc_pxl(scores: np.ndarray, gt_masks: np.ndarray) -> float:
    """
    Calculate pixel-level pr auc score

    Args:
        scores (np.array)  : numpy array of predicted masks
        gt_mask (np.array) : numpy array of ground truth masks

    Returns:
        pr_auc_pxl (float) : pro_auc pixel level score
    """

    gt_masks = np.asarray(gt_masks)

    return average_precision_score(gt_masks.flatten(), scores.flatten())


def cal_pro_auc_pxl(scores: np.ndarray, gt_masks: np.ndarray) -> float:

    def rescale(x):
        return (x - x.min()) / (x.max() - x.min())

    """
    Calculate pixel-level pro auc score

    Args:
        scores (np.array)  : numpy array of predicted masks
        gt_mask (np.array) : numpy array of ground truth masks

    Returns:
        per_pixel_roc_auc (float) : pro_auc pixel level score
    """

    # remove the channel dimension
    gt = np.squeeze(gt_masks, axis=1)

    gt[gt <= 0.5] = 0
    gt[gt > 0.5] = 1
    gt = gt.astype(np.bool_)

    max_step = 200
    expect_fpr = 0.3

    # set the max and min scores and the delta step
    max_th = scores.max()
    min_th = scores.min()
    delta = (max_th - min_th) / max_step

    pros_mean = []
    threds = []
    fprs = []

    binary_score_maps = np.zeros_like(scores, dtype=np.bool_)

    for step in trange(max_step):
        thred = max_th - step * delta

        # segment the scores with different thresholds
        binary_score_maps[scores <= thred] = 0
        binary_score_maps[scores > thred] = 1

        pro = []
        for i in range(len(binary_score_maps)):

            # label the regions in the ground truth
            label_map = label(gt[i], connectivity=2)

            # calculate some properties for every corresponding region
            props = regionprops(label_map, binary_score_maps[i])

            # calculate the per-regione overlap
            for prop in props:
                pro.append(prop.intensity_image.sum() / prop.area)

        # append the per-region overlap
        pros_mean.append(np.array(pro).mean())

        # calculate the false positive rate
        gt_neg = ~gt
        fpr = np.logical_and(gt_neg, binary_score_maps).sum() / gt_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    fprs = np.array(fprs)

    # select the case when the false positive rates are under the expected fpr
    idx = fprs <= expect_fpr

    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)
    pros_mean_selected = rescale(pros_mean[idx])
    per_pixel_roc_auc = auc(fprs_selected, pros_mean_selected)

    return per_pixel_roc_auc


def cal_mse_pxl(scores: np.ndarray, gt_masks: np.ndarray) -> float:
    """
    Calculate pixel-level mse score

    Args:
        scores (np.array)  : numpy array of predicted masks
        gt_mask (np.array) : numpy array of ground truth masks

    Returns:
        mse (float) : mse pixel level score
    """

    gt_masks = np.asarray(gt_masks)

    return mean_squared_error(gt_masks.flatten(), scores.flatten())
