import os

import torch
from torch.optim import AdamW
from tqdm import tqdm
import wandb

from moviad.models.audio.cfa.cfa import CFA
from moviad.utilities.audio.audio_feature_exctractor import AudioFeatureExtractor
from moviad.utilities.evaluator import Evaluator

class TrainerCFA():

    """
    This class contains the code for training the CFA model

    Args:
        cfa_model (CFA): model to be trained
        train_dataloader (torch.utils.data.DataLoader): train dataloader
        test_dataloder (torch.utils.data.DataLoader): test dataloader
        device (str): device to be used for the training
    """

    def __init__(
        self,
        cfa_model: CFA,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        wandb: bool, 
        device: str,
    ):
        self.cfa_model = cfa_model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.wandb = wandb
        self.evaluator = Evaluator(self.eval_dataloader, self.device)


    def train(self, epochs: int, metrics_to_compute: list, binarizer):
        """
        Train the model by first extracting the features from the batches, transform them
        with the patch descriptor and then apply the CFA loss

        Args:
            epochs (int) : number of epochs for the training
            metrics_to_compute (list[str]) : list of metrics to compute during training
            binarizer : binarizer to use in the evaluation for anomaly masks
        """

        translate_dict = {
            "f1_img" : "img_f1",
            "img_roc_auc" : "img_roc", 
            "pr_auc_img" : "img_pr", 
            "per_pixel_rocauc": "per_pixel_rocauc",
            "f1_pxl": "f1_pxl",
            "pr_auc_pxl": "pr_auc_pxl",
            "au_pro_pxl": "au_pro_pxl",
            "mse_pxl": "mse_pxl", 
            "f1_tmp" : "f1_tmp",
            "tmp_auc_roc" : "tmp_auc_roc",
            "pr_auc_tmp" : "pr_auc_tmp",
        }

        params = [{'params' : self.cfa_model.parameters()},]
        optimizer     = AdamW(params        = params,
                              lr            = 1e-3,
                              weight_decay  = 5e-4,
                              amsgrad       = True )


        best_metrics = {}
        for metric in metrics_to_compute:
            best_metrics[metric] = 0

        best_model_state = None

        for epoch in range(epochs):

            self.cfa_model.train()

            print(f"EPOCH: {epoch}")

            for batch in tqdm(self.train_dataloader):
                optimizer.zero_grad()

                loss = self.cfa_model(batch.to(self.device))
                loss.backward()
                optimizer.step()

            if binarizer:
                metrics = self.evaluator.evaluate(
                    self.cfa_model, 
                    metrics_to_compute=metrics_to_compute,
                    metrics_to_dict=True,
                    binarize_masks=binarizer,
                )
            else: 
                metrics = self.evaluator.evaluate(
                    self.cfa_model, 
                    metrics_to_compute=metrics_to_compute,
                    metrics_to_dict=True,
                )

            # update the best metrics and save the best model
            for metric in metrics_to_compute:
                if metric == "f1_pxl": 
                    if metrics[metric] > best_metrics[metric]:
                        print("New best model in f1_pxl")
                        print(f"Old f1_pxl: {best_metrics[metric]} New f1_pxl: {metrics[metric]}")
                        print("Saving best model")
                        best_model_state = self.cfa_model.state_dict()
                if metrics[metric] > best_metrics[metric]:
                    best_metrics[metric] = metrics[metric]


            print("End epoch performances:")
            for metric in metrics_to_compute:
                print(f"{metric}: {metrics[metric]}") 
            
            wandb_log_dict = { 
                "epoch": epoch, 
                "train_loss": loss, 
            }

            for metric in metrics_to_compute:
                wandb_log_dict[f"best_{translate_dict[metric]}"] = best_metrics[metric]

            if self.wandb:
                wandb.log(
                    wandb_log_dict
                )

        return best_metrics, best_model_state
