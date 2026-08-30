from tqdm import * 
import copy

import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from moviad.models.audio.stfpm.stfpm import STFPM
from moviad.utilities.evaluator import Evaluator

class TrainerSTFPM:

    """
    This class contains the code for training the STFPM model

    Args:
        stfpm (STFPM): model to be trained
        train_dataloader (torch.utils.data.DataLoader): train dataloader
        test_dataloder (torch.utils.data.DataLoader): test dataloader
        device (str): device to be used for the training
    """

    def __init__(
        self,
        stfpm: STFPM,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloder: torch.utils.data.DataLoader,
        wandb: bool, 
        device: str,
    ):
        self.stfpm = stfpm
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloder
        self.device = device
        self.wandb = wandb
        self.evaluator = (
            Evaluator(self.test_dataloader, self.device)
            if self.test_dataloader is not None
            else None
        )


    def _stfpm_loss(teacher_features, student_features):
        return torch.sum((teacher_features - student_features) ** 2, 1).mean()

    def train(self, epochs: int, metrics_to_compute: list, binarizer):
        """
        Train the model

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
        
        self.stfpm.to(self.device)

        optimizer = torch.optim.SGD(self.stfpm.student.model.parameters(), 0.4, momentum=0.9, weight_decay=1e-4)

        best_metrics = {}
        for metric in metrics_to_compute:
            best_metrics[metric] = 0

        best_model_state = None

        for epoch in trange(epochs): 

            self.stfpm.train()

            print(f"EPOCH: {epoch}")

            #train the model
            for batch in tqdm(self.train_dataloader): 

                batch = batch.to(self.device)
                teacher_features, student_features = self.stfpm(batch)

                loss = 0
                for i in range(len(student_features)):

                    teacher_features[i] = F.normalize(teacher_features[i], dim=1)
                    student_features[i] = F.normalize(student_features[i], dim=1) 
                    loss += TrainerSTFPM._stfpm_loss(teacher_features[i], student_features[i])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.evaluator is None:
                metrics = {}
            elif binarizer:
                metrics = self.evaluator.evaluate(
                    self.stfpm, 
                    metrics_to_compute=metrics_to_compute,
                    metrics_to_dict=True,
                    binarize_masks=binarizer,
                )
            else: 
                metrics = self.evaluator.evaluate(
                    self.stfpm, 
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
                        best_model_state = self.stfpm.state_dict()
                if metrics[metric] > best_metrics[metric]:
                    best_metrics[metric] = metrics[metric]

            print("End epoch performances:")
            for metric in metrics_to_compute:
                print(f"{metric}: {metrics[metric]}") 
            print(f"Loss: {loss}")
            
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








            






                





    


            






        
