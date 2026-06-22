import os, time, datetime
from typing import Union

from tqdm import trange
import pandas as pd, numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from ..models.stfpm.stfpm import Stfpm
from ..datasets.mvtec_dataset import MVTecDataset
from ..utilities.configurations import TaskType, Split


def save_logs(logs, category, log_dirpath, log_filename):
    df = pd.DataFrame(logs)
    dirpath = os.path.join(log_dirpath, category)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    fp = os.path.join(dirpath, log_filename)
    # if the file already exists, append the new logs
    if os.path.exists(fp):
        df = pd.concat([pd.read_csv(fp), df], ignore_index=True)
    df.to_csv(os.path.join(dirpath, log_filename), index=False)


def train_model(
    model: Stfpm,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    category: str,
    model_save_path: str,
    log_dirpath=None,
    seed=None,
    early_stopping: Union[float, bool] = False,
):
    """
    Train the student-teacher feature-pyramid model and save checkpoints
    for each category.

    Args:
        model: stfpm model
        train_loader: torch dataloader for the training dataset
        val_loader: val_loader: torch dataloader for the validation dataset
        epochs: number of epochs to train the model
        device: where to run the model
        category: name of the mvtec category where to save the model
        model_save_path: directory where to create the category subdirectory
            and save the model
        log_dirpath: directory where to save the training logs
        seed: seed for reproducibility
        early_stopping: if a float is provided, the training will stop if the validation
            loss difference between the current and the previous epoch is less than the
            provided value.
    """
    model.seed = seed
    model.epochs = epochs
    model.category = category

    if model.seed is not None:
        torch.manual_seed(model.seed)

    min_err = 10000
    prev_val_loss = 100000

    if "micronet" in model.student.model_name:
        optimizer = torch.optim.SGD(
            model.student.parameters(), 0.04, momentum=0.9, weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.SGD(
            model.student.parameters(), 0.4, momentum=0.9, weight_decay=1e-4
        )

    # simple loss function of STFPM
    def loss_fn(t_feat, s_feat):
        return torch.sum((t_feat - s_feat) ** 2, 1).mean()

    logs = []
    for epoch in trange(epochs, desc="Train stfpm"):
        model.train()
        mean_loss = 0

        # train the model
        for batch_img in train_loader:

            t_feat, s_feat = model(batch_img.to(device))

            loss = loss_fn(t_feat[0], s_feat[0])
            for i in range(1, len(t_feat)):
                t_feat[i] = F.normalize(t_feat[i], dim=1)
                s_feat[i] = F.normalize(s_feat[i], dim=1)
                loss += loss_fn(t_feat[i], s_feat[i])

            print("[%d/%d] loss: %f" % (epoch, epochs, loss.item()))
            mean_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss /= len(train_loader)

        # evaluate the model
        with torch.no_grad():
            model.eval()
            val_loss = torch.zeros(1, device=device)
            for batch_imgs in val_loader:
                # NOTE: train and val losses are computed in different ways, maybe we can make them the same?
                anomaly_maps, _ = model(batch_imgs.to(device))
                val_loss += anomaly_maps.mean()
            val_loss /= len(val_loader)

        log_dict = {
            "epochs": epoch,
            "val_loss": val_loss.cpu(),
            "train_loss": mean_loss,
        }
        logs.append(log_dict)

        # save best checkpoint
        if val_loss < min_err:
            min_err = val_loss

            model_filename = model.model_filename()
            save_path = os.path.join(model_save_path, model.category, model_filename)
            dir_name = os.path.dirname(save_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            torch.save(model.state_dict(), save_path)

        if early_stopping not in [False, None] and epoch > 0:
            if np.abs(val_loss.cpu() - prev_val_loss) < early_stopping:
                print(f"Early stopping at epoch {epoch+1}/{epochs}")
                break
        prev_val_loss = val_loss.cpu()

    logs_df = pd.DataFrame(logs)
    if log_dirpath is not None:
        assert model.category is not None
        logs_path = os.path.join(
            log_dirpath,
            model.category,
            "train_logs",
            model.model_filename() + "_train_logs.csv",
        )
        dirpath = os.path.dirname(logs_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        logs_df.to_csv(logs_path, index=False)

    return logs_df, save_path


def train_param_grid_step(
    config,
    batch_size,
    backbone_model_name,
    device,
    img_input_size,
    img_output_size,
    early_stopping=False,
    checkpoint_dir="./snapshots",
    normalize_dataset=True,
):
    category = config["category"]
    ad_layers = config["ad_layers"]
    dataset_path = config["dataset_path"]
    student_bootstrap_layer = config.get("student_bootstrap_layer", None)
    epochs = config["epochs"]
    log_dirpath = config.get("log_dirpath", None)
    seed = config.get("seed", None)

    print(
        f"TRAIN | cat: {category}, ad_layers: {ad_layers}, epochs: {epochs}, seed: {seed}, early_stopping: {early_stopping}, bootstrap: {student_bootstrap_layer}"
    )

    if seed is not None:
        torch.manual_seed(seed)

    start_time = time.time()

    train_dataset = MVTecDataset(
        TaskType.SEGMENTATION,
        dataset_path,
        category,
        Split.TRAIN,
        img_size=img_input_size,
        norm=normalize_dataset,
    )
    train_dataset, val_dataset = train_test_split(
        train_dataset, test_size=0.2, random_state=seed
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True
    )

    model = Stfpm(
        input_size=img_input_size,
        output_size=img_output_size,
        ad_layers=ad_layers,
        backbone_model_name=backbone_model_name,
        student_bootstrap_layer=student_bootstrap_layer,
    )
    model.to(device)

    train_logs, snapshot_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        device=device,
        category=category,
        model_save_path=checkpoint_dir,
        log_dirpath=log_dirpath,
        seed=seed,
        early_stopping=early_stopping,
    )

    train_time = time.time() - start_time

    return {
        "train_time": train_time,
        "stop_epoch": train_logs["epochs"].max() + 1,
    }, snapshot_path


default_params = {
    "categories": ["hazelnut"],
    "ad_layers": [[8, 9], [10, 11, 12]],
    "epochs": [3, 3],
    "seeds": [0, 1],
    "batch_size": 32,
    "backbone_model_name": "mobilenet_v2",
    "device": "cuda:2",
    "img_input_size": (224, 224),
    "checkpoint_dir": "snapshots",
    "student_bootstrap_layer": [6, None],
}


def train_param_grid_search(params=default_params):
    """
    Parameters:
        categories: list of categories to train the model on
        ad_layers: N list of lists of integers, each list represents the layers to be used for the AD module
        epochs: N list of integers, each integer represents the number of epochs to train the model
        seeds: list of integers, each integer represents the seed for reproducibility
        student_bootstrap_layer: N list of integers, each integer represents the layer to be used for bootstrapping

    Note:
        ad_layers, epochs, student_bootstrap_layer are lists of the same length
    """
    trained_models_filepaths = []
    log_filename = f"logs_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    boot_layers = params.get(
        "student_bootstrap_layer", [None] * len(params["ad_layers"])
    )
    for category in params["categories"]:
        for ad_layers, epochs, boot_layer in zip(
            params["ad_layers"], params["epochs"], boot_layers
        ):
            logs = []
            for seed in params["seeds"]:
                for boot_layer in params.get("student_bootstrap_layer", (None,)):
                    # config params are the parameters that change from run to run
                    config = {
                        "category": category,
                        "ad_layers": ad_layers,
                        "epochs": epochs,
                        "seed": seed,
                        "batch_size": params["batch_size"],
                        "student_bootstrap_layer": boot_layer,
                        "dataset_path": params["dataset_path"],
                        "log_dirpath": params["log_dirpath"],
                    }
                    log, snapshot_path = train_param_grid_step(
                        config,
                        params["batch_size"],
                        params["backbone_model_name"],
                        params["device"],
                        params["img_input_size"],
                        params["img_output_size"],
                        params["early_stopping"],
                        params["checkpoint_dir"],
                        params["normalize_dataset"],
                    )
                    trained_models_filepaths.append(snapshot_path)
                    if params["log_dirpath"] is not None:
                        # -- LOGGING --
                        # - config: parameters that are changed from run to run
                        # - log: results of the run
                        # - other parameters that are constant for all runs
                        logs.append(
                            {
                                **config,
                                **log,
                                "backbone_model_name": params["backbone_model_name"],
                                "img_input_size": params["img_input_size"],
                            }
                        )

            if params["log_dirpath"] is not None:
                save_logs(logs, category, params["log_dirpath"], log_filename)

    return trained_models_filepaths
