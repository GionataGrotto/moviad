from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass

from moviad.common.args import Args
from moviad.datasets.subset import training_subset
from moviad.datasets.vad_dataset import IadDataset
from moviad.entrypoints.common import load_datasets
from moviad.models.padim.padim import Padim
from moviad.trainers.trainer_padim import TrainerPadim

BATCH_SIZE = 2
IMAGE_INPUT_SIZE = (224, 224)
OUTPUT_SIZE = (224, 224)


@dataclass
class PadimArgs(Args):
    train_dataset: IadDataset | None = None
    test_dataset: IadDataset | None = None
    category: str | None = None
    backbone: str | None = None
    ad_layers: list | None = None
    model_checkpoint_save_path: str | None = None
    diagonal_convergence: bool | None = False
    results_dirpath: str | None = None
    logger = None


def train_padim(args: PadimArgs, logger=None) -> None:
    train_dataset, test_dataset = load_datasets(args.dataset_config, args.dataset_type, args.category)
    train_dataset = training_subset(train_dataset, args.subset, args.seed)
    padim = Padim(
        args.backbone,
        args.category,
        device=args.device,
        layers_idxs=args.ad_layers,
    )
    padim.to(args.device)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, pin_memory=True, drop_last=True
    )
    # evaluate the model
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    trainer = TrainerPadim(
        model=padim,
        train_dataloader=train_dataloader,
        eval_dataloader=None,
        device=args.device,
        logger=logger,
    )
    trainer.train()

    evaluator = Evaluator(test_dataloader=test_dataloader, device=args.device)

    img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(padim)

    torch.cuda.empty_cache()

    if logger is not None:
        logger.log({
            "img_roc": img_roc,
            "pxl_roc": pxl_roc,
            "f1_img": f1_img,
            "f1_pxl": f1_pxl,
            "img_pr": img_pr,
            "pxl_pr": pxl_pr,
            "pxl_pro": pxl_pro,
        })

    print("Evaluation performances:")
    print(f"""
            img_roc: {img_roc}
            pxl_roc: {pxl_roc}
            f1_img: {f1_img}
            f1_pxl: {f1_pxl}
            img_pr: {img_pr}
            pxl_pr: {pxl_pr}
            pxl_pro: {pxl_pro}
            """)


def test_padim(args: PadimArgs, logger=None) -> None:

    padim = Padim(
        args.backbone,
        args.category,
        device=args.device,
        layers_idxs=args.ad_layers,
    )
    path = padim.get_model_savepath(args.model_checkpoint_path)
    padim.load_state_dict(
        torch.load(path, map_location=args.device), strict=False
    )
    padim.to(args.device)
    print(f"Loaded model from path: {path}")

    # Evaluator
    padim.eval()

    test_dataloader = DataLoader(
        args.test_dataset, batch_size=args.batch_size, shuffle=True
    )

    # evaluate the model
    evaluator = Evaluator(test_dataloader=test_dataloader, device=args.device)
    img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(padim)

    if logger is not None:
        logger.log({
            "img_roc": img_roc,
            "pxl_roc": pxl_roc,
            "f1_img": f1_img,
            "f1_pxl": f1_pxl,
            "img_pr": img_pr,
            "pxl_pr": pxl_pr,
            "pxl_pro": pxl_pro,
        })

    print("Evaluation performances:")
    print(f"""
        img_roc: {img_roc}
        pxl_roc: {pxl_roc}
        f1_img: {f1_img}
        f1_pxl: {f1_pxl}
        img_pr: {img_pr}
        pxl_pr: {pxl_pr}
        pxl_pro: {pxl_pro}
        """)
