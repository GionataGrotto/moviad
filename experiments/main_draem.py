"""Train and test DRAEM (https://arxiv.org/abs/2108.07610) on MVTec AD.

Examples:

    uv run experiments/main_draem.py --mode train \
        --dataset_path /home/datasets/mvtec --category bottle \
        --epochs 700 --batch_size 8 --device cuda:0 --save_path ./draem_bottle.pt

    uv run experiments/main_draem.py --mode test \
        --dataset_path /home/datasets/mvtec --category bottle \
        --model_checkpoint_path ./draem_bottle.pt --device cuda:0

The paper simulates the anomalies with textures taken from the DTD dataset:
pass its ``images`` directory with ``--anomaly_source_path`` to reproduce that
setup. Without it the textures are synthesised from Perlin noise, so no extra
dataset is required.
"""

import argparse

import torch
from torch.utils.data import DataLoader

from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.datasets.mvtec import MVTecDataset
from moviad.models.draem.draem import DRAEM, DRAEMTrainArgs
from moviad.trainers.trainer import Trainer
from moviad.utilities.configurations import Split
from moviad.utilities.evaluation.evaluator import Evaluator
from moviad.utilities.evaluation.metrics import AvgPrec, F1, MetricLvl, ProAuc, RocAuc


def get_args():
    parser = argparse.ArgumentParser(description="DRAEM training and testing entrypoint")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="whether to train the model or to evaluate a checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="path of the MVTec AD dataset root")
    parser.add_argument("--category", type=str, required=True,
                        help="MVTec AD category to train on, e.g. bottle")
    parser.add_argument("--anomaly_source_path", type=str, default=None,
                        help="directory of the texture images used as anomaly source (DTD)")
    parser.add_argument("--img_size", type=int, nargs=2, default=(256, 256),
                        help="spatial size of the input images")
    parser.add_argument("--epochs", type=int, default=DRAEM.DEFAULT_PARAMETERS["epochs"])
    parser.add_argument("--batch_size", type=int, default=DRAEM.DEFAULT_PARAMETERS["batch_size"])
    parser.add_argument("--learning_rate", type=float,
                        default=DRAEM.DEFAULT_PARAMETERS["learning_rate"])
    parser.add_argument("--evaluation_epoch_interval", type=int, default=10)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default=None,
                        help="where to save the trained model weights")
    parser.add_argument("--model_checkpoint_path", type=str, default=None,
                        help="checkpoint to load before testing")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="wandb project name, logging is disabled when omitted")
    return parser.parse_args()


def build_datasets(args):
    dataset_arguments = DatasetArguments(
        dataset_path=args.dataset_path,
        img_size=tuple(args.img_size),
        gt_mask_size=tuple(args.img_size),
        image_transform_list=None,
    )
    train_dataset = MVTecDataset(dataset_arguments, category=args.category, split=Split.TRAIN)
    test_dataset = MVTecDataset(dataset_arguments, category=args.category, split=Split.TEST)
    return train_dataset, test_dataset


def build_metrics():
    return [
        RocAuc(MetricLvl.IMAGE),
        RocAuc(MetricLvl.PIXEL),
        AvgPrec(MetricLvl.IMAGE),
        AvgPrec(MetricLvl.PIXEL),
        F1(MetricLvl.IMAGE),
        F1(MetricLvl.PIXEL),
        ProAuc(MetricLvl.PIXEL),
    ]


def build_model(args):
    return DRAEM(
        anomaly_source_path=args.anomaly_source_path,
        input_size=tuple(args.img_size),
    ).to(torch.device(args.device))


def train_draem(args, logger=None):
    train_dataset, test_dataset = build_datasets(args)
    model = build_model(args)

    training_args = DRAEMTrainArgs(
        batch_size=args.batch_size,
        epochs=args.epochs,
        evaluation_epoch_interval=args.evaluation_epoch_interval,
        lr=args.learning_rate,
    )

    trainer = Trainer(
        training_args,
        model,
        train_dataset,
        test_dataset,
        metrics=build_metrics(),
        device=torch.device(args.device),
        logger=logger,
        save_path=args.save_path,
        saving_criteria=lambda best_metrics, results: results["img_roc_auc"]
        > best_metrics["img_roc_auc"],
    )
    trainer.train()


def test_draem(args):
    _, test_dataset = build_datasets(args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args)
    if args.model_checkpoint_path is None:
        raise ValueError("--model_checkpoint_path is required in test mode")
    model.load_model(args.model_checkpoint_path)
    model.eval()

    results = Evaluator.evaluate(
        model, test_dataloader, build_metrics(), torch.device(args.device)
    )
    print("\n".join(f"{name}: {value}" for name, value in results.items()))


def main():
    args = get_args()
    torch.manual_seed(args.seed)

    logger = None
    if args.wandb_project is not None:
        import wandb

        wandb.init(project=args.wandb_project, name=f"draem_{args.category}")
        logger = wandb

    if args.mode == "train":
        train_draem(args, logger)
    else:
        test_draem(args)


if __name__ == "__main__":
    main()
