import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_benchmark"))

from benchmark_common import build_fit_loaders  # noqa: E402
from moviad.datasets.dcase2026_task2 import DCASE2026Task2Dataset  # noqa: E402

SR = 16000


def make_dataset(root: Path, sources: int = 8, targets: int = 2) -> None:
    folder = root / "toy" / "toy"
    folder.mkdir(parents=True)
    for domain, count in (("source", sources), ("target", targets)):
        for index in range(count):
            wave = torch.randn(2, SR // 2) + (5.0 if domain == "target" else 0.0)
            torchaudio.save(str(folder / f"section_00_{domain}_train_normal_{index:04d}_a.wav"), wave, SR)


def run(tmp_path, method="patchcore", **options):
    config = {"batch_size": 4, "num_workers": 0, **options}
    train_ds = DCASE2026Task2Dataset(tmp_path, "toy", "train", train_domains="all", channel=0, target_sample_rate=SR)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4)
    return train_ds, train_loader, build_fit_loaders(config, method, tmp_path, "toy", 0, SR, 3, train_ds, train_loader)


def test_without_options_the_train_loader_is_unchanged(tmp_path):
    make_dataset(tmp_path)
    _, train_loader, (fit_loader, target_loader) = run(tmp_path)
    assert fit_loader is train_loader and target_loader is None


def test_patchcore_quota_separates_the_domains_and_augments_the_target(tmp_path):
    make_dataset(tmp_path)
    _, _, (fit_loader, target_loader) = run(tmp_path, patchcore_target_fraction=0.25, target_augment_copies=3)
    assert len(fit_loader.dataset) == 8  # source clips only
    assert len(target_loader.dataset) == 2 + 2 * 3  # the originals and 3 augmentations of each
    originals = torch.cat([batch for batch in target_loader])
    assert (originals.mean(dim=1).abs() > 1.0).float().mean() > 0.5  # the clips of this loader are target clips (level shifted)


def test_exact_duplicates_when_the_augmentation_probability_is_zero(tmp_path):
    make_dataset(tmp_path)
    _, _, (fit_loader, target_loader) = run(tmp_path, patchcore_target_fraction=0.25, target_augment_copies=3, target_augment_probability=0.0)
    target = torch.cat([batch for batch in target_loader])
    assert len(target) == 2 + 2 * 3
    originals = target[:2]
    for copy in target[2:]:
        assert any(torch.allclose(copy, original) for original in originals)  # every copy is one of the 2 original clips


def test_gradient_methods_get_the_target_clips_weighted(tmp_path):
    make_dataset(tmp_path)
    _, _, (fit_loader, target_loader) = run(tmp_path, method="dinomaly", train_target_weight="balance", target_augment_copies=2)
    assert target_loader is None and len(fit_loader.dataset) == 8 + 2 + 2 * 2
    sampled = torch.cat([batch for batch in fit_loader])
    # balanced: about half of the samples are target clips (level around 5), far more than their 6 / 14 share
    share = (sampled.mean(dim=1) > 2.0).float().mean().item()
    assert share > 0.35
