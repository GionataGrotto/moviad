import torch

from paper_benchmark import benchmark_common as common


def config(**extra):
    base = common.load_config(None)
    base.update(device="cpu", pretrained=False, memory_bank_size=64, **extra)
    return base


def test_make_model_passes_the_band_options_to_patchcore():
    plain = common.make_model("patchcore", config(), torch.device("cpu"), (501, 64), "fan")
    assert not plain.band_banks and plain.score_standardization == "none"

    banded = common.make_model(
        "patchcore", config(patchcore_band_banks=True, patchcore_score_std="band", score_std_holdout=0.2),
        torch.device("cpu"), (501, 64), "fan",
    )
    assert banded.band_banks and banded.score_standardization == "band" and banded.holdout_fraction == 0.2


def test_make_model_passes_the_band_options_to_padim():
    plain = common.make_model("padim", config(), torch.device("cpu"), (501, 64), "fan")
    assert not plain.per_band and plain.score_standardization == "none"

    banded = common.make_model(
        "padim", config(padim_per_band=True, padim_score_std="band", padim_diag_cov=False),
        torch.device("cpu"), (501, 64), "fan",
    )
    assert banded.per_band and banded.score_standardization == "band" and not banded.diag_cov


def test_make_model_passes_the_distance_to_dinomaly():
    assert common.make_model("dinomaly", config(dinomaly_pretrained=False), torch.device("cpu"), (224, 224), "fan").distance == "cosine"
    l2 = common.make_model("dinomaly", config(dinomaly_pretrained=False, dinomaly_distance="l2"), torch.device("cpu"), (224, 224), "fan")
    assert l2.distance == "l2"
