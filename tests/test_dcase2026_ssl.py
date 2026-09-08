from argparse import Namespace

import torch

from paper_benchmark.run_dcase2026_task2_ssl import (
    _augment_waveforms,
    _load_encoder_into_model,
    _nt_xent_loss,
    _ssl_config,
    _target_sample_rate,
)


def _args(**overrides):
    values = {
        "ssl_epochs": None,
        "ssl_batch_size": None,
        "ssl_learning_rate": None,
        "ssl_device": None,
        "ssl_channel": None,
        "detector_channel": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_ssl_defaults_to_far_to_near():
    config = _ssl_config({}, _args())
    assert config["pretrain_channel"] == 1
    assert config["detector_channel"] == 0


def test_ssl_cli_overrides_channels():
    config = _ssl_config({}, _args(ssl_channel=0, detector_channel=1))
    assert config["pretrain_channel"] == 0
    assert config["detector_channel"] == 1


def test_ssl_augmentation_preserves_shape_and_changes_audio():
    torch.manual_seed(7)
    config = _ssl_config({}, _args())
    waveform = torch.ones(3, 100)
    augmented = _augment_waveforms(waveform, config)
    assert augmented.shape == waveform.shape
    assert torch.isfinite(augmented).all()
    assert not torch.equal(augmented, waveform)


def test_nt_xent_is_finite_and_differentiable():
    first = torch.randn(4, 8, requires_grad=True)
    second = torch.randn(4, 8, requires_grad=True)
    loss = _nt_xent_loss(first, second, temperature=0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert first.grad is not None
    assert second.grad is not None


def test_ssl_frontend_sample_rates_match_backbones():
    assert _target_sample_rate("Cnn14") == 44100
    assert _target_sample_rate("HTSAT-base") == 48000


class _Extractor:
    def __init__(self):
        self.model = torch.nn.Linear(3, 2)


class _AudioModel:
    def __init__(self):
        self.feature_extractor = _Extractor()


class _PadimModel:
    def __init__(self):
        self.backbone_model = _Extractor()


class _StfpmModel:
    def __init__(self):
        self.teacher = _Extractor()
        self.student = _Extractor()


class _DinomalyModel:
    def __init__(self):
        self.encoder = torch.nn.Linear(3, 2)


def test_ssl_encoder_is_loaded_into_every_model_family():
    reference = torch.nn.Linear(3, 2)
    with torch.no_grad():
        reference.weight.fill_(2.0)
        reference.bias.fill_(3.0)
    checkpoint = {"encoder_state_dict": reference.state_dict()}
    models = {
        "patchcore": _AudioModel(),
        "cfa": _AudioModel(),
        "padim": _PadimModel(),
        "stfpm": _StfpmModel(),
        "dinomaly": _DinomalyModel(),
    }
    for method, model in models.items():
        _load_encoder_into_model(model, method, checkpoint)

    extractors = [
        models["patchcore"].feature_extractor,
        models["cfa"].feature_extractor,
        models["padim"].backbone_model,
        models["stfpm"].teacher,
        models["stfpm"].student,
    ]
    for extractor in extractors:
        torch.testing.assert_close(extractor.model.weight, reference.weight)
        torch.testing.assert_close(extractor.model.bias, reference.bias)
    torch.testing.assert_close(models["dinomaly"].encoder.weight, reference.weight)
    torch.testing.assert_close(models["dinomaly"].encoder.bias, reference.bias)
