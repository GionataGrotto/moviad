import torch

from moviad.utilities.audio.audio_feature_exctractor import _checkpoint_state_dict


def test_checkpoint_state_dict_accepts_direct_encoder_weights():
    weights = {"base.conv.weight": torch.ones(1)}
    assert _checkpoint_state_dict(weights) == weights


def test_checkpoint_state_dict_unwraps_training_checkpoint_and_prefix():
    weights = {"module.base.conv.weight": torch.ones(1)}
    checkpoint = {
        "epoch": 12,
        "name": "cnn14",
        "state_dict": weights,
        "optimizer": {"lr": 1e-4},
    }
    extracted = _checkpoint_state_dict(checkpoint)
    assert list(extracted) == ["base.conv.weight"]
    torch.testing.assert_close(extracted["base.conv.weight"], torch.ones(1))


def test_checkpoint_state_dict_accepts_model_state_dict_format():
    weights = {"audio_encoder.base.conv.weight": torch.ones(1)}
    extracted = _checkpoint_state_dict({"model_state_dict": weights})
    assert list(extracted) == ["base.conv.weight"]


def test_checkpoint_state_dict_extracts_audio_branch_from_full_clap_checkpoint():
    checkpoint = {
        "state_dict": {
            "audio_branch.base.conv.weight": torch.ones(1),
            "audio_branch.projection.linear1.weight": torch.ones(1),
            "text_branch.encoder.layer.0.weight": torch.ones(1),
            "audio_projection.0.weight": torch.ones(1),
        }
    }
    extracted = _checkpoint_state_dict(checkpoint)
    assert list(extracted) == ["base.conv.weight", "projection.linear1.weight"]
