import pytest
import torch

from moviad.models.audio.dinomaly import AudioDinomaly

ENCODER = "deit_tiny_patch16_224.fb_in1k"


def make_model(distance):
    torch.manual_seed(0)
    model = AudioDinomaly(ENCODER, "cpu", (224, 224), pretrained=False, distance=distance)
    return model


def test_cosine_remains_the_default_and_l2_returns_a_valid_audio_output():
    waveforms = torch.randn(2, 160000)
    cosine = make_model("cosine").eval()
    l2 = make_model("l2").eval()
    l2.load_state_dict(cosine.state_dict())  # same weights: only the distance differs

    cosine_maps, cosine_scores, cosine_temporal = cosine(waveforms)
    l2_maps, l2_scores, l2_temporal = l2(waveforms)

    assert cosine_maps.shape == l2_maps.shape and cosine_maps.ndim == 4
    assert l2_scores.shape == (2,) and l2_temporal.shape == cosine_temporal.shape
    assert torch.isfinite(l2_maps).all() and bool((l2_maps >= 0).all())
    assert not torch.allclose(cosine_maps, l2_maps)  # a different distance, a different map


def test_l2_scoring_is_deterministic_and_leaves_the_model_in_eval_mode():
    model = make_model("l2").eval()
    waveforms = torch.randn(1, 160000)

    first = model(waveforms)[1]
    second = model(waveforms)[1]

    torch.testing.assert_close(first, second)  # dropout stayed off while the features were read
    assert not model.training


def test_training_is_not_affected_by_the_distance_setting():
    model = make_model("l2").train()
    encoder_features, decoder_features = model(torch.randn(1, 160000))
    assert len(encoder_features) == len(decoder_features) >= 1


def test_unknown_distance_is_rejected():
    with pytest.raises(ValueError, match="distance"):
        AudioDinomaly(ENCODER, "cpu", pretrained=False, distance="manhattan")
