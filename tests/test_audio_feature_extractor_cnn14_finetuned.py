"""Regression tests for AudioFeatureExtractor's Cnn14_finetuned checkpoint loading.

``_load_cnn14_finetuned`` loads a plain, non-contrastive Cnn14 (e.g. the
official PANN AudioSet-classification checkpoint from Kong et al.), as
opposed to ``_load_cnn14``'s CLAP audio-text encoder. Before this fix it built
the model with ``classes_num=1`` (a local ``classes_num = 527`` was already
computed and simply never used) and passed the raw checkpoint straight to
``load_state_dict``, which fails for the official release: its weights are
wrapped under a top-level "model" key and include the model's own
spectrogram/logmel frontend (``spectrogram_extractor.*``,
``logmel_extractor.melW``), parameters this project's Cnn14 wrapper does not
own since the spectrogram is computed separately.

These pin the fix against a synthetic checkpoint shaped like the real one,
so they do not need the real ~327MB file.
"""

import torch

from moviad.backbones.clap.clap import Cnn14
from moviad.utilities.audio.audio_feature_extractor import AudioFeatureExtractor


def _fake_pann_checkpoint():
    """A tiny checkpoint reproducing the official PANN release's structure:
    weights wrapped under "model", plus the model's own frontend parameters
    that must be filtered out before loading into this project's Cnn14.
    """
    real_model = Cnn14(classes_num=527, out_emb=2048)
    state = {f"model_prefix_unused": None}  # placeholder, overwritten below
    state = dict(real_model.state_dict())
    # official releases also ship their own frontend, which this repo's
    # Cnn14 wrapper does not have as submodules
    state["spectrogram_extractor.stft.conv_real.weight"] = torch.zeros(513, 1, 1024)
    state["logmel_extractor.melW"] = torch.zeros(513, 64)
    return {"iteration": 12345, "model": state}


def test_cnn14_finetuned_loads_a_wrapped_official_style_checkpoint(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "fake_pann.pth"
    torch.save(_fake_pann_checkpoint(), checkpoint_path)

    extractor = AudioFeatureExtractor(
        "Cnn14_finetuned",
        ["conv_block2", "conv_block3", "conv_block4"],
        torch.device("cpu"),
        frozen=True,
        pre_trained=True,
        checkpoint_path=checkpoint_path,
    )

    assert isinstance(extractor.model, Cnn14)
    # classes_num really is 527, matching fc_audioset's shape, not the old
    # hardcoded 1 (which would have made load_state_dict raise before we
    # ever got here).
    assert extractor.model.fc_audioset.out_features == 527

    extractor.eval()
    features = extractor(torch.randn(2, 44100 * 2))

    assert len(features) == 3
    for feature_map in features:
        assert feature_map.ndim == 4
        assert torch.isfinite(feature_map).all()


def test_cnn14_finetuned_matches_contrastive_cnn14_feature_shapes():
    """Same architecture, same layer names: a drop-in swap for the contrastive
    Cnn14 path, so PaDiM's EMBEDDING_SIZES table needs no changes to use it.
    """
    layers = ["conv_block2", "conv_block3", "conv_block4"]
    device = torch.device("cpu")
    waveform = torch.randn(2, 44100 * 2)

    contrastive = AudioFeatureExtractor("Cnn14", layers, device, frozen=True, pre_trained=False)
    contrastive.eval()
    contrastive_shapes = [tuple(f.shape) for f in contrastive(waveform)]

    finetuned = AudioFeatureExtractor.__new__(AudioFeatureExtractor)
    torch.nn.Module.__init__(finetuned)
    finetuned.frozen = True
    finetuned.model_name = "Cnn14_finetuned"
    finetuned.layers_idx = layers
    finetuned.device = device
    finetuned.spectrogram_transform_enabled = True
    finetuned.checkpoint_path = None
    finetuned.model = Cnn14(classes_num=527, out_emb=2048)
    finetuned.spectrogram_extractor, finetuned.logmel_extractor, finetuned.spectro_transform = (
        AudioFeatureExtractor._load_spectrogram_transform("Cnn14_finetuned")
    )
    finetuned.attach_hook()
    finetuned.model.eval()

    finetuned_shapes = [tuple(f.shape) for f in finetuned(waveform)]

    assert finetuned_shapes == contrastive_shapes
