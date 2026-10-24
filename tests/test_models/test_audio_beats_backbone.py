import sys
import textwrap

import pytest
import torch
from torch.utils.data import DataLoader

from moviad.models.audio.cfa.cfa import CFA
from moviad.models.audio.padim.padim import Padim
from moviad.models.audio.patchcore.patchcore import PatchCore
from moviad.utilities.audio.beats_feature_extractor import BEATsFeatureExtractor
from paper_benchmark import benchmark_common as common

DIM = 768
FREQ = 8
FAKE_BEATS = textwrap.dedent(
    f"""
    import torch
    from torch import nn

    class BEATsConfig:
        def __init__(self, cfg):
            self.layers = cfg["layers"]

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear({DIM}, {DIM})

        def forward(self, x):  # time-first [tokens, B, D], returns a tuple like the real blocks
            return torch.tanh(self.linear(x)), None

    class Encoder(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(cfg.layers)])

    class BEATs(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.patch_embedding = nn.Conv2d(1, 4, kernel_size=16, stride=16)
            self.encoder = Encoder(cfg)

        def extract_features(self, source, padding_mask=None):
            assert padding_mask is not None and padding_mask.shape == source.shape
            x = source.unfold(-1, {DIM}, {DIM}).transpose(0, 1)  # [tokens, B, D]; tokens = time * 8 + band
            for layer in self.encoder.layers:
                x, _ = layer(x)
            return x.transpose(0, 1), padding_mask
    """
)
SAMPLES = FREQ * DIM * 3  # 3 time patches of 8 bands


@pytest.fixture
def fake_beats(tmp_path):
    (tmp_path / "BEATs.py").write_text(FAKE_BEATS)
    sys.modules.pop("BEATs", None)
    sys.path.insert(0, str(tmp_path))
    import BEATs  # noqa: PLC0415

    reference = BEATs.BEATs(BEATs.BEATsConfig({"layers": 12}))
    checkpoint = tmp_path / "beats.pt"
    torch.save({"cfg": {"layers": 12}, "model": reference.state_dict()}, checkpoint)
    yield tmp_path, checkpoint, reference
    sys.path.remove(str(tmp_path))
    sys.modules.pop("BEATs", None)


def waveform_loader(count=6, batch_size=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(torch.randn(count, SAMPLES, generator=generator), batch_size=batch_size)


def test_tokens_are_folded_back_onto_the_time_by_band_grid(fake_beats):
    code_dir, checkpoint, reference = fake_beats
    extractor = BEATsFeatureExtractor(checkpoint, code_dir=code_dir, layers=(3, 12))
    waveforms = torch.randn(2, SAMPLES)

    maps = extractor(waveforms)

    assert [tuple(m.shape) for m in maps] == [(2, DIM, 3, FREQ)] * 2
    tokens = waveforms.unfold(-1, DIM, DIM).transpose(0, 1)
    outputs = {}
    for index, block in enumerate(reference.encoder.layers, start=1):
        tokens = block(tokens)[0]
        outputs[index] = tokens
    for layer, feature_map in zip((3, 12), maps):
        for time, band in ((0, 0), (1, 5), (2, 7)):  # token index = time * 8 + band
            torch.testing.assert_close(feature_map[:, :, time, band], outputs[layer][time * FREQ + band])
    assert extractor.spec_shape == torch.Size([2, 1, *BEATsFeatureExtractor.fbank_shape(SAMPLES)])


def test_fbank_shape_matches_the_kaldi_framing():
    assert BEATsFeatureExtractor.fbank_shape(160000) == (998, 128)  # 10 s at 16 kHz
    assert BEATsFeatureExtractor.fbank_shape(192000) == (1198, 128)


def test_frozen_extractor_has_no_gradients_and_a_student_can_train(fake_beats):
    code_dir, checkpoint, _ = fake_beats
    teacher = BEATsFeatureExtractor(checkpoint, code_dir=code_dir, layers=(4,), frozen=True)
    student = BEATsFeatureExtractor(checkpoint, code_dir=code_dir, layers=(4,), frozen=False, pretrained=False)
    waveforms = torch.randn(1, SAMPLES)

    assert not teacher(waveforms)[0].requires_grad and not teacher.model.training
    assert student(waveforms)[0].requires_grad and student.model.training
    assert not torch.equal(student.model.encoder.layers[0].linear.weight, teacher.model.encoder.layers[0].linear.weight)


def test_extractor_validates_layers_and_checkpoint(fake_beats, tmp_path):
    code_dir, checkpoint, _ = fake_beats
    with pytest.raises(ValueError, match="block numbers"):
        BEATsFeatureExtractor(checkpoint, code_dir=code_dir, layers=(0,))
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        BEATsFeatureExtractor(tmp_path / "missing.pt", code_dir=code_dir)


def make_beats_config(code_dir, checkpoint, layers=(8, 10, 12)):
    return {"backbone": "BEATs", "layers": list(layers), "beats_checkpoint": str(checkpoint), "beats_code_dir": str(code_dir)}


def test_patchcore_runs_on_beats_feature_maps(fake_beats):
    code_dir, checkpoint, _ = fake_beats
    extractor = common.make_feature_extractor(make_beats_config(code_dir, checkpoint, (4, 8)), torch.device("cpu"))
    input_size = BEATsFeatureExtractor.fbank_shape(SAMPLES)
    model = PatchCore(torch.device("cpu"), input_size, extractor, memory_bank_size=50, num_neighbors=3, blur=False)

    model.fit(waveform_loader())
    model.eval()
    maps, scores, temporal = model(torch.randn(2, SAMPLES))

    assert maps.shape == (2, 1, *input_size) and model.memory_bank.shape[1] == 2 * DIM
    assert torch.isfinite(scores).all() and temporal.shape == (2, input_size[0])


def test_padim_runs_on_beats_feature_maps(fake_beats):
    code_dir, checkpoint, _ = fake_beats
    config = make_beats_config(code_dir, checkpoint, (8, 10, 12))
    extractor = common.make_feature_extractor(config, torch.device("cpu"))
    total, default = Padim.lookup_embedding_size("BEATs", config["layers"])
    assert total == 3 * DIM and 1 <= default <= total
    input_size = BEATsFeatureExtractor.fbank_shape(SAMPLES)
    model = Padim("BEATs", "fan", torch.device("cpu"), config["layers"], diag_cov=True, img_size=input_size,
                  backbone_model=extractor, embedding_dim=16)

    model.fit(waveform_loader(), progress=False)
    model.eval()
    maps, scores, _ = model(torch.randn(2, SAMPLES))

    assert maps.shape == (2, 1, *input_size) and torch.isfinite(scores).all()
    assert model.fit_grid == (3, FREQ)


def test_cfa_runs_on_beats_feature_maps(fake_beats):
    code_dir, checkpoint, _ = fake_beats
    extractor = common.make_feature_extractor(make_beats_config(code_dir, checkpoint, (6, 12)), torch.device("cpu"))
    model = CFA(extractor, "BEATs", torch.device("cpu"), gamma_c=2, gamma_d=1)

    model.initialize_memory_bank(waveform_loader())
    model.eval()
    maps, scores, temporal = model(torch.randn(2, SAMPLES))

    input_size = BEATsFeatureExtractor.fbank_shape(SAMPLES)
    assert maps.shape == (2, 1, *input_size) and torch.isfinite(scores).all() and temporal.shape == (2, input_size[0])


def test_benchmark_helpers_understand_the_beats_backbone(fake_beats, tmp_path):
    code_dir, checkpoint, _ = fake_beats
    config = make_beats_config(code_dir, checkpoint)
    assert common.method_sample_rate("patchcore", config) == 16000
    assert common.method_sample_rate("patchcore", {"backbone": "Cnn14"}) == 44100
    assert common.spectrogram_size(config, torch.zeros(160000)) == (998, 128)
    common.check_audio_checkpoint(config, ["patchcore"])
    # the knn_beats_* keys of an existing kNN config are accepted as a fallback
    fallback = {"backbone": "BEATs", "layers": [4], "knn_beats_checkpoint": str(checkpoint), "knn_beats_code_dir": str(code_dir)}
    assert isinstance(common.make_feature_extractor(fallback, torch.device("cpu")), BEATsFeatureExtractor)
    with pytest.raises(ValueError, match="beats_checkpoint"):
        common.check_audio_checkpoint({"backbone": "BEATs"}, ["patchcore"])
    with pytest.raises(FileNotFoundError):
        common.check_audio_checkpoint({"backbone": "BEATs", "beats_checkpoint": str(tmp_path / "x.pt")}, ["padim"])
    with pytest.raises(ValueError, match="not to BEATs"):
        common.attach_spectrogram_preprocessing("patchcore", object(), [], {**config, "spec_band_norm": "zscore"}, torch.device("cpu"))


def test_stfpm_trains_a_pretrained_beats_student_with_the_configured_optimizer(fake_beats, monkeypatch):
    from moviad.trainers.audio import trainer_stfpm

    code_dir, checkpoint, _ = fake_beats
    config = make_beats_config(code_dir, checkpoint, (6, 12))
    config.update(stfpm_optimizer="adamw", stfpm_lr=1e-3, epochs=2, stfpm_student_pretrained=True)
    recorded = []
    original = trainer_stfpm.build_optimizer
    monkeypatch.setattr(
        trainer_stfpm, "build_optimizer", lambda *args, **kwargs: recorded.append(args[1:]) or original(*args, **kwargs)
    )
    input_size = BEATsFeatureExtractor.fbank_shape(SAMPLES)
    model = common.make_model("stfpm", config, torch.device("cpu"), input_size, "fan")
    before = model.student.model.encoder.layers[0].linear.weight.clone()

    common.fit_model("stfpm", model, waveform_loader(), None, config, torch.device("cpu"), False, evaluate_during_training=False)

    assert recorded == [("adamw", 1e-3, 1e-4)]  # optimizer, learning rate and weight decay reach the trainer
    assert not torch.equal(model.student.model.encoder.layers[0].linear.weight, before)  # the student learned
    assert torch.equal(
        model.teacher.model.encoder.layers[0].linear.weight, model.teacher.model.encoder.layers[0].linear.weight
    ) and not model.teacher.model.training  # the teacher stays frozen in eval mode
    model.eval()
    maps, scores, temporal = model(torch.randn(2, SAMPLES))
    assert maps.shape == (2, 1, *input_size) and torch.isfinite(scores).all() and temporal.shape == (2, input_size[0])
