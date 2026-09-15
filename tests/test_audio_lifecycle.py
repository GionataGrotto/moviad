import copy

import torch
from torch import nn

from moviad.utilities.audio.audio_feature_extractor import AudioFeatureExtractor
from moviad.models.audio.stfpm.stfpm import STFPM
from paper_benchmark.benchmark_common import limited, audio_sample_rate


def extractor(frozen=False):
    # Exercise the real module lifecycle without downloading a backbone.
    obj = AudioFeatureExtractor.__new__(AudioFeatureExtractor)
    nn.Module.__init__(obj)
    obj.device = torch.device("cpu")
    obj.frozen = frozen
    obj.model = nn.Sequential(nn.BatchNorm2d(2), nn.Dropout(0.5))
    obj.spectrogram_extractor = nn.Identity()
    obj.logmel_extractor = nn.Identity()
    obj.spectro_transform = nn.Identity()
    return obj


def test_stfpm_eval_preserves_batchnorm_and_predictions():
    model = STFPM(extractor(True), extractor())
    model.train()
    assert not model.teacher.model.training
    assert model.student.model.training
    model.eval()
    assert not model.student.model.training
    x = torch.randn(3, 2, 4, 4)
    before = model.student.model[0].running_mean.clone()
    a = model.student.model(x)
    b = model.student.model(x)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(before, model.student.model[0].running_mean)


def test_stfpm_checkpoint_restores_student_and_teacher():
    original = STFPM(extractor(True), extractor()).eval()
    with torch.no_grad():
        original.student.model[0].weight.fill_(3)
    state = copy.deepcopy(original.state_dict())
    assert "student.model.0.weight" in state
    restored = STFPM(extractor(True), extractor()).eval()
    restored.load_state_dict(state)
    x = torch.randn(3, 2, 4, 4)
    torch.testing.assert_close(original.student.model(x), restored.student.model(x))
    torch.testing.assert_close(original.teacher.model(x), restored.teacher.model(x))


def test_frozen_extractor_stays_in_eval_with_parent_training():
    model = nn.Sequential(extractor(True))
    model.train()
    assert not model[0].model.training


def test_debug_batches_repeat_each_epoch():
    batches = limited([1, 2, 3], True, 2)
    assert list(batches) == [1, 2]
    assert list(batches) == [1, 2]
    assert len(batches) == 2


def test_frontend_sample_rates():
    assert audio_sample_rate({"backbone": "Cnn14"}) == 44100
    assert audio_sample_rate({"backbone": "HTSAT-base"}) == 48000
