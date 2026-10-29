import torch
from torch.utils.data import DataLoader, TensorDataset

from moviad.datasets.dcase2026_task2 import ls_denoise
from moviad.models.audio.knn import SubbandMahalanobis, SubbandStatsEncoder

SR = 16000


def tone(freq, n=SR, amplitude=1.0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    t = torch.arange(n) / SR
    return amplitude * torch.sin(2 * torch.pi * freq * t) + 0.01 * torch.randn(n, generator=generator)


def test_ls_denoise_removes_the_component_the_far_microphone_predicts():
    noise = torch.randn(SR, generator=torch.Generator().manual_seed(1))
    machine = tone(1000)
    near = machine + noise
    cleaned = ls_denoise(near, noise)
    assert (cleaned - machine).pow(2).mean() < 0.2 * (near - machine).pow(2).mean()


def test_ls_denoise_strength_zero_is_identity_and_half_is_between():
    near, far = torch.randn(SR), torch.randn(SR)
    assert torch.allclose(ls_denoise(near, far, 0.0), near, atol=1e-4)
    assert ls_denoise(near, far, 0.5).pow(2).mean() > ls_denoise(near, far, 1.0).pow(2).mean()


def test_subband_encoder_shape_and_mahalanobis_flags_a_shifted_spectrum():
    clips = torch.stack([tone(1000, seed=i) + 0.1 * torch.randn(SR, generator=torch.Generator().manual_seed(i)) for i in range(30)])
    encoder = SubbandStatsEncoder(n_bands=16)
    assert encoder(clips[:2])[0].shape == (2, 48)
    model = SubbandMahalanobis(encoder, torch.device("cpu"))
    model.fit(DataLoader(TensorDataset(clips), batch_size=10))
    model.eval()
    normal = model(clips[:5])[1]
    shifted = model(torch.stack([tone(3000, seed=i) for i in range(5)]))[1]
    assert shifted.min() > normal.max()


def test_ls_variants_follow_a_changing_noise_and_a_given_filter():
    from moviad.datasets.dcase2026_task2 import cross_spectra

    first, second = torch.randn(SR, generator=torch.Generator().manual_seed(2)), torch.randn(SR, generator=torch.Generator().manual_seed(3))
    far = torch.cat([first, second])
    near = torch.cat([2.0 * first, 0.5 * second]) + 0.01 * torch.randn(2 * SR, generator=torch.Generator().manual_seed(4))
    one_filter = ls_denoise(near, far)
    windowed = ls_denoise(near, far, segment_frames=60)
    assert windowed.pow(2).mean() < 0.5 * one_filter.pow(2).mean()  # the gain changes: one filter cannot follow it
    cross, power = cross_spectra(near, far)
    assert torch.allclose(ls_denoise(near, far, transfer=cross / power), one_filter, atol=1e-4)


def test_dataset_channels_ls_win_and_ls_global(tmp_path):
    import torchaudio

    from moviad.datasets.dcase2026_task2 import DCASE2026Task2Dataset

    machine = tmp_path / "toy" / "toy"
    machine.mkdir(parents=True)
    for i, name in enumerate(["train_normal_0000_a", "train_normal_0001_a", "test_anomaly_0000_a"]):
        generator = torch.Generator().manual_seed(i)
        far = torch.randn(SR, generator=generator)
        torchaudio.save(str(machine / f"section_00_source_{name}.wav"), torch.stack([far + tone(500), far]), SR)
    for channel in ("ls_win", "ls_global"):
        train = DCASE2026Task2Dataset(tmp_path, "toy", "train", channel=channel, target_sample_rate=16000)
        test = DCASE2026Task2Dataset(tmp_path, "toy", "test", channel=channel, target_sample_rate=16000)
        assert train[0].shape == (SR,) and test[0][0].shape == (SR,)
        assert train[0].pow(2).mean() < 0.9  # the shared noise (variance 1) is largely removed


def test_extra_statistics_and_alternative_scorers():
    clips = torch.stack([tone(1000, seed=i) + 0.1 * torch.randn(SR, generator=torch.Generator().manual_seed(i)) for i in range(30)])
    encoder = SubbandStatsEncoder(n_bands=8, stats=("mean", "std", "kurtosis", "q10", "q90", "corr"))
    assert encoder(clips[:2])[0].shape == (2, 8 * 5 + 7)
    shifted = torch.stack([tone(3000, seed=i) for i in range(5)])
    for scorer in ("mahalanobis", "knn", "iforest"):
        model = SubbandMahalanobis(encoder, torch.device("cpu"), scorer=scorer)
        model.fit(DataLoader(TensorDataset(clips), batch_size=10))
        model.eval()
        assert model(shifted)[1].mean() > model(clips[:5])[1].mean(), scorer


def test_unknown_statistic_or_scorer_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        SubbandStatsEncoder(stats=("median",))
    with pytest.raises(ValueError):
        SubbandMahalanobis(SubbandStatsEncoder(), torch.device("cpu"), scorer="svm")


def test_unlabelled_evaluation_test_clips_are_loaded_for_any_requested_domain(tmp_path):
    import torchaudio

    from moviad.datasets.dcase2026_task2 import DCASE2026Task2Dataset

    machine = tmp_path / "toy" / "test"
    machine.mkdir(parents=True)
    for index in (1, 2):
        torchaudio.save(str(machine / f"section_00_{index:04d}.wav"), torch.randn(2, SR), SR)
    dataset = DCASE2026Task2Dataset(tmp_path, "toy", "test", domain="source", target_sample_rate=16000)
    assert len(dataset) == 2
    assert dataset[0][1] == -1 and dataset[0][2].endswith("section_00_0001.wav")
