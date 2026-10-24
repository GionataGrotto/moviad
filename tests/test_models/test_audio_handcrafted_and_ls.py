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
