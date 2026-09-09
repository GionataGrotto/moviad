import torch


def build_model():
    from moviad.models.audio.draem import AudioDRAEM

    return AudioDRAEM(
        device="cpu",
        image_size=(64, 64),
        base_width_reconstructive=4,
        base_width_discriminative=4,
    )


def test_audio_draem_waveform_contract_and_training_step():
    from moviad.models.draem.draem import DRAEMTrainArgs
    from moviad.utilities.faithfulness import audio_spectro_transform

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        model = build_model()

        assert audio_spectro_transform(model) is model.spectrogram_transform

        waveform = torch.randn(1, 4096)

        model.eval()
        with torch.no_grad():
            anomaly_map, anomaly_score, temporal_score = model(waveform)

        assert anomaly_map.ndim == 4
        assert anomaly_map.shape[0:2] == (1, 1)
        assert anomaly_score.shape == (1,)
        # (batch, time), like the temporal ground truth masks
        assert temporal_score.ndim == 2
        assert temporal_score.shape == (1, anomaly_map.shape[2])

        model.train()
        args = DRAEMTrainArgs(batch_size=1, epochs=1)
        args.init_train(model)
        loss = model.train_step((waveform,), args)

        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))
    finally:
        torch.set_num_threads(previous_threads)


def test_audio_draem_normalizes_spectrogram_to_unit_range():
    torch.manual_seed(0)
    model = build_model()
    waveform = torch.randn(2, 4096)

    images, original_size = model._to_image(waveform)

    assert images.shape == (2, 3, 64, 64)
    assert images.min() >= 0.0 and images.max() <= 1.0
    # every sample spans the full range after per-sample min-max normalisation
    per_sample = images.flatten(start_dim=1)
    assert torch.allclose(per_sample.amin(dim=1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(per_sample.amax(dim=1), torch.ones(2), atol=1e-6)
    assert len(original_size) == 2


def test_audio_draem_rejects_unsupported_arguments():
    import pytest

    from moviad.models.audio.draem import AudioDRAEM

    with pytest.raises(ValueError):
        AudioDRAEM(device="cpu", image_size=(16, 16))

    with pytest.raises(ValueError):
        AudioDRAEM(device="cpu", spectrogram_backbone="not-a-backbone")
