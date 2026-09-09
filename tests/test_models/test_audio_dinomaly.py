import pytest
import torch


pytest.importorskip("timm")


def test_audio_dinomaly_waveform_contract_and_training_step():
    from moviad.models.audio.dinomaly import AudioDinomaly
    from moviad.models.dinomaly.dinomaly import DinomalyTrainArgs

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = AudioDinomaly(
            "deit_tiny_patch16_224.fb_in1k",
            device="cpu",
            pretrained=False,
        )
        from moviad.utilities.faithfulness import audio_spectro_transform

        assert audio_spectro_transform(model) is model.spectrogram_transform
        waveform = torch.randn(1, 4096)

        model.eval()
        anomaly_map, anomaly_score, temporal_score = model(waveform)
        assert anomaly_map.ndim == 4
        assert anomaly_map.shape[0:2] == (1, 1)
        assert anomaly_score.shape == (1,)
        assert temporal_score.ndim == 2

        model.train()
        args = DinomalyTrainArgs(batch_size=1, epochs=1)
        args.total_iters = 1
        args.init_train(model)
        loss = model.train_step((waveform,), args)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))
    finally:
        torch.set_num_threads(previous_threads)
