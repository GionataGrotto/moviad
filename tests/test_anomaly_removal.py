import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_benchmark"))

import anomaly_removal as ar  # noqa: E402
from moviad.utilities.audio.beats_feature_extractor import BEATsFbank, BEATsFeatureExtractor  # noqa: E402

SR = 16000


def make_clip(seconds=4.0, seed=0):
    """Stationary noise + a 1 kHz tone between 1.5 s and 2.5 s: ``(mixture, background, tone)``."""
    generator = torch.Generator().manual_seed(seed)
    n = int(SR * seconds)
    background = torch.randn(n, generator=generator) * 0.05
    t = torch.arange(n) / SR
    tone = torch.sin(2 * torch.pi * 1000 * t) * 0.5 * ((t > 1.5) & (t < 2.5)).float()
    return background + tone, background, tone


def tone_mask(frames, bins=128, low=40, high=80):
    """The region of the tone in the filterbank grid: the frames of 1.5-2.5 s and the mel bins around 1 kHz."""
    mask = torch.zeros(frames, bins)
    mask[145:255, low:high] = 1.0
    return mask


def test_the_filterbank_grid_maps_onto_the_stft_grid_in_time_and_frequency():
    frames = BEATsFeatureExtractor.fbank_shape(SR * 4)[0]
    grid = ar.to_stft_grid(torch.zeros(frames, 128), 401)
    assert grid.shape == (ar.N_FFT // 2 + 1, 401)
    band = torch.zeros(frames, 128)
    band[:, 100] = 1.0  # a high mel bin
    grid = ar.to_stft_grid(band, 401)
    peak = int(grid.mean(dim=1).argmax())
    assert peak * SR / ar.N_FFT > 3000  # it lands on a high frequency
    column = torch.zeros(frames, 128)
    column[200, :] = 1.0  # one filterbank frame: samples 32000-32400, centred at 2.0125 s
    grid = ar.to_stft_grid(column, 401)
    assert abs(int(grid.mean(dim=0).argmax()) * ar.HOP / SR - 2.0125) < 0.02


def test_removing_the_masked_region_brings_the_audio_back_towards_the_background():
    mixture, background, tone = make_clip()
    frames = BEATsFeatureExtractor.fbank_shape(mixture.shape[0])[0]
    cleaned = ar.remove_anomaly(mixture, tone_mask(frames, low=35, high=85))
    assert cleaned.shape == mixture.shape
    assert ar.si_sdr(cleaned, background) > ar.si_sdr(mixture, background) + 5  # the tone is gone: several dB closer to the background
    assert cleaned.abs().max() < mixture.abs().max()


def test_an_empty_mask_changes_nothing_and_the_processing_never_adds_energy():
    mixture, _, _ = make_clip()
    frames = BEATsFeatureExtractor.fbank_shape(mixture.shape[0])[0]
    untouched = ar.remove_anomaly(mixture, torch.zeros(frames, 128))
    assert torch.allclose(untouched, mixture, atol=1e-4)
    everything = ar.remove_anomaly(mixture, torch.ones(frames, 128))
    assert everything.pow(2).sum() <= mixture.pow(2).sum() * 1.001


def test_si_sdr_is_scale_invariant_and_high_for_a_copy():
    generator = torch.Generator().manual_seed(0)
    reference = torch.randn(8000, generator=generator)
    assert ar.si_sdr(reference * 3.0, reference) > 60
    noisy = reference + 0.3 * torch.randn(8000, generator=generator)  # about 10 dB
    assert 8 < ar.si_sdr(noisy, reference) < 12


def test_the_mask_is_the_map_above_the_threshold_with_soft_edges():
    anomaly_map = torch.zeros(40, 20)
    anomaly_map[10:20, 5:10] = 2.0
    mask = ar.maps_to_mask(anomaly_map, tau=1.0)
    assert mask[15, 7] == 1.0 and mask[35, 15] == 0.0
    assert 0 < mask[8, 7] < 1  # a soft edge around the region
    assert ar.maps_to_mask(anomaly_map, tau=5.0).sum() == 0


def test_the_examples_are_anomalous_clips_of_different_classes():
    categories = {1: "dog", 2: "dog", 3: "frog", 5: "sheep", 6: "frog"}
    assert ar.pick_examples(categories, [0, 1, 1, 1, 0, 1, 1], 3) == [1, 3, 5]
    assert ar.pick_examples(categories, [0, 1, 1, 1, 0, 1, 1], 2) == [1, 3]
    assert ar.percentile_of(0.5, np.array([0.1, 0.2, 0.9, 1.0])) == 0.5


class EnergyDetector:
    """A stand-in for a model: the map is the energy of the filterbank above the median of its frequency, the score the largest value."""

    def __init__(self):
        self.fbank = BEATsFbank()

    def __call__(self, waveforms):
        spec = self.fbank(waveforms)[:, 0]  # (B, frames, 128)
        maps = (spec - spec.median(dim=1, keepdim=True).values).unsqueeze(1)
        return maps, maps.flatten(1).amax(dim=1), None


def test_the_experiment_scores_the_cleaned_clip_with_every_detector_and_against_the_true_background(tmp_path):
    mixture, background, tone = make_clip()
    frames = BEATsFeatureExtractor.fbank_shape(mixture.shape[0])[0]
    detector = ar.Detector("energy", EnergyDetector(), torch.device("cpu"))
    normal_maps, normal_scores = [], []
    for seed in range(1, 31):  # enough normal clips for the percentile not to be decided by a few of them
        _, normal, _ = make_clip(seed=seed)
        m, s = detector.map_and_score(normal)
        normal_maps.append(m)
        normal_scores.append(s)
    pixels = np.concatenate([m.ravel() for m in normal_maps])
    detector.thresholds = {0.9: float(np.quantile(pixels, 0.9)), 0.99: float(np.quantile(pixels, 0.99))}
    detector.normal_scores = np.array(normal_scores)
    saved = {}
    truth = tone_mask(frames, low=30, high=90).numpy()
    rows = ar.process_example({"energy": detector}, mixture, tone, truth, "demo", quantiles=(0.99, 0.9), modes=("replace", "cap"),
                              save=lambda v, w: saved.update({v: w}), extras=())
    assert {(r["mask"], r["threshold"]) for r in rows} == {("energy", "q0.99"), ("energy", "q0.9"), ("energy", "area"), ("oracle", "truth")}
    assert len(rows) == 8 and set(saved) == {"energy", "oracle"}  # saved: the main threshold (the first) and the oracle
    for row in rows:
        if row["mask"] == "oracle" or row["threshold"] == "q0.99":
            continue
        assert row["anomaly_removed_db"] > 6 and row["si_sdr_gain"] > 3
        assert row["pct_energy_mixture"] == 1.0 and row["pct_energy_cleaned"] <= row["pct_energy_mixture"]
    oracle = [r for r in rows if r["mask"] == "oracle" and r["mode"] == "replace"][0]
    assert oracle["anomaly_removed_db"] > 10 and oracle["background_error_db"] < -5
    assert oracle["score_energy_cleaned"] < 0.6 * oracle["score_energy_mixture"]
    tight = tone_mask(frames, low=41, high=47).numpy()  # the pixels that really hold the tone
    assert ar.localisation_metrics(detector.map_and_score(mixture)[0], tight, detector.thresholds[0.99])["pixel_auc"] > 0.9
    assert all(np.isfinite(r["control_error_db"]) and 0 <= r["pct_energy_control"] <= 1 for r in rows)
    assert ar.summary_table(rows, ["energy"], "replace").count(chr(10)) == 5
    assert ar.aggregate_table(rows, ["energy"]).count(chr(10)) == 9


def _toy_detector(limit_quantile=0.99):
    detector = ar.Detector("energy", EnergyDetector(), torch.device("cpu"))
    maps, scores = [], []
    for seed in range(1, 21):
        m, s = detector.map_and_score(make_clip(seed=seed)[1])
        maps.append(m)
        scores.append(s)
    pixels = np.concatenate([m.ravel() for m in maps])
    detector.thresholds = {0.95: float(np.quantile(pixels, 0.95))}
    detector.normal_scores = np.array(scores)
    return detector


def test_every_family_of_methods_gives_rows_with_the_control_and_the_files_to_save():
    mixture, background, tone = make_clip()
    frames = BEATsFeatureExtractor.fbank_shape(mixture.shape[0])[0]
    detector = _toy_detector()
    saved = {}
    rows = ar.process_example({"energy": detector}, mixture, tone, tone_mask(frames, low=30, high=90).numpy(), "demo", quantiles=(0.95,),
                              modes=("sample", "nmf"), save=lambda k, w: saved.update({k: w}), main_mode="sample", iterations=2)
    thresholds = {r["threshold"] for r in rows}
    assert {"q0.95", "area", "rel0.5", "q0.95+fine", "q0.95x2", "truth"} <= thresholds
    assert {"energy", "energy+fine", "energy+iter", "oracle"} <= set(saved)
    assert {r["mode"] for r in rows} == {"sample", "nmf"} and all(k in ar.AFTER_FILES for k in ("patchcore", "dinomaly", "patchcore+fine", "dinomaly+fine", "fused", "patchcore+iter", "dinomaly+iter", "audiosep", "oracle"))
    oracle = [r for r in rows if r["mask"] == "oracle" and r["mode"] == "nmf"][0]
    assert oracle["anomaly_removed_db"] > 8


def test_a_separator_is_subtracted_and_a_normal_clip_is_left_alone_by_the_iterative_removal():
    mixture, background, tone = make_clip()
    frames = BEATsFeatureExtractor.fbank_shape(mixture.shape[0])[0]
    detector = _toy_detector()
    prompts = []

    def perfect(waveform, prompt):  # a separator that knows the tone
        prompts.append(prompt)
        return tone

    rows = ar.process_example({"energy": detector}, mixture, tone, tone_mask(frames, low=30, high=90).numpy(), "demo", quantiles=(0.95,),
                              modes=("sample",), extras=("iterative",), separator=perfect, prompt="car horn", iterations=2)
    audiosep = [r for r in rows if r["mask"] == "audiosep"][0]
    assert audiosep["mode"] == "subtract" and audiosep["anomaly_removed_db"] > 40 and prompts == ["car horn", "car horn"]  # the clip and its control
    iterative = [r for r in rows if r["threshold"] == "q0.95x2"][0]
    normal_clip = make_clip(seed=99)[1]
    cleaned, union = ar.iterative_remove(detector, normal_clip, detector.thresholds[0.95], "sample", 2, float("inf"))
    assert torch.equal(cleaned, normal_clip) and union.sum() == 0  # the score is below the limit: nothing to remove
    assert iterative["anomaly_removed_db"] > 3


def test_the_fine_mask_keeps_only_what_sticks_out_and_the_fusion_needs_both_models():
    mixture, _, tone = make_clip()
    frames = BEATsFeatureExtractor.fbank_shape(mixture.shape[0])[0]
    coarse = ar.to_stft_grid(tone_mask(frames, low=20, high=110), ar.stft_frames(mixture.shape[0]))
    fine = ar.refine_mask(mixture, coarse, margin_db=6.0)
    assert fine.mean() < coarse.mean() and fine.mean() > 0  # the tone (1.5-2.5 s, around 1 kHz) is inside the large coarse mask
    maps = {"a": np.array([[2.0, 0.5], [2.0, 2.0]]), "b": np.array([[2.0, 2.0], [0.5, 2.0]])}
    taus = {"a": 1.0, "b": 1.0}
    assert (ar.fuse_maps(maps, taus, "and") > 1).tolist() == [[True, False], [False, True]]
    assert (ar.fuse_maps(maps, taus, "avg") > 1).all()
    assert ar.normalise_map(np.array([2.0]), 2.0)[0] == 1.0 and ar.normalise_map(np.array([0.0]), -1.0)[0] == 2.0
    assert 0.0 < ar.adaptive_threshold(np.array([0.0, 4.0]), 0.5) == 2.0


def test_the_removal_metrics_tell_an_untouched_clip_from_a_clean_one():
    mixture, background, tone = make_clip()
    untouched = ar.removal_metrics(mixture, mixture, tone)
    assert abs(untouched["anomaly_left"] - 1) < 1e-6 and abs(untouched["anomaly_removed_db"]) < 1e-3
    clean = ar.removal_metrics(background, mixture, tone)
    assert clean["anomaly_removed_db"] > 40 and clean["background_error_db"] < -40
    half = ar.removal_metrics(background + 0.5 * tone, mixture, tone)
    assert abs(half["anomaly_removed_db"] - 6.02) < 0.1


def test_replace_fills_the_masked_cells_with_the_background_and_cap_only_lowers_them():
    generator = torch.Generator().manual_seed(3)
    background = torch.randn(SR * 4, generator=generator) * 0.05
    frames = BEATsFeatureExtractor.fbank_shape(background.shape[0])[0]
    mask = tone_mask(frames, low=35, high=85)
    quiet_hole = background.clone()
    quiet_hole[24000:40000] *= 0.1  # a region quieter than the background: cap leaves it, replace raises it
    replaced = ar.remove_anomaly(quiet_hole, mask, mode="replace")
    capped = ar.remove_anomaly(quiet_hole, mask, mode="cap")
    assert replaced[24000:40000].pow(2).mean() > capped[24000:40000].pow(2).mean()
    sampled = ar.remove_anomaly(quiet_hole, mask, mode="sample")
    assert sampled.shape == quiet_hole.shape and torch.isfinite(sampled).all()
    mixture, background_only, tone = make_clip()
    cleaned = ar.remove_anomaly(mixture, tone_mask(frames, low=35, high=85), mode="sample")
    assert ar.removal_metrics(cleaned, mixture, tone)["anomaly_removed_db"] > 8
    nmf = ar.remove_anomaly(quiet_hole, mask, mode="nmf")
    assert nmf.shape == quiet_hole.shape and torch.isfinite(nmf).all()
    with pytest.raises(ValueError):
        ar.remove_anomaly(quiet_hole, mask, mode="other")


# ---------------------------------------------------------------------------------------------- saved models
class _LevelExtractor:
    """Waveform [B, 32] -> a feature map [B, 6, 8, 4] whose level is the waveform (as in the PatchCore target tests)."""

    quantized = False

    def __init__(self):
        self.device = torch.device("cpu")

    def to(self, device):
        return self

    def __call__(self, batch):
        grid = batch.reshape(batch.shape[0], 1, 8, 4)
        return [grid.expand(-1, 6, -1, -1).contiguous()]


def _patchcore():
    from moviad.models.audio.patchcore.patchcore import PatchCore

    model = PatchCore(device=torch.device("cpu"), input_size=(8, 4), feature_extractor=_LevelExtractor(), memory_bank_size=64, num_neighbors=3,
                      blur=False, band_banks=True, score_standardization="band", holdout_fraction=0.25)
    model.feature_pooler = torch.nn.Identity()
    return model


def test_a_saved_patchcore_scores_exactly_like_the_one_that_was_trained(tmp_path):
    clips = torch.utils.data.DataLoader(torch.randn(40, 32) * 0.1, batch_size=4)
    trained = _patchcore()
    trained.fit(clips)
    trained.eval()
    meta = ar.checkpoint_meta("patchcore", {"backbone": "BEATs", "patchcore_memory_fraction": 0.01}, "siren", 0.0, 1, 40)
    ar.save_checkpoint("patchcore", trained, tmp_path, meta)
    assert (tmp_path / "patchcore.pt").exists() and not any(k.startswith("feature_extractor.") for k in torch.load(tmp_path / "patchcore.pt"))

    fresh = _patchcore()
    assert ar.load_checkpoint("patchcore", fresh, tmp_path, meta) is True
    fresh.eval()
    test = torch.randn(6, 32) * 0.1
    test[3] += 2.0  # an anomalous clip
    for a, b in zip(trained(test), fresh(test)):
        assert torch.allclose(a, b)


def test_a_saved_dinomaly_restores_its_trainable_part_and_a_changed_setting_means_training_again(tmp_path):
    from types import SimpleNamespace

    def model():
        return SimpleNamespace(trainable=torch.nn.ModuleList([torch.nn.Linear(3, 3), torch.nn.Linear(3, 2)]))

    trained = model()
    meta = ar.checkpoint_meta("dinomaly", {"backbone": "BEATs", "epochs": 12}, "drilling", -6.0, 1, 500)
    ar.save_checkpoint("dinomaly", trained, tmp_path, meta)
    fresh = model()
    assert ar.load_checkpoint("dinomaly", fresh, tmp_path, meta) is True
    for a, b in zip(trained.trainable.parameters(), fresh.trainable.parameters()):
        assert torch.equal(a, b)
    other = ar.checkpoint_meta("dinomaly", {"backbone": "BEATs", "epochs": 4}, "drilling", -6.0, 1, 500)  # trained for fewer epochs
    assert ar.load_checkpoint("dinomaly", model(), tmp_path, other) is False
    assert ar.load_checkpoint("dinomaly", model(), tmp_path / "missing", meta) is False


def test_the_base_config_gets_the_defaults_of_the_benchmark(tmp_path):
    """The DCASE config has no memory_bank_size: the models are built here without the runners that add the defaults (a KeyError on the server)."""
    import json

    from run_beats_benchmark import build_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"device": "auto", "batch_size": 4, "num_workers": 0, "knn_beats_checkpoint": "x.pt", "knn_beats_code_dir": "beats",
                                "mimii": {"dataset_path": "m"}, "envmix": {"urban_path": "u", "esc50_path": "e", "seeds": [0]}}), encoding="utf-8")
    base = ar.load_base_config(path)
    for variant in ("patchcore_beats_std_1pct", "dinomaly_beats"):
        config = build_config(base, "envmix", variant, "out", [1], [1])
        assert config["memory_bank_size"] and config["batch_size"] == 4 and config["backbone"] == "BEATs" and config["envmix"]["esc50_path"] == "e"
    assert build_config(base, "envmix", "patchcore_beats_std_1pct", "out", [1], [1])["patchcore_memory_fraction"] == 0.01


def test_a_wav_is_written_with_the_standard_library_and_reads_back(tmp_path):
    import wave

    t = torch.arange(16000) / SR
    tone = torch.sin(2 * torch.pi * 440 * t)
    ar.save_wav(tmp_path / "sub" / "tone.wav", tone, scale=0.5)
    with wave.open(str(tmp_path / "sub" / "tone.wav"), "rb") as handle:
        assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()) == (1, 2, 16000, 16000)
        data = np.frombuffer(handle.readframes(16000), dtype=np.int16)
    assert abs(data.max() / 32767 - 0.5) < 0.01  # the scale is applied
    ar.save_wav(tmp_path / "loud.wav", tone * 10, scale=1.0)  # clipped, never wrapped around
    with wave.open(str(tmp_path / "loud.wav"), "rb") as handle:
        assert np.frombuffer(handle.readframes(16000), dtype=np.int16).max() == 32767
