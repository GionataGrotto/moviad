def test_import_audio_datasets():
    from moviad.datasets.audio_dataset import AudioAnomalyDataset, SpectrogramBinarizer
    from moviad.datasets.mimi_dataset import MIMIDataset

    assert AudioAnomalyDataset is not None
    assert SpectrogramBinarizer is not None
    assert MIMIDataset is not None


def test_import_audio_models_and_trainers():
    from moviad.models.vad_model import VADModel
    from moviad.backbones.clap import Cnn14
    from moviad.models.audio.cfa.cfa import CFA
    from moviad.models.audio.padim.padim import Padim
    from moviad.models.audio.patchcore.patchcore import PatchCore
    from moviad.models.audio.stfpm.stfpm import STFPM
    from moviad.trainers.audio.trainer_patchcore import TrainerPatchCore
    from moviad.utilities.audio import AudioEvaluator, AudioFeatureExtractor

    assert Cnn14 is not None
    assert CFA is not None
    assert Padim is not None
    assert PatchCore is not None
    assert STFPM is not None
    assert TrainerPatchCore is not None
    assert AudioEvaluator is not None
    assert AudioFeatureExtractor is not None
    assert issubclass(CFA, VADModel)
    assert issubclass(Padim, VADModel)
    assert issubclass(PatchCore, VADModel)
    assert issubclass(STFPM, VADModel)
