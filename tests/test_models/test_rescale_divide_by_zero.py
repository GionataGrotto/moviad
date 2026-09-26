"""Regression tests: a completely uniform score map used to turn into NaN.

``rescale`` normalizes a score/heatmap to [0, 1] via ``(x - min) / (max -
min)``. Whenever every value is identical (a degenerate but real case: e.g.
a memory bank distance that comes out constant, or a fully silent audio
clip), ``max - min`` is 0 and every value becomes NaN, silently corrupting
the anomaly map, the derived image/temporal scores, and any metric computed
from them downstream (sklearn's roc_auc_score raises on NaN input rather
than failing quietly, but the resulting crash points nowhere near this
function). These pin the fix: a uniform input returns a finite, valid map
instead of NaN.
"""

import torch

from moviad.models.audio.cfa.cfa import CFA
from moviad.models.audio.patchcore.anomaly_map import AnomalyMapGenerator as AudioAnomalyMapGenerator
from moviad.models.cfa.cfa import CFA as VisualCFA
from moviad.models.patchcore.anomaly_map import AnomalyMapGenerator as VisualAnomalyMapGenerator


def test_audio_patchcore_rescale_handles_a_uniform_map():
    uniform = torch.full((2, 1, 4, 4), 3.0)

    result = AudioAnomalyMapGenerator.rescale(uniform)

    assert torch.isfinite(result).all()
    assert torch.equal(result, torch.zeros_like(uniform))


def test_visual_patchcore_rescale_handles_a_uniform_map():
    uniform = torch.full((2, 1, 4, 4), -1.5)

    result = VisualAnomalyMapGenerator.rescale(uniform)

    assert torch.isfinite(result).all()
    assert torch.equal(result, torch.zeros_like(uniform))


def test_audio_cfa_rescale_handles_a_uniform_map():
    uniform = torch.zeros(2, 8, 8)

    result = CFA.rescale(uniform)

    assert torch.isfinite(result).all()
    assert torch.equal(result, torch.zeros_like(uniform))


def test_visual_cfa_rescale_handles_a_uniform_map():
    uniform = torch.zeros(2, 8, 8)

    result = VisualCFA.rescale(uniform)

    assert torch.isfinite(result).all()
    assert torch.equal(result, torch.zeros_like(uniform))


def test_rescale_still_normalizes_a_varying_map_to_unit_range():
    """The guard must not change behaviour on the common, non-degenerate case."""
    varying = torch.tensor([[[0.0, 5.0], [10.0, 2.5]]])

    result = AudioAnomalyMapGenerator.rescale(varying)

    assert torch.isclose(result.min(), torch.tensor(0.0))
    assert torch.isclose(result.max(), torch.tensor(1.0))
