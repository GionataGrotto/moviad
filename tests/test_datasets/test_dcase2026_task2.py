from pathlib import Path

import pytest

from moviad.datasets.dcase2026_task2 import parse_dcase_filename


def test_parse_dcase_filename():
    record = parse_dcase_filename(
        Path("section_00_target_test_anomaly_0001_noAttribute.wav"), "fan"
    )
    assert record.machine_type == "fan"
    assert record.section == "00"
    assert record.domain == "target"
    assert record.split == "test"
    assert record.label == 1


def test_parse_rejects_unrelated_file():
    with pytest.raises(ValueError):
        parse_dcase_filename(Path("attributes_00.csv"), "fan")
