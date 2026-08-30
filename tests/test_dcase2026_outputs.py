from pathlib import Path

from paper_benchmark.run_dcase2026_task2 import _write_pairs


def test_dcase_output_is_headerless_and_sorted(tmp_path):
    output = tmp_path / "scores.csv"
    _write_pairs(output, ["b.wav", "a.wav"], [2.0, 1.0])
    assert output.read_text(encoding="utf-8") == "a.wav,1.0\nb.wav,2.0\n"
