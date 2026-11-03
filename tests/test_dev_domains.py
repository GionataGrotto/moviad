import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_benchmark"))

from dev_domains import dev_domain_metrics  # noqa: E402


def write(folder: Path, machine: str, source_good: bool, target_good: bool) -> None:
    rows = []
    for domain, good in (("source", source_good), ("target", target_good)):
        for i in range(10):
            label = i % 2
            score = label + 0.1 * i if good else (1 - label) + 0.1 * i
            rows.append(f"section_00_{domain}_test_{'anomaly' if label else 'normal'}_{i:04d}.wav,{score}\n")
    (folder / f"anomaly_score_{machine}_section_00_test.csv").write_text("".join(rows))


def test_source_target_and_official_score(tmp_path):
    write(tmp_path, "m1", True, True)
    write(tmp_path, "m2", True, False)
    metrics = dev_domain_metrics(tmp_path)
    assert set(metrics["source"]) == {"m1", "m2"}
    assert metrics["source"]["m1"] == 1.0 and metrics["target"]["m2"] < 0.5
    assert 0 < metrics["official"] < 100


def test_a_source_only_run_has_no_official_score(tmp_path):
    rows = [f"section_00_source_test_{'anomaly' if i % 2 else 'normal'}_{i:04d}.wav,{i}\n" for i in range(10)]
    (tmp_path / "anomaly_score_m1_section_00_test.csv").write_text("".join(rows))
    assert dev_domain_metrics(tmp_path)["official"] is None
