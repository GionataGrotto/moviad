import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_benchmark"))

from error_analysis import analyse, attribute_of  # noqa: E402


def test_groups_by_attribute_and_lists_missed_anomalies():
    files, attributes, scores = [], {}, {}
    for i, (attr, label, score) in enumerate([("A", 0, 0.1), ("A", 1, 0.9), ("B", 0, 0.5), ("B", 1, 0.2), ("B", 0, 0.3)]):
        name = f"section_00_source_test_{'anomaly' if label else 'normal'}_{i:04d}_{attr}.wav"
        files.append(name)
        attributes[name] = name[:-4]
        scores[name] = score
    result = analyse(scores, attributes)
    by = {g["attribute"]: g for g in result["groups"]}
    assert by["A"]["auc"] == 1.0 and by["B"]["auc"] == 0.0
    assert result["missed"][0][1] == files[3]  # the lowest-scored anomaly
    assert attribute_of(files[0], attributes) == "A"
