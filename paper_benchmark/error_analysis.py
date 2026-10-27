"""Where does a detector fail on a machine? Breaks the source-domain test clips down by attribute.

    python paper_benchmark/error_analysis.py --scores <run>/teams/moviad/patchcore --machine ToyCarEmu \
        --attributes <dev>/ground_truth_attributes

For every attribute value (e.g. ``car_B1_spd_3_mic_2``): the number of normal / anomalous clips, the AUC inside
the group, and the mean rank of its normal and anomalous clips. Then the anomalous clips that score lowest
(the missed ones) and the normal clips that score highest (false alarms) of the whole machine.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from fuse_scores import normalise, read_machine_scores


def read_attributes(path: Path) -> dict[str, str]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {row[0]: row[1] for row in csv.reader(handle) if len(row) >= 2}


def attribute_of(file: str, attributes: dict[str, str]) -> str:
    """The attribute string of a clip: the part of its ground-truth name after the clip index."""
    name = attributes.get(file, file)
    name = name[:-4] if name.endswith(".wav") else name
    parts = name.split("_")
    index = next((i for i, part in enumerate(parts) if part.isdigit() and i >= 4), None)
    return "_".join(parts[index + 1 :]) if index is not None else name


def analyse(scores: dict[str, float], attributes: dict[str, str], domain: str = "source") -> dict:
    files = [f for f in sorted(scores) if f"_{domain}_" in f]
    values = np.array([scores[f] for f in files])
    ranks = dict(zip(files, normalise(values, "rank")))
    labels = {f: int("_anomaly_" in f) for f in files}
    groups: dict[str, list[str]] = {}
    for f in files:
        groups.setdefault(attribute_of(f, attributes), []).append(f)
    table = []
    for attribute, members in sorted(groups.items()):
        y = [labels[f] for f in members]
        auc = roc_auc_score(y, [scores[f] for f in members]) if len(set(y)) == 2 else None
        normal = [ranks[f] for f in members if not labels[f]]
        anomalous = [ranks[f] for f in members if labels[f]]
        table.append({"attribute": attribute, "normal": len(normal), "anomaly": len(anomalous), "auc": auc,
                      "normal_rank": float(np.mean(normal)) if normal else None,
                      "anomaly_rank": float(np.mean(anomalous)) if anomalous else None})
    anomalies = sorted((ranks[f], f) for f in files if labels[f])
    normals = sorted(((ranks[f], f) for f in files if not labels[f]), reverse=True)
    return {"groups": table, "missed": anomalies, "false_alarms": normals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, required=True, help="a score folder (…/teams/moviad/<method>)")
    parser.add_argument("--machine", required=True)
    parser.add_argument("--attributes", type=Path, required=True, help="the ground_truth_attributes folder or one of its csv files")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    scores = read_machine_scores(args.scores)[args.machine]
    path = args.attributes
    if path.is_dir():
        path = path / f"ground_truth_{args.machine}_section_00_test.csv"
    result = analyse(scores, read_attributes(path))
    print(f"{'attribute':40s} normal anomaly   AUC  rank(normal) rank(anomaly)")
    for g in result["groups"]:
        auc = "  -  " if g["auc"] is None else f"{g['auc']:.2f}"
        nr = "-" if g["normal_rank"] is None else f"{g['normal_rank']:.2f}"
        ar = "-" if g["anomaly_rank"] is None else f"{g['anomaly_rank']:.2f}"
        print(f"{g['attribute']:40s} {g['normal']:6d} {g['anomaly']:7d}  {auc:>5s}  {nr:>11s}  {ar:>12s}")
    print("\nmissed anomalies (lowest rank):")
    for rank, f in result["missed"][: args.top]:
        print(f"  {rank:.2f}  {f}")
    print("false alarms (highest-ranked normal clips):")
    for rank, f in result["false_alarms"][: args.top]:
        print(f"  {rank:.2f}  {f}")


if __name__ == "__main__":
    main()
