"""Compare every reported scene metric, never replacing missing runs with zeros."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def read_results(root):
    records = {}
    for path in sorted(root.glob("*/*/results.json")):
        data = json.loads(path.read_text())
        if "ours_30000" in data:
            records[(path.parent.parent.name, path.parent.name)] = data["ours_30000"]
    temperature = root / "ThermoScenes1_3dgs/batch_test_evaluation_results.csv"
    if temperature.exists():
        with temperature.open() as stream:
            for row in csv.DictReader(stream):
                key = ("ThermoScenes1_3dgs", row["Scene"])
                if key in records and row["Path"].endswith("ours_30000"):
                    records[key].update({m: float(row[m]) for m in ("MAE", "MAE_roi")})
    return records


def direction(metric):
    return -1 if "LPIPS" in metric or "MAE" in metric else 1


def compare(candidate, baseline):
    common = sorted(candidate.keys() & baseline.keys())
    report = {}
    for dataset in sorted({key[0] for key in common}):
        keys = [key for key in common if key[0] == dataset]
        metric_names = set.union(*(set(baseline[key]) for key in keys))
        summary = {}
        for metric in sorted(metric_names):
            if not all(metric in candidate[key] and metric in baseline[key] for key in keys):
                summary[metric] = {"status": "missing", "improved": False}
                continue
            old = statistics.mean(baseline[key][metric] for key in keys)
            new = statistics.mean(candidate[key][metric] for key in keys)
            delta = direction(metric) * (new - old)
            summary[metric] = {"baseline": old, "candidate": new, "oriented_improvement": delta,
                               "relative_improvement": delta / max(abs(old), 1e-12), "improved": delta > 0}
        report[dataset] = {"scenes": [key[1] for key in keys], "scene_count": len(keys),
                           "all_metrics_improved": all(v["improved"] for v in summary.values()),
                           "metrics": summary,
                           "per_scene": {key[1]: {m: direction(m)*(candidate[key][m]-baseline[key][m])
                                                  for m in baseline[key] if m in candidate[key]} for key in keys}}
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, default=Path("output/JADA_batch"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(read_results(args.candidate), read_results(args.baseline))
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
