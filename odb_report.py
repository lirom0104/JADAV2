"""Summarize all ten scenes at iteration 30000, optionally compare a baseline."""

import argparse
import csv
import json
import math
from pathlib import Path


SCENES = (
    "Building", "DailyStuff", "Dimsum", "Ebike", "IronIngot", "LandScape",
    "Parterre", "RoadBlock", "RotaryKiln", "Truck",
)
METRICS = (
    "color_PSNR", "thermal_PSNR", "color_SSIM", "thermal_SSIM",
    "color_LPIPS", "thermal_LPIPS",
)


def read_scene(root, scene):
    path = root / scene / "results.json"
    result = json.loads(path.read_text())["ours_30000"]
    values = {key: float(result[key]) for key in METRICS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"Non-finite metric in {path}")
    return values


def summarize(candidate_root):
    rows = {scene: read_scene(candidate_root, scene) for scene in SCENES}
    return {
        "iteration": 30000,
        "scene_count": len(SCENES),
        "candidate_root": str(candidate_root.resolve()),
        "mean": {key: sum(row[key] for row in rows.values()) / len(SCENES)
                 for key in METRICS},
        "per_scene": rows,
    }


def compare(baseline_root, candidate_root):
    rows = {}
    for scene in SCENES:
        baseline = read_scene(baseline_root, scene)
        candidate = read_scene(candidate_root, scene)
        rows[scene] = {
            key: {"baseline": baseline[key], "candidate": candidate[key],
                  "delta": candidate[key] - baseline[key]}
            for key in METRICS
        }
    means = {
        key: {
            column: sum(rows[scene][key][column] for scene in SCENES) / len(SCENES)
            for column in ("baseline", "candidate", "delta")
        }
        for key in METRICS
    }
    return {
        "iteration": 30000,
        "scene_count": len(SCENES),
        "baseline_root": str(baseline_root.resolve()),
        "candidate_root": str(candidate_root.resolve()),
        "mean": means,
        "psnr_threshold_met": max(means[k]["delta"] for k in METRICS[:2]) >= 0.2,
        "all_mean_metrics_improve": all(
            means[k]["delta"] < 0 if k.endswith("LPIPS") else means[k]["delta"] > 0
            for k in METRICS
        ),
        "improved_scene_counts": {
            k: sum(rows[s][k]["delta"] < 0 if k.endswith("LPIPS")
                   else rows[s][k]["delta"] > 0 for s in SCENES)
            for k in METRICS
        },
        "per_scene": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="Optional baseline for comparison")
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.candidate)
    summary_path = args.candidate / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    with (args.candidate / "metrics_30000.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("scene", *METRICS))
        for scene, values in summary["per_scene"].items():
            writer.writerow((scene, *(values[key] for key in METRICS)))
        writer.writerow(("MEAN", *(summary["mean"][key] for key in METRICS)))
    print("Ten-scene means at iteration 30000:")
    for key, value in summary["mean"].items():
        print(f"{key}: {value:.6f}")
    print(summary_path)

    if args.baseline is not None:
        report = compare(args.baseline, args.candidate)
        path = args.candidate / "comparison_30000.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        for key, values in report["mean"].items():
            print(f"{key}: {values['baseline']:.6f} -> {values['candidate']:.6f} "
                  f"({values['delta']:+.6f})")
        print(f"PSNR threshold met: {report['psnr_threshold_met']}")
        print(f"All mean metrics improve: {report['all_mean_metrics_improve']}")
        print(path)
