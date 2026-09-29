"""Audit the two full dataset means without mixing scene-specific variants.

This is an artifact/coverage report, not a reproducibility or quality gate.
Incomplete or inconsistent exports never receive a full-dataset mean.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from PIL import Image

from saved_settings import options
from cwgc_compare import direction, read_results

PRIMARY = ("color_PSNR", "color_SSIM", "color_LPIPS",
           "thermal_PSNR", "thermal_SSIM", "thermal_LPIPS")


def target_hashes(scene):
    result, sizes = {}, {}
    for modality in ("color", "thermal"):
        paths = sorted((scene / "test/ours_30000" / ("gt_" + modality)).glob("*.png"))
        if not paths:
            raise ValueError("Missing held-out targets for " + modality)
        for path in paths:
            with Image.open(path) as image:
                key = modality + "/" + path.name
                result[key] = hashlib.sha256(
                    image.convert("RGB").tobytes()).hexdigest()
                sizes[key] = image.size
    return result, sizes


def audit_dataset(candidate_root, supplied_root, dataset, mode="camera_only", source_snapshot=None):
    if mode != "camera_only":
        raise ValueError(f"Unknown experiment mode: {mode}")
    supplied = {k: v for k, v in read_results(supplied_root).items() if k[0] == dataset}
    candidate = {k: v for k, v in read_results(candidate_root).items() if k[0] == dataset}
    expected, available = set(supplied), set(candidate)
    issues, signatures, checked_sources, rows = [], {}, {}, {}
    if len(expected) != 10:
        issues.append("Supplied dataset must contain ten 30000-step scene results")
    for key in sorted(expected & available):
        scene = candidate_root / key[0] / key[1]
        try:
            metrics = candidate[key]
            if not all(m in metrics and math.isfinite(metrics[m]) for m in PRIMARY):
                raise ValueError("Missing or non-finite primary metric")
            verified = json.loads((scene / ".evaluation/verified.json").read_text())
            original_metrics = json.loads((scene / "results.json").read_text())["ours_30000"]
            if verified.get("metrics") != original_metrics:
                raise ValueError("Verified metrics differ from results.json")
            actual, actual_sizes = target_hashes(scene)
            if actual != verified.get("target_sha256"):
                raise ValueError("Current targets differ from verified target hashes")
            supplied_hashes, supplied_sizes = target_hashes(supplied_root / key[0] / key[1])
            if actual != supplied_hashes or actual_sizes != supplied_sizes:
                raise ValueError("Held-out targets differ from the supplied baseline")
            model, optimization = options(candidate_root, key, "cfg_args"), options(candidate_root, key)
            if model is None or optimization is None:
                raise ValueError("Missing saved training settings")
            if optimization.get("iterations") != 30000:
                raise ValueError("Expected a 30000-step endpoint")
            if mode == "camera_only":
                if model.get("use_camera_calibration") is not True:
                    raise ValueError("Camera-only mode requires camera calibration")
                # Historical CameraOnly configs retain removed switches. Missing
                # switches in new runs mean the removed features are absent.
                expected_settings = {"use_cwgc": False, "late_rmse_weight": 0.0, "late_lr_final_factor": 1.0,
                                     "near_camera_prune_ratio": 0.0, "cmo_use_thermal_densification": False,
                                     "cmo_preserve_rgb_densification": False, "seed": 0}
                wrong = [name for name, value in expected_settings.items()
                         if optimization.get(name, value if name != "seed" else None) != value]
                if wrong:
                    raise ValueError("Camera-only settings mismatch: " + ", ".join(wrong))
            if not (scene / "point_cloud/iteration_30000/point_cloud.ply").is_file():
                raise ValueError("Missing 30000-step Gaussian export")
            source_path = Path(model["source_path"])
            if source_path.name != key[1] or source_path.parent.name != dataset:
                raise ValueError("Saved source path does not identify this scene")
            manifest = json.loads((scene / "run_manifest.json").read_text())
            hashes = manifest.get("source_sha256", {})
            if not hashes or "train.py" not in hashes:
                raise ValueError("Missing source provenance")
            source = Path(source_snapshot) if source_snapshot is not None else Path(manifest["source"])
            fingerprint = json.dumps(hashes, sort_keys=True)
            cache_key = (str(source), fingerprint)
            if cache_key not in checked_sources:
                changed = [name for name, value in hashes.items()
                           if not (source / name).is_file() or
                           hashlib.sha256((source / name).read_bytes()).hexdigest() != value]
                checked_sources[cache_key] = changed
            if checked_sources[cache_key]:
                raise ValueError("Frozen training source changed: " + ", ".join(checked_sources[cache_key]))
            model = {k: v for k, v in model.items() if k not in {"source_path", "model_path"}}
            signatures[key[1]] = {"model": model, "optimization": optimization,
                                  "source_sha256": hashes}
            rows[key[1]] = {m: {"candidate": metrics[m], "supplied": supplied[key][m],
                                  "oriented_gain": direction(m) * (metrics[m] - supplied[key][m])}
                           for m in PRIMARY}
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(key[1] + ": " + str(error))
    if signatures:
        first_name = next(iter(signatures))
        first = signatures[first_name]
        for name, signature in signatures.items():
            for group in first:
                a, b = first[group], signature[group]
                differing = [k for k in a.keys() | b.keys() if a.get(k) != b.get(k)]
                if differing:
                    issues.append(name + ": " + group + " differs from " + first_name +
                                  " in " + ", ".join(sorted(differing)))
    missing = sorted(k[1] for k in expected - available)
    unexpected = sorted(k[1] for k in available - expected)
    complete = len(expected) == 10 and not missing and not unexpected and not issues
    report = {"candidate_root": str(candidate_root), "scene_count": len(rows),
              "missing_scenes": missing, "unexpected_scenes": unexpected,
              "issues": issues, "complete_uniform_verified_exports": complete,
              "per_scene": rows, "full_dataset_mean": None}
    if complete:
        report["full_dataset_mean"] = {
            m: {field: statistics.mean(row[m][field] for row in rows.values())
                for field in ("candidate", "supplied", "oriented_gain")}
            for m in PRIMARY}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgbt-root", type=Path, required=True)
    parser.add_argument("--thermo-root", type=Path, required=True)
    parser.add_argument("--supplied", type=Path, default=Path("output/JADA_batch"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("camera_only",), default="camera_only")
    parser.add_argument("--source-snapshot", type=Path,
                        help="Verify historical source hashes against a preserved source directory")
    args = parser.parse_args()
    report = {"scope": "Fixed single-seed full-dataset exports; paired seed evidence remains separate",
              "quality_acceptance": "Not determined by this coverage/provenance report",
              "mode": args.mode,
              "datasets": {dataset: audit_dataset(root, args.supplied, dataset, args.mode, args.source_snapshot)
                           for dataset, root in (("RGBT-Scenes", args.rgbt_root),
                                                 ("ThermoScenes1_3dgs", args.thermo_root))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
