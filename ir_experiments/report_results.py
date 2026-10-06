#!/usr/bin/env python3
"""Report validated IR-kernel candidates against the fixed 30000-step baseline.

Incomplete runs are useful for design feedback but never satisfy acceptance.
The six means are scene-weighted, never weighted by each scene's view count.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from run_suite import (
    DEFAULT_BASELINE, METRICS, SCENES, finite_scalar, json_dump,
    sha256_file, utc_now, validate_artifacts,
)


def scene_metrics(root: Path, scene: str, iteration: int) -> dict[str, float]:
    raw = json.loads((root / scene / "results.json").read_text())[f"ours_{iteration}"]
    return {key: finite_scalar(raw[key]) for key in METRICS}


def report_montages(candidate: Path, baseline: Path, scene: str,
                    iteration: int, destination: Path) -> list[dict[str, Any]]:
    """Select RGB and IR median/worst PSNR-delta views independently.

    Every selection shows both modalities at that *same* view.  This makes
    thermal improvements and any RGB regressions directly inspectable.
    Native image pixels are preserved; no contrast scaling is applied.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return [{"error": "Pillow unavailable; montages not generated"}]
    candidate_views = json.loads((candidate / scene / "per_view.json").read_text())[f"ours_{iteration}"]
    baseline_views = json.loads((baseline / scene / "per_view.json").read_text())["ours_30000"]
    candidate_render = candidate / scene / "test" / f"ours_{iteration}"
    baseline_render = baseline / scene / "test" / "ours_30000"
    records = []
    destination.mkdir(parents=True, exist_ok=True)
    for modality in ("color", "thermal"):
        metric = f"{modality}_PSNR"
        ordered = sorted(
            ((finite_scalar(candidate_views[metric][name]) - finite_scalar(baseline_views[metric][name]), name)
             for name in candidate_views[metric]), key=lambda entry: (entry[0], entry[1]),
        )
        for label, (delta, name) in (("worst", ordered[0]), ("median", ordered[len(ordered) // 2])):
            images = []
            for channel in ("color", "thermal"):
                for source in (candidate_render / f"gt_{channel}" / name,
                               baseline_render / f"renders_{channel}" / name,
                               candidate_render / f"renders_{channel}" / name):
                    with Image.open(source) as original:
                        images.append(original.convert("RGB"))
            tile_width, tile_height = images[0].size
            if any(image.size != (tile_width, tile_height) for image in images):
                raise ValueError(f"inconsistent montage resolutions for {scene}/{name}")
            header, row_label = 44, 23
            canvas = Image.new("RGB", (tile_width * 3, header + (tile_height + row_label) * 2), "#202020")
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 5), f"{scene} | {modality} {label} delta view {name} | delta PSNR {delta:+.5f} dB", fill="white")
            for col, column_name in enumerate(("Ground truth", "Baseline ODB 30000", f"IR kernel {iteration}")):
                draw.text((8 + col * tile_width, 24), column_name, fill="white")
            for row, row_name in enumerate(("RGB", "IR")):
                top = header + row * (tile_height + row_label)
                draw.text((8, top + 4), row_name, fill="white")
                for col in range(3):
                    canvas.paste(images[row * 3 + col], (col * tile_width, top + row_label))
            path = destination / f"{scene}_{modality}_{label}_{Path(name).stem}.png"
            canvas.save(path)
            records.append({"scene": scene, "selection_metric": metric, "selection": label,
                            "view": name, "delta": delta, "path": str(path.relative_to(candidate)),
                            "sha256": sha256_file(path), "native_image_size": [tile_width, tile_height]})
    return records


def provenance_check(candidate: Path, scene: str, iteration: int,
                     metadata: dict[str, Any]) -> dict[str, Any]:
    issues = []
    status_path = candidate / f"{scene}.status.json"
    try:
        status = json.loads(status_path.read_text())
        for stage in ("train", "render", "metrics"):
            if status[stage]["return_code"] != 0:
                issues.append(f"{stage} failed")
        command = status["train"]["command"]
        if "--start_checkpoint" in command:
            issues.append("training loaded a checkpoint")
        if command[command.index("--iterations") + 1] != str(iteration):
            issues.append("training budget differs from report iteration")
        if "--use_ir_kernel" not in command:
            issues.append("IR kernel was not enabled")
        if not metadata.get("code_snapshot", {}).get("archive_sha256"):
            issues.append("code archive provenance missing")
        executed_source = Path(metadata.get("executed_source_root", "missing"))
        for stage, script in (("train", "train.py"), ("render", "render.py"), ("metrics", "metrics.py")):
            if Path(status[stage]["command"][1]) != executed_source / script:
                issues.append(f"{stage} did not execute the frozen source")
        if metadata.get("cuda_visible_devices") != "0":
            issues.append("physical GPU 0 provenance missing")
        if metadata.get("iterations") != iteration:
            issues.append("run metadata budget differs from report iteration")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        issues.append(str(exc))
    return {"verified": not issues, "issues": issues}


def build_report(candidate: Path, baseline: Path, iteration: int,
                 with_montages: bool, rgb_judgment: str,
                 rgb_reason: str | None) -> dict[str, Any]:
    metadata_path = candidate / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    rows = {}
    completed = []
    for scene in SCENES:
        base = scene_metrics(baseline, scene, 30000)
        row: dict[str, Any] = {"baseline": base, "candidate": None, "delta": None,
                               "status": "missing", "validation": None}
        if (candidate / scene / "results.json").is_file():
            try:
                values = scene_metrics(candidate, scene, iteration)
                validation = validate_artifacts(candidate, scene, baseline, iteration)
                provenance = provenance_check(candidate, scene, iteration, metadata)
                row.update(candidate=values, delta={key: values[key] - base[key] for key in METRICS},
                           validation=validation, provenance=provenance)
                row["status"] = "validated" if validation["valid"] and provenance["verified"] else "unverified"
                if row["status"] == "validated":
                    completed.append(scene)
                    if with_montages:
                        row["visualizations"] = report_montages(candidate, baseline, scene, iteration, candidate / "visualizations")
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                row["status"], row["error"] = "invalid", str(exc)
        rows[scene] = row
    uniform_configuration = True
    if completed:
        first = rows[completed[0]]["validation"]
        first_model = {key: value for key, value in first["saved_model_configuration"].items() if key not in ("source_path", "model_path")}
        for scene in completed[1:]:
            config = rows[scene]["validation"]
            model = {key: value for key, value in config["saved_model_configuration"].items() if key not in ("source_path", "model_path")}
            if model != first_model or config["saved_optimization"] != first["saved_optimization"]:
                uniform_configuration = False
    source_issues = []
    try:
        snapshot = metadata["code_snapshot"]
        source_root = Path(metadata["executed_source_root"])
        archive = source_root.parent / snapshot["archive"]
        if sha256_file(archive) != snapshot["archive_sha256"]:
            source_issues.append("source archive hash mismatch")
        for relative, digest in snapshot["file_sha256"].items():
            if sha256_file(source_root / relative) != digest:
                source_issues.append(f"frozen source hash mismatch: {relative}")
    except (OSError, KeyError, TypeError) as exc:
        source_issues.append(str(exc))
    complete = (len(completed) == len(SCENES) and iteration == 30000
                and uniform_configuration and not source_issues)
    mean = {}
    if completed:
        for key in METRICS:
            mean[key] = {column: sum(rows[scene][column][key] for scene in completed) / len(completed)
                         for column in ("baseline", "candidate", "delta")}
    ir_pass = complete and mean["thermal_PSNR"]["candidate"] >= 26.4
    visual_ready = complete and all(
        len(rows[scene].get("visualizations", [])) == 4
        and all("path" in item for item in rows[scene]["visualizations"])
        for scene in SCENES)
    goal_achieved = ir_pass and visual_ready and rgb_judgment == "no_obvious_degradation" and bool(rgb_reason)
    return {
        "created_utc": utc_now(), "candidate_root": str(candidate), "baseline_root": str(baseline),
        "iteration": iteration, "required_scenes": list(SCENES), "validated_scenes": completed,
        "missing_or_unverified_scenes": [scene for scene in SCENES if scene not in completed],
        "complete_ten_scene_30000_evaluation": complete,
        "uniform_training_configuration": uniform_configuration,
        "source_provenance_issues": source_issues,
        "mean_scope": "all ten scenes" if complete else f"PARTIAL: {len(completed)}/10 validated scenes at {iteration} iterations",
        "mean": mean, "baseline_ten_scene_mean": {key: sum(rows[s]["baseline"][key] for s in SCENES) / len(SCENES) for key in METRICS},
        "ir_target": 26.4, "ir_target_met": ir_pass if complete else None,
        "rgb_judgment": rgb_judgment, "rgb_reason": rgb_reason,
        "visualizations_complete": visual_ready, "goal_achieved": goal_achieved,
        "test_metrics_used_for_design_selection": True,
        "per_scene": rows,
    }


def write_report(candidate: Path, report: dict[str, Any]) -> None:
    json_dump(candidate / "comparison.json", report)
    columns = [(metric, variant) for metric in METRICS for variant in ("baseline", "candidate", "delta")]
    with (candidate / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene", "status"] + [f"{variant}_{metric}" for metric, variant in columns])
        for scene, row in report["per_scene"].items():
            writer.writerow([scene, row["status"]] + [row[variant][metric] if row.get(variant) else "" for metric, variant in columns])
        if report["mean"]:
            writer.writerow(["MEAN", report["mean_scope"]] + [report["mean"][metric][variant] for metric, variant in columns])
    lines = [
        "# IR Gaussian experiment comparison", "",
        f"Evaluation: **{report['mean_scope']}**. Candidate iteration: {report['iteration']}.", "",
        "All means give equal weight to scenes. Partial means cannot establish the target.", "",
        f"IR target at 26.4 dB: **{report['ir_target_met'] if report['ir_target_met'] is not None else 'not assessed (incomplete)'}**.", "",
        f"RGB judgment: **{report['rgb_judgment']}**. {report['rgb_reason'] or 'Metric and same-view visual review is pending.'}", "",
        f"Goal achieved: **{report['goal_achieved']}**.", "",
        "Test metrics were permitted in design selection; all learned parameters are fitted on training views.", "",
    ]
    for metric in METRICS:
        lines += [f"## {metric}", "", "| Scene | Baseline | Candidate | Delta | Status |",
                  "|---|---:|---:|---:|---|"]
        for scene, row in report["per_scene"].items():
            base = f"{row['baseline'][metric]:.6f}"
            cand = f"{row['candidate'][metric]:.6f}" if row["candidate"] else "—"
            delta = f"{row['delta'][metric]:+.6f}" if row["delta"] else "—"
            lines.append(f"| {scene} | {base} | {cand} | {delta} | {row['status']} |")
        if report["mean"]:
            average = report["mean"][metric]
            lines.append(f"| **Mean** | **{average['baseline']:.6f}** | **{average['candidate']:.6f}** | **{average['delta']:+.6f}** | {report['mean_scope']} |")
        lines.append("")
    lines += ["## Same-view visualization selection", "", "Worst and median PSNR changes are selected independently for RGB and IR; every montage shows both modalities with ground truth, baseline, and candidate at native resolution.", ""]
    for scene, row in report["per_scene"].items():
        for artifact in row.get("visualizations", []):
            if "path" in artifact:
                lines.append(f"- [{scene} {artifact['selection_metric']} {artifact['selection']} ({artifact['view']})]({artifact['path']})")
            else:
                lines.append(f"- {scene}: {artifact['error']}")
    (candidate / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--no-montages", action="store_true")
    parser.add_argument("--rgb-judgment", choices=("pending", "no_obvious_degradation", "obvious_degradation"), default="pending")
    parser.add_argument("--rgb-reason", help="Evidence-based reason after reviewing means and same-view images")
    args = parser.parse_args()
    if args.rgb_judgment != "pending" and not args.rgb_reason:
        parser.error("--rgb-reason is required for a completed RGB assessment")
    candidate, baseline = args.candidate.resolve(), args.baseline.resolve()
    if not candidate.is_dir():
        parser.error("candidate directory does not exist")
    report = build_report(candidate, baseline, args.iteration, not args.no_montages, args.rgb_judgment, args.rgb_reason)
    write_report(candidate, report)
    print(json.dumps({key: report[key] for key in ("mean_scope", "mean", "ir_target_met", "rgb_judgment", "goal_achieved")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
