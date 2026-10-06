#!/usr/bin/env python3
"""CPU-only quality attribution for saved official and kernel-ablation PNGs.

This script never renders, trains, initializes CUDA, or writes official results.
It uses the candidate's frozen utils.loss_utils.ssim and image_utils.psnr on
float32 NCHW tensors with the same PNG conversion as metrics.py. Edge-bin
statistics are diagnostic and do not replace the official all-view metrics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shlex
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    candidate = args.candidate.resolve()
    baseline = args.baseline.resolve()
    scene = args.scene
    require(scene == Path(scene).name, "Scene must be a single directory name")
    method = "ours_30000"
    scene_root = candidate / scene
    official = scene_root / "test" / method
    original = baseline / scene / "test" / method
    ablation = candidate / "diagnostics" / "kernel_ablation" / scene
    destination = candidate / "diagnostics" / "kernel_quality_cpu" / scene
    require(not destination.exists(), f"Preserve existing diagnostic: {destination}")
    ablation_report = json.loads((ablation / "report.json").read_text())
    for key in ("all_views_on_pngs_reproduce_official_pixels", "all_views_base_ir_bitwise_equal",
                "all_views_rgb_bitwise_equal", "all_test_views_evaluated"):
        require(ablation_report[key] is True, f"Missing same-model attribution evidence: {key}")
    candidate_results = json.loads((scene_root / "results.json").read_text())[method]
    baseline_results = json.loads((baseline / scene / "results.json").read_text())[method]
    official_views = json.loads((scene_root / "per_view.json").read_text())[method]
    baseline_views = json.loads((baseline / scene / "per_view.json").read_text())[method]
    names = sorted(official_views["thermal_PSNR"])
    require(names == sorted(baseline_views["thermal_PSNR"]), "Baseline test view set differs")
    require(len(names) == ablation_report["test_view_count"], "Ablation test view count differs")
    for folder in (official / "renders_thermal", official / "gt_thermal",
                   original / "renders_thermal", original / "gt_thermal",
                   ablation / "renders_thermal"):
        require(sorted(p.name for p in folder.glob("*.png")) == names,
                f"Incomplete or unexpected image set: {folder}")
    source = candidate / "code_snapshot" / "source"
    for relative in ("utils/loss_utils.py", "utils/image_utils.py", "metrics.py"):
        require(sha256(source / relative) == sha256(Path(__file__).resolve().parents[1] / relative),
                f"Current/frozen evaluation source differs: {relative}")
    sys.path.insert(0, str(source))
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    import torch.nn.functional as F
    from torchvision.transforms.functional import to_tensor
    from utils.loss_utils import ssim
    from utils.image_utils import psnr

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), "CUDA unexpectedly initialized")

    def read_image(path):
        with Image.open(path) as image:
            require(image.mode == "RGB", f"Expected three-channel official PNG: {path}")
            return np.array(image, copy=True), to_tensor(image).unsqueeze(0)[:, :3, :, :]

    def aggregate(values):
        # metrics.py stacks Python scalar tensors into float32 before its scene mean.
        return float(torch.tensor(values, dtype=torch.float32).mean())

    def edge_strength(gt):
        # RGB thermal values are pseudocolor. Use RMS vector gradient, no assumed
        # thermal-palette inverse or photometric luminance conversion.
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        ky = kx.T.contiguous()
        gx = F.conv2d(F.pad(gt, (1, 1, 1, 1), mode="reflect"),
                      kx.reshape(1, 1, 3, 3).expand(3, 1, 3, 3), groups=3)
        gy = F.conv2d(F.pad(gt, (1, 1, 1, 1), mode="reflect"),
                      ky.reshape(1, 1, 3, 3).expand(3, 1, 3, 3), groups=3)
        return ((gx.square() + gy.square()).mean(1).sqrt()[0]).numpy()

    rows, cached, input_sha = [], {}, {}
    all_edges = []
    for name in names:
        paths = {"gt": official / "gt_thermal" / name,
                 "baseline": original / "renders_thermal" / name,
                 "off": ablation / "renders_thermal" / name,
                 "on": official / "renders_thermal" / name}
        arrays, tensors = {}, {}
        for key, path in paths.items():
            arrays[key], tensors[key] = read_image(path)
        baseline_gt, _ = read_image(original / "gt_thermal" / name)
        require(np.array_equal(baseline_gt, arrays["gt"]), f"Baseline GT differs: {name}")
        require(all(t.shape == tensors["gt"].shape for t in tensors.values()),
                f"View resolutions differ: {name}")
        record = {"filename": name, "image_shape_hwc": list(arrays["gt"].shape)}
        errors = {}
        for key in ("baseline", "off", "on"):
            record[key] = {"SSIM": float(ssim(tensors[key], tensors["gt"])),
                           "PSNR": float(psnr(tensors[key], tensors["gt"]).mean())}
            errors[key] = ((tensors[key] - tensors["gt"]).square().mean(1)[0]).numpy()
            record[key]["MSE"] = float(errors[key].mean(dtype=np.float64))
        for metric in ("SSIM", "PSNR", "MSE"):
            record[f"kernel_delta_{metric}"] = record["on"][metric] - record["off"][metric]
            record[f"base_delta_{metric}"] = record["off"][metric] - record["baseline"][metric]
            record[f"total_delta_{metric}"] = record["on"][metric] - record["baseline"][metric]
        for key, reference in (("on", official_views), ("baseline", baseline_views)):
            for metric in ("SSIM", "PSNR"):
                official_value = reference[f"thermal_{metric}"][name]
                # PSNR per-view files serialize tensors as singleton lists.
                if isinstance(official_value, list):
                    official_value = official_value[0]
                record[key][f"official_{metric}_error"] = record[key][metric] - official_value
                require(abs(record[key][f"official_{metric}_error"]) <= args.metric_tolerance,
                        f"CPU {metric} does not reproduce official {key} view {name}")
        edge = edge_strength(tensors["gt"])
        all_edges.append(edge.ravel())
        cached[name] = {"arrays": arrays, "errors": errors, "edge": edge}
        input_sha[name] = {key: sha256(path) for key, path in paths.items()}
        rows.append(record)
        print(f"{name}: on/off SSIM {record['on']['SSIM']:.7f}/{record['off']['SSIM']:.7f}; "
              f"kernel PSNR {record['kernel_delta_PSNR']:+.6f} dB", flush=True)

    thresholds = np.quantile(np.concatenate(all_edges), [.25, .5, .75]).tolist()
    del all_edges
    quartiles = []
    for q in range(4):
        count = 0
        sums = {key: 0.0 for key in ("baseline", "off", "on")}
        for name in names:
            data = cached[name]
            labels = np.searchsorted(np.asarray(thresholds), data["edge"], side="right")
            mask = labels == q
            count += int(mask.sum())
            for key in sums:
                sums[key] += float(data["errors"][key][mask].sum(dtype=np.float64))
        mse = {key: value / count for key, value in sums.items()}
        quartiles.append({"quartile": q + 1, "pixel_count": count,
                          "edge_lower_inclusive": 0 if q == 0 else thresholds[q - 1],
                          "edge_upper_exclusive": thresholds[q] if q < 3 else None,
                          "MSE": mse, "kernel_MSE_gain": mse["off"] - mse["on"],
                          "kernel_MSE_reduction_percent": 100 * (mse["off"] - mse["on"]) / mse["off"],
                          "base_MSE_gain": mse["baseline"] - mse["off"],
                          "total_MSE_gain": mse["baseline"] - mse["on"]})
    summary = {key: {metric: aggregate([r[key][metric] for r in rows])
                     for metric in ("SSIM", "PSNR", "MSE")}
               for key in ("baseline", "off", "on")}
    for key, reference in (("on", candidate_results), ("baseline", baseline_results)):
        for metric in ("SSIM", "PSNR"):
            require(abs(summary[key][metric] - reference[f"thermal_{metric}"]) <= args.metric_tolerance,
                    f"CPU scene-mean {metric} does not reproduce official {key}")
    effects = {which: {metric: summary[end][metric] - summary[start][metric]
                       for metric in ("SSIM", "PSNR", "MSE")}
               for which, start, end in (("base_vs_baseline", "baseline", "off"),
                                         ("kernel_on_vs_off", "off", "on"),
                                         ("total_vs_baseline", "baseline", "on"))}
    distribution = {}
    for metric in ("SSIM", "PSNR"):
        ordered = sorted(rows, key=lambda row: row[f"kernel_delta_{metric}"])
        distribution[metric] = {"improved_views": sum(r[f"kernel_delta_{metric}"] > 0 for r in rows),
                                "regressed_views": sum(r[f"kernel_delta_{metric}"] < 0 for r in rows),
                                "unchanged_views": sum(r[f"kernel_delta_{metric}"] == 0 for r in rows),
                                "worst_5": [{"filename": r["filename"], "delta": r[f"kernel_delta_{metric}"]} for r in ordered[:5]],
                                "best_5": [{"filename": r["filename"], "delta": r[f"kernel_delta_{metric}"]} for r in reversed(ordered[-5:])]}

    destination.mkdir(parents=True)
    chosen = {}
    for label, key, reverse in (("worst_psnr", "kernel_delta_PSNR", False),
                                ("best_psnr", "kernel_delta_PSNR", True),
                                ("worst_ssim", "kernel_delta_SSIM", False)):
        chosen[label] = sorted(rows, key=lambda row: row[key], reverse=reverse)[0]["filename"]
    chosen["median_psnr"] = sorted(rows, key=lambda row: row["kernel_delta_PSNR"])[len(rows) // 2]["filename"]
    montages = []
    for label, name in chosen.items():
        data = cached[name]
        height, width, _ = data["arrays"]["gt"].shape
        panel = Image.new("RGB", (width * 4, height + 48), "white")
        draw = ImageDraw.Draw(panel)
        for col, key in enumerate(("gt", "baseline", "off", "on")):
            draw.text((width * col + 6, 5), f"{scene} {name} {label}: {key}", fill="black")
            panel.paste(Image.fromarray(data["arrays"][key]), (width * col, 48))
        full_path = destination / f"{label}_{Path(name).stem}_full.png"
        panel.save(full_path)

        # Locate native-resolution 128x128 crop windows by average kernel MSE
        # gain/loss; this selection is explicit and includes both signs.
        gain = torch.from_numpy(data["errors"]["off"] - data["errors"]["on"])[None, None]
        crop_size = 128
        pooled = F.avg_pool2d(gain, crop_size, stride=8)[0, 0]
        crop_panel = Image.new("RGB", (crop_size * 4, (crop_size + 36) * 2), "white")
        crop_draw = ImageDraw.Draw(crop_panel)
        crop_records = []
        for row_number, (crop_label, flat) in enumerate((("MSE loss", int(pooled.argmin())),
                                                        ("MSE gain", int(pooled.argmax())))):
            y = flat // pooled.shape[1] * 8
            x = flat % pooled.shape[1] * 8
            crop_records.append({"kind": crop_label, "xyxy": [x, y, x + crop_size, y + crop_size],
                                 "mean_kernel_MSE_gain": float(pooled.flatten()[flat])})
            for col, key in enumerate(("gt", "baseline", "off", "on")):
                top = row_number * (crop_size + 36)
                crop_draw.text((col * crop_size + 3, top + 3), f"{key} {crop_label}", fill="black")
                crop_panel.paste(Image.fromarray(data["arrays"][key][y:y + crop_size, x:x + crop_size]),
                                 (col * crop_size, top + 36))
        crop_path = destination / f"{label}_{Path(name).stem}_crops.png"
        crop_panel.save(crop_path)
        montages.append({"selection": label, "filename": name,
                         "full_image": full_path.name, "crop_image": crop_path.name,
                         "crop_selection": crop_records})

    require(not torch.cuda.is_initialized(), "CUDA unexpectedly initialized")
    report = {"diagnostic_only": True, "acceptance_claim": False,
              "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "scene": scene, "all_test_views_evaluated": True, "test_view_count": len(names),
              "command": shlex.join(sys.argv), "python": sys.executable,
              "script_sha256": sha256(__file__), "torch_version": torch.__version__,
              "torch_threads": torch.get_num_threads(), "cuda_initialized": torch.cuda.is_initialized(),
              "candidate": str(candidate), "baseline": str(baseline),
              "frozen_metric_source_sha256": {name: sha256(source / name) for name in
                                              ("utils/loss_utils.py", "utils/image_utils.py", "metrics.py")},
              "metric_protocol": "Official PNG RGB/255 float32 NCHW; frozen repository SSIM and PSNR; float32 mean across all test views; CPU only.",
              "official_metric_tolerance": args.metric_tolerance,
              "same_model_ablation_report_sha256": sha256(ablation / "report.json"),
              "summary": summary, "effects": effects, "kernel_view_distribution": distribution,
              "edge_method": "Sobel x/y kernels divided by 8, reflect padding, RMS gradient magnitude over three pseudocolor channels. Global pooled GT edge-strength quartiles; tied thresholds assigned upward via searchsorted(side=right). MSE averages over channels then selected pixels. Positive MSE gain means improvement.",
              "edge_quartile_thresholds": thresholds, "edge_quartiles": quartiles,
              "per_view": rows, "input_sha256": input_sha, "montages": montages,
              "limitations": [f"{args.scene} only; cannot establish ten-scene target or full-suite RGB quality.",
                              "CPU/GPU floating-point convolution roundoff is checked against official per-view and aggregate metrics.",
                              "Kernel on/off differences isolate inference contribution. Base-off vs baseline captures all training/backend/export variation; this diagnostic does not identify its cause.",
                              "Saved PNG quantization/clamp is included. Edge-conditioned MSE is diagnostic, not an acceptance metric."]}
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [f"# {scene} kernel quality attribution (CPU PNG diagnostic)", "",
             f"All {len(names)} test views; frozen repository NCHW SSIM/PSNR. CUDA initialized: false.",
             "Official on and baseline per-view and scene values reproduced within "
             f"{args.metric_tolerance:g} absolute tolerance. This is not ten-scene acceptance.", "",
             "| Render | IR PSNR | IR SSIM | Pixel MSE |", "|---|---:|---:|---:|"]
    for key in ("baseline", "off", "on"):
        lines.append(f"| {key} | {summary[key]['PSNR']:.8f} | {summary[key]['SSIM']:.8f} | {summary[key]['MSE']:.9f} |")
    lines.extend(["", "| Difference | PSNR | SSIM |", "|---|---:|---:|"])
    for key, values in effects.items():
        lines.append(f"| {key} | {values['PSNR']:+.8f} | {values['SSIM']:+.8f} |")
    lines.extend(["", "Kernel per-view changes: " + "; ".join(
        f"{metric}: {distribution[metric]['improved_views']} improved, {distribution[metric]['regressed_views']} regressed"
        for metric in ("PSNR", "SSIM")) + ".", "",
        "GT edge strength uses channel-vector Sobel RMS; quartiles pool every pixel of every test view.", "",
        "| GT edge quartile | Pixels | Off MSE | On MSE | Kernel MSE reduction |",
        "|---|---:|---:|---:|---:|"])
    for row in quartiles:
        lines.append(f"| {row['quartile']} | {row['pixel_count']} | {row['MSE']['off']:.9f} | {row['MSE']['on']:.9f} | {row['kernel_MSE_reduction_percent']:+.3f}% |")
    lines.extend(["", "Per-view metrics, hashes, worst/best lists, explicit crop coordinates and full protocol are in `report.json`.",
                  "Montages list GT, fixed baseline, same-model off, and official on, in that order. Crops include both strongest local MSE loss and gain.",
                  "Base-off versus fixed-baseline differences combine all base training/backend/export variation; no causal explanation for that residual difference is established here.", ""])
    (destination / "report.md").write_text("\n".join(lines))
    print(json.dumps({"destination": str(destination), "summary": summary, "effects": effects,
                      "distribution": distribution, "edge_quartiles": quartiles}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("/home/lf/code/Our_Project-New-2/output/odb_30k_20261003_101217_1051565"))
    parser.add_argument("--scene", default="Building")
    parser.add_argument("--metric-tolerance", type=float, default=2e-5)
    run(parser.parse_args())
