#!/usr/bin/env python3
"""Measure frozen IR-kernel gains on every training view, without fitting.

CPU-only preflight:
  python ir_experiments/diagnose_kernel_generalization.py \
    --candidate output/ir_kernel_v1_20261005_run2 --scenes Building DailyStuff --preflight

GPU-0 diagnostic (no optimizer, parameter updates, or training iterations):
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/lf/miniconda3/envs/thermalgaussian/lib \
  /home/lf/miniconda3/envs/thermalgaussian/bin/python \
  ir_experiments/diagnose_kernel_generalization.py \
    --candidate output/ir_kernel_v1_20261005_run2 --scenes Building DailyStuff

Only candidate/diagnostics/kernel_generalization/SCENE/report.json is written.
The official model/results/PNGs and frozen source/runtime are read only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import os
from pathlib import Path
import shlex
import sys

# Importing frozen modules must not add bytecode files to their source tree.
sys.dont_write_bytecode = True
import ablate_kernel as ablation


METRICS = ("PSNR", "MSE", "SSIM")


def immutable_files(data):
    """Protect every completed-scene artifact, including models and official PNGs."""
    return sorted(path for path in data["folder"].rglob("*") if path.is_file())


def file_hashes(paths, root):
    return {str(path.relative_to(root)): ablation.sha256(path) for path in paths}


def verify_hashes(hashes, root):
    for relative, expected in hashes.items():
        ablation.require(ablation.sha256(root / relative) == expected,
                         f"Input artifact changed during diagnostic: {relative}")


def verify_frozen(data):
    source, runtime = data["source"], data["runtime"]
    snapshot_path = source.parent / "code_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    for relative, expected in snapshot["file_sha256"].items():
        if relative.endswith(".py") and not relative.startswith("submodules/"):
            ablation.require(ablation.sha256(source / relative) == expected,
                             f"Frozen source changed: {relative}")
    manifest_path = runtime / "manifest.json"
    extensions = json.loads(manifest_path.read_text())
    for package, record in extensions.items():
        for relative, expected in record["files"].items():
            ablation.require(ablation.sha256(runtime / package / relative) == expected,
                             f"Frozen runtime changed: {package}/{relative}")
    return {"source_snapshot_sha256": ablation.sha256(snapshot_path),
            "runtime_manifest_sha256": ablation.sha256(manifest_path)}


def preflight(candidate, scenes):
    ablation.require(bool(scenes), "At least one scene is required")
    ablation.require(len(set(scenes)) == len(scenes), "Duplicate scene names")
    records = []
    for name in scenes:
        data = ablation.preflight(candidate, name)
        destination = data["candidate"] / "diagnostics" / "kernel_generalization" / name
        ablation.require(not destination.exists(), f"Diagnostic already exists: {destination}")
        data["destination"] = destination
        records.append(data)
    provenance = verify_frozen(records[0])
    return records, provenance


def png_metrics(prediction, target, psnr, ssim, device):
    """Official PNG quantization and NCHW all-channel metrics."""
    import torch
    from torchvision.transforms.functional import to_tensor
    a = to_tensor(prediction).unsqueeze(0).to(device)
    b = to_tensor(target).unsqueeze(0).to(device)
    ablation.require(a.shape == b.shape and a.shape[1] == 3, "PNG dimensions/channels differ")
    ablation.require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), "Nonfinite PNG data")
    result = {"PSNR": float(psnr(a, b).mean().item()),
              "MSE": float((a - b).square().mean().item()),
              "SSIM": float(ssim(a, b).mean().item())}
    ablation.require(all(math.isfinite(value) for value in result.values()), "Nonfinite image metric")
    return result


def summarize(rows):
    import torch
    ablation.require(bool(rows), "No views were evaluated")
    means = {mode: {metric: torch.tensor([row[mode][metric] for row in rows]).mean().item()
                    for metric in METRICS} for mode in ("off", "on")}
    return {"view_count": len(rows), "off": means["off"], "on": means["on"],
            "on_minus_off": {metric: means["on"][metric] - means["off"][metric] for metric in METRICS},
            "improved_views": {metric: sum((row["on"][metric] < row["off"][metric])
                                            if metric == "MSE" else (row["on"][metric] > row["off"][metric])
                                            for row in rows) for metric in METRICS},
            "aggregation": "Equal-view arithmetic mean in float32; PSNR uses all channels/pixels within each view"}


def existing_test_summary(data, psnr, ssim):
    """Reuse existing all-test ablation PNGs; never render or fit test images."""
    folder = data["candidate"] / "diagnostics" / "kernel_ablation" / data["folder"].name
    path = folder / "report.json"
    if not path.is_file():
        return {"available": False, "reason": "No existing all-test-view ablation report", "path": str(path)}, {}
    saved = json.loads(path.read_text())
    ablation.require(saved.get("iteration") == ablation.ITERATION
                     and saved.get("scene") == data["folder"].name
                     and Path(saved["candidate_root"]).resolve() == data["candidate"],
                     "Existing ablation belongs to a different export")
    ablation.require(saved.get("all_test_views_evaluated") is True
                     and saved.get("all_views_on_pngs_reproduce_official_pixels") is True,
                     "Existing ablation did not verify all official views")
    indexed = {row["filename"]: row for row in saved["per_view"]}
    ablation.require(len(indexed) == len(saved["per_view"]) and set(indexed) == set(data["names"]),
                     "Existing ablation has incomplete/duplicate test views")
    hashes = {str(path.relative_to(data["candidate"])): ablation.sha256(path)}
    for relative, expected in saved.get("official_render_sha256", {}).items():
        actual = (data["candidate"] / relative).resolve()
        ablation.require(actual.is_relative_to(data["official"]), "Invalid ablation official image path")
        ablation.require(ablation.sha256(actual) == expected, "Official render differs from the ablation")
    rows = []
    for name in data["names"]:
        off_path = folder / "renders_thermal" / name
        digest = ablation.sha256(off_path)
        ablation.require(digest == indexed[name]["off_png_sha256"], f"Existing off PNG changed: {name}")
        hashes[str(off_path.relative_to(data["candidate"]))] = digest
        gt = ablation.load_rgb(data["official"] / "gt_thermal" / name)
        on = png_metrics(ablation.load_rgb(data["official"] / "renders_thermal" / name), gt, psnr, ssim, "cuda")
        off = png_metrics(ablation.load_rgb(off_path), gt, psnr, ssim, "cuda")
        ablation.require(math.isclose(on["PSNR"], indexed[name]["on_ir_psnr"], rel_tol=0, abs_tol=1e-5)
                         and math.isclose(off["PSNR"], indexed[name]["off_ir_psnr"], rel_tol=0, abs_tol=1e-5),
                         f"Existing ablation PSNR did not reproduce: {name}")
        rows.append({"filename": name, "on": on, "off": off})
    summary = summarize(rows)
    ablation.require(math.isclose(summary["on"]["PSNR"], data["results"]["thermal_PSNR"], rel_tol=0, abs_tol=1e-5)
                     and math.isclose(summary["on"]["SSIM"], data["results"]["thermal_SSIM"], rel_tol=0, abs_tol=2e-5),
                     "Official test mean metrics did not reproduce")
    return {"available": True, "path": str(path), "report_sha256": hashes[str(path.relative_to(data["candidate"]))],
            "all_test_views_evaluated": True, "saved_mean_kernel_gain_db": saved["mean_kernel_gain_db"],
            "summary": summary, "per_view": rows,
            "scope": "Recomputed from existing official on/GT PNGs and validated kernel-off diagnostic PNGs; no test rendering or fitting"}, hashes


def load_runtime(data):
    source, runtime = data["source"], data["runtime"]
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(source))
    import torch
    import scene
    import gaussian_renderer
    import arguments
    import utils.image_utils as image_utils
    import utils.loss_utils as loss_utils
    import diff_gaussian_rasterization
    import simple_knn._C as knn
    for module, expected in ((scene, source), (gaussian_renderer, source), (arguments, source),
                             (image_utils, source), (loss_utils, source),
                             (diff_gaussian_rasterization, runtime), (knn, runtime)):
        ablation.require(Path(module.__file__).resolve().is_relative_to(expected),
                         f"Unexpected imported module: {module.__file__}")
    ablation.require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                     "Exactly one CUDA device must be visible: physical GPU 0")
    torch.cuda.set_device(0)
    return scene, gaussian_renderer.render, arguments.PipelineParams, image_utils.psnr, loss_utils.ssim


def diagnose(data, provenance, runtime):
    import torch
    scene_module, render, PipelineParams, psnr, ssim = runtime
    candidate, folder = data["candidate"], data["folder"]
    protected = file_hashes(immutable_files(data), candidate)
    checkpoint = torch.load(folder / f"chkpnt{ablation.ITERATION}.pth", map_location="cpu", weights_only=False)
    checkpoint_model, iteration = ablation.checkpoint_contents(checkpoint)
    del checkpoint_model, checkpoint
    gc.collect()
    cfg = dict(data["cfg"], model_path=str(folder))
    model = scene_module.GaussianModel(cfg["sh_degree"], **{key: cfg[key] for key in ablation.MODEL_KEYS})
    scene = scene_module.Scene(argparse.Namespace(**cfg), model, load_iteration=iteration, shuffle=False)
    ablation.require(model.use_ir_kernel and model.ir_kernel_runtime_enabled, "Export disabled IR kernels")
    ablation.require(model.ir_kernel_amplitude == float(cfg["ir_kernel_amplitude"]), "Export amplitude/config mismatch")
    stats = ablation.kernel_stats(model._ir_kernel)
    kernel_before = model._ir_kernel.detach().cpu().clone()
    cameras = scene.getTrainCameras()
    ablation.require(len(cameras) > 0, "No training cameras")
    parser = argparse.ArgumentParser(add_help=False)
    group = PipelineParams(parser)
    status_path = candidate / f"{folder.name}.status.json"
    tokens = []
    if status_path.is_file():
        command = json.loads(status_path.read_text()).get("render", {}).get("command", [])
        command = shlex.split(command) if isinstance(command, str) else command
        tokens = command[2:]
    parsed, _ = parser.parse_known_args(tokens)
    pipe = group.extract(parsed)
    background = torch.tensor([1.0 if cfg["white_background"] else 0.0] * 3, device="cuda")
    rows = []
    with torch.no_grad():
        for idx, camera in enumerate(cameras):
            for modality, value in (("RGB", camera.original_image), ("IR", camera.original_thermal)):
                ablation.require(bool(torch.isfinite(value).all()), f"Nonfinite training {modality} data: {idx}")
            gt_raw = camera.original_thermal.to("cuda")
            model.ir_kernel_runtime_enabled = False
            off_output = render(camera, model, pipe, background)
            # Retain just the two sensor images, releasing per-anchor render
            # dictionaries before computing the other mode.
            off_rgb = off_output["render_color"].detach()
            off_ir = off_output["render_thermal"].detach()
            del off_output
            ablation.clear_render_cache(model)
            model.ir_kernel_runtime_enabled = True
            on_output = render(camera, model, pipe, background)
            ablation.require(torch.equal(off_rgb, on_output["render_color"]), f"RGB changed under toggle: train view {idx}")
            ablation.require(torch.equal(off_ir, on_output["render_thermal_base"]), f"Base IR changed under toggle: train view {idx}")
            on_ir = on_output["render_thermal"].detach()
            for value in (off_rgb, off_ir, on_ir, on_output["ir_kernel_residual"]):
                ablation.require(bool(torch.isfinite(value).all()), f"Nonfinite rendered training image: {idx}")
            ablation.require(gt_raw.shape == on_ir.shape == off_ir.shape, f"Training image shape mismatch: {idx}")
            raw_mse = {"off": (off_ir - gt_raw).square().mean().item(), "on": (on_ir - gt_raw).square().mean().item()}
            ablation.require(all(math.isfinite(value) for value in raw_mse.values()),
                             f"Nonfinite raw fitting MSE: {idx}")
            gt_png = ablation.png_image(gt_raw)
            off = png_metrics(ablation.png_image(off_ir), gt_png, psnr, ssim, "cuda")
            on = png_metrics(ablation.png_image(on_ir), gt_png, psnr, ssim, "cuda")
            rows.append({"training_view_index": idx, "camera_image_name": camera.image_name,
                         "image_size": [camera.image_width, camera.image_height], "off": off, "on": on,
                         "on_minus_off": {metric: on[metric] - off[metric] for metric in METRICS},
                         "raw_unclipped_fit_mse": raw_mse})
            del off_rgb, off_ir, on_ir, on_output, gt_raw, gt_png
            ablation.clear_render_cache(model)
            if (idx + 1) % 10 == 0 or idx + 1 == len(cameras):
                print(f"{folder.name}: measured {idx + 1}/{len(cameras)} training views", flush=True)
        train_summary = summarize(rows)
        raw_mean = {mode: torch.tensor([row["raw_unclipped_fit_mse"][mode] for row in rows]).mean().item()
                    for mode in ("off", "on")}
        test, diagnostic_inputs = existing_test_summary(data, psnr, ssim)
        ablation.require(torch.equal(kernel_before, model._ir_kernel.detach().cpu()), "IR kernel parameters changed")
    ablation.require(model.optimizer is None, "Unexpected optimizer in export-only diagnostic")
    ablation.require(file_hashes(immutable_files(data), candidate) == protected,
                     "Official artifact contents or file set changed during diagnostic")
    verify_hashes(diagnostic_inputs, candidate)
    ablation.require(verify_frozen(data) == provenance, "Frozen source/runtime provenance changed")
    report = {"diagnostic_only": True, "accepted_final_candidate": False,
              "scope": "All training views of an already-completed exported model; inference only, no fitting or extra training iterations",
              "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "scene": folder.name, "candidate_root": str(candidate), "iteration": iteration,
              "command": shlex.join([sys.executable, *sys.argv]),
              "environment": {"CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                              "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
                              "gpu_name": torch.cuda.get_device_name(0)},
              "diagnostic_script_sha256": ablation.sha256(__file__),
              "ablation_helper_sha256": ablation.sha256(ablation.__file__),
              "frozen_source_root": str(data["source"]), "frozen_runtime_root": str(data["runtime"]),
              "provenance": provenance, "pipeline": vars(pipe), "export_kernel": stats,
              "all_training_views_evaluated": len(rows) == len(cameras),
              "training_camera_count": len(cameras),
              "training_sampling_camera_count": len(scene.getTrainSamplingCameras(paired_only=cfg["use_paired_views"])),
              "all_training_views_rgb_bitwise_equal": True, "all_training_views_base_ir_bitwise_equal": True,
              "all_input_and_rendered_images_finite": True, "kernel_tensor_bitwise_unchanged": True,
              "optimizer_created": False, "parameter_updates": 0, "additional_training_iterations": 0,
              "metric_convention": "torchvision.save_image PNG quantization in memory, frozen PSNR/SSIM on NCHW tensors, equal-view float32 averages",
              "training_summary": train_summary,
              "training_raw_unclipped_fit_mse": {**raw_mean, "on_minus_off": raw_mean["on"] - raw_mean["off"]},
              "training_per_view": rows, "existing_test_ablation": test,
              "official_artifacts_unchanged": True, "official_artifact_sha256": protected,
              "existing_diagnostic_input_sha256": diagnostic_inputs,
              "limitations": ["Training images were used to fit this export; their scores are in-sample diagnostics, not acceptance.",
                              "The PLY/module export may be EMA-smoothed; raw fitting MSE is not the historical training-log loss.",
                              "On/off toggles isolate the residual at inference, not a retrained no-kernel counterfactual.",
                              "Train/test camera distributions differ. A gain gap can suggest generalization issues but alone does not prove their cause.",
                              "No LPIPS, runtime benchmark, gradient, optimization or test-set fitting is performed."]}
    if test["available"]:
        report["train_minus_test_kernel_gain"] = {
            metric: train_summary["on_minus_off"][metric] - test["summary"]["on_minus_off"][metric]
            for metric in METRICS}
    # Only create the new output directory after every guard passes. Reports
    # are never placed in a scene's official model/test directories.
    data["destination"].mkdir(parents=True, exist_ok=False)
    path = data["destination"] / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"diagnostic_only": True, "scene": folder.name,
                      "training_summary": train_summary, "test_summary_available": test["available"],
                      "report": str(path)}, indent=2), flush=True)
    del model, scene, cameras, kernel_before, background
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--scenes", "--scene", dest="scenes", nargs="+", required=True)
    parser.add_argument("--preflight", action="store_true", help="CPU-only validation; no torch import, CUDA access, or file writes")
    args = parser.parse_args()
    try:
        data, provenance = preflight(args.candidate, args.scenes)
        if args.preflight:
            print(json.dumps({"ready": True, "cpu_only": True, "scenes": args.scenes,
                              "provenance": provenance, "torch_imported": "torch" in sys.modules,
                              "destinations": [str(item["destination"]) for item in data]}, indent=2))
            return
        ablation.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Set CUDA_VISIBLE_DEVICES=0")
        runtime = load_runtime(data[0])
        for item in data:
            diagnose(item, provenance, runtime)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Kernel generalization diagnostic failed: {exc}\n")


if __name__ == "__main__":
    main()
