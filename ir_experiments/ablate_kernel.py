#!/usr/bin/env python3
"""Diagnose a completed 30000-step model's IR kernel; never an acceptance test.

Example (run only when GPU 0 is available for this diagnostic):
  CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH=/home/lf/miniconda3/envs/thermalgaussian/lib \
  /home/lf/miniconda3/envs/thermalgaussian/bin/python \
  ir_experiments/ablate_kernel.py --candidate output/ir_kernel_v1_20261005_run2 \
  --scene Building

The exported model and frozen renderer are used unchanged. Official images,
metrics and the baseline are read only. Only diagnostics/kernel_ablation/SCENE
is written. No RNG seed is set and no parameters are fitted.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time


ITERATION = 30000
METHOD = "ours_30000"
MODEL_KEYS = (
    "use_bgfc", "use_at_gom", "bgfc_hidden_dim", "bgfc_gate_init_bias",
    "bgfc_thermal_grayscale_context", "bgfc_rgb_luma_transfer_only",
    "use_render_calibration", "use_color_refinement", "color_refinement_hidden_dim",
    "color_refinement_max_residual", "use_detail_basis", "detail_basis_mode",
    "detail_basis_scale", "detail_basis_thermal_scale", "use_ir_kernel",
    "ir_kernel_amplitude",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_namespace(path):
    call = ast.parse(Path(path).read_text(), mode="eval").body
    require(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id == "Namespace" and not call.args,
            f"Invalid saved Namespace: {path}")
    require(all(item.arg is not None for item in call.keywords), "Namespace unpacking is unsupported")
    return {item.arg: ast.literal_eval(item.value) for item in call.keywords}


def finite(value):
    value = float(value)
    require(math.isfinite(value), "Nonfinite official metric")
    return value


def preflight(candidate, scene_name):
    """Reject incomplete exports before importing torch or initializing CUDA."""
    candidate = candidate.resolve()
    require(scene_name and Path(scene_name).name == scene_name
            and scene_name not in (".", ".."), "--scene must be one directory name")
    folder = candidate / scene_name
    source = candidate / "code_snapshot" / "source"
    runtime = candidate / "runtime_packages"
    official = folder / "test" / METHOD
    required = [folder / "results.json", folder / "per_view.json", folder / "cfg_args",
                folder / "training_receipt.json", folder / f"chkpnt{ITERATION}.pth",
                folder / "point_cloud" / f"iteration_{ITERATION}" / "point_cloud.ply",
                folder / "point_cloud" / f"iteration_{ITERATION}" / "feature_modules.pth",
                source / "render.py", source / "utils" / "image_utils.py",
                source.parent / "code_snapshot.json", runtime / "manifest.json"]
    for path in required:
        require(path.is_file(), f"Completed {ITERATION}-step candidate required; missing {path}")
    results = json.loads((folder / "results.json").read_text())
    require(METHOD in results, f"Official results lack {METHOD}")
    per_view = json.loads((folder / "per_view.json").read_text())[METHOD]
    names = sorted(per_view["thermal_PSNR"])
    require(names and all(Path(name).name == name and name.endswith(".png") for name in names),
            "Invalid official test image names")
    require(set(per_view["color_PSNR"]) == set(names), "RGB/IR official views differ")
    for modality in ("color", "thermal"):
        finite(results[METHOD][f"{modality}_PSNR"])
        for value in per_view[f"{modality}_PSNR"].values():
            finite(value)
        for kind in ("renders", "gt"):
            directory = official / f"{kind}_{modality}"
            require(directory.is_dir() and {p.name for p in directory.iterdir()} == set(names),
                    f"Official view files differ from per_view.json: {directory}")
    receipt = json.loads((folder / "training_receipt.json").read_text())
    require(receipt.get("final_iteration") == ITERATION
            and receipt.get("cumulative_iterations") == ITERATION,
            "Training receipt must show exactly 30000 cumulative iterations")
    cfg = read_namespace(folder / "cfg_args")
    require(cfg.get("use_ir_kernel") is True, "Candidate cfg_args must enable IR kernels")
    for key in ("sh_degree", *MODEL_KEYS):
        require(key in cfg, f"Missing model configuration: {key}")
    return {"candidate": candidate, "folder": folder, "source": source, "runtime": runtime,
            "official": official, "names": names, "cfg": cfg, "receipt": receipt,
            "results": results[METHOD], "per_view": per_view}


def png_image(tensor):
    """Use the actual official save_image quantizer, including rounding/clamp."""
    from PIL import Image
    from torchvision.utils import save_image
    buffer = io.BytesIO()
    save_image(tensor, buffer, format="PNG")
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.copy()


def load_rgb(path):
    from PIL import Image
    with Image.open(path) as image:
        require(image.mode == "RGB", f"Expected official RGB PNG: {path}")
        return image.copy()


def images_equal(left, right):
    return left.mode == right.mode and left.size == right.size and left.tobytes() == right.tobytes()


def png_psnr(prediction, target, psnr, device):
    from torchvision.transforms.functional import to_tensor
    # NCHW is essential: the training logger's CHW input averages channel PSNRs.
    a = to_tensor(prediction).unsqueeze(0).to(device)
    b = to_tensor(target).unsqueeze(0).to(device)
    return float(psnr(a, b).mean().item())


def kernel_stats(raw):
    import torch
    raw = raw.detach()
    require(raw.ndim == 2 and raw.shape[0] > 0 and raw.shape[1] == 10,
            "Expected nonempty [num_gaussians, 10] IR kernel tensor")
    require(bool(torch.isfinite(raw).all()), "IR kernel contains nonfinite values")
    changed = raw.abs().sum(0)
    require(bool((changed > 0).all()), "Every one of the ten kernel columns must have learned nonzero values")
    return {"gaussian_count": raw.shape[0], "parameter_count": raw.numel(),
            "raw_parameter_bytes": raw.numel() * raw.element_size(),
            "all_ten_columns_finite_and_nonzero": True,
            "abs_sum_by_column": changed.cpu().tolist(),
            "abs_mean_by_column": raw.abs().mean(0).cpu().tolist(),
            "abs_max_by_column": raw.abs().max(0).values.cpu().tolist(),
            "bounded_abs_mean_by_column": raw.tanh().abs().mean(0).cpu().tolist()}


def checkpoint_contents(checkpoint):
    if isinstance(checkpoint, tuple) and len(checkpoint) >= 2:
        model, iteration = checkpoint[:2]
    elif isinstance(checkpoint, dict):
        model, iteration = checkpoint["model_params"], checkpoint["iteration"]
    else:
        raise ValueError("Unsupported checkpoint container")
    require(iteration == ITERATION, "Checkpoint iteration is not 30000")
    require(isinstance(model, dict) and "ir_kernel" in model, "Checkpoint lacks IR kernels")
    return model, iteration


def clear_render_cache(model):
    # The refinement head retains the latest residual tensors for training logs.
    model.last_color_refinement_residual = None
    model.last_thermal_refinement_residual = None


def measure_inference(model, camera, render, pipe, background, repeats):
    import torch
    measurements = {}
    for enabled in (False, True):
        model.ir_kernel_runtime_enabled = enabled
        for _ in range(3):
            output = render(camera, model, pipe, background)
            del output
        clear_render_cache(model)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = render(camera, model, pipe, background)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        del output
        clear_render_cache(model)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            output = render(camera, model, pipe, background)
            del output
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        clear_render_cache(model)
        measurements["on" if enabled else "off"] = {
            "repeats": repeats, "seconds_per_render": elapsed / repeats,
            "resident_allocated_bytes_before_memory_pass": baseline,
            "peak_allocated_bytes_single_render": peak,
            "incremental_peak_allocated_bytes": peak - baseline,
        }
    measurements["incremental_peak_overhead_bytes"] = (
        measurements["on"]["incremental_peak_allocated_bytes"]
        - measurements["off"]["incremental_peak_allocated_bytes"])
    measurements["time_ratio_on_over_off"] = (measurements["on"]["seconds_per_render"]
                                               / measurements["off"]["seconds_per_render"])
    measurements["scope"] = ("One first test view, inference under no_grad, three warmups per mode; "
                             "memory uses a separate single-render pass with no retained prior outputs. "
                             "Both modes retain kernel model storage. No training/backward cost measured.")
    return measurements


def run(args):
    data = preflight(args.candidate, args.scene)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Set CUDA_VISIBLE_DEVICES=0")
    require(args.timing_repeats > 0, "--timing-repeats must be positive")
    # Source and extension provenance are checked before imports can fall back
    # to current workspace modules or a subsequently rebuilt shared package.
    source, runtime = data["source"], data["runtime"]
    snapshot = json.loads((source.parent / "code_snapshot.json").read_text())
    for relative, digest in snapshot["file_sha256"].items():
        if relative.endswith(".py") and not relative.startswith("submodules/"):
            require(sha256(source / relative) == digest, f"Frozen source changed: {relative}")
    extensions = json.loads((runtime / "manifest.json").read_text())
    for package, record in extensions.items():
        for relative, digest in record["files"].items():
            require(sha256(runtime / package / relative) == digest, f"Frozen extension changed: {package}/{relative}")
    # Keep the frozen pure-Python source first, followed by the matching frozen
    # extension packages.  The current worktree remains after both entries.
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(source))
    import torch
    import scene as scene_module
    import gaussian_renderer as renderer_module
    import diff_gaussian_rasterization as rasterizer_module
    import simple_knn._C as knn_module
    from arguments import PipelineParams
    from utils.image_utils import psnr
    for module, expected in ((scene_module, source), (renderer_module, source),
                             (rasterizer_module, runtime), (knn_module, runtime)):
        require(Path(module.__file__).resolve().is_relative_to(expected),
                f"Unexpected module import: {module.__file__}")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "The diagnostic requires exactly one visible CUDA device: physical GPU 0")
    torch.cuda.set_device(0)
    checkpoint_path = data["folder"] / f"chkpnt{ITERATION}.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model, checkpoint_iteration = checkpoint_contents(checkpoint)
    checkpoint_kernel = kernel_stats(checkpoint_model["ir_kernel"])
    del checkpoint_model, checkpoint
    gc.collect()
    cfg = dict(data["cfg"], model_path=str(data["folder"]))
    model = scene_module.GaussianModel(cfg["sh_degree"], **{key: cfg[key] for key in MODEL_KEYS})
    scene = scene_module.Scene(argparse.Namespace(**cfg), model, load_iteration=ITERATION, shuffle=False)
    require(model.use_ir_kernel and model.ir_kernel_runtime_enabled, "PLY/module reload disabled IR kernels")
    require(model.ir_kernel_amplitude == float(cfg["ir_kernel_amplitude"]),
            "PLY/module amplitude differs from cfg_args")
    stats = kernel_stats(model._ir_kernel)
    require(stats["gaussian_count"] == checkpoint_kernel["gaussian_count"],
            "Checkpoint/export Gaussian counts differ")
    cameras = scene.getTestCameras()
    require(len(cameras) == len(data["names"]) and len(cameras) > 0, "Test camera count differs from official views")
    require(data["names"] == [f"{idx:05d}.png" for idx in range(len(cameras))],
            "Official filenames differ from render.py enumeration")
    pipe_parser = argparse.ArgumentParser(add_help=False)
    pipe_group = PipelineParams(pipe_parser)
    status_path = data["candidate"] / f"{args.scene}.status.json"
    render_command = []
    if status_path.is_file():
        status = json.loads(status_path.read_text())
        render_command = status.get("render", {}).get("command", [])
    if isinstance(render_command, str):
        render_command = shlex.split(render_command)
    # A recorded command starts with interpreter and render.py.  Fall back to
    # defaults if an older status record omitted the command entirely.
    pipeline_tokens = render_command[2:] if len(render_command) >= 2 else []
    pipe_args, _ = pipe_parser.parse_known_args(pipeline_tokens)
    pipe = pipe_group.extract(pipe_args)
    background = torch.tensor([1.0 if cfg["white_background"] else 0.0] * 3, device="cuda")
    destination = data["candidate"] / "diagnostics" / "kernel_ablation" / args.scene
    output_images = destination / "renders_thermal"
    require(not destination.exists(), f"Diagnostic already exists; preserve its evidence: {destination}")
    output_images.mkdir(parents=True)
    records = []
    on_metrics = []
    off_metrics = []
    official_hashes = {}
    render = renderer_module.render
    with torch.no_grad():
        for idx, camera in enumerate(cameras):
            name = data["names"][idx]
            model.ir_kernel_runtime_enabled = False
            off = render(camera, model, pipe, background)
            model.ir_kernel_runtime_enabled = True
            on = render(camera, model, pipe, background)
            require(torch.equal(off["render_color"], on["render_color"]), f"RGB changed with IR toggle: {name}")
            require(torch.equal(off["render_thermal"], on["render_thermal_base"]), f"Base IR changed: {name}")
            for key in ("render_color", "render_thermal", "ir_kernel_residual"):
                require(bool(torch.isfinite(on[key]).all()), f"Nonfinite render {key}: {name}")
            on_png = png_image(on["render_thermal"])
            off_png = png_image(off["render_thermal"])
            for modality, image in (("color", png_image(on["render_color"])), ("thermal", on_png)):
                path = data["official"] / f"renders_{modality}" / name
                require(images_equal(image, load_rgb(path)), f"On-render PNG pixels do not reproduce official {modality}/{name}")
                official_hashes[str(path.relative_to(data["candidate"]))] = sha256(path)
            gt_path = data["official"] / "gt_thermal" / name
            gt = load_rgb(gt_path)
            require(images_equal(png_image(camera.original_thermal.cuda()), gt),
                    f"Test camera/official IR ground truth mismatch: {name}")
            require(images_equal(png_image(camera.original_image.cuda()), load_rgb(data["official"] / "gt_color" / name)),
                    f"Test camera/official RGB ground truth mismatch: {name}")
            on_psnr = png_psnr(on_png, gt, psnr, "cuda")
            off_psnr = png_psnr(off_png, gt, psnr, "cuda")
            require(math.isclose(on_psnr, data["per_view"]["thermal_PSNR"][name], rel_tol=0, abs_tol=1e-5),
                    f"Official per-view PSNR does not reproduce: {name}")
            off_png.save(output_images / name)
            on_metrics.append(on_psnr)
            off_metrics.append(off_psnr)
            records.append({"filename": name, "camera_image_name": camera.image_name,
                            "on_ir_psnr": on_psnr, "off_ir_psnr": off_psnr,
                            "kernel_gain_db": on_psnr - off_psnr,
                            "residual_abs_mean": on["ir_kernel_residual"].abs().mean().item(),
                            "residual_abs_max": on["ir_kernel_residual"].abs().max().item(),
                            "off_png_sha256": sha256(output_images / name),
                            "official_gt_sha256": sha256(gt_path)})
            del off, on
            clear_render_cache(model)
        # Match metrics.py's float32 scene reduction, not channel averaging.
        mean_on = torch.tensor(on_metrics).mean().item()
        mean_off = torch.tensor(off_metrics).mean().item()
        require(math.isclose(mean_on, data["results"]["thermal_PSNR"], rel_tol=0, abs_tol=1e-5),
                "Official scene PSNR does not reproduce")
        timing = measure_inference(model, cameras[0], render, pipe, background, args.timing_repeats)
    for relative, digest in official_hashes.items():
        require(sha256(data["candidate"] / relative) == digest, f"Official PNG changed during diagnostic: {relative}")
    report = {
        "diagnostic_only": True, "accepted_final_candidate": False,
        "scope": "Same completed model, all test views, IR kernel enabled versus disabled; not ten-scene acceptance",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candidate_root": str(data["candidate"]), "scene": args.scene, "iteration": ITERATION,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "gpu_name": torch.cuda.get_device_name(0),
        "command": shlex.join([sys.executable, *sys.argv]),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "frozen_source_root": str(source), "frozen_runtime_root": str(runtime),
        "source_snapshot_sha256": sha256(source.parent / "code_snapshot.json"),
        "diagnostic_script_sha256": sha256(__file__), "pipeline": vars(pipe),
        "checkpoint_iteration": checkpoint_iteration, "training_receipt": data["receipt"],
        "export_kernel": stats, "checkpoint_kernel": checkpoint_kernel,
        "checkpoint_vs_export_note": "Official export may use EMA; ablation renders the PLY/module export, never the raw checkpoint.",
        "all_views_rgb_bitwise_equal": True, "all_views_base_ir_bitwise_equal": True,
        "all_views_on_pngs_reproduce_official_pixels": True,
        "test_view_count": len(records), "all_test_views_evaluated": True,
        "mean_ir_psnr_on": mean_on, "mean_ir_psnr_off": mean_off,
        "mean_kernel_gain_db": mean_on - mean_off,
        "mean_of_per_view_gain_db": sum(record["kernel_gain_db"] for record in records) / len(records),
        "metric_convention": "Frozen utils.image_utils.psnr on NCHW tensors from torchvision-quantized PNGs; float32 scene mean",
        "per_view": records, "inference_measurement": timing,
        "official_input_sha256": {name: sha256(data["folder"] / name) for name in ("results.json", "per_view.json", "cfg_args")},
        "official_render_sha256": official_hashes,
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"diagnostic_only": True, "scene": args.scene, "views": len(records),
                      "mean_ir_psnr_on": mean_on, "mean_ir_psnr_off": mean_off,
                      "mean_kernel_gain_db": mean_on - mean_off, "report": str(destination / "report.json")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--timing-repeats", type=int, default=20)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"IR kernel diagnostic failed: {exc}\n")


if __name__ == "__main__":
    main()
