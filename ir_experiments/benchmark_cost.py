#!/usr/bin/env python3
"""Measure completed frozen exports without changing parameters or artifacts.

CPU-only readiness check:
  python ir_experiments/benchmark_cost.py --candidate RUN --output NEW_DIR --preflight

Benchmark (only after the entire suite has finished):
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/lf/miniconda3/envs/thermalgaussian/lib \
  /home/lf/miniconda3/envs/thermalgaussian/bin/python \
  ir_experiments/benchmark_cost.py --candidate RUN --output NEW_DIR
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import importlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

# Importing these helpers does not import torch or initialize CUDA.
from ablate_kernel import MODEL_KEYS, clear_render_cache, preflight, read_namespace, require, sha256

SCENES = ("Building", "DailyStuff", "Dimsum", "Ebike", "IronIngot", "LandScape",
          "Parterre", "RoadBlock", "RotaryKiln", "Truck")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def active_suites():
    """Read actual /proc handles rather than trusting stale run status files."""
    processes = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / "cmdline").read_bytes().split(b"\0")
            tokens = [item.decode(errors="replace") for item in command if item]
        except (OSError, PermissionError):
            continue
        suite = any(Path(token).name == "run_suite.py" for token in tokens) and "run" in tokens
        # A child train.py can outlive a lost runner. Detect our workspace's
        # direct/frozen training processes too; retain only device visibility.
        train = next((token for token in tokens if Path(token).name == "train.py"), None)
        if train:
            try:
                working = (proc / "cwd").resolve()
                path = (working / train).resolve()
                in_workspace = path.is_relative_to(Path(__file__).resolve().parents[1])
                visible = next((item.split(b"=", 1)[1].decode() for item in
                                (proc / "environ").read_bytes().split(b"\0")
                                if item.startswith(b"CUDA_VISIBLE_DEVICES=")), None)
                suite = suite or (in_workspace and (visible is None or "0" in visible.split(",")))
            except OSError:
                # Unknown visibility of a known workspace training process
                # is insufficient evidence that the GPU is available.
                suite = suite or str(Path(__file__).resolve().parents[1]) in train
        if suite:
            processes.append({"pid": int(proc.name), "command": tokens})
    return processes


def cpu_readiness(args):
    """No torch, nvidia-smi, output writes, or CUDA API calls."""
    candidate, output = args.candidate.resolve(), args.output.resolve()
    errors = []
    if output.exists():
        errors.append(f"Output must be a new directory: {output}")
    if output.is_relative_to(candidate) or candidate.is_relative_to(output):
        errors.append("Diagnostic output must be separate from the official candidate directory")
    status_path = candidate / "run_status.json"
    try:
        status = json.loads(status_path.read_text())
        if status.get("state") != "finished" or status.get("ok") is not True:
            errors.append("The official suite has not finished successfully")
        scene_status = status.get("scenes", [])
        if {item.get("scene") for item in scene_status if item.get("ok")} != set(SCENES):
            errors.append("All ten scenes must finish before the cost benchmark")
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"Cannot verify suite completion: {exc}")
    suites = active_suites()
    if suites:
        errors.append("An actual suite runner or workspace GPU-0 training process is still active")
    scene_data = []
    if not errors:
        for name in args.scenes:
            try:
                data = preflight(candidate, name)
                receipt = data["receipt"]
                require(receipt.get("initialization") == "dataset"
                        and receipt.get("start_checkpoint") is None
                        and receipt.get("resumed_iteration") == 0
                        and receipt.get("optimizer_updates_this_process") == 30000,
                        f"Invalid from-scratch training receipt: {name}")
                scene_data.append(data)
            except (OSError, ValueError, KeyError) as exc:
                errors.append(str(exc))
    return {"ready": not errors, "cpu_only": True, "errors": errors,
            "active_suite_processes": suites, "candidate": str(candidate),
            "visible_proc_count": sum(path.name.isdigit() for path in Path("/proc").iterdir()),
            "output": str(output), "scenes": args.scenes}, scene_data


def gpu_snapshot(allow_shared):
    suites = active_suites()
    require(not suites, f"An active suite appeared; aborting benchmark: {suites}")
    result = subprocess.run(["nvidia-smi", "-i", "0", "-q", "-x"],
                            capture_output=True, text=True, check=True, timeout=20)
    document = ET.fromstring(result.stdout)
    devices = document.findall("gpu")
    require(len(devices) == 1, "Cannot identify exactly physical GPU 0 with nvidia-smi")
    device = devices[0]
    process_table = device.find("processes")
    require(process_table is not None and (process_table.text or "").strip() != "N/A",
            "nvidia-smi cannot inspect GPU processes; isolation is unverified")
    processes = []
    own_process_memory = None
    for process in device.findall("./processes/process_info"):
        pid_text = process.findtext("pid")
        require(pid_text and pid_text.isdigit(), "GPU process visibility is incomplete")
        pid = int(pid_text)
        if pid == os.getpid():
            own_process_memory = process.findtext("used_memory")
        if pid != os.getpid():
            try:
                tokens = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
                executable_names = [Path(token.decode(errors="replace")).name for token in tokens if token]
                # A namespaced /proc may conceal host GPU tasks. Such tasks
                # cannot be verified unrelated even with --allow-shared-gpu.
                inspectable = bool(executable_names)
            except OSError:
                executable_names, inspectable = [], False
            require(not allow_shared or inspectable,
                    f"Cannot inspect GPU PID {pid}; shared mode cannot rule out active training")
            require(not any(name in ("train.py", "run_suite.py") for name in executable_names),
                    f"GPU PID {pid} is active training; --allow-shared-gpu does not permit it")
            processes.append({"pid": pid, "type": process.findtext("type"),
                              "name": process.findtext("process_name"),
                              "command_inspectable": inspectable,
                              "used_memory": process.findtext("used_memory")})
    snapshot = {"utc": utc(), "gpu_uuid": device.findtext("uuid"),
                "gpu_name": device.findtext("product_name"),
                "pci_bus_id": device.findtext("./pci/pci_bus_id"),
                "gpu_utilization": device.findtext("./utilization/gpu_util"),
                "temperature": device.findtext("./temperature/gpu_temp"),
                "graphics_clock": device.findtext("./clocks/graphics_clock"),
                "sm_clock": device.findtext("./clocks/sm_clock"),
                "device_memory_used": device.findtext("./fb_memory_usage/used"),
                "benchmark_process_device_memory": own_process_memory,
                "other_processes": processes}
    require(allow_shared or not processes,
            "Other GPU 0 processes exist; benchmark refused. To explicitly collect non-isolated "
            f"measurements use --allow-shared-gpu. Observed processes: {processes}")
    return snapshot


def sample_indices(count, selection, representative_count):
    require(count > 0, "No cameras to benchmark")
    if selection == "all" or count <= representative_count:
        return list(range(count))
    return sorted({round(index * (count - 1) / (representative_count - 1))
                   for index in range(representative_count)})


def verify_frozen(data):
    source, runtime = data["source"], data["runtime"]
    snapshot = json.loads((source.parent / "code_snapshot.json").read_text())
    for name, digest in snapshot["file_sha256"].items():
        require(sha256(source / name) == digest, f"Frozen source changed: {name}")
    manifest = json.loads((runtime / "manifest.json").read_text())
    for package, record in manifest.items():
        for name, digest in record["files"].items():
            require(sha256(runtime / package / name) == digest,
                    f"Frozen runtime changed: {package}/{name}")
    return {"source_manifest_sha256": sha256(source.parent / "code_snapshot.json"),
            "runtime_manifest_sha256": sha256(runtime / "manifest.json")}


def imported_runtime(data):
    # No .pyc files are written into the frozen source/runtime trees.
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(data["runtime"]), str(data["source"])]
    import torch
    import scene
    import gaussian_renderer
    import diff_gaussian_rasterization
    import simple_knn._C
    from arguments import PipelineParams
    for module, root in ((scene, data["source"]), (gaussian_renderer, data["source"]),
                         (diff_gaussian_rasterization, data["runtime"]),
                         (simple_knn._C, data["runtime"])):
        require(Path(module.__file__).resolve().is_relative_to(root),
                f"Unexpected runtime module: {module.__file__}")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Exactly physical GPU 0 must be visible")
    torch.cuda.set_device(0)
    return torch, scene, gaussian_renderer.render, PipelineParams


def model_parameters(model, torch):
    seen = set()
    for value in vars(model).values():
        parameters = value.parameters() if isinstance(value, torch.nn.Module) else [value]
        for parameter in parameters:
            if isinstance(parameter, torch.nn.Parameter) and id(parameter) not in seen:
                seen.add(id(parameter))
                yield parameter


def measure(torch, model, operation, warmup, repeats):
    """Separate warmed timing from a single operation's allocator peak."""
    for _ in range(warmup):
        operation()
        clear_render_cache(model)
    torch.cuda.synchronize()
    wall_samples, device_samples = [], []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        begin.record()
        operation()
        end.record()
        torch.cuda.synchronize()
        wall_samples.append(1000.0 * (time.perf_counter() - started))
        device_samples.append(begin.elapsed_time(end))
        clear_render_cache(model)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    operation()
    torch.cuda.synchronize()
    memory = {"resident_allocated_bytes": allocated, "resident_reserved_bytes": reserved,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    memory["incremental_peak_allocated_bytes"] = memory["peak_allocated_bytes"] - allocated
    memory["incremental_peak_reserved_bytes"] = memory["peak_reserved_bytes"] - reserved
    clear_render_cache(model)
    return {"warmup": warmup, "repeats": repeats,
            "wall_ms_samples": wall_samples, "cuda_event_ms_samples": device_samples,
            "mean_wall_ms": statistics.mean(wall_samples),
            "median_wall_ms": statistics.median(wall_samples),
            "mean_cuda_event_ms": statistics.mean(device_samples), **memory}


def input_hashes(data):
    folder = data["folder"]
    names = ["cfg_args", "optimization_args", "training_receipt.json", "results.json", "per_view.json",
             "point_cloud/iteration_30000/point_cloud.ply",
             "point_cloud/iteration_30000/feature_modules.pth"]
    names += [str(path.relative_to(folder)) for path in data["official"].rglob("*.png")]
    return {name: sha256(folder / name) for name in names}


def fitting_weights(optimization):
    ssim_weight = float(optimization.get("ir_kernel_ssim_weight", 0.0))
    require(math.isfinite(ssim_weight) and ssim_weight >= 0.0,
            "Saved ir_kernel_ssim_weight must be finite and nonnegative")
    return {"mse": optimization["ir_kernel_mse_weight"],
            "regularization": optimization["ir_kernel_reg_weight"], "ssim": ssim_weight}


def load_frozen_ssim(source):
    module = importlib.import_module("utils.loss_utils")
    require(Path(module.__file__).resolve() == (source / "utils/loss_utils.py").resolve(),
            f"SSIM must come from the frozen candidate: {module.__file__}")
    return module.ssim


def ir_fitting_loss(torch, output, target, model, weights, frozen_ssim=None):
    """Match the saved IR-only objective, keeping absent/zero SSIM a no-op."""
    prediction = output["render_thermal_base"].detach() + output["ir_kernel_residual"]
    loss = (weights["mse"] * torch.nn.functional.mse_loss(prediction, target)
            + weights["regularization"] * model.get_ir_kernel_regularization())
    if weights["ssim"] > 0.0:
        require(frozen_ssim is not None, "Positive SSIM weight requires frozen repository SSIM")
        ssim_loss = 1.0 - frozen_ssim(prediction.unsqueeze(0), target.unsqueeze(0))
        loss = loss + weights["ssim"] * ssim_loss
    return loss


def benchmark_scene(args, data, modules, observations):
    torch, scene_module, render, PipelineParams = modules
    hashes = input_hashes(data)
    cfg = dict(data["cfg"], model_path=str(data["folder"]))
    model = scene_module.GaussianModel(cfg["sh_degree"], **{key: cfg[key] for key in MODEL_KEYS})
    scene = scene_module.Scene(argparse.Namespace(**cfg), model, load_iteration=30000, shuffle=False)
    require(model.use_ir_kernel and model.ir_kernel_runtime_enabled, "Export disabled the IR kernel")
    parser = argparse.ArgumentParser(add_help=False)
    group = PipelineParams(parser)
    status = json.loads((data["candidate"] / f"{data['folder'].name}.status.json").read_text())
    command = status["render"]["command"]
    if isinstance(command, str):
        command = shlex.split(command)
    pipeline_args, _ = parser.parse_known_args(command[2:])
    pipe = group.extract(pipeline_args)
    background = torch.tensor([float(cfg["white_background"])] * 3, device="cuda")
    kernel = model._ir_kernel
    require(tuple(kernel.shape) == (model.get_xyz.shape[0], 10), "Unexpected IR kernel dimensions")
    original_kernel = kernel.detach().cpu().clone()
    storage = {"gaussian_count": kernel.shape[0], "parameter_count": kernel.numel(),
               "dtype": str(kernel.dtype), "element_bytes": kernel.element_size(),
               "raw_parameter_bytes": kernel.numel() * kernel.element_size(),
               "optimizer_state_allocated": False, "kernel_storage_present_in_both_on_off_modes": True}
    cameras = scene.getTestCameras()
    require(len(cameras) == len(data["names"]), "Test camera count differs from official output")
    indices = sample_indices(len(cameras), args.views, args.representative_count)
    records = []
    with torch.no_grad():
        for ordinal, index in enumerate(indices):
            camera = cameras[index]
            row = {"view_index": index, "camera_image_name": camera.image_name,
                   "width": camera.image_width, "height": camera.image_height, "split": "test"}
            # Alternating order reduces systematic on/off bias from slow drift.
            modes = (False, True) if ordinal % 2 == 0 else (True, False)
            row["mode_order"] = ["on" if enabled else "off" for enabled in modes]
            for enabled in modes:
                observations.append(gpu_snapshot(args.allow_shared_gpu))
                model.ir_kernel_runtime_enabled = enabled
                def inference():
                    output = render(camera, model, pipe, background)
                    del output
                row["on" if enabled else "off"] = measure(torch, model, inference, args.warmup, args.repeats)
                observations.append(gpu_snapshot(args.allow_shared_gpu))
            row["wall_time_ratio_on_over_off"] = row["on"]["mean_wall_ms"] / row["off"]["mean_wall_ms"]
            row["incremental_peak_overhead_bytes"] = row["on"]["incremental_peak_allocated_bytes"] - row["off"]["incremental_peak_allocated_bytes"]
            records.append(row)
    gradient_rows = []
    fit_configuration = {"enabled": bool(args.fit_gradient), "optimizer_created": False,
                         "optimizer_updates": 0, "parameter_updates": 0, "additional_training_iterations": 0}
    if args.fit_gradient:
        model.ir_kernel_runtime_enabled = True
        for parameter in model_parameters(model, torch):
            parameter.requires_grad_(parameter is kernel)
            parameter.grad = None
        optimization = read_namespace(data["folder"] / "optimization_args")
        weights = fitting_weights(optimization)
        frozen_ssim = load_frozen_ssim(data["source"]) if weights["ssim"] > 0.0 else None
        fit_configuration.update(
            component_weights=weights,
            loss_scope="MSE + parameter regularization + (1 - SSIM)" if frozen_ssim else "MSE + parameter regularization",
            prediction="raw detached base IR + differentiable kernel residual; no clamp or quantization",
            split="train", differentiated_parameters="IR kernel only",
            includes_full_render_forward=True, includes_full_training_backward=False,
            ssim_source=str(data["source"] / "utils/loss_utils.py") if frozen_ssim else None,
            ssim_source_sha256=sha256(data["source"] / "utils/loss_utils.py") if frozen_ssim else None,
        )
        training_cameras = scene.getTrainCameras()
        # Gradients are formed on training views only and are never applied.
        for index in sample_indices(len(training_cameras), args.views, args.representative_count):
            observations.append(gpu_snapshot(args.allow_shared_gpu))
            camera = training_cameras[index]
            target = camera.original_thermal.cuda()
            def fit_gradient(check=False):
                output = render(camera, model, pipe, background)
                loss = ir_fitting_loss(torch, output, target, model, weights, frozen_ssim)
                gradient, = torch.autograd.grad(loss, kernel, create_graph=False, retain_graph=False)
                if check:
                    require(bool(torch.isfinite(loss)) and bool(torch.isfinite(gradient).all()),
                            "Non-finite IR fitting loss or gradient")
                del gradient, loss, output
            fit_gradient(check=True)
            clear_render_cache(model)
            result = measure(torch, model, fit_gradient, args.warmup, args.repeats)
            result.update(view_index=index, camera_image_name=camera.image_name, split="train",
                          finite_loss_and_gradient=True)
            gradient_rows.append(result)
            del target
            observations.append(gpu_snapshot(args.allow_shared_gpu))
        require(kernel.grad is None, "Diagnostic unexpectedly accumulated parameter gradients")
    require(torch.equal(original_kernel, kernel.detach().cpu()), "IR parameter values changed during diagnostic")
    for name, digest in hashes.items():
        require(sha256(data["folder"] / name) == digest, f"Official input changed: {name}")
    result = {"scene": data["folder"].name, "iteration": 30000, "parameter_storage": storage,
              "total_test_views": len(cameras), "tested_view_indices": indices,
              "pipeline": vars(pipe), "data_device": cfg["data_device"],
              "inference": records, "fit_gradient_diagnostic": gradient_rows,
              "fit_gradient_configuration": fit_configuration,
              "official_input_sha256_before_and_after": hashes, "ir_parameters_unchanged": True}
    for mode in ("on", "off"):
        result[f"{mode}_mean_wall_ms"] = statistics.mean(row[mode]["mean_wall_ms"] for row in records)
        result[f"{mode}_max_peak_allocated_bytes"] = max(row[mode]["peak_allocated_bytes"] for row in records)
        result[f"{mode}_max_incremental_peak_allocated_bytes"] = max(row[mode]["incremental_peak_allocated_bytes"] for row in records)
    result["wall_time_ratio_on_over_off"] = result["on_mean_wall_ms"] / result["off_mean_wall_ms"]
    del scene, model, kernel, cameras, background
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(args):
    readiness, data = cpu_readiness(args)
    if args.preflight:
        print(json.dumps(readiness, indent=2))
        return 0 if readiness["ready"] else 2
    require(readiness["ready"], "; ".join(readiness["errors"]))
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Set CUDA_VISIBLE_DEVICES=0")
    observations = [gpu_snapshot(args.allow_shared_gpu)]
    provenance = verify_frozen(data[0])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "benchmark_cost.py")
    shutil.copy2(Path(__file__).with_name("ablate_kernel.py"), output / "ablate_kernel.py")
    report = {"state": "running", "started_utc": utc(), "candidate": str(args.candidate.resolve()),
              "diagnostic_only": True, "command": shlex.join([sys.executable, *sys.argv]),
              "pid": os.getpid(), "cuda_visible_devices": "0", "provenance": provenance,
              "diagnostic_script_sha256": sha256(__file__),
              "helper_script_sha256": sha256(Path(__file__).with_name("ablate_kernel.py")),
              "source_root": str(data[0]["source"]), "runtime_root": str(data[0]["runtime"]),
              "view_selection": args.views, "representative_count": args.representative_count,
              "scenes": [], "gpu_observations": observations,
              "limitations": [
                  "On/off compares the same completed exported model; it is not a baseline retraining time comparison.",
                  "Both modes retain the IR parameter tensor; raw IR storage is reported separately.",
                  "Allocator resident memory includes the model and all cameras/images loaded by frozen Scene; peaks exclude driver and non-PyTorch allocations.",
                  "Memory uses a separate single warmed operation; timing excludes loading, file I/O, warmup, GPU process polling and empty_cache.",
                  "The optional gradient diagnostic computes full rendering plus saved IR-only MSE/regularization and positive-weight SSIM, then autograd.grad on training views; all other model parameters are frozen. Each scene records component weights and exact scope.",
                  "No optimizer is created, no parameters are updated, and no training iterations are added. This does not measure full training backward, densification, Adam state, or end-to-end training overhead.",
                  "GPU process sampling is not an exclusive reservation and cannot exclude tasks that appear between samples; device clocks and temperature are recorded.",
              ]}
    write_json(output / "report.json", report)
    try:
        modules = imported_runtime(data[0])
        for scene_data in data:
            print(f"[{utc()}] Cost benchmark: {scene_data['folder'].name}", flush=True)
            record = benchmark_scene(args, scene_data, modules, observations)
            report["scenes"].append(record)
            write_json(output / f"{record['scene']}.json", record)
            write_json(output / "report.json", report)
        require(verify_frozen(data[0]) == provenance, "Frozen provenance changed during diagnostics")
        report["state"] = "finished"
    except BaseException as exc:
        report["state"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_utc"] = utc()
        report["shared_gpu_observed"] = any(item["other_processes"] for item in observations)
        report["isolation_assessment"] = ("NON-ISOLATED: other GPU processes were observed"
            if report["shared_gpu_observed"] else "No other GPU processes observed at sample boundaries; no exclusive reservation")
        if report["state"] != "finished":
            report["isolation_assessment"] = "INCOMPLETE/ABORTED: do not interpret this run as an isolated estimate"
        write_json(output / "report.json", report)
    columns = ["scene", "off_mean_wall_ms", "on_mean_wall_ms", "wall_time_ratio_on_over_off",
               "off_max_peak_allocated_bytes", "on_max_peak_allocated_bytes",
               "off_max_incremental_peak_allocated_bytes", "on_max_incremental_peak_allocated_bytes"]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: record[key] for key in columns} for record in report["scenes"])
    print(json.dumps({"report": str(output / "report.json"), "scenes": len(report["scenes"]),
                      "isolation_assessment": report["isolation_assessment"]}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scenes", choices=SCENES, nargs="+", default=list(SCENES))
    parser.add_argument("--views", choices=("all", "representative"), default="representative")
    parser.add_argument("--representative-count", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--fit-gradient", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true", help="explicitly label measurements non-isolated if other GPU processes exist")
    parser.add_argument("--preflight", action="store_true", help="CPU-only readiness check; no CUDA or output writes")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1 or args.representative_count < 2:
        parser.error("warmup/repeats must be positive; representative count must be at least 2")
    if len(args.scenes) != len(set(args.scenes)):
        parser.error("duplicate scenes are not allowed")
    try:
        return run(args)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, ET.ParseError) as exc:
        parser.exit(1, f"Cost benchmark failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
