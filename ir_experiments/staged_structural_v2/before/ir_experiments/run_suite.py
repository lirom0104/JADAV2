#!/usr/bin/env python3
"""Run the IR-kernel experiment one scene at a time on physical GPU 0.

This is deliberately a small, self-contained runner.  It does not source or
invoke any of the historical batch scripts.  A run directory is immutable
once created and contains commands, environment, logs, timing and validation
state for every scene.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any


SCENES = (
    "Building", "DailyStuff", "Dimsum", "Ebike", "IronIngot", "LandScape",
    "Parterre", "RoadBlock", "RotaryKiln", "Truck",
)
METRICS = (
    "color_PSNR", "thermal_PSNR", "color_SSIM", "thermal_SSIM",
    "color_LPIPS", "thermal_LPIPS",
)
DEFAULT_DATASET = Path("/home/lf/data/thermal3dgs/RGBT-Scenes")
DEFAULT_BASELINE = Path(
    "/home/lf/code/Our_Project-New-2/output/"
    "odb_30k_20261003_101217_1051565"
)
DEFAULT_PYTHON = Path("/home/lf/miniconda3/envs/thermalgaussian/bin/python")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_dump(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_namespace(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(), mode="eval").body
    if not isinstance(tree, ast.Call) or not isinstance(tree.func, ast.Name) or tree.func.id != "Namespace" or tree.args:
        raise ValueError(f"invalid saved Namespace: {path}")
    return {item.arg: ast.literal_eval(item.value) for item in tree.keywords}


def snapshot_code(repo: Path, destination: Path) -> dict[str, Any]:
    """Archive source files and record hashes, including uncommitted files."""
    destination.mkdir(parents=True)
    include = []
    excluded = {".git", "__pycache__", "output", "build", ".pytest_cache", "initial_state"}
    suffixes = {".py", ".md", ".sh", ".json", ".yml", ".yaml", ".cu", ".cpp", ".h", ".hpp", ".toml", ".txt"}
    for folder, dirs, names in os.walk(repo):
        dirs[:] = [name for name in dirs if name not in excluded
                   and not (Path(folder) / name).is_relative_to(destination.parent)]
        for name in names:
            path = Path(folder) / name
            if path.is_file() and path.suffix in suffixes:
                include.append(path)
    include.sort()
    frozen_repo = destination / "source"
    frozen_repo.mkdir()
    for path in include:
        copied = frozen_repo / path.relative_to(repo)
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copied)
    archive = destination / "code_snapshot.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in include:
            tar.add(frozen_repo / path.relative_to(repo), arcname=path.relative_to(repo))
    hashes = {str(path.relative_to(repo)): sha256_file(frozen_repo / path.relative_to(repo)) for path in include}
    try:
        status = subprocess.run(
            ["git", "status", "--short"], cwd=repo, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        ).stdout
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--binary"], cwd=repo, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        ).stdout
    except OSError as exc:
        status, commit, diff = f"git unavailable: {exc}", "unknown", ""
    (destination / "git_diff.patch").write_text(diff)
    metadata = {
        "created_utc": utc_now(),
        "git_commit": commit,
        "git_status": status,
        "archive": str(archive.name),
        "archive_sha256": sha256_file(archive),
        "executed_source_root": str(frozen_repo),
        "file_sha256": hashes,
    }
    json_dump(destination / "code_snapshot.json", metadata)
    return metadata


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def freeze_extensions(python: Path, root: Path) -> dict[str, Any]:
    """Keep the two CUDA packages stable if a shared environment is rebuilt."""
    script = (
        "import importlib.util,json; "
        "out={}; "
        "\nfor n in ('diff_gaussian_rasterization','simple_knn'):\n"
        " s=importlib.util.find_spec(n); out[n]=s.origin or next(iter(s.submodule_search_locations),None)\n"
        "print(json.dumps(out))"
    )
    packages = json.loads(subprocess.check_output([str(python), "-c", script], text=True))
    destination = root / "runtime_packages"
    destination.mkdir()
    manifest = {}
    for name, origin in packages.items():
        source = Path(origin)
        if source.is_file():
            source = source.parent
        target = destination / name
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        manifest[name] = {"original_path": str(source), "files": {
            str(path.relative_to(target)): sha256_file(path)
            for path in sorted(target.rglob("*")) if path.is_file()
        }}
    json_dump(destination / "manifest.json", manifest)
    return {"path": str(destination), "packages": manifest}


def base_train_command(
    repo: Path, python: Path, dataset_root: Path, model_root: Path,
    scene: str, iterations: int, kernel_start: int = 18000,
    kernel_lr: float = 0.003, kernel_regularization: float = 0.0001,
    kernel_amplitude: float = 0.2, kernel_mse_weight: float = 1.0,
    port: int = 6027, detail_start: int = 26000,
) -> list[str]:
    """The fixed baseline ODB-26000 settings plus the IR Gaussian branch."""
    command = [
        str(python), str(repo / "train.py"),
        "-s", str(dataset_root / scene), "-m", str(model_root / scene),
        "--iterations", str(iterations), "--test_iterations", str(iterations),
        "--save_iterations", str(iterations), "--checkpoint_iterations", str(iterations), "--quiet",
        "--port", str(port),
        # Baseline model-side ODB options.
        "--use_bgfc", "--use_at_gom", "--use_paired_views",
        "--use_camera_calibration", "--use_render_calibration",
        "--use_color_refinement", "--use_detail_basis",
        "--detail_basis_mode", "oriented_hermite",
        "--detail_basis_scale", "0.08", "--detail_basis_thermal_scale", "0.06",
        # ODB is activated at iteration 26000 in the common baseline schedule.
        "--detail_basis_start_iter", str(detail_start), "--detail_basis_lr", "0.001",
        "--detail_basis_reg_weight", "0.005", "--detail_edge_weight", "0.05",
        # IR-only learned Gaussian kernel.
        "--use_ir_kernel", "--ir_kernel_amplitude", str(kernel_amplitude),
        "--ir_kernel_start_iter", str(kernel_start), "--ir_kernel_lr", str(kernel_lr),
        "--ir_kernel_reg_weight", str(kernel_regularization),
        "--ir_kernel_mse_weight", str(kernel_mse_weight),
    ]
    # There is intentionally no --start_checkpoint: every candidate starts
    # from the dataset initialization.  Keep that invariant visible in logs.
    return command


def run_process(
    command: list[str], env: dict[str, str], log_path: Path, cwd: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    started_at = utc_now()
    with log_path.open("w") as log:
        log.write(f"started_utc={started_at}\n")
        log.write(f"cwd={cwd}\n")
        log.write(f"env_CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}\n")
        log.write(f"command={command_text(command)}\n\n")
        log.flush()
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        pid = process.pid
        process_path = log_path.with_suffix(".process.json")
        process_info = {"pid": pid, "parent_pid": os.getpid(), "started_utc": started_at,
                        "state": "running", "command": command, "cwd": str(cwd)}
        json_dump(process_path, process_info)
        json_dump(log_path.parent.parent / "active_process.json", process_info)
        print(f"[{started_at}] PID {pid}: {command_text(command)}", flush=True)
        before = process.poll()
        return_code = process.wait()
        after = process.poll()
        log.write(f"\nreturn_code={return_code}\n")
    finished = time.monotonic()
    result = {
        "pid": pid, "started_utc": started_at, "finished_utc": utc_now(),
        "elapsed_seconds": finished - started, "return_code": return_code,
        "process_state_before_wait": before, "process_state_after_wait": after,
        "log": str(log_path), "command": command, "command_shell": command_text(command),
    }
    json_dump(process_path, {**process_info, **result, "state": "finished"})
    json_dump(log_path.parent.parent / "active_process.json", {**process_info, **result, "state": "finished"})
    print(f"[{utc_now()}] PID {pid} finished, return code {return_code}", flush=True)
    return result


def _png_files(path: Path) -> list[Path]:
    return sorted(p for p in path.glob("*.png") if p.is_file())


def compare_pngs(expected_dir: Path, actual_dir: Path) -> dict[str, Any]:
    expected, actual = _png_files(expected_dir), _png_files(actual_dir)
    expected_names = {p.name for p in expected}
    actual_names = {p.name for p in actual}
    report: dict[str, Any] = {
        "expected_count": len(expected), "actual_count": len(actual),
        "missing": sorted(expected_names - actual_names),
        "extra": sorted(actual_names - expected_names), "pixel_exact": True,
        "comparison": "sha256",
    }
    try:
        from PIL import Image  # type: ignore
        report["comparison"] = "Pillow pixel comparison"
        for name in sorted(expected_names & actual_names):
            with Image.open(expected_dir / name) as lhs, Image.open(actual_dir / name) as rhs:
                if lhs.mode != rhs.mode or lhs.size != rhs.size or lhs.tobytes() != rhs.tobytes():
                    report["pixel_exact"] = False
                    report.setdefault("mismatched", []).append(name)
    except ImportError:
        for name in sorted(expected_names & actual_names):
            if sha256_file(expected_dir / name) != sha256_file(actual_dir / name):
                report["pixel_exact"] = False
                report.setdefault("mismatched", []).append(name)
    return report


def finite_scalar(value: Any) -> float:
    # metrics.py has historically emitted PSNR as one-element lists per view.
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"invalid metric scalar: {value!r}")
    return float(value)


def validate_artifacts(model_root: Path, scene: str, baseline_root: Path,
                       iterations: int) -> dict[str, Any]:
    scene_root = model_root / scene
    method = f"ours_{iterations}"
    point_cloud = scene_root / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    checkpoint = scene_root / f"chkpnt{iterations}.pth"
    render_root = scene_root / "test" / method
    result_path, view_path = scene_root / "results.json", scene_root / "per_view.json"
    validation: dict[str, Any] = {
        "scene": scene, "method": method, "model_root": str(scene_root),
        "point_cloud": {"path": str(point_cloud), "exists": point_cloud.is_file()},
        "checkpoint": {"path": str(checkpoint), "exists": checkpoint.is_file()},
        "render_root": {"path": str(render_root), "exists": render_root.is_dir()},
        "valid": False,
    }
    if not result_path.is_file() or not view_path.is_file():
        validation["error"] = "results.json or per_view.json missing"
        return validation
    try:
        if not point_cloud.is_file() or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise ValueError("model or checkpoint artifact missing/empty")
        with point_cloud.open("rb") as handle:
            header = []
            for _ in range(2048):
                line = handle.readline()
                header.append(line.decode("ascii"))
                if line == b"end_header\n":
                    break
            else:
                raise ValueError("PLY header missing or too long")
        header_text = "".join(header)
        if any(f"property float ir_kernel_{index}\n" not in header_text for index in range(10)):
            raise ValueError("PLY does not export all ten IR-kernel parameters")
        vertices = [int(line.split()[2]) for line in header if line.startswith("element vertex ")]
        if not vertices or vertices[0] <= 0:
            raise ValueError("PLY has no Gaussians")
        feature_modules = point_cloud.parent / "feature_modules.pth"
        if not feature_modules.is_file() or feature_modules.stat().st_size == 0:
            raise ValueError("feature module export missing/empty")
        model_config = read_namespace(scene_root / "cfg_args")
        baseline_config = read_namespace(baseline_root / scene / "cfg_args")
        for key in ("resolution", "images", "thermal", "white_background", "eval", "use_paired_views", "use_camera_calibration"):
            if model_config[key] != baseline_config[key]:
                raise ValueError(f"dataset/render protocol differs from baseline: {key}")
        optimization = read_namespace(scene_root / "optimization_args")
        if optimization["iterations"] != iterations or not model_config["use_ir_kernel"]:
            raise ValueError("saved configuration has incorrect budget or disabled IR kernel")
        receipt = json.loads((scene_root / "training_receipt.json").read_text())
        if (receipt.get("initialization") != "dataset"
                or receipt.get("start_checkpoint") is not None
                or receipt.get("resumed_iteration") != 0
                or receipt.get("final_iteration") != iterations
                or receipt.get("optimizer_updates_this_process") != iterations
                or receipt.get("cumulative_iterations") != iterations
                or receipt.get("cuda_visible_devices") != "0"):
            raise ValueError("training receipt does not prove from-scratch exact-budget GPU-0 training")
        validation["training_receipt"] = receipt
        validation["gaussian_count"] = vertices[0]
        validation["saved_optimization"] = optimization
        validation["saved_model_configuration"] = model_config
        results = json.loads(result_path.read_text())
        per_view = json.loads(view_path.read_text())
        values = results[method]
        values = {key: finite_scalar(values[key]) for key in METRICS}
        views = per_view[method]
        counts = {key: len(views[key]) for key in METRICS}
        image_names = {p.name for p in _png_files(render_root / "gt_color")}
        if not image_names or any(set(views[key]) != image_names for key in METRICS):
            raise ValueError("per-view metric names differ from rendered image names")
        if any({p.name for p in _png_files(render_root / d)} != image_names for d in ("renders_color", "renders_thermal", "gt_thermal")):
            raise ValueError(f"invalid per-view counts or render files: {counts}")
        for key in METRICS:
            view_mean = sum(finite_scalar(value) for value in views[key].values()) / len(image_names)
            if not math.isclose(view_mean, values[key], rel_tol=2e-6, abs_tol=2e-6):
                raise ValueError(f"aggregate {key} differs from all per-view values")
        validation["metrics"] = {key: float(values[key]) for key in METRICS}
        validation["per_view_counts"] = counts
        validation["metric_file_sha256"] = {path.name: sha256_file(path) for path in (result_path, view_path)}
        validation["model_artifact_sizes"] = {str(path.relative_to(scene_root)): path.stat().st_size for path in (point_cloud, checkpoint, feature_modules)}
        validation["gt_color"] = compare_pngs(
            baseline_root / scene / "test" / "ours_30000" / "gt_color", render_root / "gt_color")
        validation["gt_thermal"] = compare_pngs(
            baseline_root / scene / "test" / "ours_30000" / "gt_thermal", render_root / "gt_thermal")
        validation["valid"] = bool(
            validation["point_cloud"]["exists"] and validation["checkpoint"]["exists"]
            and validation["render_root"]["exists"] and all(
                validation[key]["pixel_exact"] and validation[key]["expected_count"] > 0
                and not validation[key]["missing"] and not validation[key]["extra"]
                for key in ("gt_color", "gt_thermal")
            )
        )
    except (OSError, KeyError, TypeError, ValueError, SyntaxError, json.JSONDecodeError) as exc:
        validation["error"] = str(exc)
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="train, render and validate serially")
    run.add_argument("--root", required=True, type=Path, help="new run output directory")
    run.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    run.add_argument("--iterations", type=int, default=30000)
    run.add_argument("--debug", action="store_true", help="allow subset/short smoke runs")
    run.add_argument("--ir-kernel-start-iter", type=int, default=18000)
    run.add_argument("--ir-kernel-lr", type=float, default=0.003)
    run.add_argument("--ir-kernel-reg-weight", type=float, default=0.0001)
    run.add_argument("--ir-kernel-amplitude", type=float, default=0.2)
    run.add_argument("--ir-kernel-mse-weight", type=float, default=1.0)
    run.add_argument("--detail-basis-start-iter", type=int, default=26000)
    run.add_argument("--port", type=int, default=6027)
    run.add_argument("--dry-run", action="store_true", help="print plan without creating files or using CUDA")
    run.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    run.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    run.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    run.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def run_suite(args: argparse.Namespace) -> int:
    repo, root = args.repo.resolve(), args.root.resolve()
    scenes = list(args.scenes)
    if len(set(scenes)) != len(scenes):
        raise SystemExit("duplicate scenes are not allowed")
    if set(scenes) == set(SCENES) and args.iterations != 30000:
        raise SystemExit("the full ten-scene suite requires --iterations 30000; use a scene subset for smoke")
    if args.iterations != 30000 and not args.debug:
        raise SystemExit("short smoke runs require --debug and an explicit --iterations")
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.iterations != 30000 and (args.ir_kernel_start_iter != 0 or args.detail_basis_start_iter != 0):
        raise SystemExit("smoke requires --ir-kernel-start-iter 0 --detail-basis-start-iter 0")
    commands = {scene: base_train_command(
        repo, args.python, args.dataset_root, root, scene, args.iterations,
        args.ir_kernel_start_iter, args.ir_kernel_lr, args.ir_kernel_reg_weight,
        args.ir_kernel_amplitude, args.ir_kernel_mse_weight, args.port, args.detail_basis_start_iter) for scene in scenes}
    if args.dry_run:
        print(json.dumps({"CUDA_VISIBLE_DEVICES": "0", "root": str(root), "commands": commands}, indent=2))
        return 0
    if root.exists():
        raise SystemExit(f"run root already exists: {root}")
    if not args.python.is_file():
        raise SystemExit(f"Python executable not found: {args.python}")
    for scene in scenes:
        for subfolder in ("rgb/train", "rgb/test", "thermal/train", "thermal/test", "sparse/0"):
            if not (args.dataset_root / scene / subfolder).is_dir():
                raise SystemExit(f"missing dataset folder: {args.dataset_root / scene / subfolder}")
        if not (args.baseline_root / scene / "results.json").is_file():
            raise SystemExit(f"missing fixed baseline result for {scene}")
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    snapshot = snapshot_code(repo, root / "code_snapshot")
    runtime = freeze_extensions(args.python, root)
    original_repo = repo
    repo = Path(snapshot["executed_source_root"])
    commands = {scene: base_train_command(
        repo, args.python, args.dataset_root, root, scene, args.iterations,
        args.ir_kernel_start_iter, args.ir_kernel_lr, args.ir_kernel_reg_weight,
        args.ir_kernel_amplitude, args.ir_kernel_mse_weight, args.port, args.detail_basis_start_iter) for scene in scenes}
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    env["LD_LIBRARY_PATH"] = str(args.python.resolve().parent.parent / "lib") + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["PYTHONPATH"] = os.pathsep.join((runtime["path"], str(repo)))
    metadata = {
        "created_utc": utc_now(), "repo": str(original_repo), "executed_source_root": str(repo), "root": str(root),
        "runner_pid": os.getpid(), "runner_command": sys.argv,
        "dataset_root": str(args.dataset_root.resolve()), "baseline_root": str(args.baseline_root.resolve()),
        "python": str(args.python.resolve()), "scenes": scenes, "iterations": args.iterations,
        "debug": bool(args.debug), "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "kernel_configuration": {key: value for key, value in vars(args).items() if key.startswith("ir_kernel_")},
        "initialization": "dataset, no start checkpoint", "seed": "unchanged train.py safe_state default",
        "environment": {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "PYTHONUNBUFFERED", "PYTHONPATH", "LD_LIBRARY_PATH", "PATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS") if key in env},
        "code_snapshot": snapshot,
        "runtime_extensions": runtime,
    }
    json_dump(root / "run_metadata.json", metadata)
    baseline_reference = {}
    for scene in SCENES:
        baseline_scene = args.baseline_root / scene
        baseline_reference[scene] = {
            "results": json.loads((baseline_scene / "results.json").read_text())["ours_30000"],
            "file_sha256": {name: sha256_file(baseline_scene / name)
                            for name in ("results.json", "per_view.json", "cfg_args", "optimization_args")},
        }
    json_dump(root / "baseline_reference.json", baseline_reference)
    statuses = []
    for scene in scenes:
        scene_log = root / "logs" / f"{scene}.train.log"
        train = run_process(commands[scene], env, scene_log, repo)
        render = None
        metrics = None
        validation = None
        model_exists = (root / scene / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply").is_file()
        if train["return_code"] == 0 and model_exists:
            render_cmd = [str(args.python), str(repo / "render.py"), "-m", str(root / scene), "--iteration", str(args.iterations), "--skip_train", "--quiet"]
            render = run_process(render_cmd, env, root / "logs" / f"{scene}.render.log", repo)
        if render is not None and render["return_code"] == 0:
            metrics_cmd = [str(args.python), str(repo / "metrics.py"), "--model_paths", str(root / scene)]
            metrics = run_process(metrics_cmd, env, root / "logs" / f"{scene}.metrics.log", repo)
        if metrics is not None and metrics["return_code"] == 0:
            validation = validate_artifacts(root, scene, args.baseline_root, args.iterations)
        status = {"scene": scene, "train": train, "render": render, "metrics": metrics, "validation": validation,
                  "ok": bool(validation and validation.get("valid"))}
        statuses.append(status)
        json_dump(root / f"{scene}.status.json", status)
        json_dump(root / "run_status.json", {"state": "running", "scene_count": len(statuses), "scenes": statuses})
        if not status["ok"]:
            # Do not spend more GPU time after a failed stage or invalid export.
            break
    summary = {"state": "finished", "finished_utc": utc_now(), "scene_count": len(statuses), "planned_scene_count": len(scenes), "ok": len(statuses) == len(scenes) and all(s["ok"] for s in statuses), "scenes": statuses}
    json_dump(root / "run_status.json", summary)
    if not args.skip_report:
        report_cmd = [str(args.python), str(repo / "ir_experiments" / "report_results.py"), "--candidate", str(root), "--baseline", str(args.baseline_root), "--iteration", str(args.iterations)]
        report = run_process(report_cmd, env, root / "logs" / "report.log", repo)
        summary["report"] = report
        summary["ok"] = summary["ok"] and report["return_code"] == 0
        json_dump(root / "run_status.json", summary)
    return 0 if summary["ok"] else 1


def main() -> int:
    args = parse_args()
    if args.action == "run":
        return run_suite(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
