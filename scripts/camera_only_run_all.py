"""Train both datasets with camera calibration as the only added method.

The original BGFC/AT-GOM/CMO model remains enabled. Existing batch outputs supply test
targets and comparison metrics; their model parameters are never loaded.
"""

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

from batch_common import JOBS, PRIMARY, stop_workers

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "output/CWGC_seed0_20260925_130204_2918293"
TRAIN_ARGS = [
    "--iterations", "30000", "--use_camera_calibration",
    "--save_iterations", "30000", "--checkpoint_iterations", "30000",
]


def commands(output, reference):
    workers = []
    for dataset, gpu, port, _, scenes in JOBS:
        workers.append([
            sys.executable, "-u", str(ROOT / "scripts/cwgc_run.py"),
            "--root", str(output), "--gpu", str(gpu), "--seed", "0",
            "--port", str(port), "--reference-root", str(reference), "--scenes",
            *[f"{dataset}/{scene}" for scene in scenes], "--", *TRAIN_ARGS,
        ])
    report = [
        sys.executable, str(ROOT / "scripts/cwgc_dataset_report.py"),
        "--rgbt-root", str(output), "--thermo-root", str(output),
        "--supplied", str(reference), "--mode", "camera_only",
        "--output", str(output / "comparison.json"),
    ]
    return workers, report


def preflight(output, reference, dry_run):
    if output == reference or reference in output.parents or output in reference.parents:
        raise ValueError("输出目录与已有结果目录不能重叠。")
    if output.exists():
        raise FileExistsError(f"输出目录已存在，请使用新目录：{output}")
    for script in ("train.py", "render.py", "metrics.py", "extra_metrics.py", "wendu.py",
                   "scripts/cwgc_run.py", "scripts/cwgc_dataset_report.py"):
        if not (ROOT / script).is_file():
            raise FileNotFoundError(ROOT / script)
    for dataset, _, port, data, scenes in JOBS:
        for scene in scenes:
            for part in ("sparse/0", "rgb/train", "rgb/test", "thermal/train", "thermal/test"):
                if not (data / scene / part).is_dir():
                    raise FileNotFoundError(f"数据目录缺失：{data / scene / part}")
            saved = reference / dataset / scene
            results = json.loads((saved / "results.json").read_text()).get("ours_30000", {})
            if not all(metric in results and math.isfinite(results[metric]) for metric in PRIMARY):
                raise ValueError(f"已有结果缺少有效的 30000 步指标：{saved}")
            for modality in ("color", "thermal"):
                target = saved / "test/ours_30000" / f"gt_{modality}"
                if not any(target.glob("*.png")):
                    raise FileNotFoundError(f"缺少用于校验的测试目标：{target}")
        if not dry_run:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
    if not dry_run:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1")
        subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available() and "
                        "torch.cuda.device_count() >= 2, '需要可用的 GPU 0 和 GPU 1'"],
                       env=env, check=True)


def write_summary(output, reference):
    report = json.loads((output / "comparison.json").read_text())
    datasets = report["datasets"]
    if report.get("mode") != "camera_only" or set(datasets) != {job[0] for job in JOBS}:
        raise ValueError("报告未完整包含仅相机修正配置下的两个数据集。")
    for dataset, result in datasets.items():
        if not result["complete_uniform_verified_exports"] or result["scene_count"] != 10:
            raise RuntimeError(f"{dataset} 完整性检查未通过；请查看 comparison.json。")
    lines = ["仅相机投影修正：seed=0，每个场景从零训练 30000 步。",
             "原模型的 BGFC/AT-GOM/CMO 保留，启用相机投影修正。",
             f"比较对象（已有组合配置结果）：{reference}",
             "Gain 正数为改善：PSNR/SSIM = CameraOnly - Reference，LPIPS = Reference - CameraOnly。",
             "这是与已有组合配置的比较，不是与原始 JADA 基线的比较。"]
    with (output / "comparison_means.csv").open("w", newline="", encoding="utf-8-sig") as means, \
            (output / "comparison_per_scene.csv").open("w", newline="", encoding="utf-8-sig") as scenes:
        mean_writer, scene_writer = csv.writer(means), csv.writer(scenes)
        mean_writer.writerow(["dataset", "scene_count", "metric", "reference", "camera_only", "gain"])
        scene_writer.writerow(["dataset", "scene", "metric", "reference", "camera_only", "gain"])
        for dataset, result in datasets.items():
            lines.extend(["", f"{dataset}：10 个场景平均值",
                          f"{'Metric':<18} {'Reference':>12} {'CameraOnly':>12} {'Gain':>12}"])
            for metric in PRIMARY:
                value = result["full_dataset_mean"][metric]
                old, new, gain = (value[key] for key in ("supplied", "candidate", "oriented_gain"))
                lines.append(f"{metric:<18} {old:>12.5f} {new:>12.5f} {gain:>+12.5f}")
                mean_writer.writerow([dataset, 10, metric, old, new, gain])
            for scene, metrics in result["per_scene"].items():
                for metric in PRIMARY:
                    value = metrics[metric]
                    scene_writer.writerow([dataset, scene, metric, value["supplied"],
                                           value["candidate"], value["oriented_gain"]])
    summary = "\n".join(lines) + "\n"
    (output / "comparison.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="新输出目录，默认使用时间戳")
    parser.add_argument("--reference-root", type=Path, default=REFERENCE,
                        help="已有的两个数据集结果，用于测试目标校验和指标比较")
    parser.add_argument("--dry-run", action="store_true", help="检查数据路径并预览命令，不训练或创建输出目录")
    args = parser.parse_args()
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / "output" / f"CameraOnly_seed0_{stamp}_{os.getpid()}"
    output = (output if output.is_absolute() else ROOT / output).resolve()
    reference = args.reference_root
    reference = (reference if reference.is_absolute() else ROOT / reference).resolve()
    preflight(output, reference, args.dry_run)
    worker_commands, report_command = commands(output, reference)
    print(f"输出目录：{output}", flush=True)
    print("配置：原模型 + 相机投影修正。", flush=True)
    print("GPU 0：RGBT-Scenes，GPU 1：ThermoScenes1_3dgs；各 10 个场景，每场景 30000 步。", flush=True)
    if args.dry_run:
        for command in [*worker_commands, report_command]:
            print(shlex.join(command))
        print("预览完成，未启动训练，未创建输出目录。")
        return 0
    output.mkdir(parents=True, exist_ok=False)
    settings = {"mode": "camera_only", "seed": 0, "iterations": 30000,
                "source": str(ROOT), "reference_root": str(reference),
                "commands": worker_commands, "comparison_command": report_command}
    (output / "run_settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    workers, streams, readers, finished = [], [], [], set()
    try:
        for (dataset, gpu, _, _, _), command in zip(JOBS, worker_commands):
            logfile = output / f"gpu{gpu}_{dataset}.log"
            stream = logfile.open("w", encoding="utf-8")
            streams.append(stream)
            workers.append(subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                            stderr=subprocess.STDOUT, start_new_session=True))
            readers.append(logfile.open(encoding="utf-8", errors="replace"))
            print(f"[GPU {gpu}] 已启动 {dataset}，日志：{logfile}", flush=True)
        print("每个场景的训练进度见其 train.log。", flush=True)
        while len(finished) < len(workers):
            for index, (process, reader) in enumerate(zip(workers, readers)):
                if index in finished:
                    continue
                for line in reader.readlines():
                    if line.startswith("Starting ") or any(
                            line.rstrip().endswith(": " + stage)
                            for stage in ("render", "metrics", "extra_metrics")):
                        print(f"[GPU {JOBS[index][1]}] {line.rstrip()}", flush=True)
                code = process.poll()
                if code is None:
                    continue
                if code != 0:
                    raise RuntimeError(f"{JOBS[index][0]} 失败（退出码 {code}）；查看 {reader.name} 和场景 train.log。")
                finished.add(index)
                print(f"[完成] {JOBS[index][0]} 的训练与评估。", flush=True)
            if len(finished) < len(workers):
                time.sleep(5)
        print("两个数据集已完成，正在校验配置和汇总指标……", flush=True)
        with (output / "comparison_full.log").open("w") as stream:
            subprocess.run(report_command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        write_summary(output, reference)
        print(f"全部完成：{output / 'comparison_means.csv'}", flush=True)
        return 0
    finally:
        stop_workers(workers)
        for stream in [*readers, *streams]:
            stream.close()


def interrupted(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已停止本次训练，已有输出保留。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"运行失败：{error}", file=sys.stderr)
        raise SystemExit(1)
