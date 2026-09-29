"""Reproducible sequential scene jobs on one GPU; use two workers for two GPUs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cwgc_evaluate import evaluate

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"RGBT-Scenes": Path("/home/lf/data/thermal3dgs/RGBT-Scenes"),
            "ThermoScenes1_3dgs": Path("/home/lf/data/ThermoScenes1_3dgs")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True, help="dataset/scene")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=6020)
    parser.add_argument("--wait-log", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--reference-root", type=Path,
                        help="Existing results used to verify the evaluation targets")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    output = args.root.resolve()
    baseline = (ROOT/"output/JADA_batch").resolve()
    if output == baseline or baseline in output.parents:
        raise ValueError("Do not overwrite the supplied baseline")
    if args.reference_root is not None:
        reference = args.reference_root.resolve()
        if not reference.is_dir():
            raise FileNotFoundError(reference)
        if output == reference or reference in output.parents or output in reference.parents:
            raise ValueError("Output and reference directories must not overlap")
    output.mkdir(parents=True, exist_ok=True)
    if args.wait_log:
        print(f"Waiting for {args.wait_log}", flush=True)
        while True:
            if args.wait_log.exists():
                with args.wait_log.open("rb") as stream:
                    stream.seek(max(0, args.wait_log.stat().st_size - 8192))
                    tail = stream.read().decode(errors="replace")
                if "Training complete." in tail:
                    break
                if "Traceback (most recent call last)" in tail:
                    raise RuntimeError("Preceding training failed")
            time.sleep(15)
    source = args.source.resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTHONUNBUFFERED="1")
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    for item in args.scenes:
        dataset, scene = item.split("/", 1)
        data = DATASETS[dataset] / scene
        model = output / dataset / scene
        if model.exists() and any(model.iterdir()):
            raise FileExistsError(f"Run directory already contains artifacts: {model}")
        model.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "train.py", "-s", str(data), "-m", str(model),
                   "--port", str(args.port), "--seed", str(args.seed),
                   "--test_iterations", "7000", "15000", "30000", "--save_iterations", "7000", "30000",
                   "--checkpoint_iterations", "4000", "7000", "10000", "15000", "20000", "25000", "30000"]
        command += extra
        hashes = {}
        for p in source.rglob("*.py"):
            relative = p.relative_to(source)
            if any(part in {"output", "submodules", ".git"} for part in relative.parts):
                continue
            hashes[str(relative)] = hashlib.sha256(p.read_bytes()).hexdigest()
        manifest = {"command": command, "source": str(source), "source_sha256": hashes, "gpu": args.gpu,
                    "reference_root": str(args.reference_root.resolve()) if args.reference_root else str(baseline)}
        if "--start_checkpoint" in command:
            checkpoint = Path(command[command.index("--start_checkpoint")+1]).resolve()
            manifest["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        (model / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Starting {model}", flush=True)
        with (model / "train.log").open("w") as stream:
            subprocess.run(command, cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        evaluate(model, args.gpu, reference_root=args.reference_root)
    if any(item.startswith("ThermoScenes1_3dgs/") for item in args.scenes):
        subprocess.run([sys.executable, "wendu.py", "--data_root", str(DATASETS["ThermoScenes1_3dgs"]),
                        "--output_root", str(output/"ThermoScenes1_3dgs")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
