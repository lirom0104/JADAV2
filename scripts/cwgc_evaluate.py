"""Run the unchanged saved-model evaluation pipeline and verify its artifacts."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"color_PSNR", "color_SSIM", "color_LPIPS", "thermal_PSNR",
            "thermal_SSIM", "thermal_LPIPS", "thermal_Boundary_Fscore", "hot_region_IoU"}


def reference_scene(model, baseline):
    """Resolve the supplied scene from saved input data, not output naming.

    Training outputs can have arbitrary names. Parse only the literal saved
    source_path rather than executing the serialized argparse Namespace.
    """
    config = ast.parse((model / "cfg_args").read_text(), mode="eval").body
    if (not isinstance(config, ast.Call) or not isinstance(config.func, ast.Name)
            or config.func.id != "Namespace" or config.args):
        raise ValueError("Expected a saved argparse Namespace in cfg_args")
    sources = [ast.literal_eval(item.value) for item in config.keywords
               if item.arg == "source_path"]
    if len(sources) != 1 or not isinstance(sources[0], str):
        raise ValueError("cfg_args must contain exactly one literal source_path")
    source = Path(sources[0]).resolve()
    if source.parent.name not in ("RGBT-Scenes", "ThermoScenes1_3dgs"):
        raise ValueError(f"Unrecognized reference dataset: {source.parent.name}")
    reference = baseline / source.parent.name / source.name
    if not reference.is_dir():
        raise ValueError(f"No supplied reference scene: {reference}")
    return reference


def evaluate(model, gpu, iteration=30000, target_iteration=30000, reference_root=None):
    from PIL import Image
    model = model.resolve()
    baseline = (Path(reference_root) if reference_root is not None else ROOT / "output/JADA_batch").resolve()
    if model == baseline or baseline in model.parents:
        raise ValueError("Evaluation must not overwrite the reference results")
    reference = reference_scene(model, baseline)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    log_dir = model / ".evaluation"
    log_dir.mkdir(parents=True, exist_ok=True)
    stages = [
        ("render", ["render.py", "-m", str(model), "--iteration", str(iteration), "--skip_train"]),
        ("metrics", ["metrics.py", "-m", str(model)]),
        ("extra_metrics", ["extra_metrics.py", "-m", str(model)]),
    ]
    for name, args in stages:
        print(f"{model.name}: {name}", flush=True)
        with (log_dir / f"{name}.log").open("w") as stream:
            subprocess.run([sys.executable, *args], cwd=ROOT, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    metrics = json.loads((model / "results.json").read_text())[f"ours_{iteration}"]
    if not REQUIRED <= metrics.keys():
        raise RuntimeError(f"Incomplete metrics for {model}: {metrics.keys()}")
    import math
    if not all(math.isfinite(metrics[k]) for k in REQUIRED):
        raise RuntimeError(f"Non-finite metrics for {model}")
    # Check that comparison uses the exact same held-out targets/order.
    old = reference / "test" / f"ours_{target_iteration}"
    new = model / "test" / f"ours_{iteration}"
    hashes = {}
    for modality in ("color", "thermal"):
        expected = sorted((old / f"gt_{modality}").glob("*.png"))
        actual = sorted((new / f"gt_{modality}").glob("*.png"))
        if len(expected) != len(actual) or not expected:
            raise RuntimeError("Held-out view count differs from supplied baseline")
        for reference, generated in zip(expected, actual):
            with Image.open(reference) as image:
                old_hash = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            with Image.open(generated) as image:
                new_hash = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            if reference.name != generated.name or old_hash != new_hash:
                raise RuntimeError(f"Held-out target mismatch: {reference} versus {generated}")
            hashes[f"{modality}/{generated.name}"] = new_hash
    (log_dir / "verified.json").write_text(json.dumps({"metrics": metrics, "target_sha256": hashes}, indent=2))
    print(json.dumps({"model": str(model), "metrics": metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--target-iteration", type=int, default=30000,
                        help="Supplied-baseline target directory used for hash verification")
    parser.add_argument("--reference-root", type=Path,
                        help="Existing dataset/scene results containing the original test targets")
    parser.add_argument("--wait-log", type=Path)
    parser.add_argument("--temperature", action="store_true")
    args = parser.parse_args()
    if args.wait_log:
        print(f"Waiting for training: {args.wait_log}", flush=True)
        while True:
            if args.wait_log.exists():
                with args.wait_log.open("rb") as stream:
                    stream.seek(max(0, args.wait_log.stat().st_size - 8192))
                    tail = stream.read().decode(errors="replace")
                if "Training complete." in tail:
                    break
                if "Traceback (most recent call last)" in tail:
                    raise RuntimeError(f"Training failed; inspect {args.wait_log}")
            time.sleep(15)
    for model in args.models:
        evaluate(model, args.gpu, args.iteration, args.target_iteration, args.reference_root)
    if args.temperature:
        roots = {model.resolve().parent for model in args.models if model.parent.name == "ThermoScenes1_3dgs"}
        for root in roots:
            subprocess.run([sys.executable, "wendu.py", "--data_root", "/home/lf/data/ThermoScenes1_3dgs",
                            "--output_root", str(root)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
