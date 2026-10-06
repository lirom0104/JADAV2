#!/usr/bin/env python3
"""Compare original JADA and installed thermalgaussian rasterizers on GPU 0.

Runs small synthetic forward/backward cases only; never trains or modifies an
environment. The thermalgaussian extension receives pstb=None explicitly.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = {
    "JADA": "/home/lf/miniconda3/envs/JADA/bin/python",
    "thermalgaussian": "/home/lf/miniconda3/envs/thermalgaussian/bin/python",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(label, output):
    import inspect
    import math
    import numpy as np
    import torch
    import diff_gaussian_rasterization as backend

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    assert torch.cuda.is_available(), "CUDA is inaccessible; rerun with approved GPU access"
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    device = "cuda:0"
    height, width = 40, 48
    arrays = {}
    summaries = {}

    def leaf(values):
        return torch.tensor(values, dtype=torch.float32, device=device, requires_grad=True)

    def run_case(kind, thermal_sign=1.0, perturb=None):
        single = kind == "signed_single"
        means = [[0.04, -0.03, 2.0]] if single else [
            [-0.22, -0.15, 1.7], [0.08, -0.09, 2.0], [0.26, 0.17, 2.5],
            [-0.17, 0.24, 2.1], [0.02, 0.01, 3.0],
        ]
        count = len(means)
        params = {
            "means3D": leaf(means),
            "means2D": leaf([[0.0, 0.0, 0.0]] * count),
            "opacities": leaf([[0.36 + 0.08 * i] for i in range(count)]),
        }
        if kind == "covariance":
            params["cov3D_precomp"] = leaf([
                [0.010 + i * 0.002, 0.0002, 0.0001, 0.016, 0.0003, 0.012]
                for i in range(count)
            ])
        else:
            params["scales"] = leaf([[0.09 + 0.008 * i, 0.13, 0.11] for i in range(count)])
            params["rotations"] = leaf([[1.0, 0.0, 0.0, 0.0]] * count)
        if kind == "sh":
            params["thermal_shs"] = leaf([
                [[0.3 * math.sin(i * 16 + k + c) for c in range(3)] for k in range(16)]
                for i in range(count)
            ])
            params["color_shs"] = leaf([
                [[0.25 * math.cos(i * 16 + k + c) for c in range(3)] for k in range(16)]
                for i in range(count)
            ])
        else:
            if single:
                thermal = [[thermal_sign * value for value in (-0.7, -0.2, -0.4)]]
            elif kind == "signed_mixed":
                thermal = [[(-1.0 if (i + c) % 2 else 1.0) * (0.2 + 0.07 * i + 0.1 * c)
                            for c in range(3)] for i in range(count)]
            else:
                thermal = [[0.2 + 0.07 * i + 0.1 * c for c in range(3)] for i in range(count)]
            params["thermals_precomp"] = leaf(thermal)
            params["colors_precomp"] = leaf([
                [(-1.0 if kind == "signed_mixed" and (i + c) % 2 else 1.0)
                 * (0.15 + 0.05 * i + 0.07 * c) for c in range(3)] for i in range(count)
            ])
        if perturb:
            name, index, delta = perturb
            with torch.no_grad():
                params[name][index] += delta
        tanx, tany, near, far = 0.65, 0.55, 0.01, 100.0
        projection = torch.zeros((4, 4), dtype=torch.float32, device=device)
        projection[0, 0] = 1.0 / tanx
        projection[1, 1] = 1.0 / tany
        projection[2, 2] = far / (far - near)
        projection[2, 3] = -(far * near) / (far - near)
        projection[3, 2] = 1.0
        settings = backend.GaussianRasterizationSettings(
            image_height=height, image_width=width, tanfovx=tanx, tanfovy=tany,
            bg=torch.zeros(3, device=device), scale_modifier=1.0,
            viewmatrix=torch.eye(4, device=device), projmatrix=projection.T.contiguous(),
            sh_degree=3, campos=torch.zeros(3, device=device), prefiltered=False, debug=False,
        )
        rasterizer = backend.GaussianRasterizer(settings)
        extra = {"pstb": None} if "pstb" in inspect.signature(rasterizer.forward).parameters else {}
        thermal, color, radii = rasterizer(**params, **extra)
        weights = torch.linspace(-0.3, 0.7, thermal.numel(), device=device).reshape_as(thermal)
        loss = (thermal * weights).mean() + (color * weights.flip(-1) * 0.7).mean()
        loss.backward()
        torch.cuda.synchronize()
        result = {"thermal": thermal.detach().cpu().numpy(),
                  "color": color.detach().cpu().numpy(), "radii": radii.detach().cpu().numpy(),
                  "loss": np.array(loss.item())}
        for name, value in params.items():
            assert value.grad is not None, f"Missing gradient: {kind}/{name}"
            result["grad_" + name] = value.grad.detach().cpu().numpy()
        assert all(np.isfinite(value).all() for value in result.values()), kind
        assert (result["radii"] > 0).all(), kind
        return result

    for case in ("positive", "signed_mixed", "signed_single", "sh", "covariance"):
        result = run_case(case)
        arrays.update({case + "/" + key: value for key, value in result.items()})
        summaries[case] = {"thermal_min": float(result["thermal"].min()),
                           "thermal_max": float(result["thermal"].max()),
                           "color_min": float(result["color"].min()),
                           "color_max": float(result["color"].max())}
    signed = run_case("signed_single")
    unsigned = run_case("signed_single", thermal_sign=-1.0)
    sign_error = float(np.max(np.abs(signed["thermal"] + unsigned["thermal"])))
    assert signed["thermal"].min() < -0.01, "Signed thermal colors were clipped"
    assert sign_error < 1e-7, "Thermal signed-color linearity failed"
    finite_difference = {}
    for name, index in (("thermals_precomp", (0, 0)), ("opacities", (0, 0))):
        epsilon = 0.001
        plus = run_case("signed_single", perturb=(name, index, epsilon))["loss"].item()
        minus = run_case("signed_single", perturb=(name, index, -epsilon))["loss"].item()
        numeric = (plus - minus) / (2 * epsilon)
        analytic = float(signed["grad_" + name][index])
        error = abs(numeric - analytic)
        finite_difference[name] = {"analytic": analytic, "numeric": numeric,
                                   "absolute_error": error}
        assert error <= 2e-6 + abs(numeric) * 2e-3, (name, finite_difference[name])
    np.savez(output / (label + ".npz"), **arrays)
    metadata = {
        "environment": label, "python_executable": sys.executable,
        "python_version": sys.version, "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "device": torch.cuda.get_device_name(0), "backend_wrapper": backend.__file__,
        "backend_wrapper_sha256": sha256(backend.__file__),
        "backend_extension": backend._C.__file__, "backend_extension_sha256": sha256(backend._C.__file__),
        "supports_pstb": "pstb" in inspect.signature(backend.GaussianRasterizer.forward).parameters,
        "pstb_argument": "None explicitly" if label == "thermalgaussian" else "unsupported",
        "cases": summaries, "signed_color_linearity_max_absolute_error": sign_error,
        "signed_color_finite_differences": finite_difference,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(0),
    }
    (output / (label + ".json")).write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"environment": label, "cases": list(summaries), "passed": True}))


def compare(output):
    import numpy as np
    output.mkdir(parents=True, exist_ok=True)
    commands = []
    for label, python in ENVIRONMENTS.items():
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["LD_LIBRARY_PATH"] = str(Path(python).parent.parent / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        command = [python, str(Path(__file__).resolve()), "--worker", label, "--output", str(output)]
        commands.append({"command": command, "CUDA_VISIBLE_DEVICES": "0", "LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"]})
        completed = subprocess.run(command, env=env, capture_output=True, text=True, cwd=ROOT)
        (output / (label + ".log")).write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"{label} exited {completed.returncode}; see {output / (label + '.log')}")
    original, extended = (np.load(output / (label + ".npz")) for label in ENVIRONMENTS)
    assert set(original.files) == set(extended.files)
    comparisons = {}
    for key in original.files:
        a, b = original[key], extended[key]
        comparisons[key] = {
            "shape": list(a.shape), "max_absolute_error": float(np.max(np.abs(a - b))),
            "reference_max_absolute": float(np.max(np.abs(a))),
            "bitwise_equal": bool(np.array_equal(a, b)),
            "within_tolerance": bool(np.allclose(a, b, atol=1e-6, rtol=1e-5)),
        }
    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Five synthetic GPU 0 forward/backward cases; no training or environment changes",
        "passed": all(item["within_tolerance"] for item in comparisons.values()),
        "all_bitwise_equal": all(item["bitwise_equal"] for item in comparisons.values()),
        "absolute_tolerance": 1e-6, "relative_tolerance": 1e-5,
        "workspace_wrapper_sha256": sha256(ROOT / "submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py"),
        "script_sha256": sha256(__file__), "commands": commands,
        "environments": {label: json.loads((output / (label + ".json")).read_text()) for label in ENVIRONMENTS},
        "comparisons": comparisons,
        "limitations": "Synthetic coverage cannot prove equivalence for every possible rasterizer input or concurrency ordering.",
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "all_bitwise_equal": report["all_bitwise_equal"],
                      "tensor_comparisons": len(comparisons),
                      "max_absolute_error": max(item["max_absolute_error"] for item in comparisons.values()),
                      "report": str(output / "comparison.json")}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=list(ENVIRONMENTS))
    parser.add_argument("--output", type=Path, default=ROOT / "ir_experiments/backend_audit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args.worker, args.output)
    else:
        compare(args.output)
