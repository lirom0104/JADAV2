#!/usr/bin/env python3
"""Inspect real training updates and exported IR kernels on physical GPU 0."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scene import Scene, GaussianModel
from gaussian_renderer import render
from run_suite import read_namespace, validate_artifacts, DEFAULT_BASELINE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scene", default="Ebike")
    parser.add_argument("--iteration", type=int, default=80)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    assert torch.cuda.device_count() == 1
    folder = args.root / args.scene
    checkpoint = torch.load(folder / f"chkpnt{args.iteration}.pth", map_location="cpu", weights_only=False)
    state, iteration = checkpoint[:2]
    group = next(group for group in state["optimizer_state"]["param_groups"] if group["name"] == "ir_kernel")
    moments = state["optimizer_state"]["state"][group["params"][0]]
    coefficients = state["ir_kernel"]
    changed = coefficients.abs().sum(0)
    gradients = moments["exp_avg_sq"].sum(0)
    assert torch.isfinite(coefficients).all() and torch.all(changed > 0)
    assert torch.isfinite(gradients).all() and torch.all(gradients > 0)
    cfg = read_namespace(folder / "cfg_args")
    cfg["model_path"] = str(folder.resolve())
    dataset = argparse.Namespace(**cfg)
    keys = ("use_bgfc", "use_at_gom", "bgfc_hidden_dim", "bgfc_gate_init_bias",
            "bgfc_thermal_grayscale_context", "bgfc_rgb_luma_transfer_only",
            "use_render_calibration", "use_color_refinement", "color_refinement_hidden_dim",
            "color_refinement_max_residual", "use_detail_basis", "detail_basis_mode",
            "detail_basis_scale", "detail_basis_thermal_scale", "use_ir_kernel", "ir_kernel_amplitude")
    model = GaussianModel(dataset.sh_degree, **{key: cfg[key] for key in keys})
    scene = Scene(dataset, model, load_iteration=iteration, shuffle=False)
    camera = scene.getTestCameras()[0]
    background = torch.zeros(3, device="cuda")
    pipe = argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    timings = {}
    with torch.no_grad():
        model.ir_kernel_runtime_enabled = False
        base = render(camera, model, pipe, background)
        model.ir_kernel_runtime_enabled = True
        kernel = render(camera, model, pipe, background)
        assert torch.equal(base["render_color"], kernel["render_color"])
        assert torch.equal(base["render_thermal"], kernel["render_thermal_base"])
        assert kernel["ir_kernel_residual"].abs().max() > 0
        torch.testing.assert_close(kernel["render_thermal"], base["render_thermal"] + kernel["ir_kernel_residual"])
        for enabled in (False, True):
            model.ir_kernel_runtime_enabled = enabled
            for _ in range(3):
                render(camera, model, pipe, background)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            for _ in range(20):
                render(camera, model, pipe, background)
            torch.cuda.synchronize()
            timings[str(enabled)] = {"seconds_per_render": (time.perf_counter() - started) / 20,
                                     "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    validation = validate_artifacts(args.root, args.scene, DEFAULT_BASELINE, iteration)
    assert validation["valid"], validation
    report = {"iteration": iteration, "accepted_final_candidate": False,
              "scope": "80-iteration engineering smoke test only",
              "raw_parameter_abs_sum_by_column": changed.tolist(),
              "adam_second_moment_sum_by_column": gradients.tolist(),
              "rgb_forward_bitwise_equal_with_kernel_enabled": True,
              "thermal_base_bitwise_equal_with_kernel_enabled": True,
              "kernel_residual_abs_mean": kernel["ir_kernel_residual"].abs().mean().item(),
              "kernel_residual_abs_max": kernel["ir_kernel_residual"].abs().max().item(),
              "timing": timings, "artifact_validation": validation}
    (args.root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "iteration": iteration, "timing": timings}))


if __name__ == "__main__":
    main()
