#!/usr/bin/env python3
"""CPU checks for this unapplied patch; never import train.py or launch training."""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
STAGE = Path(__file__).resolve().parent
MANIFEST = json.loads((STAGE / "manifest.json").read_text())
ROOT = Path(MANIFEST["repo_root"])
ROOT_STATE = "before"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_node(path, name):
    return next(node for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.FunctionDef) and node.name == name)


def compiled_statements(statements, filename):
    return compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(statements), type_ignores=[])),
                   str(filename), "exec")


def kernel_loss_block(path):
    training = function_node(path, "training")
    blocks = [node for node in ast.walk(training) if isinstance(node, ast.If)
              and any(isinstance(child, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "ir_prediction"
                              for target in child.targets) for child in node.body)]
    assert len(blocks) == 1, "Cannot uniquely identify the actual training IR loss block"
    return compiled_statements(blocks, path)


BEFORE_LOSS = kernel_loss_block(STAGE / "before/train.py")
AFTER_LOSS = kernel_loss_block(STAGE / "after/train.py")
LOSS_UTILS = module_at("staged_check_loss_utils", ROOT / "utils/loss_utils.py")
RUNNER = module_at("staged_check_runner", STAGE / "after/ir_experiments/run_suite.py")
ORIGINAL_RUNNER = module_at("original_check_runner", STAGE / "before/ir_experiments/run_suite.py")
ARGUMENTS = module_at("staged_check_arguments", STAGE / "after/arguments/__init__.py")
regularization_namespace = {"torch": torch}
exec(compiled_statements([function_node(ROOT / "scene/gaussian_model.py", "get_ir_kernel_regularization")],
                         ROOT / "scene/gaussian_model.py"), regularization_namespace)


class KernelFixture:
    """Synthetic render output; use the unchanged real Gaussian regularizer."""
    get_ir_kernel_regularization = regularization_namespace["get_ir_kernel_regularization"]
    use_ir_kernel = True

    def __init__(self, kernel, active):
        self._ir_kernel = kernel
        self.ir_kernel_runtime_enabled = active


def graph_signature(node):
    if node is None:
        return None
    return (type(node).__name__, tuple(graph_signature(edge[0]) for edge in node.next_functions))


def evaluate(code, weight, *, active=True, base_objective=False):
    # Deterministic fixtures intentionally include values outside [0, 1].
    # The new objective must not clamp the raw training prediction.
    shape = (3, 13, 15)
    shared = torch.linspace(-0.1, 0.1, math.prod(shape)).reshape(shape).requires_grad_()
    thermal_parent = torch.linspace(-0.2, 1.2, math.prod(shape)).reshape(shape).requires_grad_()
    rgb_parent = torch.linspace(0.05, 0.95, math.prod(shape)).reshape(shape).requires_grad_()
    kernel = torch.linspace(-0.4, 0.4, 8 * 10).reshape(8, 10).requires_grad_()
    projection = torch.cos(torch.arange(math.prod(shape) * kernel.numel()).float() * 0.007)
    projection = projection.reshape(math.prod(shape), kernel.numel()) / kernel.numel()
    residual = (projection @ torch.tanh(kernel).flatten()).reshape(shape)
    thermal = thermal_parent + shared * 0.2
    rgb = rgb_parent + shared * 0.3
    target = torch.linspace(0.15, 0.85, math.prod(shape)).reshape(shape)
    base_loss = F.mse_loss(rgb, target) + F.mse_loss(thermal, target) if base_objective else torch.zeros(())
    calls = []

    def observed_ssim(prediction, truth):
        calls.append((tuple(prediction.shape), tuple(truth.shape)))
        assert prediction.ndim == truth.ndim == 4 and prediction.shape[0] == 1
        assert prediction.min() < 0 and prediction.max() > 1, "Prediction was clamped"
        return LOSS_UTILS.ssim(prediction, truth)

    namespace = dict(gaussians=KernelFixture(kernel, active), thermal=thermal,
                     render_pkg={"ir_kernel_residual": residual}, gt_thermal=target,
                     opt=SimpleNamespace(ir_kernel_mse_weight=1.0, ir_kernel_reg_weight=1e-4),
                     ir_kernel_ssim_weight=weight, loss=base_loss, F=F, ssim=observed_ssim)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    exec(code, namespace)
    assert random.getstate() == rng[0]
    state = np.random.get_state()
    assert state[0] == rng[1][0] and np.array_equal(state[1], rng[1][1]) and state[2:] == rng[1][2:]
    assert torch.equal(torch.get_rng_state(), rng[2]), "Loss code consumed CPU RNG"
    loss = namespace["loss"]
    gradients = torch.autograd.grad(loss, (kernel, thermal_parent, rgb_parent, shared),
                                    allow_unused=True) if loss.requires_grad else (None,) * 4
    return dict(value=loss.detach(), gradients=gradients, graph=graph_signature(loss.grad_fn), calls=calls,
                prediction=namespace.get("ir_prediction"), namespace=namespace)


def parsed_runner_args(arguments):
    with patch.object(sys, "argv", [str(STAGE / "after/ir_experiments/run_suite.py"), *arguments]):
        return RUNNER.parse_args()


class StagedChecks(unittest.TestCase):
    def test_manifest_and_protected_sources(self):
        for version, hashes in (("before", MANIFEST["base_sha256"]), ("after", MANIFEST["staged_sha256"])):
            for name, digest in hashes.items():
                self.assertEqual(sha256(STAGE / version / name), digest, name)
        for name, digest in MANIFEST["protected_root_sha256"].items():
            if ROOT_STATE == "after" and name in MANIFEST["changed_files"]:
                digest = MANIFEST["staged_sha256"][name]
            self.assertEqual(sha256(ROOT / name), digest, name)
        frozen = Path(MANIFEST["protected_frozen_source_root"])
        for name, digest in MANIFEST["protected_frozen_sha256"].items():
            self.assertEqual(sha256(frozen / name), digest, name)
        self.assertEqual(sha256(STAGE / MANIFEST["patch_file"]), MANIFEST["patch_sha256"])

    def test_syntax_and_patch_application_in_temporary_copy(self):
        for version in ("before", "after"):
            for name in MANIFEST["changed_files"]:
                path = STAGE / version / name
                compile(path.read_text(), str(path), "exec")
        with tempfile.TemporaryDirectory(prefix="ir_structural_patch_") as directory:
            destination = Path(directory)
            for name in MANIFEST["changed_files"]:
                path = destination / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((STAGE / "before" / name).read_bytes())
            for flags in (("--check",), ()):
                subprocess.run(["git", "apply", *flags, str(STAGE / MANIFEST["patch_file"])],
                               cwd=destination, check=True, capture_output=True, text=True)
            for name, digest in MANIFEST["staged_sha256"].items():
                self.assertEqual(sha256(destination / name), digest)

    def test_zero_weight_preserves_actual_v1_graph_value_gradients_and_rng(self):
        for active in (False, True):
            original = evaluate(BEFORE_LOSS, 0.0, active=active, base_objective=True)
            staged = evaluate(AFTER_LOSS, 0.0, active=active, base_objective=True)
            self.assertEqual(staged["calls"], [])
            self.assertEqual(original["graph"], staged["graph"])
            self.assertTrue(torch.equal(original["value"], staged["value"]))
            for old, new in zip(original["gradients"], staged["gradients"]):
                self.assertTrue(old is new is None or (old is not None and new is not None and torch.equal(old, new)))

    def test_positive_weight_uses_nchw_and_reaches_only_kernel(self):
        fitted = evaluate(AFTER_LOSS, 0.01)
        original = evaluate(BEFORE_LOSS, 0.0)
        self.assertEqual(fitted["calls"], [((1, 3, 13, 15), (1, 3, 13, 15))])
        self.assertTrue(torch.isfinite(fitted["value"]))
        kernel_gradient, *parent_gradients = fitted["gradients"]
        self.assertTrue(torch.isfinite(kernel_gradient).all() and kernel_gradient.abs().max() > 0)
        self.assertEqual(parent_gradients, [None, None, None])
        self.assertFalse(torch.equal(kernel_gradient, original["gradients"][0]))
        self.assertIn("ir_kernel_ssim_loss", fitted["namespace"])
        inactive = evaluate(AFTER_LOSS, 0.01, active=False)
        self.assertEqual(inactive["calls"], [])
        self.assertEqual(inactive["gradients"], (None,) * 4)

    def test_positive_weight_preserves_base_rgb_and_shared_gradients(self):
        original = evaluate(BEFORE_LOSS, 0.0, base_objective=True)
        staged = evaluate(AFTER_LOSS, 0.01, base_objective=True)
        for old, new in zip(original["gradients"][1:], staged["gradients"][1:]):
            self.assertTrue(torch.equal(old, new))

    def test_training_validation_and_backward_compatible_missing_option(self):
        training = function_node(STAGE / "after/train.py", "training")
        initialization = compiled_statements(training.body[:2], STAGE / "after/train.py")
        for weight in (-0.01, float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                exec(initialization, dict(opt=SimpleNamespace(ir_kernel_ssim_weight=weight), math=math))
        namespace = dict(opt=SimpleNamespace(), math=math)
        exec(initialization, namespace)
        self.assertEqual(namespace["ir_kernel_ssim_weight"], 0.0)

    def test_training_arguments_and_saved_optimization_option(self):
        parser = argparse.ArgumentParser()
        options = ARGUMENTS.OptimizationParams(parser)
        for argv, expected in (([], 0.0), (["--ir_kernel_ssim_weight", "0.01"], 0.01)):
            extracted = options.extract(parser.parse_args(argv))
            self.assertEqual(extracted.ir_kernel_ssim_weight, expected)
            saved = repr(argparse.Namespace(**vars(extracted)))
            with tempfile.TemporaryDirectory(prefix="ir_structural_option_") as directory:
                path = Path(directory) / "optimization_args"
                path.write_text(saved)
                self.assertEqual(RUNNER.read_namespace(path)["ir_kernel_ssim_weight"], expected)

    def test_runner_zero_keeps_exact_v1_command(self):
        positional = (ROOT, Path(sys.executable), Path("/dataset"), Path("/unused"),
                      "Building", 30000, 18000, 0.003, 0.0001, 0.2, 1.0, 6027, 26000)
        self.assertEqual(ORIGINAL_RUNNER.base_train_command(*positional), RUNNER.base_train_command(*positional))
        command = RUNNER.base_train_command(*positional, kernel_ssim_weight=0.01)
        self.assertEqual(command[-2:], ["--ir_kernel_ssim_weight", "0.01"])
        self.assertNotIn("--start_checkpoint", command)

    def test_runner_dry_run_and_invalid_weights(self):
        with tempfile.TemporaryDirectory(prefix="ir_structural_dry_") as directory:
            root = Path(directory) / "new_run"
            common = ["run", "--root", str(root), "--repo", str(ROOT), "--dry-run"]
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(RUNNER.run_suite(parsed_runner_args(common + ["--ir-kernel-ssim-weight", "0.01"])), 0)
            plan = json.loads(stream.getvalue())
            self.assertEqual(plan["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(set(plan["commands"]), set(RUNNER.SCENES))
            for command in plan["commands"].values():
                self.assertEqual(command[command.index("--ir_kernel_ssim_weight") + 1], "0.01")
                self.assertEqual(command[command.index("--iterations") + 1], "30000")
                self.assertNotIn("--start_checkpoint", command)
            self.assertFalse(root.exists())
            for value in ("-0.01", "nan", "inf", "-inf"):
                with self.assertRaisesRegex(SystemExit, "finite and nonnegative"):
                    RUNNER.run_suite(parsed_runner_args(common + ["--ir-kernel-ssim-weight=" + value]))
            self.assertFalse(root.exists())

    def test_runner_frozen_command_and_metadata_with_all_launches_stubbed(self):
        # Exercise the actual post-snapshot command site and metadata, without
        # calling snapshot, extension discovery, subprocesses, or a GPU.
        with tempfile.TemporaryDirectory(prefix="ir_structural_metadata_") as directory:
            fixture = Path(directory)
            baseline = fixture / "baseline"
            dataset = fixture / "dataset"
            for scene in RUNNER.SCENES:
                folder = baseline / scene
                folder.mkdir(parents=True)
                (folder / "results.json").write_text('{"ours_30000": {}}')
                for name in ("per_view.json", "cfg_args", "optimization_args"):
                    (folder / name).write_text("{}")
            for name in ("rgb/train", "rgb/test", "thermal/train", "thermal/test", "sparse/0"):
                (dataset / "Building" / name).mkdir(parents=True)
            root = fixture / "new_run"
            frozen = root / "code_snapshot/source"
            args = parsed_runner_args(["run", "--root", str(root), "--scenes", "Building", "--skip-report",
                                       "--repo", str(ROOT), "--python", sys.executable,
                                       "--dataset-root", str(dataset), "--baseline-root", str(baseline),
                                       "--ir-kernel-ssim-weight", "0.01"])
            with patch.object(RUNNER, "snapshot_code", return_value={"executed_source_root": str(frozen)}), \
                    patch.object(RUNNER, "freeze_extensions", return_value={"path": str(root / "runtime_packages")}), \
                    patch.object(RUNNER, "run_process", return_value={"return_code": 7}) as process:
                self.assertEqual(RUNNER.run_suite(args), 1)
            process.assert_called_once()
            command, environment = process.call_args.args[:2]
            self.assertEqual(command[1], str(frozen / "train.py"))
            self.assertEqual(command[-2:], ["--ir_kernel_ssim_weight", "0.01"])
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
            metadata = json.loads((root / "run_metadata.json").read_text())
            self.assertEqual(metadata["kernel_configuration"]["ir_kernel_ssim_weight"], 0.01)
            self.assertEqual(metadata["iterations"], 30000)
            self.assertEqual(metadata["initialization"], "dataset, no start checkpoint")


def main():
    global ROOT_STATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--root-state", choices=("before", "after"), default="before")
    args = parser.parse_args()
    ROOT_STATE = args.root_state
    assert not torch.cuda.is_initialized()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(StagedChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors),
                  cuda_initialized=torch.cuda.is_initialized(), torch_version=torch.__version__,
                  root_state=ROOT_STATE,
                  python=sys.executable, command=[sys.executable, *sys.argv],
                  check_script_sha256=sha256(Path(__file__)), patch_sha256=MANIFEST["patch_sha256"],
                  staged_sha256=MANIFEST["staged_sha256"], limitations=MANIFEST["limitations"])
    assert not report["cuda_initialized"]
    if args.report:
        destination = args.report.resolve()
        if not destination.is_relative_to(STAGE):
            raise ValueError("Check report must stay inside the staging directory")
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
