#!/usr/bin/env python3
"""Focused CPU validation of optional benchmark IR fitting loss; no GPU jobs."""

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

import benchmark_cost as benchmark

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "ir_experiments/staged_structural_v2"
MANIFEST = json.loads((STAGE / "manifest.json").read_text())
FROZEN = Path(MANIFEST["protected_frozen_source_root"])
sys.path.insert(0, str(FROZEN))
SSIM = benchmark.load_frozen_ssim(FROZEN)
torch.set_num_threads(1)


def graph(node):
    return None if node is None else (type(node).__name__, tuple(graph(edge[0]) for edge in node.next_functions))


class Fixture:
    ir_kernel_runtime_enabled = True

    def __init__(self):
        self.base = torch.linspace(-0.3, 1.3, 3 * 13 * 15).reshape(3, 13, 15).requires_grad_()
        self.kernel = torch.linspace(-0.1, 0.1, 3 * 13 * 15).reshape(3, 13, 15).requires_grad_()
        self.target = torch.linspace(0.1, 0.9, 3 * 13 * 15).reshape(3, 13, 15)
        self.output = {"render_thermal_base": self.base, "ir_kernel_residual": self.kernel.tanh() * 0.2}

    def get_ir_kernel_regularization(self):
        return self.kernel.tanh().square().mean()


def original_loss(fixture):
    prediction = fixture.output["render_thermal_base"].detach() + fixture.output["ir_kernel_residual"]
    return (1.0 * torch.nn.functional.mse_loss(prediction, fixture.target)
            + 0.0001 * fixture.get_ir_kernel_regularization())


def staged_loss(fixture, weight):
    path = STAGE / "after/train.py"
    tree = ast.parse(path.read_text())
    training = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "training")
    blocks = [node for node in ast.walk(training) if isinstance(node, ast.If)
              and any(isinstance(child, ast.Assign) and any(isinstance(target, ast.Name)
                      and target.id == "ir_prediction" for target in child.targets) for child in node.body)]
    assert len(blocks) == 1
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(blocks), type_ignores=[]))
    namespace = dict(gaussians=fixture, thermal=fixture.base, render_pkg=fixture.output,
                     gt_thermal=fixture.target, ir_kernel_ssim_weight=weight,
                     opt=SimpleNamespace(ir_kernel_mse_weight=1.0, ir_kernel_reg_weight=0.0001),
                     F=torch.nn.functional, ssim=SSIM, loss=torch.zeros(()))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["loss"]


class FittingDiagnosticChecks(unittest.TestCase):
    def test_absent_and_zero_preserve_original_graph_value_gradient_and_rng(self):
        for config in ({}, {"ir_kernel_ssim_weight": 0.0}):
            fixture, reference = Fixture(), Fixture()
            weights = benchmark.fitting_weights(dict(ir_kernel_mse_weight=1.0, ir_kernel_reg_weight=0.0001, **config))
            state = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
            def forbidden(*_):
                self.fail("SSIM called at absent/zero weight")
            loss = benchmark.ir_fitting_loss(torch, fixture.output, fixture.target, fixture, weights, forbidden)
            expected = original_loss(reference)
            self.assertEqual(graph(loss.grad_fn), graph(expected.grad_fn))
            self.assertTrue(torch.equal(loss, expected))
            actual_gradient, base_gradient = torch.autograd.grad(loss, (fixture.kernel, fixture.base), allow_unused=True)
            expected_gradient, = torch.autograd.grad(expected, reference.kernel)
            self.assertTrue(torch.equal(actual_gradient, expected_gradient))
            self.assertIsNone(base_gradient)
            self.assertEqual(random.getstate(), state[0])
            now = np.random.get_state()
            self.assertEqual(now[0], state[1][0])
            self.assertTrue(np.array_equal(now[1], state[1][1]))
            self.assertEqual(now[2:], state[1][2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), state[2]))

    def test_positive_matches_staged_loss_and_preserves_parameters(self):
        fixture, reference = Fixture(), Fixture()
        before = fixture.kernel.detach().clone()
        calls = []
        def observed(prediction, target):
            calls.append(tuple(prediction.shape))
            self.assertLess(prediction.min(), 0)
            self.assertGreater(prediction.max(), 1)
            return SSIM(prediction, target)
        loss = benchmark.ir_fitting_loss(torch, fixture.output, fixture.target, fixture,
                                         {"mse": 1.0, "regularization": 0.0001, "ssim": 0.01}, observed)
        expected = staged_loss(reference, 0.01)
        self.assertTrue(torch.equal(loss, expected))
        gradient, parent = torch.autograd.grad(loss, (fixture.kernel, fixture.base), allow_unused=True)
        reference_gradient, = torch.autograd.grad(expected, reference.kernel)
        self.assertTrue(torch.equal(gradient, reference_gradient))
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertIsNone(parent)
        self.assertEqual(calls, [(1, 3, 13, 15)])
        self.assertTrue(torch.equal(before, fixture.kernel.detach()))
        self.assertIsNone(fixture.kernel.grad)

    def test_saved_ssim_rejects_invalid_weights(self):
        for weight in (-0.01, float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                benchmark.fitting_weights({"ir_kernel_mse_weight": 1.0, "ir_kernel_reg_weight": 0.0001,
                                           "ir_kernel_ssim_weight": weight})

    def test_positive_requires_ssim(self):
        fixture = Fixture()
        with self.assertRaisesRegex(ValueError, "requires frozen"):
            benchmark.ir_fitting_loss(torch, fixture.output, fixture.target, fixture,
                                      {"mse": 1.0, "regularization": 0.0001, "ssim": 0.01})

    def test_ssim_import_must_match_frozen_root(self):
        self.assertIs(benchmark.load_frozen_ssim(FROZEN), SSIM)
        with self.assertRaisesRegex(ValueError, "frozen candidate"):
            benchmark.load_frozen_ssim(ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_initialized()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FittingDiagnosticChecks))
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
              "errors": len(result.errors), "failures": len(result.failures),
              "cuda_initialized": torch.cuda.is_initialized(), "torch_version": torch.__version__,
              "command": [sys.executable, *sys.argv],
              "benchmark_sha256": benchmark.sha256(Path(benchmark.__file__)),
              "check_sha256": benchmark.sha256(__file__),
              "frozen_ssim_sha256": benchmark.sha256(FROZEN / "utils/loss_utils.py"),
              "staged_train_sha256": benchmark.sha256(STAGE / "after/train.py"),
              "limitations": ["Synthetic CPU render outputs; no rasterization, CUDA timing, actual scene fitting, or quality evidence."]}
    assert not report["cuda_initialized"]
    benchmark.write_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
