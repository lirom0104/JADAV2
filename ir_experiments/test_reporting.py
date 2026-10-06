"""CPU tests for experiment acceptance guards; no training or GPU usage."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from run_suite import (
    METRICS, SCENES, base_train_command, compare_pngs, json_dump,
    snapshot_code, validate_artifacts,
)
from report_results import build_report


class AcceptanceGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.baseline = self.root / "baseline"
        self.candidate = self.root / "candidate"
        self.candidate.mkdir()
        config = dict(resolution=-1, images="images", thermal="thermal", white_background=False,
                      eval=False, use_paired_views=True, use_camera_calibration=True)
        metrics = {key: (27.0 if "PSNR" in key else 0.9) for key in METRICS}
        self.metrics = metrics
        for scene in SCENES:
            base = self.baseline / scene
            base.mkdir(parents=True)
            json_dump(base / "results.json", {"ours_30000": metrics})
            (base / "cfg_args").write_text(repr(argparse.Namespace(**config)))
        scene = "Building"
        folder = self.candidate / scene
        folder.mkdir()
        (folder / "cfg_args").write_text(repr(argparse.Namespace(**config, use_ir_kernel=True)))
        (folder / "optimization_args").write_text(repr(argparse.Namespace(iterations=30000)))
        json_dump(folder / "training_receipt.json", {
            "initialization": "dataset", "start_checkpoint": None, "resumed_iteration": 0,
            "final_iteration": 30000, "optimizer_updates_this_process": 30000,
            "cumulative_iterations": 30000, "cuda_visible_devices": "0",
        })
        cloud = folder / "point_cloud" / "iteration_30000"
        cloud.mkdir(parents=True)
        (cloud / "point_cloud.ply").write_text("ply\nformat ascii 1.0\nelement vertex 1\n" + "".join(f"property float ir_kernel_{i}\n" for i in range(10)) + "end_header\n" + " ".join(["0"] * 10) + "\n")
        (cloud / "feature_modules.pth").write_bytes(b"fixture")
        (folder / "chkpnt30000.pth").write_bytes(b"fixture")
        self.folder = folder
        self.write_metrics()
        for root in (self.baseline, self.candidate):
            for channel in ("gt_color", "gt_thermal", "renders_color", "renders_thermal"):
                images = root / scene / "test" / "ours_30000" / channel
                images.mkdir(parents=True)
                for name in ("00000.png", "00001.png"):
                    Image.new("RGB", (3, 2), (13, 24, 35)).save(images / name)

    def tearDown(self):
        self.temp.cleanup()

    def write_metrics(self):
        json_dump(self.folder / "results.json", {"ours_30000": self.metrics})
        json_dump(self.folder / "per_view.json", {"ours_30000": {key: {name: value for name in ("00000.png", "00001.png")} for key, value in self.metrics.items()}})

    def validate(self):
        return validate_artifacts(self.candidate, "Building", self.baseline, 30000)

    def test_valid_outputs_then_swallowed_metric_failure(self):
        self.assertTrue(self.validate()["valid"])
        (self.folder / "results.json").unlink()
        self.assertFalse(self.validate()["valid"])

    def test_actual_updates_are_required(self):
        receipt_path = self.folder / "training_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["optimizer_updates_this_process"] = 29999
        json_dump(receipt_path, receipt)
        self.assertFalse(self.validate()["valid"])

    def test_gt_pixels_and_extra_views_are_rejected(self):
        images = self.folder / "test" / "ours_30000" / "gt_thermal"
        Image.new("RGB", (3, 2), (0, 0, 0)).save(images / "00001.png")
        self.assertFalse(self.validate()["valid"])
        Image.new("RGB", (3, 2), (13, 24, 35)).save(images / "00001.png")
        Image.new("RGB", (3, 2)).save(images / "00002.png")
        self.assertFalse(self.validate()["valid"])

    def test_missing_and_nonfinite_per_view_metrics_are_rejected(self):
        path = self.folder / "per_view.json"
        views = json.loads(path.read_text())
        del views["ours_30000"]["thermal_PSNR"]["00001.png"]
        json_dump(path, views)
        self.assertFalse(self.validate()["valid"])
        self.write_metrics()
        views = json.loads(path.read_text())
        views["ours_30000"]["thermal_PSNR"]["00001.png"] = float("nan")
        json_dump(path, views)
        self.assertFalse(self.validate()["valid"])

    def test_partial_run_cannot_pass_target(self):
        report = build_report(self.candidate, self.baseline, 30000, False,
                              "no_obvious_degradation", "unit fixture")
        self.assertFalse(report["goal_achieved"])
        self.assertIsNone(report["ir_target_met"])
        self.assertEqual(len(report["per_scene"]), 10)

    def test_snapshot_is_frozen_and_command_has_no_resume(self):
        source = self.root / "source"
        source.mkdir()
        (source / "train.py").write_text("old = True\n")
        archive = snapshot_code(source, self.root / "snapshot")
        (source / "train.py").write_text("old = False\n")
        frozen = Path(archive["executed_source_root"])
        self.assertEqual((frozen / "train.py").read_text(), "old = True\n")
        command = base_train_command(frozen, Path("python"), self.root, self.candidate, "Building", 30000)
        self.assertEqual(command[1], str(frozen / "train.py"))
        self.assertNotIn("--start_checkpoint", command)
        self.assertIn("--checkpoint_iterations", command)


if __name__ == "__main__":
    unittest.main()
