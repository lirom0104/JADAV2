import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cwgc_dataset_report import audit_dataset


class DatasetReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.candidate, self.supplied = self.root / "candidate", self.root / "supplied"
        self.dataset = "RGBT-Scenes"
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "train.py").write_text("# fixed training source\n")
        source_hash = hashlib.sha256((self.source / "train.py").read_bytes()).hexdigest()
        for i in range(10):
            name = "scene" + str(i)
            base = {"color_PSNR": 20.0 + i, "thermal_PSNR": 30.0 + i,
                    "color_SSIM": .7, "thermal_SSIM": .8,
                    "color_LPIPS": .3, "thermal_LPIPS": .2}
            trial = {**base, "color_PSNR": base["color_PSNR"] + 1,
                     "thermal_PSNR": base["thermal_PSNR"] + 2,
                     "color_LPIPS": .28, "thermal_LPIPS": .19}
            for folder, metrics in ((self.supplied, base), (self.candidate, trial)):
                scene = folder / self.dataset / name
                scene.mkdir(parents=True)
                (scene / "results.json").write_text(json.dumps({"ours_30000": metrics}))
                hashes = {}
                for modality in ("color", "thermal"):
                    path = scene / "test/ours_30000" / ("gt_" + modality) / "00000.png"
                    path.parent.mkdir(parents=True)
                    image = Image.new("RGB", (2, 2), (i, 20, 30))
                    image.save(path)
                    hashes[modality + "/00000.png"] = hashlib.sha256(image.tobytes()).hexdigest()
                if folder == self.candidate:
                    (scene / ".evaluation").mkdir()
                    (scene / ".evaluation/verified.json").write_text(json.dumps(
                        {"metrics": metrics, "target_sha256": hashes}))
                    (scene / "cfg_args").write_text(
                        "Namespace(source_path=" + repr(str(self.root / self.dataset / name)) +
                        ", model_path=" + repr(str(scene)) + ", resolution=-1, use_camera_calibration=True)")
                    (scene / "optimization_args").write_text(
                        "Namespace(iterations=30000, seed=0, feature_lr=0.0025)")
                    (scene / "run_manifest.json").write_text(json.dumps(
                        {"source": str(self.source), "source_sha256": {"train.py": source_hash}}))
                    ply = scene / "point_cloud/iteration_30000/point_cloud.ply"
                    ply.parent.mkdir(parents=True)
                    ply.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")

    def audit(self):
        return audit_dataset(self.candidate, self.supplied, self.dataset)

    def test_ten_scene_mean_preserves_metric_direction(self):
        result = self.audit()
        self.assertTrue(result["complete_uniform_verified_exports"])
        means = result["full_dataset_mean"]
        self.assertEqual(means["color_PSNR"]["candidate"], 25.5)
        self.assertEqual(means["thermal_PSNR"]["oriented_gain"], 2)
        self.assertAlmostEqual(means["color_LPIPS"]["oriented_gain"], .02)

    def test_missing_scene_cannot_produce_full_mean(self):
        (self.candidate / self.dataset / "scene9/results.json").unlink()
        result = self.audit()
        self.assertEqual(result["missing_scenes"], ["scene9"])
        self.assertFalse(result["complete_uniform_verified_exports"])
        self.assertIsNone(result["full_dataset_mean"])

    def test_mixed_scene_strategy_is_rejected(self):
        path = self.candidate / self.dataset / "scene9/optimization_args"
        path.write_text(path.read_text().replace("feature_lr=0.0025", "feature_lr=0.005"))
        result = self.audit()
        self.assertFalse(result["complete_uniform_verified_exports"])
        self.assertTrue(any("feature_lr" in item for item in result["issues"]))
        self.assertIsNone(result["full_dataset_mean"])

    def test_refreshed_verifier_cannot_hide_changed_targets(self):
        scene = self.candidate / self.dataset / "scene3"
        image = Image.new("RGB", (2, 2), (200, 200, 200))
        image.save(scene / "test/ours_30000/gt_color/00000.png")
        path = scene / ".evaluation/verified.json"
        item = json.loads(path.read_text())
        item["target_sha256"]["color/00000.png"] = hashlib.sha256(image.tobytes()).hexdigest()
        path.write_text(json.dumps(item))
        result = self.audit()
        self.assertFalse(result["complete_uniform_verified_exports"])
        self.assertTrue(any("supplied baseline" in item for item in result["issues"]))

    def test_modified_frozen_source_is_rejected(self):
        (self.source / "train.py").write_text("# changed after training\n")
        result = self.audit()
        self.assertFalse(result["complete_uniform_verified_exports"])
        self.assertTrue(any("Frozen training source changed" in item for item in result["issues"]))

    def test_preserved_source_must_match_historical_hashes(self):
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        (snapshot / "train.py").write_bytes((self.source / "train.py").read_bytes())
        (self.source / "train.py").write_text("# cleaned training source\n")
        result = audit_dataset(self.candidate, self.supplied, self.dataset, source_snapshot=snapshot)
        self.assertTrue(result["complete_uniform_verified_exports"], result["issues"])
        (snapshot / "train.py").write_text("# incorrect snapshot\n")
        result = audit_dataset(self.candidate, self.supplied, self.dataset, source_snapshot=snapshot)
        self.assertFalse(result["complete_uniform_verified_exports"])
        self.assertTrue(any("Frozen training source changed" in item for item in result["issues"]))

    def test_identical_pixel_bytes_with_changed_dimensions_are_rejected(self):
        path = self.candidate / self.dataset / "scene3/test/ours_30000/gt_color/00000.png"
        with Image.open(path) as image:
            pixels = image.tobytes()
        Image.frombytes("RGB", (1, 4), pixels).save(path)
        result = self.audit()
        self.assertFalse(result["complete_uniform_verified_exports"])
        self.assertTrue(any("supplied baseline" in item for item in result["issues"]))

    def configure_legacy_camera_only(self):
        for scene in (self.candidate / self.dataset).iterdir():
            (scene / "optimization_args").write_text(
                "Namespace(iterations=30000, seed=0, use_cwgc=False, late_rmse_weight=0.0, "
                "late_lr_final_factor=1.0, near_camera_prune_ratio=0.0, "
                "cmo_use_thermal_densification=False, cmo_preserve_rgb_densification=False)")

    def test_historical_camera_only_configuration_is_readable(self):
        self.configure_legacy_camera_only()
        result = audit_dataset(self.candidate, self.supplied, self.dataset, "camera_only")
        self.assertTrue(result["complete_uniform_verified_exports"], result["issues"])
        self.assertEqual(result["full_dataset_mean"]["thermal_PSNR"]["oriented_gain"], 2)
        self.assertTrue(self.audit()["complete_uniform_verified_exports"])

    def test_camera_only_mode_rejects_reintroduced_rmse(self):
        self.configure_legacy_camera_only()
        path = self.candidate / self.dataset / "scene0/optimization_args"
        path.write_text(path.read_text().replace("late_rmse_weight=0.0", "late_rmse_weight=0.5"))
        result = audit_dataset(self.candidate, self.supplied, self.dataset, "camera_only")
        self.assertFalse(result["complete_uniform_verified_exports"])
        self.assertTrue(any("late_rmse_weight" in issue for issue in result["issues"]))

    def test_camera_only_mode_requires_calibrated_rendering(self):
        self.configure_legacy_camera_only()
        path = self.candidate / self.dataset / "scene0/cfg_args"
        path.write_text(path.read_text().replace("use_camera_calibration=True", "use_camera_calibration=False"))
        result = audit_dataset(self.candidate, self.supplied, self.dataset, "camera_only")
        self.assertFalse(result["complete_uniform_verified_exports"])
        self.assertTrue(any("requires camera calibration" in issue for issue in result["issues"]))


if __name__ == "__main__":
    unittest.main()
