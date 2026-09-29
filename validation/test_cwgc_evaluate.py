import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "cwgc_evaluate", Path(__file__).resolve().parents[1] / "scripts/cwgc_evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class ReferenceSceneTests(unittest.TestCase):
    def test_flat_output_name_resolves_saved_source_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, baseline = root / "arbitrary_trial", root / "supplied"
            model.mkdir()
            reference = baseline / "RGBT-Scenes" / "DailyStuff"
            reference.mkdir(parents=True)
            (model / "cfg_args").write_text(
                "Namespace(source_path='/data/RGBT-Scenes/DailyStuff', eval=False)")
            self.assertEqual(evaluation.reference_scene(model, baseline), reference)

    def test_nonliteral_source_is_rejected_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cfg_args").write_text("Namespace(source_path=str('/data/RGBT-Scenes/X'))")
            with self.assertRaises(ValueError):
                evaluation.reference_scene(root, root)

    def test_missing_reference_fails_before_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cfg_args").write_text("Namespace(source_path='/data/RGBT-Scenes/missing')")
            with self.assertRaisesRegex(ValueError, "No supplied reference scene"):
                evaluation.reference_scene(root, root / "supplied")
