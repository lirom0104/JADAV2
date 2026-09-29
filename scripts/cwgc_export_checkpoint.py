"""Export a trusted local training checkpoint without any optimizer updates.

Use separate output directories for raw/EMA or context ablations. The original
training artifacts and supplied baseline are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from arguments import ModelParams, OptimizationParams
from scene import GaussianModel
from saved_settings import saved_namespace


def export(checkpoint, destination, save_context=False):
    checkpoint, destination = checkpoint.resolve(), destination.resolve()
    baseline = Path(__file__).resolve().parents[1] / "output/JADA_batch"
    if destination == baseline or baseline in destination.parents:
        raise ValueError("Refusing to write to the supplied baseline")
    if destination == checkpoint.parent or (destination.exists() and any(destination.iterdir())):
        raise FileExistsError("Export requires a new, empty destination")
    parser = argparse.ArgumentParser()
    model_group, optimization_group = ModelParams(parser), OptimizationParams(parser)
    namespace = parser.parse_args([])
    for filename in ("cfg_args", "optimization_args"):
        vars(namespace).update(saved_namespace(checkpoint.parent / filename))
    data, options = model_group.extract(namespace), optimization_group.extract(namespace)
    state = torch.load(checkpoint, weights_only=False)
    if not isinstance(state, tuple) or len(state) < 2:
        raise ValueError("Expected a locally produced tuple checkpoint")
    iteration = int(state[1])
    if not 0 < iteration <= 30000:
        raise ValueError("Only checkpoints within the 30000-step budget may be exported")
    keys = ("use_bgfc", "use_at_gom", "bgfc_hidden_dim", "bgfc_gate_init_bias",
            "bgfc_thermal_grayscale_context", "bgfc_rgb_luma_transfer_only",
            "use_render_calibration", "use_color_refinement", "color_refinement_hidden_dim",
            "color_refinement_max_residual")
    model = GaussianModel(data.sh_degree, **{key: getattr(data, key) for key in keys})
    model.restore(state[0], options)
    model.set_color_refinement_runtime_enabled(iteration > options.color_refinement_start_iter)
    model.save_cmo_states_enabled = save_context
    destination.mkdir(parents=True, exist_ok=True)
    point_path = destination / "point_cloud" / f"iteration_{iteration}"
    model.save_ply(str(point_path / "point_cloud.ply"))
    model.save_feature_modules(str(point_path))
    model.save_cmo_states(str(point_path))
    config = saved_namespace(checkpoint.parent / "cfg_args")
    config["model_path"] = str(destination)
    (destination / "cfg_args").write_text(str(argparse.Namespace(**config)))
    # Optimization settings describe the source run; the manifest explicitly
    # distinguishes this raw export from that run's possible EMA export.
    (destination / "optimization_args").write_bytes((checkpoint.parent / "optimization_args").read_bytes())
    manifest = {"source_checkpoint": str(checkpoint), "iteration": iteration,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "parameter_state": "raw", "save_context": save_context,
                "additional_optimizer_updates": 0,
                "source_uses_ema_export": options.use_ema_export}
    (destination / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--save-context", action="store_true")
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.save_context)
