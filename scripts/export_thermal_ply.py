"""Export the saved thermal rendering branch as a standard 3DGS PLY.

BGFC appearance and AT-GOM geometry/opacity are baked into the Gaussian
parameters. Image-space calibration, refinement and lens distortion remain
outside the PLY representation. The original model files are read only.
"""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from plyfile import PlyData, PlyElement
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene import GaussianModel
from scene.colmap_loader import (
    read_extrinsics_binary, read_extrinsics_text,
    read_intrinsics_binary, read_intrinsics_text,
)
from scene.dataset_readers import getNerfppNorm, readColmapCameras
from saved_settings import saved_namespace


MODEL_KEYS = (
    "use_bgfc", "use_at_gom", "bgfc_hidden_dim", "bgfc_gate_init_bias",
    "bgfc_thermal_grayscale_context", "bgfc_rgb_luma_transfer_only",
    "use_render_calibration", "use_color_refinement",
    "color_refinement_hidden_dim", "color_refinement_max_residual",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_extent(source):
    """Use exactly the training-camera radius used by Scene on reload."""
    sparse = source / "sparse/0"
    if (sparse / "images.bin").exists() and (sparse / "cameras.bin").exists():
        extrinsics = read_extrinsics_binary(str(sparse / "images.bin"))
        intrinsics = read_intrinsics_binary(str(sparse / "cameras.bin"))
    else:
        extrinsics = read_extrinsics_text(str(sparse / "images.txt"))
        intrinsics = read_intrinsics_text(str(sparse / "cameras.txt"))
    with redirect_stdout(io.StringIO()):
        cameras = readColmapCameras(
            extrinsics, intrinsics, str(source / "rgb/train"),
            str(source / "thermal/train"),
        )
    try:
        if not cameras:
            raise ValueError(f"No paired training cameras: {source}")
        return float(getNerfppNorm(sorted(cameras, key=lambda c: c.image_name))["radius"])
    finally:
        for camera in cameras:
            camera.image.close()
            camera.thermal.close()


def cpu(tensor):
    return tensor.detach().cpu().numpy()


def validate(path, expected, params, sh_degree):
    """Read the file using the standard 3DGS field layout and activations."""
    ply = PlyData.read(str(path))
    vertices = ply["vertex"].data
    if ply.text or ply.byte_order != "<" or vertices.dtype.names != expected.dtype.names:
        raise ValueError(f"Invalid standard binary 3DGS schema: {path}")
    if vertices.shape != expected.shape:
        raise ValueError(f"Vertex count mismatch: {path}")
    for name in expected.dtype.names:
        np.testing.assert_array_equal(vertices[name], expected[name])
        if not np.isfinite(vertices[name]).all():
            raise ValueError(f"Nonfinite {name}: {path}")

    def fields(names):
        array = np.stack([vertices[name] for name in names], axis=1)
        return torch.as_tensor(array, device=params["means3D"].device)

    count = len(vertices)
    coefficients = (sh_degree + 1) ** 2
    dc = fields([f"f_dc_{i}" for i in range(3)]).reshape(count, 3, 1)
    rest = fields([f"f_rest_{i}" for i in range(3 * (coefficients - 1))])
    features = torch.cat((dc, rest.reshape(count, 3, coefficients - 1)), dim=2)
    rotation = fields([f"rot_{i}" for i in range(4)])
    decoded = {
        "means3D": fields(["x", "y", "z"]),
        "features": features.transpose(1, 2),
        "opacity": torch.sigmoid(fields(["opacity"])),
        "scales": torch.exp(fields([f"scale_{i}" for i in range(3)])),
        "rotations": torch.nn.functional.normalize(rotation, dim=1),
    }
    errors = {}
    for key, actual in decoded.items():
        torch.testing.assert_close(actual, params[key], rtol=1e-6, atol=1e-7)
        errors[key] = float((actual - params[key]).abs().max())
    return errors


@torch.no_grad()
def export_scene(scene, iteration):
    folder = scene / "point_cloud" / f"iteration_{iteration}"
    source = folder / "point_cloud.ply"
    destination = folder / "point_cloud_thermal.ply"
    if destination.exists():
        raise FileExistsError(destination)
    config = saved_namespace(scene / "cfg_args")
    source_hash = sha256(source)
    model = GaussianModel(config["sh_degree"], **{
        key: config[key] for key in MODEL_KEYS if key in config
    })
    modules = folder / "feature_modules.pth"
    if (model.use_bgfc or model.use_render_calibration or model.use_color_refinement) and not modules.is_file():
        raise FileNotFoundError(modules)
    model.load_ply(str(source))
    model.load_feature_modules(str(folder))
    model.load_cmo_states(str(folder))
    extent = scene_extent(Path(config["source_path"]))
    model.constrain_thermal_scaling(extent)
    params = model.get_thermal_render_params()
    features = params["features"]
    log_scales = model._scaling
    if model.use_at_gom:
        log_scales = log_scales + model._at_gom_log_scale_residual
    logits = model._opacity_base + model._at_gom_opacity_bias_th
    count = len(params["means3D"])
    rest_count = 3 * ((model.max_sh_degree + 1) ** 2 - 1)
    names = (["x", "y", "z", "nx", "ny", "nz"]
             + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(rest_count)]
             + ["opacity"] + [f"scale_{i}" for i in range(3)]
             + [f"rot_{i}" for i in range(4)])
    matrix = np.concatenate((
        cpu(params["means3D"]), np.zeros((count, 3), dtype=np.float32),
        cpu(features[:, :1].transpose(1, 2).flatten(start_dim=1)),
        cpu(features[:, 1:].transpose(1, 2).flatten(start_dim=1)),
        cpu(logits), cpu(log_scales), cpu(params["rotations"]),
    ), axis=1).astype("<f4", copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Nonfinite export parameters: {scene}")
    vertices = np.ascontiguousarray(matrix).view([(name, "<f4") for name in names]).reshape(-1)
    with tempfile.NamedTemporaryFile(dir=folder, prefix=".thermal_", suffix=".ply", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        PlyData([PlyElement.describe(vertices, "vertex")], text=False, byte_order="<").write(str(temporary))
        errors = validate(temporary, vertices, params, model.max_sh_degree)
        if sha256(source) != source_hash:
            raise RuntimeError(f"Source PLY changed while exporting: {source}")
        # A hard link publishes the complete file without overwriting an existing path.
        destination.hardlink_to(temporary)
        destination.chmod(0o664)
    finally:
        temporary.unlink(missing_ok=True)
    result = {
        "dataset": scene.parent.name, "scene": scene.name,
        "source": str(source), "output": str(destination),
        "source_sha256": source_hash, "output_sha256": sha256(destination),
        "vertices": count, "bytes": destination.stat().st_size,
        "sh_degree": model.max_sh_degree, "scene_extent": extent,
        "bgfc_baked": model.use_bgfc, "at_gom_baked": model.use_at_gom,
        "cmo_context": "saved" if (folder / "cmo_states.pth").exists() else "zero (same as render.py reload)",
        "render_calibration_enabled_in_source": model.use_render_calibration,
        "thermal_calibration_scale_not_applied": cpu(model._render_calibration_thermal_scale).flatten().tolist(),
        "thermal_calibration_bias_not_applied": cpu(model._render_calibration_thermal_bias).flatten().tolist(),
        "image_refinement_enabled_in_source": model.use_color_refinement,
        "roundtrip_max_abs_error": errors, "validated": True,
    }
    del params, features, model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["RGBT-Scenes", "ThermoScenes1_3dgs"])
    parser.add_argument("--iteration", type=int, default=30000)
    args = parser.parse_args()
    root = args.root.resolve()
    scenes = []
    for dataset in args.datasets:
        children = sorted(path for path in (root / dataset).iterdir() if path.is_dir())
        if not children:
            raise ValueError(f"No scenes in {root / dataset}")
        for scene in children:
            folder = scene / "point_cloud" / f"iteration_{args.iteration}"
            for required in (scene / "cfg_args", folder / "point_cloud.ply"):
                if not required.is_file():
                    raise FileNotFoundError(required)
            if (folder / "point_cloud_thermal.ply").exists():
                raise FileExistsError(folder / "point_cloud_thermal.ply")
        scenes.extend(children)
    report_path = root / "thermal_ply_export_manifest.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report = {
        "format": "standard binary little-endian 3DGS PLY",
        "iteration": args.iteration,
        "appearance": "effective thermal SH after BGFC, stored in standard f_dc/f_rest fields",
        "limitations": [
            "Exports the thermal Gaussian branch before image-space postprocessing.",
            "Image-space affine calibration, multimodal refinement and camera lens distortion are not encoded.",
            "This PLY is for a standard 3DGS viewer, not for resuming dual-modal training.",
        ],
        "expected_scenes": len(scenes), "complete": False, "scenes": [],
    }
    for index, scene in enumerate(scenes, 1):
        result = export_scene(scene, args.iteration)
        report["scenes"].append(result)
        report["complete"] = len(report["scenes"]) == len(scenes)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"[{index}/{len(scenes)}] {result['dataset']}/{result['scene']}: "
              f"{result['vertices']:,} Gaussians, {result['bytes'] / 1e6:.1f} MB, validated", flush=True)
    print(f"Manifest: {report_path}", flush=True)


if __name__ == "__main__":
    main()
