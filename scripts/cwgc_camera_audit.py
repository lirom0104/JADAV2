"""Audit supplied COLMAP cameras against training tracks and sensor grids.

Read-only: uses training-image membership/dimensions, sparse points, tracks,
and calibration. It neither reads held-out image pixels nor fits parameters.
This repository's supplied datasets have binary COLMAP reconstructions.
"""
import argparse
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary
from utils.camera_calibration import unpack_calibration, calibration_grid, project_sensor_pixels


def point_dictionary(path):
    points = {}
    with path.open("rb") as stream:
        count, = struct.unpack("<Q", stream.read(8))
        for _ in range(count):
            point = struct.unpack("<QdddBBBd", stream.read(43))
            points[point[0]] = point[1:4]
            track_length, = struct.unpack("<Q", stream.read(8))
            stream.seek(8*track_length, 1)
    return points


def error_summary(errors):
    if not errors:
        return {"observations": 0}
    values = np.concatenate(errors)
    return {"observations": len(values), "median_px": float(np.median(values)),
            "p90_px": float(np.quantile(values, .9)), "mean_px": float(values.mean())}


def audit_scene(scene):
    sparse = scene / "sparse/0"
    cameras = read_intrinsics_binary(sparse / "cameras.bin")
    views = read_extrinsics_binary(sparse / "images.bin")
    points = point_dictionary(sparse / "points3D.bin")
    training = {p.name: p for p in (scene / "rgb/train").iterdir() if p.is_file()}
    errors = {key: [] for key in ("centered_pinhole", "principal_point_pinhole", "full_calibration")}
    train_views, behind_camera_tracks, grids = 0, 0, {}
    for view in views.values():
        image_path = training.get(Path(view.name).name)
        if image_path is None:
            continue
        train_views += 1
        camera = cameras[view.camera_id]
        intrinsics, distortion = unpack_calibration(camera.model, camera.params)
        valid = np.array([pid in points for pid in view.point3D_ids], dtype=bool)
        if valid.any():
            xyz = np.array([points[pid] for pid in view.point3D_ids[valid]])
            xyz_camera = xyz @ view.qvec2rotmat().T + view.tvec
            front = xyz_camera[:, 2] > 1e-6
            behind_camera_tracks += int((~front).sum())
            xyz_camera = torch.from_numpy(xyz_camera[front])
            observed = view.xys[valid][front]
            for label in errors:
                parameters, lens = intrinsics.copy(), distortion.copy()
                if label != "full_calibration":
                    lens[:] = 0
                if label == "centered_pinhole":
                    parameters[2:] = [camera.width/2, camera.height/2]
                # Compare in COLMAP's pixel-center coordinates.
                projected = project_sensor_pixels(xyz_camera, parameters, lens).numpy()+.5
                errors[label].append(np.linalg.norm(projected-observed, axis=-1))
        with Image.open(image_path) as image:
            width,height = image.size
        key = (view.camera_id,width,height)
        if key in grids:
            continue
        scaled = intrinsics*np.array([width/camera.width,height/camera.height]*2)
        grid,rw,rh = calibration_grid(width,height,scaled,distortion)
        normalized = torch.from_numpy(grid).double()/torch.tensor([2*scaled[0]/rw,2*scaled[1]/rh])
        rays = torch.cat((normalized,torch.ones(height,width,1,dtype=torch.double)),-1)
        mapped = project_sensor_pixels(rays,scaled,distortion)
        yy,xx = torch.meshgrid(torch.arange(height),torch.arange(width),indexing="ij")
        residual = (mapped-torch.stack((xx,yy),-1)).norm(dim=-1)
        grids[key] = {"camera_id":view.camera_id,"model":camera.model,
                      "sensor_wh":[width,height],"raster_wh":[rw,rh],
                      "canvas_area_ratio":rw*rh/(width*height),
                      "fraction_error_gt_1px":float((residual>1).double().mean()),
                      "max_error_px":float(residual.max()),
                      "p99_error_px":float(residual.quantile(.99))}
    if not train_views:
        raise ValueError(f"No sparse cameras match training images in {scene}")
    return {"scene":str(scene.resolve()),"train_views":train_views,
            "tracks_excluded_at_nonpositive_depth":behind_camera_tracks,
            "reprojection_colmap_resolution":{k:error_summary(v) for k,v in errors.items()},
            "sensor_grid_audits":list(grids.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = {"training_tracks_only":True,"parameter_fitting":False,"scenes":[]}
    for scene in args.scenes:
        row = audit_scene(scene)
        result["scenes"].append(row)
        print(json.dumps(row),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
