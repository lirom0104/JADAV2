#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.camera_calibration import project_sensor_pixels, warp_raster_to_sensor
from utils.general_utils import build_rotation

def _resolve_head_appearance(render_params, viewpoint_camera, pc, pipe, override_precomp):
    if override_precomp is not None:
        return None, override_precomp

    if pipe.convert_SHs_python:
        shs_view = render_params["features"].transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = render_params["means3D"] - viewpoint_camera.camera_center.repeat(render_params["features"].shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        return None, torch.clamp_min(sh2rgb + 0.5, 0.0)

    return render_params["features"], None

def _render_head(
    rasterizer,
    render_params,
    means2D,
    opacity,
    viewpoint_camera,
    pc,
    pipe,
    override_precomp=None,
):
    head_shs, head_precomp = _resolve_head_appearance(
        render_params=render_params,
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        override_precomp=override_precomp,
    )

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = render_params["cov3D_precomp"]
    else:
        scales = render_params["scales"]
        rotations = render_params["rotations"]

    rendered_thermal, rendered_color, radii = rasterizer(
        means3D=render_params["means3D"],
        means2D=means2D,
        thermal_shs=head_shs,
        color_shs=head_shs,
        thermals_precomp=head_precomp,
        colors_precomp=head_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    return {
        "rendered_thermal": rendered_thermal,
        "rendered_color": rendered_color,
        "radii": radii,
        "visibility_filter": radii > 0,
        "viewspace_points": means2D,
    }


def _central_difference(field):
    """Apply zero-mean central differences channel-wise to a rendered coeff field."""
    if field.ndim != 3 or field.shape[0] != 3:
        raise ValueError("detail coefficient fields must have shape [3,H,W]")
    x_kernel = field.new_tensor([[-0.5, 0.0, 0.5]]).view(1, 1, 1, 3).expand(3, 1, 1, 3)
    y_kernel = field.new_tensor([[-0.5], [0.0], [0.5]]).view(1, 1, 3, 1).expand(3, 1, 3, 1)
    field = field.unsqueeze(0)
    dx = F.conv2d(field, x_kernel, padding=(0, 1), groups=3)
    dy = F.conv2d(field, y_kernel, padding=(1, 0), groups=3)
    return dx.squeeze(0), dy.squeeze(0)


def _second_difference(field, direction):
    kernels = {
        "xx": [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],
        "xy": [[0.25, 0.0, -0.25], [0.0, 0.0, 0.0], [-0.25, 0.0, 0.25]],
        "yy": [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]],
    }
    kernel = field.new_tensor(kernels[direction]).view(1, 1, 3, 3).expand(3, 1, 3, 3)
    return F.conv2d(field.unsqueeze(0), kernel, padding=1, groups=3).squeeze(0)


def _project_detail_points(points, camera):
    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    if hasattr(camera, "calibration_intrinsics"):
        view_points = homogeneous @ camera.world_view_transform
        return project_sensor_pixels(
            view_points[:, :3], camera.calibration_intrinsics, camera.calibration_distortion
        )
    clip_points = homogeneous @ camera.full_proj_transform
    ndc = clip_points[:, :2] / (clip_points[:, 3:4] + 1e-7)
    return torch.stack((
        (ndc[:, 0] + 1.0) * camera.image_width * 0.5,
        (ndc[:, 1] + 1.0) * camera.image_height * 0.5,
    ), dim=1)


def _oriented_detail_fields(coefficients, render_params, camera):
    scales = render_params["scales"]
    principal_scales, indices = scales.topk(2, dim=1)
    rotation = build_rotation(render_params["rotations"])
    axes = rotation.gather(2, indices[:, None, :].expand(-1, 3, -1))
    offsets = axes * principal_scales[:, None, :]
    means = render_params["means3D"]
    center = _project_detail_points(means, camera)
    projected = torch.stack([
        _project_detail_points(means + offsets[:, :, axis], camera) - center
        for axis in range(2)
    ], dim=1)
    projected_length = projected.norm(dim=2).clamp_min(1e-5)
    directions = projected / projected_length[:, :, None]
    sigma = projected_length.clamp(0.5, 8.0)
    ux, uy = directions[:, 0, 0:1], directions[:, 0, 1:2]
    vx, vy = directions[:, 1, 0:1], directions[:, 1, 1:2]
    su, sv = sigma[:, 0:1], sigma[:, 1:2]
    du, dv, duu, duv, dvv = coefficients.unbind(dim=1)
    fields = (
        du * su * ux + dv * sv * vx,
        du * su * uy + dv * sv * vy,
        duu * su.square() * ux.square() + duv * su * sv * ux * vx + dvv * sv.square() * vx.square(),
        duu * su.square() * 2 * ux * uy + duv * su * sv * (ux * vy + uy * vx) + dvv * sv.square() * 2 * vx * vy,
        duu * su.square() * uy.square() + duv * su * sv * uy * vy + dvv * sv.square() * vy.square(),
    )
    return fields


def _render_oriented_detail_basis(
    rasterizer, render_params, means2D, viewpoint_camera, pc, pipe, warp
):
    coefficients = render_params["detail_coefficients"]
    fields = _oriented_detail_fields(coefficients, render_params, viewpoint_camera)
    rendered = []
    for field in fields:
        head = _render_head(
            rasterizer=rasterizer,
            render_params=render_params,
            means2D=means2D,
            opacity=render_params["opacity"],
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            pipe=pipe,
            override_precomp=field,
        )
        rendered.append(warp(head["rendered_color"]))
    dx, _ = _central_difference(rendered[0])
    _, dy = _central_difference(rendered[1])
    return (
        dx + dy
        + _second_difference(rendered[2], "xx")
        + _second_difference(rendered[3], "xy")
        + _second_difference(rendered[4], "yy")
    )


def _render_detail_basis(
    rasterizer,
    detail_rasterizer,
    render_params,
    means2D,
    viewpoint_camera,
    pc,
    pipe,
    warp,
    image_height,
    image_width,
):
    coefficients = render_params.get("detail_coefficients")
    if coefficients is None or coefficients.numel() == 0 or not getattr(pc, "detail_basis_runtime_enabled", pc.use_detail_basis):
        zero = render_params["features"].new_zeros((3, image_height, image_width))
        return zero
    # ODB learns appearance coefficients; the base rasterization already
    # supplies geometry gradients. Keep the extra passes coefficient-only.
    render_params = {
        key: value if key == "detail_coefficients" or not torch.is_tensor(value) else value.detach()
        for key, value in render_params.items()
    }
    means2D = means2D.detach()
    if pc.detail_basis_mode == "oriented_hermite":
        return _render_oriented_detail_basis(
            detail_rasterizer, render_params, means2D, viewpoint_camera, pc, pipe, warp
        )

    rendered_fields = []
    for direction in range(min(int(coefficients.shape[1]), 4)):
        head = _render_head(
            rasterizer=detail_rasterizer,
            render_params=render_params,
            means2D=means2D,
            opacity=render_params["opacity"],
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            pipe=pipe,
            override_precomp=coefficients[:, direction, :],
        )
        # Both outputs are identical for precomputed coefficients; color is used
        # here to keep the pass independent of the modality's SH head.
        rendered_fields.append(warp(head["rendered_color"]))

    dx_x, dy_x = _central_difference(rendered_fields[0])
    dx_y, dy_y = _central_difference(rendered_fields[1])
    detail = dx_x + dy_y
    if len(rendered_fields) > 2:
        dx_45, dy_45 = _central_difference(rendered_fields[2])
        detail = detail + (dx_45 + dy_45) / math.sqrt(2.0)
    if len(rendered_fields) > 3:
        dx_135, dy_135 = _central_difference(rendered_fields[3])
        detail = detail + (dx_135 - dy_135) / math.sqrt(2.0)
    if coefficients.shape[1] > 4:
        dog_coefficients = coefficients[:, 4, :]
        wide = _render_head(
            rasterizer=detail_rasterizer,
            render_params=render_params,
            means2D=means2D,
            opacity=render_params["opacity"],
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            pipe=pipe,
            override_precomp=dog_coefficients,
        )["rendered_color"]
        radius_ratio = 0.7
        narrow_params = dict(render_params)
        narrow_params["scales"] = render_params["scales"] * radius_ratio
        narrow_params["cov3D_precomp"] = render_params["cov3D_precomp"] * radius_ratio**2
        narrow = _render_head(
            rasterizer=detail_rasterizer,
            render_params=narrow_params,
            means2D=means2D,
            opacity=render_params["opacity"],
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            pipe=pipe,
            override_precomp=dog_coefficients,
        )["rendered_color"]
        detail = detail + warp(narrow - radius_ratio**2 * wide)
    return detail

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, override_thermal = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    raster_camera = getattr(viewpoint_camera, "raster_camera", viewpoint_camera)
    tanfovx = math.tan(raster_camera.FoVx * 0.5)
    tanfovy = math.tan(raster_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(raster_camera.image_height),
        image_width=int(raster_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=raster_camera.world_view_transform,
        projmatrix=raster_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    detail_rasterizer = None
    if getattr(pc, "detail_basis_runtime_enabled", pc.use_detail_basis):
        detail_rasterizer = GaussianRasterizer(
            raster_settings=raster_settings._replace(bg=torch.zeros_like(bg_color))
        )

    bgfc_outputs = pc.get_bgfc_outputs()
    rgb_render_params = pc.get_rgb_render_params(
        scaling_modifier=scaling_modifier,
        bgfc_outputs=bgfc_outputs,
    )
    thermal_render_params = pc.get_thermal_render_params(
        scaling_modifier=scaling_modifier,
        bgfc_outputs=bgfc_outputs,
    )

    rgb_head = _render_head(
        rasterizer=rasterizer,
        render_params=rgb_render_params,
        means2D=screenspace_points,
        opacity=rgb_render_params["opacity"],
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        override_precomp=override_color,
    )

    # Keep a dedicated thermal screen-space tensor so densification can consume
    # gradients from both modalities instead of RGB only.
    thermal_screenspace_points = torch.zeros_like(
        thermal_render_params["means3D"],
        dtype=thermal_render_params["means3D"].dtype,
        requires_grad=True,
        device="cuda",
    ) + 0
    try:
        thermal_screenspace_points.retain_grad()
    except:
        pass
    thermal_head = _render_head(
        rasterizer=rasterizer,
        render_params=thermal_render_params,
        means2D=thermal_screenspace_points,
        opacity=thermal_render_params["opacity"],
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        override_precomp=override_thermal,
    )
    rendered_color = pc.apply_render_calibration(
        warp_raster_to_sensor(rgb_head["rendered_color"], viewpoint_camera), "color"
    )
    rendered_thermal = pc.apply_render_calibration(
        warp_raster_to_sensor(thermal_head["rendered_thermal"], viewpoint_camera), "thermal"
    )
    # Apply RGB detail after the shared base refinement so thermal keeps its
    # original rendering path.
    rendered_color, rendered_thermal = pc.apply_multimodal_refinement(rendered_color, rendered_thermal)
    if getattr(pc, "detail_basis_runtime_enabled", pc.use_detail_basis):
        rgb_detail = _render_detail_basis(
            rasterizer=rasterizer,
            detail_rasterizer=detail_rasterizer,
            render_params=rgb_render_params,
            means2D=screenspace_points,
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            pipe=pipe,
            warp=lambda x: warp_raster_to_sensor(x, viewpoint_camera),
            image_height=int(raster_camera.image_height),
            image_width=int(raster_camera.image_width),
        )
        rendered_color = rendered_color + pc.detail_basis_scale * rgb_detail
    else:
        rgb_detail = rendered_color.new_zeros(())
    rendered_thermal_base = rendered_thermal
    ir_kernel_residual = rendered_thermal.new_zeros(())
    if getattr(pc, "ir_kernel_runtime_enabled", False):
        # Signed, spatially localized radiance is splatted from a distinct
        # infrared Gaussian kernel. There is no image-space filter here.
        ir_params = pc.get_ir_kernel_render_params(thermal_render_params, scaling_modifier)
        ir_rasterizer = GaussianRasterizer(
            raster_settings=raster_settings._replace(bg=torch.zeros_like(bg_color))
        )
        ir_head = _render_head(
            rasterizer=ir_rasterizer,
            render_params=ir_params,
            means2D=thermal_screenspace_points.detach(),
            opacity=ir_params["opacity"],
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            pipe=pipe,
            override_precomp=ir_params["colors_precomp"],
        )
        ir_kernel_residual = warp_raster_to_sensor(ir_head["rendered_thermal"], viewpoint_camera)
        rendered_thermal = rendered_thermal_base + ir_kernel_residual
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
            "render": rendered_color,
            "render_thermal": rendered_thermal,
            "render_thermal_base": rendered_thermal_base,
            "ir_kernel_residual": ir_kernel_residual,
            "render_color": rendered_color,
            "viewspace_points": rgb_head["viewspace_points"],
            "visibility_filter": rgb_head["visibility_filter"],
            "radii": rgb_head["radii"],
            "rgb_viewspace_points": rgb_head["viewspace_points"],
            "rgb_visibility_filter": rgb_head["visibility_filter"],
            "rgb_radii": rgb_head["radii"],
            "thermal_viewspace_points": thermal_head["viewspace_points"],
            "thermal_visibility_filter": thermal_head["visibility_filter"],
            "thermal_radii": thermal_head["radii"],
            "detail_rgb_abs_mean": rgb_detail.detach().abs().mean(),
            "detail_thermal_abs_mean": rendered_thermal.new_zeros(()),
            "bgfc_gate_th2rgb_anchor": bgfc_outputs["gate_th2rgb_anchor"],
            "bgfc_gate_rgb2th_anchor": bgfc_outputs["gate_rgb2th_anchor"],
            "bgfc_stability_reg": bgfc_outputs["stability_reg"],
            "bgfc_gate_sparsity_reg": bgfc_outputs["gate_sparsity_reg"],
            "bgfc_gate_collapse_reg": bgfc_outputs["gate_collapse_reg"],
            "bgfc_gate_overlap_reg": bgfc_outputs["gate_overlap_reg"],
            "bgfc_gate_th2rgb_mean": bgfc_outputs["gate_th2rgb_mean"],
            "bgfc_gate_th2rgb_std": bgfc_outputs["gate_th2rgb_std"],
            "bgfc_gate_th2rgb_max": bgfc_outputs["gate_th2rgb_max"],
            "bgfc_gate_rgb2th_mean": bgfc_outputs["gate_rgb2th_mean"],
            "bgfc_gate_rgb2th_std": bgfc_outputs["gate_rgb2th_std"],
            "bgfc_gate_rgb2th_max": bgfc_outputs["gate_rgb2th_max"],
            "bgfc_delta_th2rgb_mag_mean": bgfc_outputs["delta_th2rgb_mag_mean"],
            "bgfc_delta_rgb2th_mag_mean": bgfc_outputs["delta_rgb2th_mag_mean"],
    }
