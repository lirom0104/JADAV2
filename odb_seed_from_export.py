"""Create a resumable ODB checkpoint from an exported Gaussian scene."""

import os
from argparse import ArgumentParser

import torch

from arguments import ModelParams, OptimizationParams, get_combined_args
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state


def main():
    parser = ArgumentParser(description="Seed ODB training from an exported scene")
    model = ModelParams(parser, sentinel=True)
    optimization = OptimizationParams(parser)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output_checkpoint", required=True)
    args = get_combined_args(parser)
    safe_state(True)
    dataset = model.extract(args)
    gaussians = GaussianModel(
        dataset.sh_degree,
        use_bgfc=dataset.use_bgfc,
        use_at_gom=dataset.use_at_gom,
        bgfc_hidden_dim=dataset.bgfc_hidden_dim,
        bgfc_gate_init_bias=dataset.bgfc_gate_init_bias,
        bgfc_thermal_grayscale_context=dataset.bgfc_thermal_grayscale_context,
        bgfc_rgb_luma_transfer_only=dataset.bgfc_rgb_luma_transfer_only,
        use_render_calibration=dataset.use_render_calibration,
        use_color_refinement=dataset.use_color_refinement,
        color_refinement_hidden_dim=dataset.color_refinement_hidden_dim,
        color_refinement_max_residual=dataset.color_refinement_max_residual,
        use_detail_basis=dataset.use_detail_basis,
        detail_basis_mode=getattr(dataset, "detail_basis_mode", None) or "screen_dog",
        detail_basis_scale=getattr(dataset, "detail_basis_scale", None) or 0.08,
        detail_basis_thermal_scale=getattr(dataset, "detail_basis_thermal_scale", None) or 0.06,
    )
    Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.training_setup(optimization.extract(args))
    os.makedirs(os.path.dirname(os.path.abspath(args.output_checkpoint)), exist_ok=True)
    torch.save((gaussians.capture(), args.iteration), args.output_checkpoint)


if __name__ == "__main__":
    main()
