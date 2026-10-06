import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel


class RGBOnlyDetailTests(unittest.TestCase):
    def test_only_rgb_detail_is_rendered(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required by the renderer")

        rgb = torch.full((3, 2, 2), 2.0, device="cuda")
        thermal = torch.full((3, 2, 2), 3.0, device="cuda")
        rgb_params = {"features": rgb, "opacity": torch.ones(1, 1, device="cuda")}
        thermal_params = {
            "features": thermal,
            "means3D": torch.zeros(1, 3, device="cuda"),
            "opacity": torch.ones(1, 1, device="cuda"),
        }
        model = SimpleNamespace(
            get_xyz=torch.zeros(1, 3, device="cuda"),
            active_sh_degree=0,
            use_detail_basis=True,
            detail_basis_runtime_enabled=True,
            detail_basis_scale=0.5,
            get_bgfc_outputs=lambda: defaultdict(lambda: torch.zeros((), device="cuda")),
            get_rgb_render_params=lambda **kwargs: rgb_params,
            get_thermal_render_params=lambda **kwargs: thermal_params,
            apply_render_calibration=lambda image, modality: image,
            apply_multimodal_refinement=lambda color, th: (color, th),
        )
        camera = SimpleNamespace(
            FoVx=1.0, FoVy=1.0, image_height=2, image_width=2,
            world_view_transform=None, full_proj_transform=None, camera_center=None,
        )

        def render_head(*args, **kwargs):
            image = kwargs["render_params"]["features"]
            return {
                "rendered_color": image,
                "rendered_thermal": image,
                "radii": torch.ones(1, device="cuda"),
                "visibility_filter": torch.ones(1, dtype=torch.bool, device="cuda"),
                "viewspace_points": kwargs["means2D"],
            }

        with patch("gaussian_renderer.GaussianRasterizationSettings") as settings, \
                patch("gaussian_renderer.GaussianRasterizer"), \
                patch("gaussian_renderer._render_head", side_effect=render_head), \
                patch("gaussian_renderer._render_detail_basis", return_value=torch.ones_like(rgb)) as detail_pass, \
                patch("gaussian_renderer.warp_raster_to_sensor", side_effect=lambda image, camera: image):
            settings.return_value = MagicMock()
            output = render(camera, model, SimpleNamespace(debug=False), torch.zeros(3, device="cuda"))

        detail_pass.assert_called_once()
        self.assertIs(detail_pass.call_args.kwargs["render_params"], rgb_params)
        torch.testing.assert_close(output["render_color"], rgb + 0.5)
        torch.testing.assert_close(output["render_thermal"], thermal)
        self.assertEqual(output["detail_thermal_abs_mean"].item(), 0.0)

    def test_thermal_detail_is_not_optimized_or_regularized(self):
        model = GaussianModel.__new__(GaussianModel)
        model.use_detail_basis = True
        model.use_at_gom = False
        model.use_render_calibration = False
        model._detail_rgb = nn.Parameter(torch.ones(2, 5, 3))
        model._detail_thermal = nn.Parameter(torch.full((2, 5, 3), 4.0))
        for name in (
            "_at_gom_opacity_bias_rgb", "_at_gom_opacity_bias_th",
            "_at_gom_center_residual", "_at_gom_log_scale_residual",
            "_render_calibration_color_scale", "_render_calibration_color_bias",
            "_render_calibration_thermal_scale", "_render_calibration_thermal_bias",
        ):
            setattr(model, name, torch.empty(0))
        model.optimizer = SimpleNamespace(param_groups=[
            {"name": "detail_rgb", "params": [model._detail_rgb], "lr": 0.001},
            {"name": "detail_thermal", "params": [model._detail_thermal], "lr": 0.001},
        ])

        model._refresh_optional_parameter_grad_flags()
        self.assertTrue(model._detail_rgb.requires_grad)
        self.assertFalse(model._detail_thermal.requires_grad)
        model.set_detail_basis_only()
        self.assertTrue(model._detail_rgb.requires_grad)
        self.assertFalse(model._detail_thermal.requires_grad)
        self.assertEqual(model.get_detail_regularization().item(), 0.5)
        model._restore_detail_learning_rate(SimpleNamespace(detail_basis_lr=0.002))
        self.assertEqual(model.optimizer.param_groups[0]["lr"], 0.002)
        self.assertEqual(model.optimizer.param_groups[1]["lr"], 0.0)


if __name__ == "__main__":
    unittest.main()
