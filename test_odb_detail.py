import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from gaussian_renderer import _render_detail_basis


class DetailBasisGradientTests(unittest.TestCase):
    def test_delayed_basis_returns_zero_before_activation(self):
        coefficients = torch.randn(2, 5, 3, requires_grad=True)
        render_params = {"detail_coefficients": coefficients, "features": torch.zeros(2, 1, 3)}
        pc = SimpleNamespace(use_detail_basis=True, detail_basis_runtime_enabled=False)
        with patch("gaussian_renderer._render_oriented_detail_basis") as detail_pass:
            output = _render_detail_basis(
                None, None, render_params, torch.zeros(2, 3), None, pc, None,
                lambda x: x, 2, 2,
            )
        self.assertTrue(torch.equal(output, torch.zeros(3, 2, 2)))
        detail_pass.assert_not_called()

    def test_extra_passes_keep_coefficient_gradients_only(self):
        coefficients = torch.randn(2, 5, 3, requires_grad=True)
        means = torch.randn(2, 3, requires_grad=True)
        screenspace = torch.randn(2, 3, requires_grad=True)
        render_params = {
            "detail_coefficients": coefficients,
            "means3D": means,
            "features": torch.randn(2, 1, 3, requires_grad=True),
        }
        pc = SimpleNamespace(use_detail_basis=True, detail_basis_mode="oriented_hermite")

        def inspect_pass(_rasterizer, params, projected, *_args):
            self.assertIs(params["detail_coefficients"], coefficients)
            self.assertFalse(params["means3D"].requires_grad)
            self.assertFalse(params["features"].requires_grad)
            self.assertFalse(projected.requires_grad)
            return coefficients.sum().expand(3, 2, 2)

        with patch("gaussian_renderer._render_oriented_detail_basis", side_effect=inspect_pass):
            output = _render_detail_basis(
                None, None, render_params, screenspace, None, pc, None,
                lambda x: x, 2, 2,
            )
        output.sum().backward()
        self.assertIsNotNone(coefficients.grad)
        self.assertIsNone(means.grad)
        self.assertIsNone(screenspace.grad)


if __name__ == "__main__":
    unittest.main()
