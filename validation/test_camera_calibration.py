from types import SimpleNamespace
import unittest

import numpy as np
import torch

from utils.camera_calibration import (calibration_grid, project_sensor_pixels,
                                      sensor_sample_neighbors, warp_raster_to_sensor)


class CameraCalibrationTests(unittest.TestCase):
    def test_inverse_map_round_trip_matches_sensor_centers(self):
        width, height = 64, 80
        intrinsics = np.array([70., 72., 35., 37.])
        distortion = np.array([-.12,.01,.002,-.001])
        grid, rw, rh = calibration_grid(width,height,intrinsics,distortion)
        normalized = torch.from_numpy(grid).double() / torch.tensor([2*intrinsics[0]/rw,2*intrinsics[1]/rh])
        points = torch.cat((normalized,torch.ones(height,width,1,dtype=torch.double)),-1)
        actual = project_sensor_pixels(points,intrinsics,distortion)
        yy,xx = torch.meshgrid(torch.arange(height),torch.arange(width),indexing="ij")
        expected = torch.stack((xx,yy),-1).double()
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=0)

    def test_contribution_sampling_matches_bilinear_image_warp(self):
        intrinsics = np.array([50.,52.,22.,17.])
        grid,rw,rh = calibration_grid(40,32,intrinsics,np.array([-.1,.01,.002,.003]))
        camera = SimpleNamespace(calibration_grid=torch.from_numpy(grid),
                                 raster_camera=SimpleNamespace(image_width=rw,image_height=rh))
        image = torch.rand(3,rh,rw,requires_grad=True)
        xy = torch.tensor([[0.,0.],[20.,17.],[39.,31.]])
        points,weights = sensor_sample_neighbors(camera,xy)
        ix,iy = points.long().unbind(-1)
        expected = (image[:,iy,ix]*weights).reshape(3,3,4).sum(-1)
        actual = warp_raster_to_sensor(image,camera)[:,xy[:,1].long(),xy[:,0].long()]
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
        ga, = torch.autograd.grad(actual.sum(),image,retain_graph=True)
        ge, = torch.autograd.grad(expected.sum(),image)
        torch.testing.assert_close(ga,ge)
