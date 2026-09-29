"""Shared dataset assignments and worker lifecycle for batch training."""

import os
from pathlib import Path
import signal
import subprocess

JOBS = (
    ("RGBT-Scenes", 0, 6120, Path("/home/lf/data/thermal3dgs/RGBT-Scenes"),
     ("Building", "DailyStuff", "Dimsum", "Ebike", "IronIngot",
      "LandScape", "Parterre", "RoadBlock", "RotaryKiln", "Truck")),
    ("ThermoScenes1_3dgs", 1, 6121, Path("/home/lf/data/ThermoScenes1_3dgs"),
     ("buildingA_spring", "buildingA_winter", "double_robot",
      "exhibition_building", "freezing_ice_cup", "heater_water_cup",
      "heater_water_kettle", "melting_ice_cup", "raspberrypi", "trees")),
)
PRIMARY = ("color_PSNR", "color_SSIM", "color_LPIPS",
           "thermal_PSNR", "thermal_SSIM", "thermal_LPIPS")


def stop_workers(workers):
    running = [p for p in workers if p.poll() is None]
    for process in running:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for process in running:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


