"""Runtime state needed for matched, reproducible checkpoint continuations."""

import random

import numpy as np
import torch


def replay_camera_sampler(cameras, iterations):
    """Recover legacy camera sampling after Scene's deterministic shuffle.

    This repository's training loop draws one Python randint per iteration;
    it does not otherwise use the Python RNG after constructing Scene.
    This does not recover the CUDA RNG or the missing export EMA state.
    """
    stack = []
    for _ in range(iterations):
        if not stack:
            stack = cameras.copy()
        stack.pop(random.randint(0, len(stack) - 1))
    return stack


def capture_runtime(viewpoint_stack, ema_export_state, model, seed):
    return {
        "seed": seed,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "viewpoint_uids": [camera.uid for camera in (viewpoint_stack or [])],
        "ema_export_state": ema_export_state,
        "cmo_start_point_count": model.cmo_start_point_count,
        "cmo_budget_reference_count": model.cmo_budget_reference_count,
    }


def restore_runtime(state, cameras, model, seed):
    if state["seed"] != seed:
        raise ValueError("Checkpoint seed differs from --seed; exact continuation requires matching seeds")
    by_uid = {camera.uid: camera for camera in cameras}
    if len(by_uid) != len(cameras):
        raise ValueError("Camera UIDs must be unique to restore the sampling order")
    stack = [by_uid[uid] for uid in state["viewpoint_uids"]]
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    model.cmo_start_point_count = state["cmo_start_point_count"]
    model.cmo_budget_reference_count = state["cmo_budget_reference_count"]
    return stack, state["ema_export_state"]
