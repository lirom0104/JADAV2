# Rasterizer backend audit

Result: **passed**. The installed thermalgaussian rasterizer with explicit `pstb=None` agrees with the original JADA rasterizer on five synthetic GPU 0 forward/backward cases: positive precomputed colors, mixed signed colors, an all-negative thermal Gaussian, spherical harmonics, and precomputed covariance.

All 20 forward outputs (thermal, color, radii, loss) were bitwise identical. All 54 tensor comparisons passed atol=1e-6 and rtol=1e-5. Maximum gradient difference: 2.98023223877e-08.

The negative-color case reaches thermal output -0.24962756037712097 and preserves exact sign linearity. Central finite differences confirm signed precomputed-color and opacity gradients (maximum absolute discrepancy about 2.27e-8).

Physical GPU selection was fixed to `CUDA_VISIBLE_DEVICES=0`; each worker saw exactly one CUDA device, NVIDIA GeForce RTX 5090. Peak allocated memory was 181760 bytes per worker. Workers ran serially. This audit did not train, change installed packages, or touch the baseline.

The JADA Python wrapper SHA-256 matches the current workspace original submodule wrapper: `90c7c64022c6bd2498f99476df3eaa606ece8fd6ccc88df0ba236807247c48ea`.

| Backend | Extension SHA-256 |
|---|---|
| JADA | `b2ff4dcfa3fea2325cd610db75e9e31f55cbed321ef64728015c8784b8f0d577` |
| thermalgaussian | `69dbb8c62800e83704213cd50042a2e417babfb333b4ff9d025cead7a1743f2a` |

Executed command:

```bash
CUDA_VISIBLE_DEVICES=0 /home/lf/miniconda3/envs/thermalgaussian/bin/python ir_experiments/verify_backend.py
```

The initial sandbox attempt could not access CUDA. The exact command was rerun with execution escalation and succeeded. Full commands, environment identities, tensor comparisons, numerical derivatives, hashes, and raw arrays are in `comparison.json`, per-environment JSON/log files, and NPZ files.

Scope limitation: synthetic coverage supports retaining the present extension with `pstb=None`; it does not mathematically prove equivalence for every input or guarantee identical long training trajectories.
