# Oriented detail basis splatting

The results below were measured before ODB was restricted to the RGB branch.
They do not describe the current RGB-only implementation; that configuration
needs a new evaluation.

`--use_detail_basis --detail_basis_mode oriented_hermite` adds five RGB detail
coefficients to each Gaussian. They multiply two first-order and three
second-order image derivatives of rendered Gaussian coefficient fields. The
derivative directions come from the two largest local 3D scale axes, projected
into the current camera. The detail residual is added only to RGB after the
shared base refinement; thermal uses the original rendering path. The stored
thermal detail coefficients remain for compatibility with older checkpoints
and PLY files, but are inactive. RGB coefficients start at zero, so an
untrained ODB model renders the same base image. `screen_dog` remains available
as an ablation. The implementation applies finite-difference filters to
rasterized coefficient fields; it is an image-space approximation of derivative
Gaussian bases, not an analytic per-pixel derivative inside the CUDA rasterizer.

## Historical ablation (not valid for this request)

The script seeds each run from the 30,000-step **EMA-exported** baseline PLY and
saved feature modules, freezes the base model, and trains ODB coefficients for
2,000 additional steps. It renders the held-out test views and runs `metrics.py`.
This historical path is retained for debugging; it is not the strict 30k
protocol used for the requested comparison.
It needs the baseline results under
`output/camera_only_20261001_142717/RGBT-Scenes` and scene data under
`/home/lf/data/thermal3dgs/RGBT-Scenes`.

The historical run used `detail_basis_lr=0.001`, `detail_basis_reg_weight=0.005`,
`detail_mse_weight=0`, `detail_edge_weight=0.05`, and iteration 32,000.
Its per-scene logs and `summary.json` remain under the path below for audit.

## Historical result (not the final 30k protocol)

The completed run is at `/tmp/odb_oriented_balanced_10scenes_20261002`.
Entries below are candidate minus the original EMA baseline; lower LPIPS is
better. Each metric is the arithmetic mean over ten scenes, with each scene's
test-view mean computed by `metrics.py`.

| Metric | Baseline | ODB | Delta |
| --- | ---: | ---: | ---: |
| RGB PSNR | 25.18081 | 25.50854 | +0.32774 dB |
| Thermal PSNR | 26.13871 | 26.15527 | +0.01656 dB |
| RGB SSIM | 0.85814 | 0.85887 | +0.00072 |
| Thermal SSIM | 0.89378 | 0.89447 | +0.00069 |
| RGB LPIPS | 0.18175 | 0.18061 | -0.00114 |
| Thermal LPIPS | 0.15395 | 0.15279 | -0.00116 |

| Scene | RGB PSNR delta | Thermal PSNR delta |
| --- | ---: | ---: |
| Building | +0.4267 | +0.0160 |
| DailyStuff | +0.0972 | +0.0060 |
| Dimsum | +0.0995 | +0.0137 |
| Ebike | +0.2746 | +0.0091 |
| IronIngot | +0.0999 | +0.0255 |
| LandScape | +0.0228 | +0.0163 |
| Parterre | +0.8076 | +0.0148 |
| RoadBlock | +1.1668 | +0.0062 |
| RotaryKiln | +0.0491 | +0.0192 |
| Truck | +0.2332 | +0.0388 |

All ten scenes improve RGB and thermal PSNR. RGB PSNR clears +0.2 dB on the
ten-scene average; thermal PSNR does not. The gains are uneven, with much of
the RGB mean coming from Parterre and RoadBlock. These results compare an
ODB-extended model at 32,000 steps against a 30,000-step baseline; the base
weights were verified unchanged after ODB training, but this is not a matched
compute comparison against 2,000 extra baseline training steps.

## Follow-up ablations

All rows below are two-scene means over DailyStuff and LandScape, measured on
held-out test views against the same 30,000-step EMA baseline. They are probes,
not replacements for the ten-scene result above. The experimental code was
removed from the main ODB path after these tests; its patch and command wrapper
are saved under `/tmp/odb_thermal_geometry_lr5_pair_20261002`.

| Variant | RGB PSNR delta | Thermal PSNR delta | Thermal SSIM delta | Thermal LPIPS delta |
| --- | ---: | ---: | ---: | ---: |
| Thermal appearance fine-tune | +0.0531 | -0.0160 | -0.00159 | +0.00183 |
| Thermal-only MSE weight 50 | +0.0600 | +0.0242 | ~0.00000 | -0.00016 |
| Thermal center/scale fine-tune with MSE | +0.0603 | +0.0527 | -0.00043 | +0.00078 |
| Fivefold thermal geometry learning rates | +0.0605 | +0.0716 | -0.00172 | +0.00348 |
| ODB sparsity weight 0.0005 | +0.0611 | +0.0113 | +0.00048 | -0.00085 |

Increasing the thermal basis scale from 0.06 to 0.12 on four weak scenes also
left thermal PSNR nearly unchanged (+0.0168 dB mean). These trials do not
support claiming a +0.2 dB thermal PSNR gain.

A joint fine-tuning probe unfroze the original model during the same 2,000
steps. On DailyStuff, RGB PSNR fell by 0.0123 dB and thermal PSNR rose by only
0.0238 dB; RGB SSIM and both LPIPS values worsened. LandScape exhausted a
31 GB GPU before training finished, so this variant has no complete two-scene
or ten-scene result.

## Strict 30k protocol

Use `odb_train_30k_10scenes.sh` for a new RGB-only evaluation. It starts each
scene from dataset initialization and trains continuously for exactly 30,000
iterations. The base model is updated throughout; ODB starts after iteration
26,000 by default and is trained for the remaining 4,000 iterations. This
preserves the original densification and EMA export schedule and does not
load a completed baseline checkpoint or add extra iterations. Held-out test
views are compared with the original 30,000-step baseline. The ten-scene
arithmetic mean is written to `summary.json` under `ODB_OUTPUT_ROOT` after
all scenes finish.

## Historical strict 30k result

The completed ten-scene run is recorded at
`/tmp/odb_delayed_26k_30k_pair_20261002`, with the comparison in
`comparison_30000.json`. Every scene was trained continuously for exactly
30,000 iterations and evaluated on the same held-out views as the baseline.
The candidate-minus-baseline arithmetic means are:

| Metric | Delta |
| --- | ---: |
| RGB PSNR | **+0.37971 dB** |
| Thermal PSNR | -0.05988 dB |
| RGB SSIM | +0.00085 |
| Thermal SSIM | +0.00051 |
| RGB LPIPS | -0.00062 |
| Thermal LPIPS | -0.00074 |

The RGB PSNR criterion is met on the ten-scene mean. RGB PSNR improves on
8/10 scenes and thermal PSNR improves on 5/10 scenes; thermal PSNR is slightly
lower on average, while both thermal SSIM and thermal LPIPS improve. This is
the matched 30k result to use for the current implementation; the 32k table
above remains historical and is not part of this conclusion.

## 自行运行完整数据集

在当前机器上直接运行，脚本默认使用 `thermalgaussian` 环境中的 Python，
无需先激活 conda 环境。默认 GPU 0、1 各运行一个场景，依次完成全部十场：

```bash
cd /home/lf/code/Our_Project-New-1-1-1
bash odb_train_30k_10scenes.sh
```

只使用一张 GPU：

```bash
ODB_GPUS="0" bash odb_train_30k_10scenes.sh
```

指定数据与结果目录：

```bash
ODB_GPUS="0 1" \
ODB_DATA_ROOT=/home/lf/data/thermal3dgs/RGBT-Scenes \
ODB_OUTPUT_ROOT=/home/lf/code/Our_Project-New-1-1-1/output/odb_my_run \
bash odb_train_30k_10scenes.sh
```

脚本沿用上述实验的训练轮数，但 ODB 现在仅用于 RGB：每场从头训练 30,000 轮，
26,000 轮后启用 ODB；`oriented_hermite`，RGB scale 为 0.08，detail 学习率 0.001，
正则权重 0.005，edge 权重 0.05。复现入口不再接受场景子集或 detail 参数覆盖。
保留原代码的训练/测试划分及其余默认参数。

每次默认创建新的 `output/odb_30k_<时间>_<进程号>/`；指定的输出目录也必须是新目录。
终端显示每场开始/完成及日志路径。每场顺序执行训练、测试集渲染、指标计算，
全部成功后才输出完整十场均值。若某场失败，该 GPU 的队列停止，其他 GPU 会完成各自
队列；脚本最终返回失败并保留日志，不会把部分场景包装成完整数据集结果。

结果目录包含：

- `<场景>.log`：该场景的运行命令、训练进度及评估日志。
- `<场景>/`：模型、30,000 轮 checkpoint、测试集渲染及 `results.json`。
- `summary.json`：十场逐场指标及各场等权平均指标。
- `metrics_30000.csv`：十场 RGB/thermal PSNR、SSIM、LPIPS 和平均值，可用表格软件打开。
- `comparison_30000.json`：发现完整基线时，额外生成相对基线的逐场和均值变化。
- `report.log`：终端汇总；`launch_script.sh`：本次运行的脚本存档。

基线默认读取 `output/camera_only_20261001_142717/RGBT-Scenes`。
可以用 `ODB_BASELINE_ROOT` 指定其他基线，或设置为 `none` 仅汇总本次指标。
基线缺失时仍会正常计算本次十场平均。可用 `ODB_PORT_BASE` 修改训练端口，
用 `ODB_PYTHON` 指定 Python 解释器。上述环境变量的相对路径以项目目录为准。

仅检查命令、不启动训练：

```bash
bash odb_train_30k_10scenes.sh --dry-run
```

`odb_report.py` 始终要求全部十场的 `ours_30000` 结果，拒绝缺失或非有限指标。
