# JADA

## Installation

The provided environment uses Python 3.10, PyTorch 2.10.0, and CUDA 12.8. An NVIDIA GPU with a compatible driver is required.

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate JADA
```

The required CUDA extensions are included directly in this repository. Install them from the repository root after activating the Conda environment:

```bash
python -m pip install --no-build-isolation ./submodules/simple-knn
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
```

Verify the installation:

```bash
python -c "import torch; from simple_knn._C import distCUDA2; from diff_gaussian_rasterization import GaussianRasterizer; print('CUDA extensions: OK')"
python train.py --help
```

The first LPIPS use may download pretrained VGG and LPIPS weights to the PyTorch cache.

## Run the included example

The repository includes a ready-to-use RGB--thermal scene in `example_scene/`. Its training and test views have already been split, and the COLMAP reconstruction is provided in `example_scene/sparse/0/`.

### Training

Train the example scene for the default 30,000 iterations:

```bash
python train.py \
    -s example_scene \
    -m output/example_scene
```

The trained Gaussian model and configuration will be saved under `output/example_scene/`.

### Rendering

Render the test views from the saved 30,000-iteration model:

```bash
python render.py \
    -m output/example_scene \
    --iteration 30000 \
    --skip_train
```

Omit `--iteration 30000` to load the latest saved iteration automatically. Remove `--skip_train` to render both the training and test views.

### Evaluation

Compute PSNR, SSIM, and LPIPS for both RGB and thermal test images:

```bash
python metrics.py -m output/example_scene
```

Compute the thermal boundary F-score and hot-region IoU:

```bash
python extra_metrics.py -m output/example_scene
```

The aggregate metrics are written to `output/example_scene/results.json`. Per-view metrics are written to `per_view.json` and `per_view_extra_metrics.json`.

## Use your own scene

JADA expects a preprocessed COLMAP scene with paired RGB and thermal images:

```text
SceneName/
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
  rgb/
    train/
    test/
  thermal/
    train/
    test/
```

Text-form COLMAP files are also supported. RGB and thermal images within each split should have matching filenames. The loader first matches exact basenames, then normalized stems, and finally sorted indices as a fallback.

Train a custom scene with:

```bash
python train.py \
    -s /path/to/SceneName \
    -m output/SceneName
```

Frequently used options include:

- `--iterations`: total training iterations.
- `--test_iterations`: iterations at which validation is run.
- `--save_iterations`: iterations at which Gaussian models are saved.
- `--checkpoint_iterations`: iterations at which resumable checkpoints are saved.
- `--start_checkpoint`: path to a checkpoint from which training is resumed.
- `--port`: network GUI port; the default is `6009`.

Render and evaluate the custom scene with:

```bash
python render.py -m output/SceneName --iteration 30000 --skip_train
python metrics.py -m output/SceneName
python extra_metrics.py -m output/SceneName
```

`render.py` reads the training configuration from `output/SceneName/cfg_args`. Multiple scenes can be evaluated in one command:

```bash
python metrics.py -m output/SceneA output/SceneB
python extra_metrics.py -m output/SceneA output/SceneB
```

## 仅相机投影修正：两个数据集一键运行

`Our_Project-New-2` 使用以下入口运行原模型加相机投影修正。
BGFC、AT-GOM、CMO 等原有功能保留；增密使用 RGB 屏幕空间梯度。
CWGC、后期 RMSE 混合、额外后期学习率退火、近相机剪枝、双模态增密及 RGB 梯度下限的实现已移除。

```bash
cd /home/lf/code/Our_Project-New-2
./run_camera_only_all.sh --dry-run
./run_camera_only_all.sh
```

脚本使用 JADA 环境的 Python，无需手动激活环境。GPU 0 顺序运行 RGBT-Scenes 的
10 个场景，GPU 1 顺序运行 ThermoScenes1_3dgs 的 10 个场景；固定 seed=0，
每场景从零训练 30000 步，然后自动渲染、计算 PSNR/SSIM/LPIPS、额外指标及温度误差。

结果写入新的 `output/CameraOnly_seed0_<时间戳>_<进程号>/`；可通过 `--output` 指定
尚不存在的输出目录。完成后查看 `comparison.txt`、`comparison_means.csv` 和
`comparison_per_scene.csv`，逐场景详细进度在各自的 `train.log` 中。

默认用保留的 `output/CWGC_seed0_20260925_130204_2918293/` 校验测试目标并比较指标，
该目录是此前 CWGC＋相机修正＋RMSE 的组合结果，**不是原始 JADA 基线**。
已有结果仅用于评估和比较，不作为模型初始化；不再依赖已删除的 `output/JADA_batch`。
其他参考结果可用 `--reference-root /path/to/results` 指定。

相机修正使用已有 COLMAP 内参和畸变参数，在无畸变画布上渲染后映射回原图网格；
训练与评估目标保持原样。单场景通过 `--use_camera_calibration` 开启，批量入口已包含该参数。
原有随机种子、断点恢复、EMA 导出和 JSON 训练日志继续可用。

`scripts/cwgc_run.py`、`cwgc_evaluate.py`、`cwgc_dataset_report.py` 等通用工具保留原文件名，
供批量训练、评估和历史结果查询使用，不包含 CWGC 训练实现。
旧 CameraOnly 配置中的已删除选项仍可供报告脚本读取；重新执行旧训练命令时需移除这些选项。
历史报告的源码校验可通过 `scripts/cwgc_dataset_report.py --source-snapshot /path/to/source_snapshot`
指定训练时的源码快照，原有哈希检查仍然生效。

## Temperature evaluation

For ThermoScenes outputs, compute temperature MAE and ROI MAE across all scenes with:

```bash
python wendu.py \
    --data_root /path/to/ThermoScenes \
    --output_root output/Thermoscenes
```

Each scene under `--data_root` must contain a `temperature_bounds.json` file with the absolute temperature range:

```json
{
  "absolute_min_temperature": -20.0,
  "absolute_max_temperature": 50.0
}
```

If `gt_csv` or `renders_csv` is absent, `wendu.py` linearly maps the corresponding 8-bit thermal PNG values to this temperature range and creates the CSV matrices. Existing CSV matrices are reused. Before evaluation, predicted and ground-truth temperatures below `-20.0` degrees Celsius are clipped to `-20.0`. `MAE` is the image-averaged global mean absolute error. `MAE_roi` is the image-averaged mean absolute error over pixels whose ground-truth temperature is above the per-image Otsu threshold.

The per-scene summary is written to:

```text
output/Thermoscenes/batch_test_evaluation_results.csv
```

## Output layout

A completed run has the following structure:

```text
output/SceneName/
  cfg_args
  cameras.json
  input.ply
  point_cloud/
    iteration_30000/
  train/
    ours_30000/
  test/
    ours_30000/
      renders_color/
      gt_color/
      renders_thermal/
      gt_thermal/
  results.json
  per_view.json
  per_view_extra_metrics.json
```

## Acknowledgements

This project builds on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Thermal Gaussian](https://github.com/chen-hangyu/Thermal-Gaussian-main). The vendored `simple-knn`, differentiable Gaussian rasterization, and GLM sources retain their original copyright and license notices.

## License

This repository inherits the non-commercial research and evaluation license of 3D Gaussian Splatting. See [LICENSE.md](LICENSE.md) and the license files included with the third-party components for details.
