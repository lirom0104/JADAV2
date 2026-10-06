# Fixed baseline audit

Baseline: `/home/lf/code/Our_Project-New-2/output/odb_30k_20261003_101217_1051565`. Dataset: `/home/lf/data/thermal3dgs/RGBT-Scenes`. Method: `ours_30000`.

All 133 test views, both modalities, were decoded and compared: baseline ground-truth pixels match the current dataset exactly. All are 640 × 480 and camera order matches sorted test filenames.

| Scene | Train | Test | IR PSNR | IR SSIM | IR LPIPS | RGB PSNR | RGB SSIM | RGB LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Building | 238 | 35 | 27.984075546 | 0.909247696 | 0.142259926 | 25.618734360 | 0.875670791 | 0.134226277 |
| DailyStuff | 68 | 10 | 22.445747375 | 0.847374260 | 0.191279367 | 22.198451996 | 0.812224209 | 0.224711463 |
| Dimsum | 134 | 20 | 27.145069122 | 0.899626434 | 0.111378595 | 24.726074219 | 0.865404725 | 0.176873922 |
| Ebike | 42 | 6 | 24.071617126 | 0.882458448 | 0.177777484 | 28.495824814 | 0.929345131 | 0.129520163 |
| IronIngot | 53 | 8 | 30.405328751 | 0.904344916 | 0.076725647 | 26.607627869 | 0.904979825 | 0.150244802 |
| LandScape | 90 | 13 | 21.866977692 | 0.837611139 | 0.278793067 | 21.847471237 | 0.731377006 | 0.261992246 |
| Parterre | 57 | 9 | 26.528123856 | 0.905011833 | 0.163619056 | 27.409404755 | 0.882124543 | 0.152520850 |
| RoadBlock | 62 | 9 | 26.967687607 | 0.917576075 | 0.183885276 | 31.013664246 | 0.928552866 | 0.195418417 |
| RotaryKiln | 92 | 14 | 27.260986328 | 0.933307052 | 0.103582464 | 22.377309799 | 0.784503400 | 0.218777090 |
| Truck | 64 | 9 | 26.549016953 | 0.889462709 | 0.126958504 | 24.440376282 | 0.858255506 | 0.200386703 |
| Equal-weight mean | — | — | 26.122463036 | 0.892602056 | 0.155625939 | 25.473493958 | 0.857243800 | 0.184467193 |

IR mean is **26.122463035583497 dB**. Target 26.4 dB requires **+0.277536964416502 dB**. Use unrounded values in the JSON for acceptance.

## Commands and settings

Baseline flags common to every scene:

```text
--use_detail_basis --detail_basis_mode oriented_hermite --detail_basis_scale 0.08 --detail_basis_lr 0.001 --detail_basis_reg_weight 0.005 --detail_edge_weight 0.05 --detail_basis_start_iter 26000 --iterations 30000 --save_iterations 30000 --checkpoint_iterations 30000
```

Candidate invocation adds new IR design flags to these baseline settings and uses current-workspace train.py, followed by:

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON" render.py -m "$OUTPUT/$SCENE" --iteration 30000 --skip_train
CUDA_VISIBLE_DEVICES=0 "$PYTHON" metrics.py -m "$OUTPUT/$SCENE"
```

Every candidate starts at dataset initialization and finishes exactly 30,000 cumulative iterations. Preserve default seed 0. Historical batch scripts were read only as evidence and must not be executed for this task.

All baseline cfg_args agree except paths: sh_degree=3, resolution=-1, white_background=False, data_device=cuda, use_bgfc=True, use_at_gom=True, use_paired_views=True, use_camera_calibration=True, bgfc_hidden_dim=32, bgfc_gate_init_bias=-2.2, bgfc_thermal_grayscale_context=True, bgfc_rgb_luma_transfer_only=True, use_render_calibration=True, use_color_refinement=True, color_refinement_hidden_dim=16, color_refinement_max_residual=0.06, use_detail_basis=True, detail_basis_mode=oriented_hermite, detail_basis_scale=0.08, detail_basis_thermal_scale=0.06, eval=False.

## Evaluation details

- **split**: Explicit rgb/train + thermal/train and rgb/test + thermal/test directories; eval flag does not alter this reader split.
- **view_order**: COLMAP-matched existing images, sorted by image_name; rendering sets shuffle=False.
- **resolution**: resolution=-1; all current input and baseline output images are 640x480, so no downsampling occurs.
- **camera_calibration**: Keep use_camera_calibration=True: render enlarged undistorted canvas and warp to original distorted sensor grid; ground truth unchanged.
- **quantization**: render.py uses torchvision.utils.save_image; metrics.py reads saved PNG RGB pixels scaled to [0,1].
- **PSNR**: MSE over all three channels and pixels per view; 20*log10(1/sqrt(MSE)); average view PSNRs per scene, then average ten scenes equally.
- **SSIM**: Repository implementation: 11x11 Gaussian window, sigma=1.5, C1=0.01^2, C2=0.03^2, grouped convolutions and zero padding.
- **LPIPS**: Repository lpipsPyTorch using VGG16 and existing normalization convention; preserve code.
- **thermal_target**: Pseudocolor RGB images, not scalar grayscale.
- **random_seed**: utils/general_utils.py safe_state seeds Python, NumPy and torch with 0; preserve unchanged.
- **failure_handling**: metrics.py catches exceptions and can exit 0 without metrics; independently require ours_30000 with finite fields and expected view counts.

## Environment and historical evidence

Baseline logs explicitly use `/home/lf/miniconda3/envs/thermalgaussian/bin/python` (currently Python 3.10.19), torch 2.10.0+cu128, torchvision 0.25.0+cu128, NumPy 2.2.6 and Pillow 12.0.0. JADA currently has Python 3.10.21, the same torch/torchvision, NumPy 2.2.5 and Pillow 12.3.0. Installed backend hashes and source hashes are recorded in JSON. The thermalgaussian rasterizer has since acquired PSTB support; backend compatibility is independently tested in `backend_audit/comparison.json`.

- Baseline logs used thermalgaussian Python; cfg_args model_path retains old Our_Project-New-1-1-1 location.
- Installed thermalgaussian rasterizer has historical optional PSTB support; see backend_audit/comparison.json for measured no-PSTB compatibility.
- Do not use old per-scene RGB thresholds or old acceptance criteria found in historical status files.
- Train-only palette diagnostic rejects universal scalar luminance: IronIngot held-training pseudocolor reconstruction PSNR 26.26001130617582 dB.
- Historical oriented-sensor raw variant improved Building/DailyStuff/RoadBlock/RotaryKiln but hurt Parterre by 2.6639576 dB; no reason to assume it generalizes uniformly.
- Fixed image blur is not universally beneficial: baseline Dimsum test diagnostic fell from 27.1450710 to 27.0358925 dB at blur radius 3.

The JSON preserves all per-image hashes/pixel checks, configs, real historical commands, final progress evidence, source hashes, package versions, and full-precision metrics. Historical training times ranged from roughly 25 to 49 minutes per scene.
