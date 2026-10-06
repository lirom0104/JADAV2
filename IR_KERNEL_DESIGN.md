# Infrared local Gaussian kernels — candidate v1

## Starting point and protocol

The initial working tree, including existing uncommitted ODB changes, is archived
in `ir_experiments/initial_state/source.tar.gz`; `tracked.diff`, `status.txt`,
`head.txt`, and SHA-256 manifest record the exact starting state. No historical
batch script is executed. The fixed baseline is
`/home/lf/code/Our_Project-New-2/output/odb_30k_20261003_101217_1051565`.

All ten scenes use the baseline's explicit dataset train/test folders, original
640×480 sensor images, default seed 0 through the unchanged `safe_state`, and
physical GPU 0 only. Final candidates start from dataset initialization and run
30,000 iterations. Evaluation uses `render.py --iteration 30000 --skip_train`
and unchanged `metrics.py`. Debug experiments and method selection may inspect
test metrics; training fits only training cameras.

## Representation

Each existing Gaussian owns ten additional raw infrared parameters
`_ir_kernel[i] = (a[3], d[3], w[3], o[1])`. The thermal images are pseudocolor RGB;
three amplitude channels preserve that representation without assuming a
universal scalar-to-color palette.

Given the parent's infrared center μ, scales s, rotation R and opacity α, form:

```
signed radiance = 0.2 tanh(a)
kernel center = stopgrad(μ) + stopgrad(R) [0.75 tanh(d) ⊙ stopgrad(s)]
kernel scales = stopgrad(s) ⊙ 0.7 exp(log(2) tanh(w))
kernel opacity = min(0.99, stopgrad(α) × 2 sigmoid(o))
```

All raw parameters initialize to zero without consuming random numbers. Thus
the initial added image is exactly zero; widths begin at 0.7 times the parent
width, locations coincide, and opacities equal the parent's except for the
0.99 rasterization cap. Widths remain within [0.35, 1.4] times parent widths;
bounded local offsets and amplitudes prevent runaway geometry or radiance.

The new kernels are projected, depth sorted, and alpha composited by the real
Gaussian rasterizer against a zero background, then warped through the same
camera calibration as the base image. Their signed infrared radiance is added
after base multimodal refinement. This is a three-dimensional Gaussian
representation, not an image filter. Position and width change its projected
spatial support; opacity changes visibility within the residual field. Shared
base geometry is detached for this pass. No added RGB image is generated.

Thermal smooth regions can receive broad, low-amplitude corrections while
narrow, displaced kernels represent thermal boundaries and hot/cold regions.
This is a design hypothesis; only complete experiments establish its benefit.

## Training

Base optimization and RGB ODB retain the baseline configuration. Kernels become
active after iteration 18,000, after base densification and late pruning.
During each remaining iteration, the new parameters minimize training-camera
MSE between `stopgrad(base_IR) + rendered_kernel` and target IR. Adam learning
rate is 0.003. The regularizer is
`1e-4 * (mean(tanh(a)^2) + 0.01 * mean(tanh(d,w,o)^2))`.

Existing reconstruction, adaptive branch weights, CMO statistics and smoothing
continue to use base IR. This isolates the new fit from shared RGB/geometry
optimization. The new parameters also participate in existing EMA export.

The pre-existing training loop skipped its last optimizer update and exported
before that update. Candidate training updates on every numbered iteration and
saves afterward, including the updated EMA. A machine-readable
`training_receipt.json` records initialization, actual updates, final iteration,
GPU visibility, elapsed time, peak allocation and parameter count.

## Persistence and lifecycle

Checkpoint version 8 stores raw kernels and configuration, with name-based
optimizer migration for older layouts. PLY exports contain all ten raw channels
and metadata for enable/amplitude; feature-module exports also preserve config.
Clone copies parent kernels. Split copies their raw local-coordinate parameters
onto the child Gaussians. Prune applies the same anchor mask and Adam-state mask.
Older checkpoints/PLYs initialize missing kernels to zero.

Checkpoint tests verify parameter and optimizer-state restoration. The inherited
checkpoint format does not save the camera sampling queue, RNG states, or export
EMA history, so an interrupted/resumed trajectory is not guaranteed to match an
uninterrupted one. The current formal suite starts each scene fresh and runs it
uninterrupted; it does not use checkpoint continuation.

## Cost and verification

There are 10N added floats: 40N bytes of raw model storage and approximately
160N bytes including gradients and two Adam moments, plus 40N bytes for EMA.
The active renderer adds one rasterizer pass per view. Its sorting buffers and
autograd intermediates depend on projected Gaussian coverage; measured runtime
and peak allocation must accompany final results.

The 80-step Ebike engineering smoke test verified nonzero updates and Adam
second moments in all ten raw columns through the real renderer, export/reload,
and bitwise-identical RGB outputs with kernels enabled/disabled. On its 5,032
Gaussians, 20-view-repeat inference measurements were 9.40 ms without the kernel
and 11.21 ms with it (about 19%); both peaks were 534,202,880 allocated bytes.
This small-model measurement is not a full-scene training-cost estimate.
The complete smoke run recorded 1,014,811,648 peak allocated training bytes.
Detailed evidence is in `ir_experiments/smoke_v1/verification.json`.

Verification must exercise true rasterizer gradients (including signed
amplitudes, location, width and opacity), zero initialization, RGB isolation,
optimizer updates, checkpoint/PLY round trips, clone/split/prune and retained
optimizer moments. Training/rendering/evaluation smoke checks precede full
30,000-iteration runs. Complete ten-scene results, same-camera images and
equal-weight averages are required before any acceptance claim.

## Evaluation audit note

The inherited training report passes a CHW tensor to `psnr`, which interprets
the first dimension as a batch and averages per-channel dB. The required
`metrics.py` passes NCHW and computes one all-channel MSE per view. Thus
training-log PSNR must not be used for acceptance. On the smoke export, the
correct IR mean is15.580463dB while the per-channel mean is18.566997dB,
explaining the apparent discrepancy without implying an export failure.
`ir_experiments/smoke_v1/metric_shape_diagnostic.json` preserves the CPU
recalculation. The official baseline/candidate metric code remains unchanged.

## Early full-budget evidence, not final acceptance

The uniform v1 suite is recorded at
`output/ir_kernel_v1_20261005_run2`. Its first two validated scenes show:

| Scene | Fixed baseline IR PSNR | Kernel disabled in candidate | Official candidate IR PSNR |
|---|---:|---:|---:|
| Building | 27.984076 | 28.074032 | 28.375807 |
| DailyStuff | 22.445747 | 21.745819 | 21.758146 |

The disabled-kernel images are auxiliary ablations of the same completed export,
not alternative candidates. All official on-kernel PNGs reproduce exactly;
RGB and base IR tensors remain bitwise identical under the toggle. The kernel
adds +0.301775 dB on Building and +0.012327 dB on DailyStuff. The latter's
decline against the fixed baseline is primarily present in its base render.
This does not establish the cause of that base-model difference.

Building also exposes a structural tradeoff: kernel-on IR SSIM is 0.899048,
versus 0.909576 with the kernel disabled. PSNR improves on 30 of 35 views while
SSIM declines on 33 of 35. CPU recomputation of the unchanged official functions
reproduces these metrics; edge-quartile MSE decreases throughout the image, but
visual review shows added local irregularity in some regions. The diagnostic
report and crop-selection coordinates are under
`diagnostics/kernel_quality_cpu/Building` in the run directory.

Full-model inference measurements so far were taken while another scene of the
suite trained on GPU 0. They are explicitly marked as concurrent measurements
and must not be presented as isolated performance estimates. An isolated
measurement after the suite remains pending, as does full-ten acceptance.
