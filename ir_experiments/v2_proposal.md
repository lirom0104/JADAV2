# Proposed uniform v2: bounded directional IR kernels with a structural training term

Status: proposal only, prepared on 2026-10-05 from the first four completed v1
scenes. The frozen v1 ten-scene suite must finish unchanged. No core code,
configuration, model, official result, or GPU job was changed for this proposal.
The proposal does not establish a quality gain or satisfy ten-scene acceptance.

Update after all-training-view diagnostics: the preferred next experiment is
now the existing ten-parameter representation with a modest structural fitting
term, deferring angular capacity. See `design_review_after_generalization.md`.
The 19-parameter design below remains a documented alternative, not the selected
next candidate. Any new run still depends on complete v1 evaluation.

## Evidence and its limits

The current comparison is `output/ir_kernel_v1_20261005_run2/comparison.json`
(report timestamp 2026-10-05 19:07:10 UTC). These are full 30,000-step scenes but
only four of the ten required scenes:

| Scene | Baseline IR PSNR | v1 IR PSNR | Change | v1 IR SSIM change | v1 IR LPIPS change |
|---|---:|---:|---:|---:|---:|
| Building | 27.984076 | 28.375807 | +0.391731 | -0.010200 | +0.007594 |
| DailyStuff | 22.445747 | 21.758146 | -0.687601 | -0.021500 | +0.024524 |
| Dimsum | 27.145069 | 27.144156 | -0.000914 | -0.012480 | +0.021138 |
| Ebike | 24.071617 | 23.997641 | -0.073977 | -0.007737 | +0.006837 |
| Four-scene mean, not acceptance | 25.411627 | 25.318937 | -0.092690 | -0.012979 | +0.015023 |

The four-scene RGB changes average +0.049623 dB PSNR, +0.000718 SSIM, and
+0.000388 LPIPS. This is incomplete evidence for the final RGB judgment.

The stronger attribution evidence is the same-export on/off diagnostic:

- Building: kernel adds **+0.301775 dB PSNR**, while SSIM changes
  **-0.010528**. PSNR improves on 30/35 views; SSIM declines on 33/35.
- DailyStuff: kernel adds **+0.012327 dB**. Its large baseline-relative deficit
  is already in the kernel-off base image. View 00007 accounts for 92.57% of
  the mean base deficit; its residual alone gains +0.036343 dB.
- Subsequent all-view ablations (2026-10-05 19:15 UTC) found kernel gains of
  **-0.119232 dB on Dimsum** (off 27.263388, on 27.144156) and
  **-0.120422 dB on Ebike** (off 24.118063, on 23.997641). Official on-kernel
  PNGs reproduced exactly, and RGB/base IR tensors were bitwise invariant.
  Thus the residual itself can harm test PSNR, even when the base model exceeds
  the fixed baseline. This strengthens the need to test generalization and
  regularization; it does not prove that added angular capacity will help.

The Building CPU image diagnostic shows MSE improvement in all four GT
edge-strength quartiles. Native-resolution inspection nonetheless finds added
small streaks/local irregularity in some sign and building areas. This supports
a kernel-specific structural cost; it does **not** show that broad smoothing or
removing all high-frequency correction would improve the result.

Read-only NumPy/PlyData inspection of all exported anchors in the four scenes
adds the following evidence. Ratios are relative to the detached parent IR
Gaussian scale. These are anchor distributions, not visibility-weighted pixel
statistics; many low-opacity anchors carry nearly zero correction.

| Scene | Gaussians | Mean absolute radiance, RGB channels | Width ratio p10 / p50 / p90 | Width axes >1.3 | Radiance channels with abs >0.18 |
|---|---:|---|---|---:|---:|
| Building | 380846 | .00790 / .01087 / .01079 | .534 / .700 / .881 | 0.712% | 0.072% |
| DailyStuff | 250763 | .01693 / .01135 / .01486 | .502 / .700 / .986 | 2.373% | 0.288% |
| Dimsum | 266137 | .01431 / .00802 / .01336 | .514 / .700 / .995 | 2.168% | 0.204% |
| Ebike | 130467 | .00451 / .01677 / .02543 | .453 / .700 / 1.108 | 3.935% | 0.395% |

All exported raw parameters are finite. The existing width bound is [0.35,1.4]
and radiance bound is [-0.2,0.2]. These values do not indicate a broadly
saturated amplitude or upper-width limit. DailyStuff's parameters are active:
its opacity-weighted mean absolute amplitude is .02646, versus .02055 for
Building, yet its test gain is much smaller. Weak gain cannot be explained by
an inactive kernel alone. Offset absolute-component p90 values are .513, .637,
.618, and .687 parent-scale units, respectively, against the .75 bound. Some
anchors use substantial displacement; increasing that freedom is not the first
choice for the observed local irregularity.

The DailyStuff source/trajectory audit found divergence by the first 10-step
log and different final Gaussian counts before kernel activation. It did not
identify the numerical cause. This proposal cannot promise to repair that
base-geometry or visibility failure. Kernel-off versus fixed baseline remains
an imperfect counterfactual because their complete training trajectories differ.

## Additional train/test evidence (2026-10-05 19:38 UTC)

Every training camera of the completed four exports was evaluated without an
optimizer, parameter update, or additional training iteration. The report is
`output/ir_kernel_v1_20261005_run2/diagnostics/kernel_generalization/summary.md`.

| Scene | Training views | Train kernel PSNR gain | Test kernel PSNR gain | Train SSIM change | Test SSIM change |
|---|---:|---:|---:|---:|---:|
| Building | 238 | +0.489233 | +0.301775 | -0.009875 | -0.010528 |
| DailyStuff | 68 | +0.742699 | +0.012327 | -0.013343 | -0.015079 |
| Dimsum | 134 | +0.242882 | -0.119232 | -0.011679 | -0.012865 |
| Ebike | 42 | +0.587097 | -0.120422 | -0.005229 | -0.008212 |

All four scenes improve training PSNR while their test gains are smaller;
Dimsum and Ebike reverse sign. This weakens a capacity-only explanation and
makes regularization/generalization the priority. It does not prove a specific
cause because train/test camera distributions differ. The directional proposal
below remains an untested hypothesis and is not yet the selected next run.
SSIM declines in both splits, supporting a structural training term directly.

Dimsum test mean MSE decreases although mean per-view PSNR decreases too. The
required metric averages logarithmic scores per view; lower arithmetic mean
MSE is not sufficient. A uniformly defined training-view relative-error or
log-MSE objective is another possible paired strategy for these learned kernels,
but must be evaluated from scratch and must not replace official evaluation.
No such objective or directional parameter has been implemented in v1.

## Evaluate the three directions

| Direction | What it could address | Evidence strength and cost | Decision for this candidate |
|---|---|---|---|
| Low-order, bounded view-dependent residual radiance | A static per-anchor correction must compromise when the required correction varies with viewing direction; a small angular term can fit that variation without moving shared geometry. | The current residual is view-independent, although the base thermal SH is already degree 3. No current diagnostic proves angular underfitting or identifies physical reflection as the cause. Nine extra floats per anchor; no extra rasterizer pass. | Include one first-order term, strongly bounded and regularized. This tests a specific representation hypothesis. |
| Wider or spatially coordinated support | Broad corrections or nearby-anchor agreement could reduce incoherent narrow residuals. | Most widths are not at the upper bound. Uniform broadening can cross heat boundaries, change residual occlusion, and add rasterization coverage; neighbor coupling needs a defensible surface-aware neighborhood. The present diagnostics do not establish the right neighborhood or width increase. | Keep v1 support bounds/initialization in v2. Do not combine an unvalidated width expansion or neighbor graph with the angular test. |
| Training-image structure term applied only to the kernel prediction | Directly penalizes the measured local-structure degradation while leaving the detached base objective untouched. | Building on/off attribution directly supports this need. SSIM can compete with the primary PSNR target, so use a modest, fixed coefficient. No inference operation or parameter fitting on test images. | Include weight 0.01 on the unchanged repository SSIM loss. |

The proposed angular radiance is a model for view-correlated thermal-image
appearance, not a calibrated emission/reflection decomposition. The observations
are pseudocolor RGB, and camera response, palette changes, base SH error or
occlusion may also produce view-correlated residuals. No thermodynamic claim is
required by the proposed parameterization.

## Exact representation

Use **19 raw IR-only floats per existing Gaussian**. Preserve the v1 ten-column
layout `(a[3], d[3], w[3], o[1])` and append `B[3,3]` in row-major order. `a`
controls static radiance; each row of `B` controls one pseudocolor channel's
first-order angular variation. `d`, `w`, and `o` retain local offset, width and
opacity meanings. Initialize all 19 columns to zero, without a random draw.

For parent IR center mu, rotation R, scale s, opacity alpha, and the renderer's
camera center c, define:

```text
q = stopgrad(R^T ((c - mu) / max(||c - mu||, 1e-8)))
angular = (0.5 / sqrt(3)) * tanh(B) @ q
radiance(q) = 0.2 * tanh(a + angular)

kernel_center = stopgrad(mu) + stopgrad(R) [0.75 tanh(d) * stopgrad(s)]
kernel_scale  = stopgrad(s) * 0.7 exp(log(2) tanh(w))
kernel_alpha  = min(0.99, stopgrad(alpha) * 2 sigmoid(o))
```

`q` uses the **parent** IR center, not the learned displaced kernel center, so
the angular appearance loss cannot improve itself by changing the view vector
through `d`. The epsilon only protects a degenerate camera/anchor coincidence.
All camera/parent inputs are detached. Each angular latent component is bounded
in absolute value by 0.5; the final radiance stays in [-0.2,0.2], independent of
the viewing direction. At `B=0` this is exactly the v1 radiance formula. At
initialization the entire added image is exactly zero.

Rasterize these precomputed signed colors in the existing single IR residual
pass, with the existing zero background, covariance path, depth sorting and
sensor warp. Add the result after base multimodal refinement. No image filter,
extra inference correction, RGB residual, shared-geometry gradient, or new
angular geometry parameter is introduced.

## Exact training choices

Use one configuration for every scene; no scene-specific switches or values:

| Setting | Proposed value |
|---|---|
| Initialization / budget | Dataset initialization, exactly 30000 numbered optimizer updates |
| Seed / GPU | Unchanged default `safe_state` seed; physical GPU 0 only |
| Kernel activation | Iteration 18001 through 30000, as in v1 |
| Kernel Adam LR | 0.003 for all 19 columns; existing Adam epsilon and other options unchanged |
| Static amplitude / angular latent bound | 0.2 / 0.5 |
| Width / offset / opacity formulas | Unchanged from v1 |
| Kernel MSE coefficient | 1.0 |
| Kernel SSIM coefficient | **0.01**, repository `ssim` on NCHW training images |
| Parameter regularizer coefficient | 0.0001 |
| Directional regularizer relative weight | 0.25; group-normalized mean, as defined below |
| Base objective, adaptive weighting, CMO, RGB ODB | Unchanged; continue to use base IR |
| Export | Existing EMA, starting at 20000 with decay 0.995, including all 19 IR columns |
| Final export/update ordering | Keep v1's update on iteration 30000 and save after that update |
| In-training evaluation | Keep v1's explicit `[30000]` schedule for this candidate |

For each sampled **training** camera:

```text
prediction = stopgrad(base_IR) + rendered_kernel
L_kernel = mean((prediction - gt_IR)^2)
           + 0.01 * (1 - SSIM(prediction[None], gt_IR[None]))
           + 1e-4 * R
R = mean(tanh(a)^2)
    + 0.25 * mean(tanh(B)^2)
    + 0.01 * mean(tanh(concat(d, w, o))^2)
```

Do not clamp `prediction` for the fitting loss; use the same raw image convention
as the current training objectives. Official rendering continues to quantize
with `torchvision.save_image`, and official metrics stay unchanged. The new
SSIM and regularization gradients must reach only IR parameters. No new
loss-based camera sampling, adaptive test weighting or test-image fitting is
introduced.

The 0.01 SSIM coefficient is an explicit method-selection choice informed by
the allowed test diagnostics, not a learned value. Building's measured MSE gain
is approximately 0.00014276, while its SSIM penalty increases by 0.01052797.
Multiplying the latter by 0.01 gives approximately 0.00010528: enough to make
that structural cost relevant while preserving a net objective improvement for
the aggregate v1 correction in this illustrative calculation. This scale check
is not a prediction about training or ten-scene outcomes. Training pixels alone
fit model parameters; test diagnostics participating in design selection must
be disclosed in the eventual report.

## Implementation, cost, and verification plan

This document does not implement the design. If selected after v1 completes,
add a representation version/config flag, e.g. `ir_kernel_mode=directional_v2`,
`ir_kernel_directional_scale=0.5`, and `ir_kernel_ssim_weight=0.01`. Freeze the
complete source/configuration/runtime packages before new runs. Record a new
checkpoint format version and explicit column count. A v1 ten-column state
loads by appending nine zeros, preserving its forward behavior; final v2
candidates must nevertheless start fresh from the dataset.

Clone copies all 19 raw values. Split copies raw local-coordinate values to the
child, using its inherited rotation as with v1. Prune and Adam moments use the
same anchor mask. Export all 19 raw values and directional configuration in
PLY/checkpoint/module metadata, and include all columns in EMA. No parameters
may be dropped silently when loading older exports.

The extra nine floats require 36N bytes of raw storage and approximately 180N
bytes including gradients, Adam's two moments and EMA, above v1. At Building's
380846 anchors that is about **13.71 MB raw / 68.55 MB total extra persistent
training storage** (decimal MB). Temporary angular tensors add further memory.
The design retains one extra IR rasterizer pass, with the same geometry/support;
angular color computation and training SSIM add work. Measure the actual cost;
the concurrent v1 timings are not isolated performance estimates.

Required checks before formal v2 runs:

1. Zero initialization and `B=0` reproduce v1 residuals exactly; deterministic
   parameter construction consumes no RNG draws.
2. Two distinct synthetic camera directions produce different residual colors
   when B is nonzero; colors remain finite/bounded. Real rasterizer gradients
   reach the nine angular columns, amplitude, position, width and opacity.
3. Kernel loss including SSIM leaves every shared/RGB parameter gradient absent
   or unchanged; toggling the kernel leaves RGB and base-IR tensors bitwise equal.
4. Checkpoint, PLY, module metadata and EMA round trips preserve configuration
   and render outputs; clone/split/prune preserve all 19 columns and Adam state.
5. A short GPU-0 smoke run verifies fit/update/export/render/official PNG metric
   flow. It is engineering evidence only and does not replace final 30000-step runs.

Do not use completed v1 models for extra fitting iterations. Once v1 finishes,
use fresh **Building, DailyStuff and Ebike** runs as one fixed three-scene pilot:
Building tests the demonstrated structural cost, DailyStuff tests weak residual
gain, and Ebike tests a smaller camera set with stronger geometric-parameter use.
Report all three regardless of result. This is a diagnosis set, not an acceptance
set or a basis for choosing scene-specific versions. If the design shows useful
effects, expand it through the remaining scenes under the same frozen 30000-step
configuration, rather than abandoning it after one weak additional scene.
Acceptance still requires the complete uniform ten-scene comparison. If the
pilot lacks useful effects, retain it and use those observations to design the
next experiment; do not assemble a per-scene winner mixture.

On completed exports, report official baseline-relative results and same-model
kernel on/off deltas separately. Also evaluate `B=0` at inference without fitting
to identify the learned angular term's direct effect, while stating that this
toggle is not a retrained DC-only counterfactual. Repeat the all-view PSNR/SSIM
diagnostic and native-image review, including the previously poor regions.

## Risks and decision boundary

- Directional capacity can overfit sparse viewing directions or absorb camera
  response rather than scene appearance. Bounds, a first-order basis and its
  separate regularizer limit but do not eliminate this risk.
- Base thermal appearance already uses degree-3 SH. The new angular residual
  may be redundant; its distinct support and detached late MSE fit are the
  reasons to test it, not evidence that it must help.
- SSIM can sacrifice PSNR, and the primary requirement remains the unrounded
  ten-scene mean IR PSNR of at least 26.4 dB. Auxiliary structure metrics do not
  replace this target.
- A displaced narrow residual can still produce irregularities, and local
  support cannot repair missing geometry/visibility or large base errors. The
  DailyStuff outlier is a particular warning against promising such a repair.
- Adding nine columns can change allocations and numerical training trajectories
  even with unchanged shared gradients and seed. RGB must be judged from the
  complete ten-scene metrics and same-view images, not inferred from graph
  isolation alone. Do not attribute changes to CUDA nondeterminism without a
  controlled diagnostic.
- This candidate changes angular capacity and its structural fitting objective
  together. Export toggles attribute direct inference contributions, but a
  causal separation of training effects would require additional fresh, uniform
  runs. The current proposal prioritizes one concrete next candidate.

The recommendation is to test this bounded 19-parameter representation and
0.01 structural term **if another candidate is needed after complete v1
evaluation**. Broadening support or adding spatial coupling should remain a
separate, evidence-driven experiment rather than a simultaneous unmeasured
change. No projected target score is claimed.

Evidence paths:

- `output/ir_kernel_v1_20261005_run2/comparison.json`
- `output/ir_kernel_v1_20261005_run2/diagnostics/kernel_ablation/Building/report.json`
- `output/ir_kernel_v1_20261005_run2/diagnostics/kernel_ablation/DailyStuff/report.json`
- `output/ir_kernel_v1_20261005_run2/diagnostics/kernel_ablation/Dimsum/report.json`
- `output/ir_kernel_v1_20261005_run2/diagnostics/kernel_ablation/Ebike/report.json`
- `output/ir_kernel_v1_20261005_run2/diagnostics/kernel_quality_cpu/Building/report.json`
- `output/ir_kernel_v1_20261005_run2/diagnostics/kernel_quality_cpu/Building/interpretation.md`
- `output/ir_kernel_v1_20261005_run2/diagnostics/base_divergence_DailyStuff/audit.json`
- The four scenes' `point_cloud/iteration_30000/point_cloud.ply` files; the table
  above was computed with CPU NumPy/PlyData using the v1 transformations.
