# Design review after seven completed v1 scenes

2026-10-05. This review uses the seven-scene `comparison.json` created at
21:02:30 UTC and the completed all-view ablations through Parterre. RoadBlock,
RotaryKiln and Truck were not included in that report. The running v1 suite
must finish unchanged. Structural-v2 remains prepared and **unselected**; this
review applies no patch, starts no run and makes no acceptance claim.

**Conditional recommendation:** if the complete v1 result fails, the staged
uniform `ir_kernel_ssim_weight=0.01` candidate remains a justified next
experiment because it directly tests the observed fitting-objective tradeoff.
Seven scenes strengthen the case for addressing structure and weaken any
claim that this modest change is already an adequate solution for test PSNR.
Do not add angular capacity, wider supports or scene-specific choices to this
same experiment.

## What the seven-scene evidence establishes

All values below use official quantized PNGs. “Total” compares the candidate
with the fixed historical baseline. “Kernel” toggles only the IR kernel on
the same completed EMA export; “base” is that export with the kernel off.

| Scene | Official IR PSNR | Total ΔPSNR | Base ΔPSNR | Kernel ΔPSNR | Kernel views losing PSNR | Total IR ΔSSIM |
|---|---:|---:|---:|---:|---:|---:|
| Building | 28.375807 | +0.391731 | +0.089956 | +0.301775 | 5/35 | -0.010200 |
| DailyStuff | 21.758146 | -0.687601 | -0.699928 | +0.012327 | 2/10 | -0.021500 |
| Dimsum | 27.144156 | -0.000914 | +0.118319 | -0.119232 | 16/20 | -0.012480 |
| Ebike | 23.997641 | -0.073977 | +0.046446 | -0.120422 | 5/6 | -0.007737 |
| IronIngot | 30.131325 | -0.274004 | -0.090246 | -0.183758 | 5/8 | -0.010099 |
| LandScape | 22.590431 | +0.723454 | +0.525501 | +0.197952 | 3/13 | +0.004605 |
| Parterre | 27.148643 | +0.620520 | +0.757683 | -0.137163 | 6/9 | -0.006653 |
| Equal-scene mean | 25.878021 | +0.099887 | +0.106819 | **-0.006932** | 42/101 pooled | -0.009152 |

The total improvement does not establish a benefit from the new representation:
its isolated mean contribution is slightly negative, with four of seven scene
means regressing. Parterre illustrates this directly: a +0.620520 dB total
gain coexists with a -0.137163 dB kernel contribution. The causes of base-model
differences remain unresolved; do not assign them to SSIM, kernel gradient
leakage or CUDA nondeterminism without further evidence.

The seven-scene official IR LPIPS mean also worsens by +0.007151 (five scene
means worsen). IR SSIM worsens in six scene means. LandScape is the apparent
SSIM exception against the historical baseline, but its same-model diagnostic
shows **kernel SSIM losses in all 13 views**, mean -0.009753. Its base SSIM
gain +0.014357 masks the kernel penalty. Existing direct on/off SSIM evidence
therefore covers five scenes: the first four plus LandScape, with 82/84 test
views losing SSIM. Do not present the official IronIngot/Parterre SSIM losses
as measurements of their isolated kernel SSIM effects; those ablation reports
currently supply PSNR, not SSIM.

All seven ablations reproduced the official on PNGs exactly and preserved RGB
and base-IR tensors bitwise. This supports attribution of their on/off effects
to the kernel. It does not prove whole-run RGB parity with the baseline. The
seven-scene RGB changes are PSNR -0.013571 dB, SSIM +0.000508 and LPIPS
+0.000661; five scene PSNR means decline. Full-ten RGB assessment remains open.
The partial IR mean is not the required ten-scene mean and cannot decide the
26.4 dB target before the remaining results exist.

## Why 0.01 SSIM is sensible, and why its adequacy is unknown

The existing all-training-view evidence is still limited to Building,
DailyStuff, Dimsum and Ebike. Their kernel train gains are
+0.489233/+0.742699/+0.242882/+0.587097 dB; test gains are
+0.301775/+0.012327/-0.119232/-0.120422 dB. SSIM declines on 464/482 training
views and 69/71 test views. A defect on fitted images gives a direct reason
to change the loss; it is not solely an unseen-camera phenomenon. Stronger
train than test gains suggest examining transfer, but different camera
distributions prevent identifying overfitting as the sole cause. Train gains
for IronIngot, LandScape and Parterre have not been measured here.

The unchanged proposed loss is:

```text
P = stopgrad(base_IR) + rendered_kernel
L = MSE(P, GT) + 0.01 * (1 - SSIM(P[None], GT[None]))
    + 1e-4 * (mean(tanh(a)^2) + 0.01 * mean(tanh(d,w,o)^2))
```

This is a structural training objective, not an explicit multi-view
consistency constraint. It can penalize local irregularity in training images
and change learned kernel amplitudes, locations and widths. It does not
directly constrain consistency between cameras, fix a base geometry error,
or resolve the difference between arithmetic MSE optimization and official
mean-per-view logarithmic PSNR. Dimsum already demonstrates that mean test MSE
can improve while mean PSNR declines.

The coefficient has a scale rationale, not a measured success guarantee:
0.01 times the four observed training SSIM penalties is approximately
0.000099/0.000133/0.000117/0.000052, versus PNG MSE reductions
0.000147/0.000178/0.000209/0.000188. Thus it is not negligible in this
aggregate calculation. Exported, quantized metrics are not raw loss gradients;
no coefficient sweep or training trial establishes that 0.01 is sufficient.

LandScape is useful counterevidence to any promised free improvement. The
kernel improves PSNR in 10/13 views while harming SSIM in all 13. Its strongest
GT-edge quartile MSE improves about 4.85%, while the middle two quartiles
slightly worsen. SSIM may suppress useful corrections along with irregular
ones. IronIngot and Parterre add genuine test-PSNR failures across multiple
views; SSIM has not been shown to cure those failures. A plausible outcome
is better structure with unchanged or worse PSNR, which must be reported as
that tradeoff rather than success toward the PSNR goal.

## Why not bundle more capacity or support changes

All seven exports contain finite learned values in every kernel column; mean
absolute bounded radiance is about 0.009–0.016 across scenes. Earlier four-scene
distribution checks found little amplitude/upper-width saturation, although
anchor statistics do not establish which parameters dominate visible pixels.
There is no measured directional residual pattern demonstrating that a new
angular term is the missing capacity. The base thermal representation already
has degree-3 SH. Added angular parameters could help a real view-dependent
effect, but they also add flexibility under already weak transfer evidence.

Broader supports remain a separate hypothesis. Increasing an upper bound does
not enforce smoothness; a larger minimum width or stronger geometry prior
would regularize more directly, with risks to boundaries and residual
visibility. Building and LandScape both show useful high-edge MSE corrections,
so blanket smoothing has a real potential cost. None of these alternatives
currently has stronger direct evidence than testing the structural objective
alone. Defer them rather than interpreting this as proof they cannot help.

## Decision evidence for a later uniform trial

The staged patch changes only the optional loss/arguments/runner. It retains
ten parameters, zero initialization, all support bounds, LR 0.003, start after
18,000, EMA and exactly 30,000 fresh dataset updates on GPU 0. Default weight
zero has passed CPU tests for unchanged actual loss-block graph, values,
gradients and RNG behavior; positive-weight isolation also passed with a
synthetic render residual. This validates preparation, not CUDA behavior or
image quality. Source and active snapshot remain untouched.

After complete v1, record its full-ten official and same-model reference values
before selecting a candidate. If structural-v2 is selected, use one fixed
0.01 value and training protocol across scenes. The previously specified
four-scene pilot can test the hypothesis, but it cannot establish acceptance
or justify excluding IronIngot/Parterre or other weak scenes from a uniform
follow-up. No per-scene checkpoint, parameter value or method selection.

Assess three distinct outcomes: (1) reduced train/test on/off SSIM penalties
and fewer irregular structures in the same views; (2) improved equal-scene
test kernel PSNR, reporting every scene/view and checking the already useful
Building/LandScape corrections; (3) the complete official result against the
fixed baseline, with the required RGB judgment. A smaller train/test gap caused
only by lost training gain is not evidence of improved transfer. Improved
SSIM alone supports the structural hypothesis, not the target-metric one.
Because every candidate is freshly trained, base-off outputs must be measured
again rather than assumed equal across runs.

Test metrics have informed this design review; they must not enter fitting or
per-scene choices. Final acceptance is still the uniform ten-scene IR target
and RGB assessment, with no claim that this modest candidate will achieve it.

## Inputs and scope

Primary inputs under `output/ir_kernel_v1_20261005_run2`:

- `comparison.json`, SHA256
  `79ae31c8c8de0a5950600b05f036cccd21b51917c4843f04e9bedf2bc09a9a66`.
- All seven `diagnostics/kernel_ablation/SCENE/report.json` files; summary SHA256
  `d86213835e9cf7c6fb9337df1ed41c47c6db53920bdb0042a9407d1723a32319`.
- Four `diagnostics/kernel_generalization/SCENE/report.json` files; summary SHA256
  `b4fad82dbda107b6e8f32fc5d694135617a6bb342886bba715a8749bc0cb5de7`.
- Building/LandScape kernel-quality reports and existing scene visual reviews.
  LandScape quality report SHA256
  `b91707ebb531db0c4a4186d063fa6ffe6ec9f034441412b273484e8e2a98124d`.
  Its limitations text contains a stale “Building only” label; scene metadata,
  13-view inputs and calculations identify LandScape. No source report edited.

Also reviewed `design_review_after_generalization.md` and the staged patch,
manifest and CPU check report. This synthesis reads existing evidence and
recomputes simple aggregates; it does not conduct a new visual/GPU experiment,
modify official artifacts, or change the running method.

Root follow-up: corrected the LandScape report limitations scene label after this review; the JSON records a documentation-only amendment. Metrics were not recomputed or changed. Current corrected report SHA256: c53cd1e315a7a11125306df52e86d6fb37b3fde4467df648ce6c12113323c7f3. The hash above identifies the version read during review. The generator now uses the actual scene name.
