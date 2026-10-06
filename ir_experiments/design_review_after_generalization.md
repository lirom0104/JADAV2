# Design review after v1 train/test diagnostics

2026-10-05. Proposal only: v1 must finish unchanged. This review supersedes my
earlier recommendation to add angular coefficients in the immediate next
candidate. It does not select or implement a run, claim acceptance, or modify
core/frozen code. No GPU was used.

**Recommendation: first test the existing 10-parameter IR representation with
a 0.01 SSIM fitting term, retaining its current support and all other settings.**
Defer increased angular capacity and support changes to separate experiments.

## Evidence that changes the recommendation

The authoritative inputs are all four `kernel_generalization/*/report.json`
files and their summary under `output/ir_kernel_v1_20261005_run2/diagnostics`.

| Scene | Train kernel PSNR gain | Test gain | Train views losing SSIM | Test views losing SSIM |
|---|---:|---:|---:|---:|
| Building | +0.489233 | +0.301775 | 227/238 | 33/35 |
| DailyStuff | +0.742699 | +0.012327 | 66/68 | 10/10 |
| Dimsum | +0.242882 | -0.119232 | 131/134 | 20/20 |
| Ebike | +0.587097 | -0.120422 | 40/42 | 6/6 |

The equal-scene mean kernel gain is about +0.515478 dB on training views and
+0.018612 dB on test views. The loss of structure also exists on fitted views:
464/482 training views and 69/71 test views lose SSIM. Thus the strongest
diagnosis is a fitting-objective tradeoff plus weak transfer to test cameras,
not demonstrated lack of capacity. Different camera distributions prevent
calling the gain gap proof of overfitting or of a particular physical cause.

This is not only a single-outlier test effect. Dimsum loses PSNR on 16/20 test
views, with median change -0.079803 dB; Ebike loses on 5/6, median -0.086884.
DailyStuff's training median gain is +0.513046 versus +0.033043 on test views.
Dimsum's mean test MSE improves despite mean PSNR declining: larger absolute
improvements on a few high-error views can outweigh losses elsewhere in MSE.
The official mean-per-view PSNR remains the decision metric.

The previously inspected exports show little amplitude/upper-width saturation:
only 0.072–0.395% of radiance channels exceed absolute 0.18 (bound 0.2), and
0.712–3.935% of width axes exceed 1.3 (bound 1.4). Width medians are 0.7 and
opacity-weighted means approximately 0.69–0.74. These are anchor statistics,
not visibility-weighted pixel contributions; they do not prove support is
optimal. Large offset p90 values (0.51–0.69 parent-scale units, bound 0.75)
also show substantial existing local geometric flexibility.

## Compare the choices

| Choice | Assessment under the new evidence |
|---|---|
| Modest SSIM term on current kernel fitting | Directly addresses a widespread defect on both train and test images. Changes optimization of genuine IR Gaussian parameters while preserving representation, inference, memory and RGB isolation. Best immediate test. SSIM is an auxiliary objective, not a guarantee of PSNR or generalization gain. |
| Nine new bounded angular coefficients | Could represent smooth view-correlated residuals, but there is no measured directional error pattern showing this is the limiting factor. The base thermal branch already has degree-3 SH; strong train gains weaken a capacity-only explanation. Additional coefficients could worsen transfer, although a correct directional model might improve it. Defer pending directional evidence. |
| Broader/smoother supports | Plausible spatial regularization, but increasing the upper-width bound adds flexibility; it does not enforce smoothness. A larger minimum width or stronger offset/width prior would constrain the representation more directly, while risking boundary blur and changed residual visibility. Building improves MSE even at strong edges, so blanket smoothing is not supported. Defer as a separate support/prior experiment if structure loss persists. |

## Exact immediate candidate and falsifiable expectations

Keep the v1 ten raw columns, zero initialization, amplitude 0.2, offsets bounded
by 0.75 parent-scale units, width formula `0.7*exp(log(2)*tanh(w))`, and opacity
formula. For each training camera use:

```text
P = stopgrad(base_IR) + rendered_kernel
L_kernel = MSE(P, GT) + 0.01*(1 - SSIM(P[None], GT[None]))
           + 1e-4*(mean(tanh(a)^2) + 0.01*mean(tanh(d,w,o)^2))
```

Use the unchanged repository SSIM, raw unclipped prediction, and detached base.
Keep kernel LR 0.003, start after 18000, existing EMA, all base/RGB objectives,
default seed, GPU 0, and fresh dataset initialization with exactly 30000 updates.
No new loss-weight schedule, scene-specific value, support change or angular
parameter is bundled into this candidate.

The coefficient is grounded in a train-view scale check: 0.01 times the measured
SSIM penalty is 0.000099/0.000133/0.000117/0.000052 across the four scenes,
against respective PNG MSE reductions 0.000147/0.000178/0.000209/0.000188.
It makes the observed structural cost relevant without making the whole v1
correction unfavorable in this approximate aggregate calculation. PNG export
statistics are not raw training gradients, so this is a scale rationale only.

If another candidate is needed after complete v1, use all four completed
diagnostic scenes as one uniform fresh pilot. Record these falsifiable
hypotheses before running, without treating them as user acceptance thresholds:

1. Equal-scene kernel-on/off SSIM penalties shrink on both train and test sets;
   previously observed streaks diminish in the same-view image checks.
2. Equal-scene test kernel PSNR gain improves above v1's +0.018612 dB. Report
   every scene and per-view distribution; improvements confined to training
   PSNR or arithmetic mean MSE do not support the generalization hypothesis.
3. Any gain-gap reduction accompanies improved test gain. A smaller gap caused
   only by destroying training gains is not supporting evidence.
4. All-view RGB/base-IR toggle equality, parameter finiteness and lifetime tests
   remain satisfied. Complete-ten RGB quality still needs empirical assessment.

If SSIM improves but test PSNR does not, the structural hypothesis is supported
while the target-metric hypothesis is not; do not describe that result as a
solution. Retain and report mixed results. Useful effects should be expanded
uniformly across scenes under the task's protocol, not dismissed after one weak
additional scene or combined with per-scene winners. Angular or support changes
would require separate, newly specified candidates.

This is a recommendation about the next experiment's evidence, not a projected
26.4 dB score. Final acceptance remains the full uniform ten-scene result and
the required RGB judgment. The new diagnostics inform method selection; model
parameters must continue to be fitted only on training images.
