# Prepared structural-v2 patch — not selected or launched

This directory stages one candidate for a later decision after the complete
ten-scene v1 run. It has not been applied to the current source or the active
v1 snapshot. No training or GPU experiment was run while preparing it.

The exact proposed source directory is
`/home/lf/code/Our_Project-New-2-1/ir_experiments/staged_structural_v2/after`.
It is a **partial source tree**, containing copies of only these changed files:

- `arguments/__init__.py`: optional optimization setting
  `ir_kernel_ssim_weight=0.0`, automatically saved in `optimization_args`.
- `train.py`: validate a finite nonnegative weight, then add
  `weight * (1 - SSIM(P[None], GT[None]))` only when the IR branch is active
  and the weight is positive. Here `P = base_IR.detach() + kernel_residual`.
  Raw predictions are neither clamped nor quantized. The existing MSE and
  regularization expression is retained exactly. Log the SSIM loss only when
  it is computed.
- `ir_experiments/run_suite.py`: accept `--ir-kernel-ssim-weight`, validate it,
  forward it at both command construction sites as `--ir_kernel_ssim_weight`,
  and include it in the existing automatic `kernel_configuration` metadata.
  Zero keeps the original train command exactly, omitting the optional flag.

No model, renderer, inference, export, checkpoint, densification, EMA or optimizer
changes are included. The representation retains its ten IR parameter columns.
Weight zero creates no new tensor operation, SSIM call, scalar logging operation
or RNG draw in the loss block. The default remains zero; **0.01 is only the
prepared candidate configuration**.

`before/` records the three current-root inputs. `structural_v2.patch` is a
reproducible unified diff from those copies to `after/`. `manifest.json`
records exact directories, base/staged hashes, patch hash, checks and protected
root/v1 source hashes. The hash guard deliberately fails if the current source
has changed, so rebase and review a stale patch rather than force-applying it.

## Focused CPU verification

From `/home/lf/code/Our_Project-New-2-1`:

```bash
python -B ir_experiments/staged_structural_v2/verify_manifest.py --root-state before
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /home/lf/miniconda3/envs/thermalgaussian/bin/python -B \
  ir_experiments/staged_structural_v2/check_cpu.py \
  --report ir_experiments/staged_structural_v2/cpu_checks.json
git apply --check ir_experiments/staged_structural_v2/structural_v2.patch
```

The tests compile and execute the **actual IR loss block extracted from the
staged `train.py` AST**, using the repository SSIM and unchanged Gaussian
regularizer. They do not import `train.py`, whose top-level LPIPS setup uses
CUDA. A deterministic synthetic residual substitutes for rasterization.
They establish:

- At weight zero, exact v1 loss values, autograd graph structure and leaf
  gradients, with no SSIM call or Python/NumPy/Torch CPU RNG consumption.
- At weight 0.01, NCHW SSIM is evaluated once on unclipped predictions,
  changes finite kernel gradients, and sends no gradient to the base/RGB/shared
  parents. Existing base/RGB/shared gradients are unchanged when their normal
  objectives are present. The inactive branch does no SSIM work.
- CLI/default/saved-setting behavior, invalid-weight rejection, dry-run commands
  for all ten scenes, exact default-zero command parity, and the post-snapshot
  command plus metadata through a runner test with **every launch stubbed**.
- Syntax, base/staged/protected-source hashes, and exact patch application to
  temporary copies under `/tmp`; no patch application to the root source.

`cpu_checks.json` records results, interpreter/version, hashes and limitations.
These checks do not establish CUDA rasterizer behavior, end-to-end training
determinism, memory/runtime cost, image quality or the 26.4 dB goal. In particular,
gradient isolation upstream of the synthetic residual relies on the unchanged
renderer; it is not newly tested on CUDA here.

## Application, only if this candidate is selected later

These are future application instructions, not actions performed by preparation.
Run the hash guard and patch check above first. Then:

```bash
git apply ir_experiments/staged_structural_v2/structural_v2.patch
python -B ir_experiments/staged_structural_v2/verify_manifest.py --root-state after
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /home/lf/miniconda3/envs/thermalgaussian/bin/python -B \
  ir_experiments/staged_structural_v2/check_cpu.py --root-state after \
  --report ir_experiments/staged_structural_v2/cpu_checks_after_application.json
```

If rollback is later desired and no further edits have been made to these files,
check and reverse the exact patch with `git apply --reverse --check` followed by
`git apply --reverse`, both given the patch path above. The active v1 frozen
snapshot is never an application or rollback target.

## Fresh 30,000-update commands, after selection and application

The four-scene diagnostic pilot proposed in the review uses one uniform
configuration. First inspect its plan by appending `--dry-run`:

```bash
/home/lf/miniconda3/envs/thermalgaussian/bin/python ir_experiments/run_suite.py run \
  --root output/ir_kernel_structural_v2_pilot_01 \
  --scenes Building DailyStuff Dimsum Ebike --iterations 30000 \
  --ir-kernel-ssim-weight 0.01
```

If a fresh complete ten-scene run is selected instead, or a uniform follow-up is
selected after the pilot, its command is:

```bash
/home/lf/miniconda3/envs/thermalgaussian/bin/python ir_experiments/run_suite.py run \
  --root output/ir_kernel_structural_v2_full10_01 --iterations 30000 \
  --ir-kernel-ssim-weight 0.01
```

Both commands require new output directories, use the unchanged default seed,
physical GPU 0, dataset initialization without a checkpoint, exactly 30,000
updates, IR activation after 18,000, LR 0.003, amplitude 0.2, MSE weight 1,
regularization weight 0.0001, existing support and EMA, and ODB after 26,000.
The runner freezes and hashes the selected source/runtime before training.
Do not execute the partial staged `after/` tree as a training repository.

A safe prospective command preview **before application** is recorded in
`prepared_full10_plan.json`; it was generated by the staged runner with
`--repo` pointing at the intended complete repository and `--dry-run`. The
printed training commands require patch application before they can run.

Use the same exported-model on/off and all-training-view diagnostics to test
the review's hypotheses: reduced train/test SSIM penalties and improved test
kernel PSNR gain, reporting all scenes/views. Increased training score alone,
or a smaller gap caused only by damaging training fit, is insufficient.
No partial-run result establishes task acceptance; full-ten IR and RGB review
remain required.

## Cost diagnostic scope

`benchmark_cost.py --fit-gradient` now reads the saved optional SSIM weight,
defaults missing weights to zero, and adds positive-weight NCHW SSIM using the
completed candidate's frozen `utils/loss_utils.py`. It records component weights,
source hashes, scope and zero optimizer/parameter updates. Missing/zero SSIM
retains the v1 MSE/regularizer calculation. Five focused CPU checks in
`../benchmark_loss_cpu_checks.json` verify zero-path parity, positive-loss and
gradient agreement with this staged training block, invalid weights and frozen
SSIM provenance. They do not establish GPU runtime or memory costs. The
inference on/off benchmark is unaffected. `review.md` records the original
review and resolution of this diagnostic follow-up. No v2 cost or image-quality
result is claimed; the training patch remains unapplied and unselected.
