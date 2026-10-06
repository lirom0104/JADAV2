# IR Gaussian experiments

`run_suite.py` is the new task-specific runner. It executes one scene at a time,
always with `CUDA_VISIBLE_DEVICES=0`, using the `thermalgaussian` interpreter.
It does not invoke the historical batch scripts. The default command uses all
ten scenes and exactly 30,000 iterations, starts each model from the dataset,
uses the default random seed, retains ODB activation at 26,000, and activates
the new IR Gaussian kernel at 18,000.

```bash
python ir_experiments/run_suite.py run --root output/ir_kernel_v1
```

The same command with `--dry-run` prints the exact planned commands without
creating files or using CUDA. A 30,000-step pilot can select several scenes:

```bash
python ir_experiments/run_suite.py run --root output/ir_kernel_v1_pilot \
  --scenes Building RoadBlock RotaryKiln
```

A short smoke run must explicitly activate both learned branches immediately:

```bash
python ir_experiments/run_suite.py run --root output/ir_kernel_smoke \
  --scenes RoadBlock --debug --iterations 40 \
  --ir-kernel-start-iter 0 --detail-basis-start-iter 0
```

Kernel settings can be changed uniformly with `--ir-kernel-amplitude`,
`--ir-kernel-lr`, `--ir-kernel-reg-weight`, `--ir-kernel-mse-weight` and
`--ir-kernel-start-iter`. Every run uses a fresh output directory. The runner
copies the current sources, including uncommitted changes, into
`code_snapshot/source/`, archives and hashes that copy, and executes all
training, rendering, evaluation and automatic reporting from that frozen copy.
Subsequent repository edits therefore do not change a running candidate.
The installed rasterizer and nearest-neighbor CUDA packages are also copied
into `runtime_packages/`, hashed, and imported from that frozen location.

The runner stops on the first failed process or invalid artifact. It records
exact argv, a shell-readable command, environment, process PID, timing, return
code and separate logs. `active_process.json` records the currently launched
child; `logs/*.process.json` retains stage-level state. Verify live processes
against their actual PID/command, rather than assuming a state file proves
that a process is still alive. It never chooses another GPU or stops unrelated
processes. Model checkpoints are saved at the requested final iteration.

After each scene it checks the PLY IR parameter export, checkpoint and feature
modules, saved configuration, and `training_receipt.json`, including actual
optimizer-update count. After rendering, all RGB and IR ground-truth PNGs must
match the fixed baseline pixel for pixel, with identical view names/counts.
Every rendered view needs all six finite per-view metrics, and the means must
agree with `results.json`. These checks detect failures that `metrics.py` may
otherwise catch and return with exit code zero.

`comparison.json`, `comparison.csv` and `comparison.md` contain all six metrics
and equal-scene-weight means. Missing scenes remain visible, and a partial run
cannot pass the 26.4 dB target. Full acceptance also requires common model and
optimization configuration, verified frozen sources, exactly 30,000 updates
from scratch, and a documented RGB assessment. Report images show ground
truth, baseline and candidate RGB/IR at the same view without rescaling pixels
or changing contrast. Both the median and worst PSNR-delta view are selected
for each modality, producing four montages per scene.

Rebuild the report after reviewing the six means and the selected same-view
images. The RGB assessment is a written judgment; no per-scene numerical
threshold has been invented:

```bash
/home/lf/miniconda3/envs/thermalgaussian/bin/python \
  ir_experiments/report_results.py --candidate output/ir_kernel_v1 \
  --rgb-judgment no_obvious_degradation \
  --rgb-reason 'Describe the measured average RGB changes and the actual visual review.'
```

Use `obvious_degradation` when the evidence supports that judgment. The default
is `pending`, which cannot mark the goal achieved. The official baseline is
always the specified `odb_30k_20261003_101217_1051565/SCENE/results.json`
`ours_30000` result, never historical comparison files. Test metrics may inform
design selection and this is disclosed in the report.

`ablate_kernel.py` isolates the new representation on a completed exported
model. It loads the frozen source and CUDA packages, renders all test views
with kernels enabled/disabled, verifies exact reproduction of the official
PNGs, and saves the per-view IR gain and RGB invariance. It never modifies
official results or performs optimization. Example:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/home/lf/miniconda3/envs/thermalgaussian/lib \
/home/lf/miniconda3/envs/thermalgaussian/bin/python \
ir_experiments/ablate_kernel.py \
  --candidate output/ir_kernel_v1_20261005_run2 --scene Building
```

This is auxiliary evidence; only the complete fixed-baseline ten-scene
comparison can establish acceptance. Run isolated timing after formal training
finishes if concurrent GPU work would contaminate runtime measurements.

`diagnose_kernel_generalization.py` measures kernel on/off PSNR, MSE and SSIM
on every training camera of a completed export. It also reports raw, unclipped
training-image MSE and recomputes test metrics from the existing ablation PNGs.
This exposes train/test gain differences without fitting anything or adding
iterations to a completed model:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/home/lf/miniconda3/envs/thermalgaussian/lib \
/home/lf/miniconda3/envs/thermalgaussian/bin/python \
ir_experiments/diagnose_kernel_generalization.py \
  --candidate output/ir_kernel_v1_20261005_run2 \
  --scenes Building DailyStuff Dimsum Ebike
```

Add `--preflight` for CPU-only checks. Reports use a new directory under
`diagnostics/kernel_generalization/SCENE`; existing reports cannot be
overwritten. Official artifacts, frozen code/runtime and parameter values are
checked for changes. Training scores are in-sample evidence, and a train/test
gain gap alone does not prove its cause. Acceptance still uses the complete
ten-scene official test results.

`benchmark_cost.py` provides a separate inference cost diagnostic after the
entire ten-scene suite has finished. Its CPU-only preflight checks completion
and actual `/proc` process handles; it does not import PyTorch, inspect a GPU,
or create output files:

```bash
python ir_experiments/benchmark_cost.py \
  --candidate output/ir_kernel_v1_20261005_run2 \
  --output output/ir_kernel_v1_cost_01 --preflight
```

Once preflight is ready, use physical GPU 0 with the frozen export:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/home/lf/miniconda3/envs/thermalgaussian/lib \
/home/lf/miniconda3/envs/thermalgaussian/bin/python \
  ir_experiments/benchmark_cost.py \
  --candidate output/ir_kernel_v1_20261005_run2 \
  --output output/ir_kernel_v1_cost_01 \
  --views all --warmup 5 --repeats 20
```

The default selects five evenly spaced test views per scene; `--views all`
measures all views, and `--scenes Building RoadBlock RotaryKiln` restricts this
auxiliary diagnostic. Results are written only to a new directory outside the
official candidate. Source/runtime hashes are checked, modules load from the
frozen run, and hashes of official model/configuration/metrics/PNG inputs are
checked before and after. Kernel values must remain identical.

Each view records IR-kernel on/off wall time and CUDA-event samples after
warmup, alternating mode order across views. A separate single-render pass
records resident and peak allocated/reserved memory plus incremental peak
memory. Both modes keep the kernel parameter storage; its exact element count,
dtype and bytes are reported separately. Resident allocator memory includes
the model and the camera/image tensors loaded by `Scene`, and excludes CUDA
driver/non-PyTorch allocations. `nvidia-smi` samples also record process/device
memory, clocks and temperature. Loading, file I/O, polling, warmup and cache
cleanup are excluded from timed samples.

The script refuses any active suite or workspace GPU-0 training process, even
if an old runner state file says it finished. It also refuses other visible
GPU-0 processes by default. `--allow-shared-gpu` explicitly allows unrelated
GPU work and marks results **NON-ISOLATED** when such a process is observed.
This flag never allows active suite training. Processes are checked before and
after each measurement block; absence at these samples is not an exclusive
GPU reservation and cannot exclude a brief task between samples.

Optional `--fit-gradient` measures full rendering plus the saved IR objective
and `autograd.grad` with respect to only IR parameters, using training views.
It reads MSE, regularization and optional SSIM weights from `optimization_args`.
Absent/zero SSIM preserves the original MSE/regularizer calculation without
an SSIM call; a positive weight includes the frozen repository's
`1 - SSIM(prediction[None], target[None])` on the raw detached base image plus
the learned residual. It applies no clamp or quantization. Per-scene
`fit_gradient_configuration` records component weights, SSIM source/hash,
measurement scope, and zero optimizer/parameter updates and added iterations.
Other model parameters are frozen; no optimizer is created, gradients are
discarded, and no parameters change. This is a diagnostic of IR fitting work,
**not** full training overhead: it excludes shared-parameter backpropagation,
Adam state, optimizer updates, densification and training I/O. It never adds
iterations to the completed 30,000-step candidate. `report.json` retains every
sample and these limitations; `summary.csv` summarizes inference costs.

`check_benchmark_loss_cpu.py` validates this optional loss calculation using
synthetic CPU outputs and the frozen SSIM implementation. It checks exact
absent/zero-weight parity, positive-weight agreement with the staged training
loss, detached base gradients, unchanged parameters and source-path guards:

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
/home/lf/miniconda3/envs/thermalgaussian/bin/python -B \
  ir_experiments/check_benchmark_loss_cpu.py \
  --report ir_experiments/benchmark_loss_cpu_checks.json
```

These focused checks do not launch CUDA or measure runtime/image quality.
