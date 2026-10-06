# Independent review of the unapplied structural-v2 patch

Reviewed patch SHA-256
`785bd0ef5f5c40e17123ab0a41a34fc5460d22979302729eb3fc3ae2720827e3`.
No blocking defect found in the three-file training/CLI patch. This review does
not select v2 or authorize replacing the live v1 suite; the complete ten-scene
v1 result remains necessary before that decision.

The original review identified one future cost-diagnostic follow-up, now
resolved without applying this training patch:

- At initial review, `ir_experiments/benchmark_cost.py` computed only v1 MSE plus
  regularization in `--fit-gradient`. On a nonzero-SSIM v2 export it would omit
  the additional convolution/autograd work. Before presenting that diagnostic
  as v2 fitting cost, read `optimization_args.ir_kernel_ssim_weight` with a
  zero fallback and conditionally add the same frozen NCHW SSIM expression,
  or explicitly label the result as an MSE-only subdiagnostic. Inference
  on/off measurements remain valid because this patch does not change the
  inference path. This omission does not affect candidate training or the
  current v1 measurements. The separate diagnostic now reads the saved SSIM
  weight (default zero), evaluates positive-weight NCHW SSIM from the frozen
  candidate, and records all component weights and zero optimizer/parameter
  updates. Five focused CPU checks passed, including exact agreement with
  this staged training loss and zero-weight parity; evidence is in
  `../benchmark_loss_cpu_checks.json`. No GPU timing result is inferred.

Confirmed by direct code inspection:

- Optimization default is `0.0`; both direct training and the runner reject
  negative/nonfinite weights. The runner accepts the hyphenated option and
  forwards the underscore training flag at both prospective and frozen-source
  command sites. The trailing defaulted function parameter preserves existing
  positional callers. Zero omits the new argv flag exactly.
- `kernel_configuration` automatically captures the option, and the normal
  saved `optimization_args` captures it for each scene. The report's existing
  full-configuration equality check covers uniform scene settings.
- The original IR MSE/regularizer arithmetic is unchanged. SSIM evaluation,
  loss addition and scalar logging are inside a positive-weight guard; default
  zero adds no tensor/autograd operation or RNG draw. There is only a new
  Python scalar validation before training.
- The optional term uses unclipped `thermal.detach() + ir_kernel_residual`,
  expanded to NCHW. Its target comes from the currently sampled training
  camera, not the evaluation loop. Existing branch activation, default seed,
  iteration budget, optimizer, EMA, renderer and export code are unchanged.

Evidence checked during this review:

- `git apply --check` passes against the current worktree without applying it.
- `verify_manifest.py --root-state before` passes, verifies exact regenerated
  patch contents, and confirms all seven named protected root/frozen files.
- Independently verified the recorded CPU-report hash and all four named
  artifact hashes. The matching report records ten passing checks and
  `cuda_initialized=false`.
- Inspected the actual CPU test implementation: it executes the real staged
  loss block, checks exact zero-weight values/graph/gradients and CPU RNG
  preservation, positive-weight NCHW/raw inputs and isolated gradients,
  invalid weights, argument persistence, dry-run commands, and the post-freeze
  command/metadata path with launches stubbed. Existing tests were not rerun.

Limitations: the synthetic residual does not exercise CUDA rasterization or
prove complete-run bitwise reproducibility. The recorded tests do not prove
quality gains, full training overhead, or the final ten-scene target. Default
zero adds a saved configuration field even though loss arithmetic and argv
remain equivalent. This review created only this report; root, frozen and
staged sources were not edited, and no GPU job was launched.
