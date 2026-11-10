# Round-4 evaluation: random-weight small models — numerical consistency and performance

Reviewer plan: PR #2 comment `6756872311` ("后续方案：随机权重小模型的数值一致性与性能评估").
Code commits: `968d624` (accuracy harness, used for the matrix run) and
`8188a41` (bench hardening + minimal mse repro; `tests/small_models.py` and
`tests/test_small_models.py` are byte-identical in both commits, so the
matrix evidence remains valid for the shipped test code). Bench run at
`8188a41`.

## 1. Setup (plan §1–§3)

Groups, one process each (registration is process-global), identical
serialized CPU-generated inputs/params/teacher target loaded bit-identically
in every group (`input_identity` checked by every compare):

| group | tree | registration |
|---|---|---|
| n | no flag_gems on PYTHONPATH | none (never imported) |
| f_target | `/tmp/opencode/flagfix-fixed` | `only_enable(include=[...])` — A: `sum, sum_dim, sum_out, sum_dim_out, abs, absolute`; B: `conv2d, linear, linear_backward` |
| f_all | `/tmp/opencode/flagfix-fixed` | full `flag_gems.enable()` |
| b (pre-fix control) | `/tmp/opencode/flagfix-e2ebase` (base + linear-5379) | full `enable()`; only diff vs F = zeros-7181 |

- Model A TinyComplexNet (min: B=2, L=16, D=8, H=8, out=4): real input →
  `Z = complex(X@Wr+br, X@Wi+bi)` → `u = abs(sum(Z,dim=1))/L`,
  `v = sum(abs(Z),dim=1)/L` → concat → real linear head → MSE vs frozen
  teacher target (own seed 7777). Variants: `contig`, genuinely
  non-contiguous (`permute→contiguous→permute`, asserted), `conj`
  (asserted `is_conj()`). matmul/complex-construct/cat/loss stay native in
  the isolated group.
- Model B TinyLayoutNet (min: N=2, Cin=3, H=W=8, Cout=4, lin=5):
  `conv2d(padding=1)` → `tanh` → `permute(0,2,3,1)` NHWC non-contiguous
  (asserted) → `F.linear` on last dim → MSE. Layout premise recorded every
  run; f_target hit-keys confirmed registered (`conv2d`, `linear`,
  `linear_backward` present in `all_registered_keys()`).
- Seeds 0/1/2; dropout/AMP/TF32 off; `torch.use_deterministic_algorithms(True)`
  supported and recorded per run (see meta `determinism`).
- Frozen thresholds (never loosened after seeing any failure): fp32
  inter/out/loss atol 1e-5 rtol 1e-4, grads/updated-params atol 1e-5
  rtol 1e-3; fp64 inter atol 1e-10 rtol 1e-8, grads atol 1e-9 rtol 1e-7.
- Native-noise gate: duplicate native run compared against the matrix native
  run — **max_abs = 0.0 (bit-identical)**, so thresholds sit far above
  measurement noise.
- LR calibration (plan allows recalibrating the task via lr candidates):
  native pre-run, rule = largest stable lr with final-loss drop ≥ 10% over
  20 plain-SGD steps. Plan candidates {1e-3, 1e-2} were too small for model
  B (drop 0.5%), so 1e-1 was added: chosen lr = **0.1 for both models**
  (A: 0.500→0.381 in 3 steps @1e-2, stable @1e-1; B: drop 29.3% @1e-1,
  finite). Locked and shared by all groups.
- Centered finite-difference check (model A, fp64, 12 seeded coords, eps
  {5e-7, 1e-6, 2e-6}): **PASS** (`fd_a_min_seed0_fp64.json`, max rel err
  and step stability within 1e-5 / 1e-4; min |Z| and min |sum(Z)| guards
  clear).

## 2. Accuracy results (29 runs, 20 compares, `summary.json`)

### f_target — the fix's declared scope: 9/9 PASS

| compare | seeds | worst max_abs | thresholds |
|---|---|---|---|
| A fp32 (contig/noncontig/conj) | 0,1,2 | 1.19e-7 – 2.88e-7 | 1e-5 |
| B fp32 | 0,1,2 | 3.58e-7 – 4.77e-7 | 1e-5 |
| A fp64/c128 | 0,1,2 | 4.4e-16 – 8.9e-16 | 1e-10 |

Single-step intermediates (u, v, h, out, loss), input/param grads, params
after one SGD step, all 20 training losses and every per-step grad/param
checkpoint pass; grad-existence/shape/dtype/finite checks pass with no
whitelists; zero-grad ratios and norms recorded (non-degenerate).

### f_all — full registration on the fixed tree: FAILS, first divergence nodes located

- **Model A, fp32: crash.** Backward of `abs` calls `sgn`;
  `flag_gems/ops/sgn.py:149` `torch.empty_like(x)` (complex64) dispatches to
  `fill_scalar_` → triton `canonicalize_dtype('complex64')` → **KeyError:
  'complex64'** (`f_all__a_*_fp32.meta.json`). The fix's 8 target files do
  not include `sgn`/`fill` — pre-existing complex gap in the pointwise
  stack, not a regression of zeros-7181.
- **Model A, fp64: crash.** triton `CompilationError`: loop-carried
  accumulator fp32 vs fp64 (libentry autotune) — same pre-existing complex
  limitation.
- **Model B, fp32: runs but silently wrong gradients** (worse than a crash;
  flagged per plan). Forward is exact (out/loss pass), but 190/192 stats
  fail with rel_l2 ≈ 1.0 on all grads. First divergence located by bisection
  (`--include` override): adding **only** `mse_loss_backward` to the
  f_target base reproduces the failure exactly. Minimal repro
  (`tests/repro_mse_scalar_grad.py`, log `mse-scalar-repro.log`): autograd
  passes a **0-dim scalar grad_output** for scalar losses; the FlagGems
  kernel loads it elementwise without broadcasting → element 0 exact,
  39/40 elements garbage; with a full-tensor grad_output the kernel is
  exact; reduction none/sum exact. `ops/mse_loss_backward.py` is identical
  in the pre-fix tree — pre-existing, outside fix scope.

### b (pre-fix control): both models crash where the fix applies

- Model A: `sum.py:301 sum_dim_comm` → `fill_scalar_` → **KeyError:
  'complex64'** — exactly the bug family zeros-7181 repairs; the fixed tree
  (f_target) runs the same path numerically correctly.
- Model B: old `linear_backward.py:170` `input.view(batch_size,
  in_features)` on the NHWC non-contiguous activation → RuntimeError; the
  fixed tree uses `.reshape(...)` and passes. This confirms the fix's
  real-path value on model B's layout (thead `linear_backward` was actually
  hit, not bypassed).

### Companion op guarantees

Complex `zeros`/`zero_`/`zeros_like` dtype/value/aliasing/unselected-storage
guarantees remain covered by the existing 210-case regression
(`results/regression-*-2026-11-10.log`, 210 passed / 0 failed); per the plan
they are kept out of the differentiable graph here.

## 3. Performance (plan §4, first round = reporting only)

Protocol: first call recorded separately (JIT/autotune, cold process),
warmup 20, 5 independent blocks × 100 iters, CUDA events device time +
synced wall clock, seeded-random case order (seed 20261110), peak
allocated/reserved recorded per case, zero_ restore timed as a separate
case, same weights/inputs/grad state per timing block (restore in per-block
setup). Groups n / f_target / f_all / b / n_rep (second native pass bounds
drift). Raw per-block times in `bench-*.json`; table in `bench-table.md`;
combined `bench-summary.json`.

### Op level (device ms/iter, median of 5×100; full table in `bench-table.md`)

| family (c64/c128) | n | f_target | note |
|---|---|---|---|
| sum/full n1024→2^20 | 0.006–0.032 | 0.132 (6.3–20.9×) | slower than native on this build |
| sum/inner (contig/strided) | 0.007–0.019 | 0.092–0.094 (5.0–13.6×) | same |
| sum/inner fwd+bwd | ~0.052 | ~0.157 (3.0×) | same |
| abs (contig) | 0.024–0.032 | 0.030–0.041 (1.1–1.3×) | mild overhead |
| abs (strided) | 0.025–0.037 | 0.040–0.056 (1.5–1.7×) | mild overhead |
| abs fwd+bwd | 0.083–0.144 | 0.092–0.152 (1.06–1.12×) | near parity |
| zeros | 0.011 | 0.010–0.011 (0.89–0.96×) | parity-to-faster |
| zero_ | 0.004–0.005 | 0.003–0.005 (0.83–1.00×) | parity-to-faster |

### Model level (medium: A large B=8/L=256/D=64/H=64; B large N=8/32×32/Cout=32)

| case | n | f_target | f_all | b |
|---|---|---|---|---|
| A contig fwd | 0.209 | 0.493 (2.36×) | 2.658 (12.73×)¹ | crash¹ |
| A contig fwd+bwd | 0.650 | 1.494 (2.30×) | crash¹ | crash¹ |
| A contig train-step | 0.749 | 1.349 (1.80×) | crash¹ | crash¹ |
| A conj fwd / fwd+bwd / step | 0.233 / 0.748 / 0.909 | 2.23× / 1.82× / 1.64× | crash¹ | crash¹ |
| B fwd | 0.101 | 0.108 (1.07×) | 1.145 (11.30×)¹ | 1.136 (11.21×)¹ |
| B fwd+bwd | 0.391 | 0.630 (1.61×) | 4.115 (10.53×) | crash² |
| B train-step | 0.455 | 0.751 (1.65×) | 5.271 (11.59×) | crash² |

¹ recorded as case-level exceptions (pre-existing complex gap / conv-overhead —
crashes are never counted as speedups); ² per-case backward exceptions in the
pre-fix tree (`linear_backward` NHWC `.view()` crash).

Native drift (second native pass, n_rep): ≤ 9.6% on model cases, ≤ ~16% on
µs-scale zero_/abs small-size rows, ≤ ~4% on large op sizes — small-size
absolute numbers are jitter-dominated; ratio conclusions rely on the
repeatable large-size rows. 58 stable >10% f_target regressions are flagged
in `bench-summary.json` (sum/abs families above); **no speedup claim is
made for this round** — on this build the fix's value is correctness
(crash → numerically correct), not speed.

## 4. Conclusions (scoped to these models)

1. **The zeros-7181 fix does what it claims in its declared scope**: with
   selective registration of the target ops (f_target), 9/9 model/seed/dtype
   compares pass with fp32 max_abs ≤ 4.8e-7 (threshold 1e-5) and fp64
   ≤ 8.9e-16 (threshold 1e-10); the pre-fix control crashes on the exact
   paths the fix targets (complex `sum_dim` backward; non-contiguous
   `linear_backward`).
2. **Full-registration FlagGems on this build is not complex-safe and has a
   mean-MSE-backward scalar-grad bug — both pre-existing and outside the
   fix's 8 target files.** f_all model A crashes (triton complex64 KeyError
   / CompilationError); f_all model B silently corrupts gradients via
   `mse_loss_backward` not broadcasting a 0-dim grad_output (minimal repro
   included). Recommended follow-ups (separate fix packages): complex
   support or explicit complex rejection in the pointwise fill/sgn path, and
   a broadcast fix in `mse_loss_backward`.
3. Native oracle noise is zero (bit-identical reruns), thresholds were
   frozen before any fix-group ran, and no result was loosened afterwards.
   Conclusions are limited to these two models and this backend build.

## 5. Artifacts

- `results/small-model-2026-11-10/matrix/` — frozen `thresholds.json`,
  `lr.json`, `summary.json` (29 runs, 20 compares, noise gate, fd), inputs,
  per-run `.pt`/`.meta.json` dumps, all compare JSONs.
- `results/small-model-2026-11-10/bench/` — `bench-{n,f_target,f_all,b,n_rep}.json`,
  `bench-summary.json`, `bench-table.md`.
- `results/small-model-2026-11-10/mse-scalar-repro.log` — minimal repro
  output; script `tests/repro_mse_scalar_grad.py`.
- Harness: `tests/small_models.py`, `tests/test_small_models.py`,
  `benchmarks/bench_small_models.py`.
