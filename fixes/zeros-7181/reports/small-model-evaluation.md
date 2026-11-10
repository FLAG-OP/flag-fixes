# Round-4/5 evaluation: random-weight small models — numerical consistency and performance

Reviewer plans: PR #2 comments `6756872311` (round-4) and `6766490204`
(round-5 fixes). This report replaces the round-4 version: the comparator is
hardened (structural checks v2), the bench is re-scoped per family with
registration assertions and dispatch verification, statistics units are
corrected, and all numbers below are regenerated from the round-5 data.
Round-4 raw evidence is preserved unchanged under `results/small-model-2026-11-10/{matrix,bench}/`;
round-5 evidence lives in `{matrix-rescored,bench-r2}/`.

Commits: `968d624` (accuracy harness; matrix run, tree clean), `fac401c`
(comparator v2 + negative selftests; re-scoring of saved dumps, CPU only),
`dd5ebea` (bench per-family rework; bench-r2 run, tree clean). Evidence
commit: `5e02b3b` (round-4) + the round-5 evidence commit of this branch.
`tests/small_models.py` / `tests/test_small_models.py` are byte-identical
between `968d624` and `fac401c` (`git diff` empty), so the matrix run
remains valid for the shipped test code; the comparator change only
re-scores stored tensors.

## 1. Setup (plan §1–§3, unchanged in substance)

Groups, one process each, identical serialized CPU-generated
inputs/params/teacher target (bit-identity asserted by every compare):

| group | tree | registration |
|---|---|---|
| n | no flag_gems on PYTHONPATH | none (never imported) |
| f_target | `/tmp/opencode/flagfix-fixed` | `only_enable(include=[...])` per family (below), expected keys asserted |
| f_all | `/tmp/opencode/flagfix-fixed` | full `flag_gems.enable()` |
| b (pre-fix control) | `/tmp/opencode/flagfix-e2ebase` (base + linear-5379) | full `enable()`; only diff vs F = zeros-7181 |

Accuracy f_target include lists (recorded per run): A =
`sum, sum_dim, sum_out, sum_dim_out, abs, absolute`; B =
`conv2d, linear, linear_backward`. Round-5 adds the same scoping to the
bench (see §4).

- Model A TinyComplexNet (min: B=2,L=16,D=8,H=8,out=4): real input →
  `Z=complex(X@Wr+br, X@Wi+bi)` → `u=abs(sum(Z,dim=1))/L`,
  `v=sum(abs(Z),dim=1)/L` → concat → real linear head → MSE vs frozen
  teacher target (seed 7777). Variants: contig / genuinely non-contiguous
  (asserted) / conj (asserted `is_conj()`).
- Model B TinyLayoutNet (min: N=2,Cin=3,H=W=8,Cout=4,lin=5): `conv2d(1)`
  → `tanh` → NHWC non-contiguous (asserted) → `F.linear` → MSE.
- Seeds 0/1/2; A additionally fp64/c128. Dropout/AMP/TF32 off;
  `use_deterministic_algorithms(True)` supported, recorded per run.
- Frozen thresholds (never loosened): fp32 inter 1e-5/1e-4, grads
  1e-5/1e-3; fp64 inter 1e-10/1e-8, grads 1e-9/1e-7. Native noise gate:
  duplicate native run **max_abs = 0.0** (bit-identical).
- LR calibration: chosen **0.1 for both models** (plan candidates
  {1e-3,1e-2} insufficient for model B; 1e-1 added under the plan's
  task-recalibration clause, reason recorded).
- Finite-difference check (A, fp64, 12 seeded coords, eps 5e-7/1e-6/2e-6):
  **PASS**.

## 2. Comparator hardening (round-5, P2) and re-scoring

`compare_dumps` (pure function, extracted from the CLI) now enforces,
before any numeric subtraction:

- **shape equality and dtype equality** per checkpoint — no broadcasting,
  no silent float64 upcast (wrong dtype or broadcastable wrong shape fail
  with explicit reasons);
- **expected key sets** for intermediates/grads/params_after and per-step
  grads/params — missing or extra keys fail;
- **training structure** — losses length, step-ID sequence must be exactly
  0..n-1 in both groups (duplicates/out-of-order/truncated steps fail),
  per-step key sets equal;
- **reference validity** — a missing or non-finite NATIVE grad is a
  *reference-experiment-invalid failure*, never a skip-as-pass.

Negative self-tests (`tests/test_comparator_negative.py`, device-free):
10/10 — control passes; wrong-dtype / broadcastable-wrong-shape / missing
trailing checkpoint / duplicate / out-of-order step IDs / missing grad key
/ non-finite ref grad / non-finite fix grad / beyond-threshold value all
FAIL as required (`matrix-rescored/comparator-negative-selftest.{json,log}`).

**Re-scoring** of all 20 saved pairs + the noise pair from the stored
`.pt` dumps (CPU only, original run evidence untouched):
**f_target 9/9 PASS + noise PASS** — identical verdicts to the original
scoring, now with structural guarantees
(`matrix-rescored/*.compare.json`). f_all/b remain FAIL as recorded.

Meta semantics renamed: `hits` → `registered_target_keys` — this records
which target keys are **enabled**; actual dispatch is verified separately
(§4, kernel-name capture). Registered ≠ called.

## 3. Accuracy results

### f_target — target-isolated group: 9/9 PASS

| compare | seeds | worst max_abs | thresholds |
|---|---|---|---|
| A fp32 (contig/noncontig/conj) | 0,1,2 | 1.19e-7 – 2.88e-7 | 1e-5 |
| B fp32 | 0,1,2 | 3.58e-7 – 4.77e-7 | 1e-5 |
| A fp64/c128 | 0,1,2 | 4.4e-16 – 8.9e-16 | 1e-10 |

Single-step intermediates, all input/param grads, params after one SGD
step, all 20 training losses and every per-step grad/param checkpoint
pass; grad existence/shape/dtype/finite enforced with no whitelists;
zero-grad ratios and norms recorded (non-degenerate).

### b (pre-fix control): both models crash where the fix applies

- Model A: `sum_dim_comm` → `fill_scalar_` → **KeyError: 'complex64'**
  (`b__a_min_seed0_fp32.meta.json`) — exactly the bug family zeros-7181
  repairs; the fixed tree runs the same path correctly.
- Model B: pre-fix `linear_backward.py:170` `.view()` on the NHWC
  non-contiguous activation → RuntimeError; the fixed tree uses
  `.reshape(...)` and passes. The fix is genuinely hit on the model path.

### f_all — full registration: FAIL, first divergence nodes located (pre-existing; follow-up items filed)

- **Model A fp32: crash.** abs backward → `sgn` (`sgn.py:149`
  `empty_like`) → `fill_scalar_` → triton KeyError 'complex64'; fp64:
  triton CompilationError (loop-carried accumulator fp32→fp64).
- **Model B fp32: silently wrong gradients** (flagged; worse than a
  crash). Forward exact, 190/192 stats fail with rel_l2≈1.0. First
  divergence by bisection: adding only `mse_loss_backward` to the f_target
  base reproduces it. Root cause: the kernel loads a **0-dim scalar
  grad_output** (what autograd passes for scalar losses) elementwise
  without broadcasting — element 0 exact, 39/40 wrong; full-tensor
  grad_output and reduction none/sum are exact.
- **Isolated same-path controls (round-5)**: both failures reproduce on
  the PRE-FIX tree (`matrix-rescored/repro-mse-scalar-btree.log`: 39/40,
  identical; `repro-complex-fill-btree.log`: 5/5 KeyError incl. abs(z)
  forward) — pre-existing attribution backed by isolated controls, not
  merely "file not in this patch". The complex family also crashes on the
  fixed tree (5/5 probes, `repro-complex-fill-fixed.log`); trigger is
  `torch.complex(p,q)`-assembled tensors (cast tensors unaffected).
- Follow-up items with repros/controls/status:
  `reports/follow-up-fix-items.md` (FIX-ITEM-1 mse scalar broadcast,
  priority; FIX-ITEM-2 complex pointwise/fill/zeros-size-branch/to_copy +
  FP64 compile).

### Companion op guarantees

Complex `zeros`/`zero_`/`zeros_like` dtype/value/aliasing/unselected-storage
remain covered by the existing 210-case regression
(`results/regression-*-2026-11-10.log`, 210 passed / 0 failed), kept out of
the differentiable graph here per the plan.

## 4. Performance (bench-r2, round-5 re-scoped; reporting only)

**P1 fix**: the round-4 bench registered model-A ops globally, so model-B /
zeros f_target timings were mis-attributed. bench-r2 runs **one process per
(family × group)** with per-family include lists and **expected-registered-
key assertions** (f_target fails hard if a target key is not registered):

| family | include | cases |
|---|---|---|
| a-sum-abs | sum/sum_dim/sum_out/sum_dim_out/abs/absolute | sum full/inner/outer + abs, c64/c128 × 2^10/2^16/2^20, contig/strided, fwd+bwd; model A fwd/fwd+bwd/train-step |
| b-linear | conv2d/linear/linear_backward | model B fwd/fwd+bwd/train-step |
| zeros | zeros/zero_/zeros_like | zeros + zero_ (restore timed separately) |

**Registered ≠ called**: outside timed regions, dispatch is verified per
family via torch.profiler CUDA kernel-name capture (`hit_verification` in
each JSON): a-sum-abs/f_target → `sum_dim_kernel_inner`,
`_absolute_complex_kernel`; b-linear/f_target → conv2d/linear triton
kernels; zeros/f_target → `zeros` triton kernel, `zero_` **native fill**
(the fix deliberately routes complex `zero_` through native `copy_(0)`).
Probe crashes on the b tree are recorded, not swallowed.

**Protocol/units (round-5 corrections)**: first call recorded separately
(cold process, JIT/autotune); warmup 20; 5 blocks × 100 iters; CUDA events
+ synced wall clock; seeded-random case order; `calls_per_s = 1000 /
ms_per_iter_median` (the round-4 `samples_per_s` formula inflated by 100×
— withdrawn), model cases additionally `samples_per_s = batch ×
calls_per_s`; "block_mean_p95_ms_per_iter" is a quantile estimate of
per-block MEANS (5 blocks) — **not a per-request tail latency**; per-case
memory now records baseline allocated/reserved before the timed window,
peak within it (window includes compile+warmup) and the delta (a lower
bound under shared process residency). Raw block times in
`bench-r2/bench2-*.json`; table `bench2-table.md`; flags are
`*_candidate_regression_gt10pct` — **a median-ratio criterion only, not a
stability verdict**; 115 such candidates (regenerated from data; the
round-4 report's 58 was stale and is withdrawn).

**Correctness pre-check → timing validity**: configs whose accuracy failed
in the matrix are labelled `*_correctness = failed → diagnostic only` in
the summary/table; their timings (all f_all/b model-level rows) are NOT
valid performance results and never count as speedups.

### Model level (ms/iter, median; full table in bench2-table.md)

| case | n | f_target | f_all | b |
|---|---|---|---|---|
| A contig fwd / fwd+bwd / step | 0.210 / 0.662 / 0.751 | 2.39× / 1.94× / 1.86× | 12.90× / crash¹ / crash¹ | crash¹ / crash¹ / crash¹ |
| A conj fwd / fwd+bwd / step | 0.236 / 0.742 / 0.834 | 2.28× / 1.88× / 1.78× | crash¹ ×3 | crash¹ ×3 |
| B fwd / fwd+bwd / step | 0.097 / 0.396 / 0.466 | **2.42× / 2.93× / 2.52×** | 12.07× / 10.56× / 11.51× (diagnostic²) | 11.88× (diagnostic²) / crash³ / crash³ |

¹ per-case exceptions recorded (pre-existing complex gap); ² correctness
failed (mse scalar bug) → diagnostic timing; ³ pre-fix `linear_backward`
NHWC `.view()` crash. Native drift (n_rep) ≤ 9% on model rows except one
row (B fwd+bwd 35% — µs-scale jitter, flagged for repeat-measurement
analysis; no conclusion drawn from it).

### Op level (summary; full table in bench2-table.md)

- sum (c64/c128, full/inner/outer, contig/strided): f_target **slower than
  native** on this build (≈3–21× at small/mid sizes, ≈1.1–2× at 2^20 fwd;
  fwd+bwd ≈3×) — candidate regressions flagged, no speedup claim.
- abs: 1.06–1.7× (strided worst) — mild overhead, fwd+bwd near parity.
- zeros: f_target creation ≈0.89–0.96× (parity-to-faster where it runs);
  **size-dependent complex branch**: the 8×8 probe dispatches, larger
  sizes hit the KeyError — recorded per-case in zeros/f_all/b.
- zero_: ≈native (0.83–1.00×) — consistent with the fix's deliberate
  native `copy_(0)` route for complex (verified by kernel capture).

## 5. Conclusions (three states, scoped to these models and this build)

1. **Target-isolated accuracy: PASS.** With per-family selective
   registration and asserted/dispatch-verified scope, 9/9 compares pass
   (fp32 ≤ 4.8e-7 vs 1e-5; fp64 ≤ 8.9e-16 vs 1e-10), re-scored by the
   hardened comparator with negative self-tests; the pre-fix control
   crashes exactly on the paths the fix repairs.
2. **Full registration: FAIL.** f_all model A crashes (complex
   assembled-tensor fill/sgn path; fp64 compile), f_all model B silently
   wrong gradients (mse scalar grad_output). Both are pre-existing —
   proven by isolated same-path controls on the pre-fix tree — and are
   filed as follow-up fix items; not covered by this patch, not
   whitelisted.
3. **Performance: re-measured with correct attribution; reporting only.**
   No speedup claim for this round: complex sum/abs and the target-ops
   model paths are slower than native on this build (candidate flags
   listed); zeros/zero_ ≈ native (by design of the fix's native fallback
   for complex). Correctness-failed configs are excluded from performance
   conclusions (diagnostic only).

The original fix remains `draft / deployable=false /
hardware_status=unverified / owner=null`; these small-model conclusions do
not replace large-model full-gradient or container platform validation.

## 6. Artifacts

- `results/small-model-2026-11-10/matrix/` — round-4 accuracy evidence
  (frozen thresholds, lr.json, 29 runs, 20 compares, fd, summary; runs at
  commit `968d624`). Unchanged.
- `results/small-model-2026-11-10/matrix-rescored/` — comparator-v2
  re-scoring of all pairs (CPU), negative selftests, mse/complex-fill
  repro logs on both trees.
- `results/small-model-2026-11-10/bench/` — round-4 bench (historical,
  kept for provenance; its f_target scope and samples_per_s units are
  superseded).
- `results/small-model-2026-11-10/bench-r2/` — round-5 bench
  (per-family, hit verification, corrected units; runs at commit
  `dd5ebea`).
- `results/small-model-2026-11-10/{provenance.json,mse-scalar-repro.log}` —
  round-4 provenance + repro log (kept).
- Harness: `tests/small_models.py`, `tests/test_small_models.py`,
  `tests/test_comparator_negative.py`, `benchmarks/bench_small_models.py`,
  `tests/repro_mse_scalar_grad.py`, `tests/repro_complex_fill.py`.
- Follow-up items: `reports/follow-up-fix-items.md`.
