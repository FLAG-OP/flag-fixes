# Round-4/5/6 evaluation: random-weight small models — numerical consistency and performance

Reviewer plans: PR #2 comments `6756872311` (round-4), `6766490204`
(round-5), `6784122575` (round-6 corrections). This version replaces the
round-5 text: the performance section is **auto-generated from
`bench2-summary.json`** (`bench-r2/perf-section-auto.md`, embedded below),
the zeros/zero_ conclusions are withdrawn and corrected, the profiler
hit-verification claims are bound to the actual event lists, and the run
provenance is stated from `run-commits.json`.

## Run provenance (round-6 correction)

| artifact | generating commit | notes |
|---|---|---|
| accuracy matrix (`matrix/`) | **`968d624`** | device runs used the harness AS OF THIS COMMIT |
| re-scoring (`matrix-rescored/`) | `fac401c` | comparator hardening **modified `tests/test_small_models.py` after the device runs (226 lines: 170+/56−, `968d624..fac401c`)**; the stored `.pt` dumps were re-scored on CPU — the device data itself was produced by the pre-hardening harness (`tests/small_models.py` unchanged, diff empty) |
| bench-r2 (`bench-r2/`) | **`dd5ebea`** | per-family bench, 15 groups |
| evidence commits | `5e02b3b` (round-4), `fb13007` (round-5) | |

Consequently: statements of the form "the original device runs used the
hardened comparator" are **not** made; the accurate statement is "device
runs at `968d624`, comparator updated afterwards at `fac401c`, stored
tensors re-scored". Machine-readable provenance lives in
`bench-r2/run-commits.json` and each JSON's own fields; the summary
numbers in §4 are generated, not hand-maintained.

## 1. Setup (unchanged in substance)

Groups, one process each, identical serialized CPU-generated
inputs/params/teacher target (bit-identity asserted by every compare):

| group | tree | registration |
|---|---|---|
| n | no flag_gems on PYTHONPATH | none (never imported) |
| f_target | `/tmp/opencode/flagfix-fixed` | `only_enable(include=[...])` per family, expected keys asserted |
| f_all | `/tmp/opencode/flagfix-fixed` | full `flag_gems.enable()` |
| b (pre-fix control) | `/tmp/opencode/flagfix-e2ebase` (base + linear-5379) | full `enable()`; only diff vs F = zeros-7181 |

Accuracy f_target include lists (recorded per run): A =
`sum, sum_dim, sum_out, sum_dim_out, abs, absolute`; B =
`conv2d, linear, linear_backward`; bench families add `zeros/zero_/
zeros_like` (see §4).

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
- LR calibration: chosen **0.1 for both models** (1e-1 added under the
  plan's task-recalibration clause, reason recorded).
- Finite-difference check (A, fp64, 12 seeded coords): **PASS**.

## 2. Comparator hardening and re-scoring

`compare_dumps` (pure function) enforces, before any numeric subtraction:
shape/dtype equality (no broadcasting, no silent upcast), expected key
sets for all checkpoint sections, training structure (losses length, step
IDs exactly 0..n-1 in both groups, no duplicates/out-of-order), and
reference validity (missing/non-finite NATIVE grad = reference-invalid
failure, never skip-as-pass).

Negative self-tests (`tests/test_comparator_negative.py`, device-free):
**10/10** — control passes; wrong dtype / broadcastable wrong shape /
missing trailing checkpoint / duplicate / out-of-order step IDs / missing
grad key / non-finite ref grad / non-finite fix grad / beyond-threshold
all FAIL as required (`matrix-rescored/comparator-negative-selftest.*`).

Re-scoring of all 20 saved pairs + noise pair from stored `.pt` dumps
(CPU only; device evidence untouched): **f_target 9/9 PASS + noise PASS**
(`matrix-rescored/*.compare.json`). f_all/b remain FAIL as recorded.
Meta semantics: `registered_target_keys` records **enabled** keys;
registered ≠ called (see §4 for dispatch evidence).

## 3. Accuracy results

### f_target — target-isolated group: 9/9 PASS

| compare | seeds | worst max_abs | thresholds |
|---|---|---|---|
| A fp32 (contig/noncontig/conj) | 0,1,2 | 1.19e-7 – 2.88e-7 | 1e-5 |
| B fp32 | 0,1,2 | 3.58e-7 – 4.77e-7 | 1e-5 |
| A fp64/c128 | 0,1,2 | 4.4e-16 – 8.9e-16 | 1e-10 |

### b (pre-fix control): both models crash where the fix applies

- Model A: `sum_dim_comm` → `fill_scalar_` → **KeyError: 'complex64'**;
- Model B: pre-fix `linear_backward.py:170` `.view()` on NHWC
  non-contiguous → RuntimeError; fixed tree uses `.reshape(...)` and
  passes. The fix is genuinely hit on the model path.

### f_all — full registration: FAIL, first divergence nodes located (pre-existing; follow-up items filed)

- **Model A fp32: crash** (abs backward → `sgn` `empty_like` →
  `fill_scalar_` → triton KeyError 'complex64'); fp64: CompilationError.
- **Model B fp32: silently wrong gradients** — `mse_loss_backward` does
  not broadcast a 0-dim scalar grad_output (element 0 exact, 39/40
  wrong; full-tensor grad_output and none/sum exact). Located by
  include-bisection.
- **Isolated same-path controls**: both failures reproduce on the
  PRE-FIX tree (`repro-mse-scalar-btree.log` 39/40 identical;
  `repro-complex-fill-btree.log` 5/5 KeyError incl. abs(z) forward) —
  pre-existing attribution backed by isolated controls. Trigger for the
  complex family: `torch.complex(p,q)`-assembled tensors (cast tensors
  unaffected). → `reports/follow-up-fix-items.md` FIX-ITEM-1/2.

## 4. Performance (bench-r2; section auto-generated from `bench2-summary.json`)

Protocol: per (family × group) processes; first call recorded separately
(cold process); warmup 20; 5 blocks × 100 iters; CUDA events + synced
wall clock; seeded-random case order; `calls_per_s = 1000 /
ms_per_iter_median`; `block_mean_p95_ms_per_iter` is a quantile estimate
of per-block means (5 blocks), not a per-request tail latency; per-case
memory baseline/peak/delta (window includes compile+warmup; delta is a
lower bound under shared residency); correctness-failed configs labelled
diagnostic. Raw block times in `bench2-*.json`.

<!-- AUTO-GENERATED content begins (perf-section-auto.md) -->

Candidate regression flags (>1.10 median ratio, criterion only, not a
stability verdict): **115** (auto-counted).

### zeros / zero_ (creation vs in-place, full table)

| case | n ms/iter | f_target | ratio | f_all | b |
|---|---:|---:|---:|---:|---:|
| zero_/c128/n1024 | 0.0039 | 0.0184 | 4.70x | 0.1883 | - |
| zero_/c128/n1048576 | 0.0054 | 1.5587 | 290.42x | 0.1896 | - |
| zero_/c128/n65536 | 0.0039 | 0.1836 | 47.41x | 0.1887 | - |
| zero_/c64/n1024 | 0.0039 | 0.0176 | 4.52x | 0.1885 | - |
| zero_/c64/n1048576 | 0.0039 | 0.9037 | 233.06x | 0.1885 | - |
| zero_/c64/n65536 | 0.0045 | 0.1135 | 25.44x | 0.1896 | - |
| zeros/c128/n1024 | 0.0105 | 0.0287 | 2.73x | - | - |
| zeros/c128/n1048576 | 0.0108 | 1.8791 | 174.79x | - | - |
| zeros/c128/n65536 | 0.0105 | 0.1896 | 17.99x | - | - |
| zeros/c64/n1024 | 0.0106 | 0.0281 | 2.66x | - | - |
| zeros/c64/n1048576 | 0.0105 | 1.0277 | 97.42x | - | - |
| zeros/c64/n65536 | 0.0105 | 0.1462 | 13.90x | - | - |

**Reading (mechanism, round-6 root-cause, round-7/8/9 corrected probe)**:
the f_target overhead grows linearly with tensor bytes (~9–14 GB/s
signature, fixed ~15 µs at 2^10). Event capture attributes it: the fix
routes complex `zero_` to `x.copy_(0)` and complex `zeros` to `empty` +
that same path; **native ATen complex-scalar copy itself stages the value
through a full-tensor pageable HtoD** — profiler memcpy ≈ 570 µs at 2^20
c64 under registration, and the same overhead appears NATIVE without
flag_gems (`copy_(0)` 0.82–0.83 ms vs native `zero_()` 0.004–0.006 ms
device memset). PRE-BUILT device-scalar copy-only net cost, **by probe
batch** (timing variation between runs, not a functional difference;
batch/device/protocol provenance in the logs): **round-7 batch
0.0071/0.0069 ms, round-8 batch 0.0089/0.0067 ms** (c64/c128; archives
`zero-staging-attribution-r7-batch.log` and
`zero-staging-attribution.log`). The earlier ~0.02 ms figure mixed in
per-call `.cuda()` construction and is superseded; device-scalar creation
is 0.336 ms single-shot (含冷启动/first-launch overhead) / 0.015 ms
50-iter average, one-off outside any steady-state copy; see
`zero-staging-attribution*.log`, `zeros-hitprobe-*.json`). So the
dominant cost is the inherited ATen complex-scalar copy path, not extra
FlagGems compute; it is a candidate performance follow-up (FIX-ITEM-3,
three candidate paths + correctness matrix), with no optimization number
promised.

**Round-7/8 probe correction** [historical / superseded: the "~0.02 ms"
and "one-off 0.0139 ms" figures below are the round-7 wording; round-8
relabels creation as 50-iter average and adds the single-shot value;
round-9 note — the copy-only numbers below are the round-7 BATCH, the
round-8 batch measured 0.0089/0.0067 ms, see the batch-labeled paragraph
above]:
the earlier "0.02 ms" device-scalar figure included a per-call
`.cuda()` construction inside the timed lambda. With the device scalar
PRE-BUILT outside timing, the copy-only net cost is **0.0071 ms (c64) /
0.0069 ms (c128)** at 2^20 (device-scalar creation 0.0139 ms 50-iter
average, 0.336 ms single-shot; native memset 0.004–0.006 ms) — the ~117×
gap versus the host-staging path (0.83/1.50 ms) is confirmed with the
corrected probe; FIX-ITEM-3 lists the three candidate paths and their
required correctness matrix (contiguous / slice-transpose / storage
offset /
c64-c128 / unselected-storage).

### sum / abs (a-sum-abs family, forward)

| case | n ms/iter | f_target | ratio |
|---|---:|---:|---:|
| sum/full/c128/n1024 | 0.0060 | 0.1370 | 22.92x |
| sum/full/c128/n1048576 | 0.0211 | 0.1397 | 6.63x |
| sum/full/c128/n65536 | 0.0099 | 0.1404 | 14.25x |
| sum/full/c64/n1024 | 0.0062 | 0.1372 | 22.30x |
| sum/full/c64/n1048576 | 0.0105 | 0.1378 | 13.08x |
| sum/full/c64/n65536 | 0.0118 | 0.1388 | 11.81x |
| sum/inner/c128/n1024 | 0.0067 | 0.0964 | 14.37x |
| sum/inner/c128/n1048576 | 0.0187 | 0.0980 | 5.24x |
| sum/inner/c128/n65536 | 0.0068 | 0.0970 | 14.17x |
| sum/inner/c64/n1024 | 0.0068 | 0.0969 | 14.29x |
| sum/inner/c64/n1048576 | 0.0080 | 0.0972 | 12.13x |
| sum/inner/c64/n65536 | 0.0069 | 0.0965 | 13.94x |
| sum/inner/strided/c128/n1024 | 0.0067 | 0.0981 | 14.53x |
| sum/inner/strided/c128/n1048576 | 0.0118 | 0.0974 | 8.27x |
| sum/inner/strided/c128/n65536 | 0.0068 | 0.0971 | 14.20x |
| sum/inner/strided/c64/n1024 | 0.0068 | 0.0967 | 14.22x |
| sum/inner/strided/c64/n1048576 | 0.0085 | 0.0977 | 11.47x |
| sum/inner/strided/c64/n65536 | 0.0070 | 0.0984 | 14.00x |
| sum/outer/c128/n1024 | 0.0079 | 0.1100 | 13.98x |
| sum/outer/c128/n1048576 | 0.0282 | 0.1115 | 3.95x |
| sum/outer/c128/n65536 | 0.0253 | 0.1100 | 4.35x |
| sum/outer/c64/n1024 | 0.0067 | 0.1085 | 16.16x |
| sum/outer/c64/n1048576 | 0.0134 | 0.1106 | 8.23x |
| sum/outer/c64/n65536 | 0.0106 | 0.1098 | 10.32x |
| abs/c128/n1024 | 0.0244 | 0.0303 | 1.24x |
| abs/c128/n1048576 | 0.0323 | 0.0405 | 1.26x |
| abs/c128/n65536 | 0.0243 | 0.0304 | 1.25x |
| abs/c64/n1024 | 0.0243 | 0.0305 | 1.26x |
| abs/c64/n1048576 | 0.0244 | 0.0304 | 1.25x |
| abs/c64/n65536 | 0.0242 | 0.0309 | 1.28x |
| abs/strided/c128/n1024 | 0.0249 | 0.0408 | 1.64x |
| abs/strided/c128/n1048576 | 0.0365 | 0.0557 | 1.52x |
| abs/strided/c128/n65536 | 0.0247 | 0.0409 | 1.66x |
| abs/strided/c64/n1024 | 0.0246 | 0.0408 | 1.65x |
| abs/strided/c64/n1048576 | 0.0245 | 0.0402 | 1.64x |
| abs/strided/c64/n65536 | 0.0245 | 0.0408 | 1.66x |

sum forward ratios span 3.95x–22.92x (no large-size exemption: even 2^20
stays ≥ 3.95x); all are candidate flags, reported not explained-away.

### sum/abs fwd+bwd

| case | n ms/iter | f_target | ratio |
|---|---:|---:|---:|
| sum/inner/fwd+bwd/c128/n1024 | 0.0525 | 0.1578 | 3.00x |
| sum/inner/fwd+bwd/c128/n1048576 | 0.0530 | 0.1574 | 2.97x |
| sum/inner/fwd+bwd/c128/n65536 | 0.0531 | 0.1572 | 2.96x |
| sum/inner/fwd+bwd/c64/n1024 | 0.0521 | 0.1561 | 2.99x |
| sum/inner/fwd+bwd/c64/n1048576 | 0.0523 | 0.1592 | 3.04x |
| sum/inner/fwd+bwd/c64/n65536 | 0.0528 | 0.1573 | 2.98x |
| sum/inner/fwd+bwd/strided/c128/n1024 | 0.0519 | 0.1575 | 3.04x |
| sum/inner/fwd+bwd/strided/c128/n1048576 | 0.0523 | 0.1589 | 3.04x |
| sum/inner/fwd+bwd/strided/c128/n65536 | 0.0522 | 0.1584 | 3.04x |
| sum/inner/fwd+bwd/strided/c64/n1024 | 0.0538 | 0.1557 | 2.89x |
| sum/inner/fwd+bwd/strided/c64/n1048576 | 0.0521 | 0.1578 | 3.03x |
| sum/inner/fwd+bwd/strided/c64/n65536 | 0.0528 | 0.1571 | 2.97x |
| abs/fwd+bwd/c128/n1024 | 0.0831 | 0.0928 | 1.12x |
| abs/fwd+bwd/c128/n1048576 | 0.1436 | 0.1521 | 1.06x |
| abs/fwd+bwd/c128/n65536 | 0.0844 | 0.0930 | 1.10x |
| abs/fwd+bwd/c64/n1024 | 0.0837 | 0.0917 | 1.09x |
| abs/fwd+bwd/c64/n1048576 | 0.0830 | 0.0930 | 1.12x |
| abs/fwd+bwd/c64/n65536 | 0.0831 | 0.0932 | 1.12x |

### model level (diagnostic labels applied)

| family | case | n | f_target | f_all | b | drift n_rep |
|---|---|---:|---:|---:|---:|---:|
| a-sum-abs | modelA/conj/fwd | 0.2355 | 0.5373 (2.28x) | - (-) | - (-) | 6.3% diagnostic:f,b |
| a-sum-abs | modelA/conj/fwd+bwd | 0.7419 | 1.3962 (1.88x) | - (-) | - (-) | 1.7% diagnostic:f,b |
| a-sum-abs | modelA/conj/train-step | 0.8336 | 1.4857 (1.78x) | - (-) | - (-) | 5.7% diagnostic:f,b |
| a-sum-abs | modelA/contig/fwd | 0.2104 | 0.5036 (2.39x) | 2.7147 (12.90x) | - (-) | 7.1% diagnostic:f,b |
| a-sum-abs | modelA/contig/fwd+bwd | 0.6621 | 1.2878 (1.94x) | - (-) | - (-) | 2.9% diagnostic:f,b |
| a-sum-abs | modelA/contig/train-step | 0.7508 | 1.3979 (1.86x) | - (-) | - (-) | 6.4% diagnostic:f,b |
| b-linear | modelB/default/fwd | 0.0970 | 0.2351 (2.42x) | 1.1707 (12.07x) | 1.1523 (11.88x) | 9.0% diagnostic:f,b |
| b-linear | modelB/default/fwd+bwd | 0.3955 | 1.1605 (2.93x) | 4.1783 (10.56x) | - (-) | 35.3% diagnostic:f,b |
| b-linear | modelB/default/train-step | 0.4660 | 1.1751 (2.52x) | 5.3628 (11.51x) | - (-) | 5.8% diagnostic:f,b |

¹ `diagnostic:f,b` = correctness-failed configs (f_all/b) → timings are
diagnostic only, not valid performance. One native drift outlier
(B fwd+bwd 35.3%) flagged for repeat-measurement analysis; no conclusion
drawn from it.

### Hit verification (registered vs probe-hit vs timed-case)

- **registered** (asserted per family; f_target fails hard if missing):
  `registered_target_keys` in each JSON.
- **probe-hit** (torch.profiler CUDA events, FULL lists in
  `zeros-hitprobe-{f_target,native}.json`): under f_target, `zeros`
  shows a **native** at::native FillFunctor + pageable HtoD memcpy and
  `zero_` shows **only** the pageable HtoD memcpy — **no Triton kernel
  name appears for the zeros family**. (The round-5 reply's claim that
  zeros hit a Triton kernel was a substring false-positive on
  `vectorized_elementwise_kernel` and is **corrected here**.) Triton
  kernel names (`sum_dim_kernel_inner`, `_absolute_complex_kernel`,
  conv2d/linear kernels) ARE observed for a-sum-abs and b-linear.
  Kernel-name recognition is by convention, raw names kept verbatim;
  unrecognizable names are marked unknown.
- **timed-case behavior** (scaling law 2^10→2^20): zeros/zero_ cost
  tracks tensor bytes (host staging, mechanism above).

### Corrections vs earlier report texts (withdrawn claims)

- zeros `0.89–0.96x` / zero_ `0.83–1.00x` (bench-r1, round-5 report):
  **withdrawn** — bench-r2 measures 2.66x–4.70x at 2^10 rising to
  97x–290x at 2^20.
- sum "large sizes only 1.1–2x" (round-5 report): **withdrawn** —
  bench-r2 sum forward ratios 3.95x–22.92x, fwd+bwd ≈3x.
- "tests files byte-identical 968d624..fac401c" (round-5 report header):
  **withdrawn** — `test_small_models.py` differs by 226 lines
  (170+/56−); see the provenance table at the top.
- Flag-count history: "58" (round-4) and "115" (round-5) were
  hand/stale counts; the current number is auto-generated above from the
  JSON.

<!-- AUTO-GENERATED content ends -->

## 5. Conclusions (three states, scoped to these models and this build)

1. **Target-isolated accuracy: PASS** — 9/9 compares (fp32 ≤ 4.8e-7 vs
   1e-5; fp64 ≤ 8.9e-16 vs 1e-10), re-scored by the hardened comparator
   with negative self-tests; the pre-fix control crashes exactly on the
   paths the fix repairs.
2. **Full registration: FAIL** — f_all model A crashes, f_all model B
   silently wrong gradients; both pre-existing (isolated same-path
   controls on the pre-fix tree), filed as follow-up items, not
   whitelisted.
3. **Performance: corrected-attribution observational report only** —
   no speedup claim. Newly and accurately reported: the zeros/zero_
   complex path carries a large, size-proportional overhead (host
   staging inherited from ATen's complex-scalar copy), now root-caused
   and filed as FIX-ITEM-3; complex sum/abs carry 1.06x–23x overheads
   (candidates listed). Whether to open a dedicated optimization item is
   left to maintainers now that the cost is quantified.

The original fix remains `draft / deployable=false /
hardware_status=unverified / owner=null`; these small-model conclusions
do not replace large-model full-gradient or container platform
validation.

## 6. Artifacts

- `results/small-model-2026-11-10/matrix/` — round-4 accuracy evidence
  (runs at `968d624`). Unchanged.
- `results/small-model-2026-11-10/matrix-rescored/` — comparator-v2
  re-scoring + negative selftests + both-tree repro logs.
- `results/small-model-2026-11-10/bench/` — round-4 bench (historical,
  superseded).
- `results/small-model-2026-11-10/bench-r2/` — round-5/6 bench: per-group
  JSONs (`dd5ebea`), `bench2-summary.json`, `bench2-table.md`,
  `perf-section-auto.md` (generated §4), `run-commits.json`,
  `zeros-hitprobe-{f_target,native}.json` (full profiler event lists),
  `zero-staging-attribution.log` (native-vs-registered mechanism
  attribution), `perf-section-meta.json`.
- Harness/tools: `tests/small_models.py`, `tests/test_small_models.py`,
  `tests/test_comparator_negative.py`,
  `benchmarks/bench_small_models.py`,
  `benchmarks/zeros_hitprobe.py`, `benchmarks/zero_staging_probe.py`,
  `benchmarks/gen_perf_section.py`,
  `tests/repro_mse_scalar_grad.py`, `tests/repro_complex_fill.py`.
- Follow-up items: `reports/follow-up-fix-items.md` (FIX-ITEM-1/2/3).
