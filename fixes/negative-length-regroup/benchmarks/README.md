# Entry A/B plan

This directory intentionally has no fabricated benchmark JSON. Run the plan on
ZW810E with the original BGE-M3 entry before changing the fix status.

1. Prepare a clean checkout of FlagScale at `89f342e690053e07dda845f1365089ec00464e09`
   and apply this patch through `flag-fixes check`/`plan` for review. Use the
   same model, data shards, batch size, gradient accumulation, seed, bf16 policy,
   dropout, and optimizer setup in both arms.
2. Run arm A with `FLAGSCALE_NEGATIVE_LENGTH_REGROUP=0` and arm B with
   `FLAGSCALE_NEGATIVE_LENGTH_REGROUP=1`. Each process must start fresh; do not
   reuse a model or optimizer from the other arm.
3. Record per optimizer step wall time, effective tokens/s, forward/backward/
   clip+optimizer segments if available, peak device memory, loss, gradient norm,
   and non-finite count. Also record batch and token fingerprints so the same
   logical examples can be compared.
4. Repeat at least three independent processes per arm in an interleaved order.
   Report paired per-process differences, mean difference, and a 95% confidence
   interval. A result inside noise is a valid conclusion.
5. If wall time is unchanged despite lower padding, capture one no-profiler run
   and one profiler trace per arm. Attribute the difference to GEMM shape
   efficiency, attention kernel selection, regroup overhead, copies, or gaps.
6. After a stable performance signal appears, run short training and retrieval
   metrics with the original optimizer update. Report any gradient or update
   parameter deltas against the native repeated-run spread.
