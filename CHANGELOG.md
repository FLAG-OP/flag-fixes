# Changelog

## 0.1.0

- Pin eight existing FLAG-OP FIX repositories as Git submodules under `third_party/`.
- Index nine fix entries with source commits, patch hashes, target snapshots and explicit validation status.
- Add `templates/fix/` and `flag-fixes new` for new fixes maintained in this repository.
- Add 1 local performance fix `gelu-kunlunxin-pow`: replace libdevice `pow(x, 2.0)` with multiplication in the KunlunXIN GELU family (P800/XPU3 microbenchmark 5.8x-16.5x, no accuracy regression; draft, hardware unverified).
- Add 1 local performance fix `adamw-fused` (first flagscale-targeted local fix): select fused AdamW in `build_optimizer` for plain CUDA fp32/bf16/fp16 parameters, with DTensor(FSDP2)/CPU guards, env override and construction-time fallback (P800/XPU3 336-tensor benchmark 27.6 -> 11.5 ms/step, launches 2,688 -> 672, bf16 max diff 1 ulp; draft, hardware unverified).
- Add 1 local integration fix `flaggems-selective-enable` (flagscale): new `flaggems_include` config routes FlagGems activation through `only_enable` so only patched ops (gelu family) register; full enable measured as a net regression on KunlunXIN (9 of 10 representative ops slower). BGE-M3 rebuilt-workload step with all three local performance fixes: 2,665.9 -> 2,291.2 ms (-14.0%). Draft, hardware unverified.
- Add 1 local memory fix `reranker-last-token-logits` (flagscale): the Qwen3-Reranker-8B P800 OOM of problem26 (`Tried to allocate 11.57 GiB` at batch 32) is the adapter's full-sequence vocabulary logits (64 pairs x 640 tokens x 151,669 vocab in bf16), not the lm_head all-gather (1.16 GiB); the adapter now projects only the scored position (LM-head output 11.57 GiB -> 18.5 MiB) with bit-identical scores and gradients. The baseline OOM is reproduced natively on 2-card FSDP2 with real weights (11.57 GiB / 20.77 GiB free / 71.66 GiB allocated, matching the platform record); a single-card fragmented heap reproduces the OOM text on the baseline (exit 1) and the fix completes (exit 0); batch 32 forward/backward peak 40.80/54.99 -> 29.94/40.61 GiB; the real trainer on one card (bf16, batch 16) drops its reserved peak 92.76 -> 79.64 GiB with identical loss over 5 steps. 2-card FSDP2 end-to-end is intermittently blocked by F-P800-1 on the validation host. Draft, hardware unverified.
- Provide catalog validation, exact source-tree patch planning, explicit apply, status, verification and LIFO rollback.
- Test failure recovery, drift/conflict refusal, idempotence and imported source consistency.
- Add CI for Python 3.10 and 3.12.

Imported fixes remain non-deployable drafts pending adaptation and platform validation. Hardware profiles, validated bundles and offline release assembly are not included in this version. See `docs/source-audit.md` for the initial source audit.
