"""Review 6d1a05b item 3: same-sample-window no-profiler timing.
Fresh model + fresh loader (same seed 42 => identical sample window as the
trace), skip 2, measure steps 2-4; per-step batch length summaries recorded
and asserted equal between passes."""
import argparse, hashlib, json, statistics, sys, time
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--trace-meta", default="results/training_trace_r36_meta.json",
                    help="trace-pass fingerprint file; default: in-repo r36 meta")
    a = ap.parse_args()
    import random, os
    random.seed(42); torch.manual_seed(42)
    os.environ.setdefault("FLAGSCALE_DEVICE", a.device)
    os.environ["FL_FLAGGEMS"] = ""
    os.environ["FL_ADAMW_FUSED"] = "0"
    from omegaconf import OmegaConf
    from flagscale.train.datasets.retrieval import build_split_loaders
    from flagscale.train.losses.retrieval import loss_for_batch
    from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
    from flagscale.models.retrieval import build_retrieval_model

    cfg = OmegaConf.load(a.conf)
    accum = int(cfg.gradient_accumulation_steps)
    dtype_name = str(cfg.model.get("torch_dtype", "bf16"))
    _r, _w, dev, _p = init_distributed(OmegaConf.create({"device_type": a.device}))
    model, processor = build_retrieval_model(str(cfg.model.model_type), cfg.model, dev)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(cfg.optimizer, trainable)
    loaders = build_split_loaders(str(cfg.task), cfg.data, int(cfg.batch_size), pin_memory=True, seed=42)
    stream = iter(loaders[0])

    # review 5ec9e6c item 1: tokenizer timing REMOVED from the measured step
    # (batches saved; tokenized AFTER timing) and fingerprints are compared
    # against the TRACE-pass fingerprint file
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(cfg.model.model_path))
    def run_step(i):
        nonlocal loader_idx
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        step_batches = []
        for micro in range(accum):
            batch = next(stream)
            step_batches.append(batch)
            with autocast_context(dev, dtype_name):
                loss = loss_for_batch(model, processor, str(cfg.task), batch, dev, float(cfg.temperature))
            (loss / accum).backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000
        lens = []
        for batch in step_batches:   # tokenization OUTSIDE timing
            g = {}
            for k in sorted(batch.keys()):
                texts = "|".join(batch[k])
                text_hash = hashlib.sha256(texts.encode()).hexdigest()[:16]
                enc = tok(batch[k], padding=True, truncation=True,
                          max_length=int(cfg.model.get("max_length", 8192)), return_tensors=None)
                mask = enc["attention_mask"]
                g[k] = {"loader_index": loader_idx,
                         "text_hash16": text_hash,
                         "max_len": max(len(m) for m in mask),
                         "valid_tokens": sum(sum(1 for x in m if x == 1) for m in mask),
                         "padded_positions": sum(len(m) for m in mask)}
                loader_idx += 1
            lens.append({"groups": g,
                          "valid_tokens": sum(x["valid_tokens"] for x in g.values()),
                          "padded_positions": sum(x["padded_positions"] for x in g.values())})
        return wall_ms, lens

    timings = []
    summaries = []
    loader_idx = 0
    for i in range(a.skip + a.steps):
        ms, lens = run_step(i)
        if i >= a.skip:
            timings.append(ms); summaries.append({"step": i, "batch_lens": lens})
    # strict pairing assertions vs the trace-pass summary (token_stats_r25.json,
    # same seed window): per-micro valid/padded tokens must match exactly
    import os
    ref_path = a.trace_meta
    import os as _os
    if not _os.path.exists(ref_path):
        print("FATAL: --trace-meta file not found:", ref_path, file=sys.stderr)
        sys.exit(2)
    pairing = {"reference": ref_path, "asserted": False}
    try:
        ref = json.load(open(ref_path))
        ref_steps = ref["batch_fingerprints"]["fingerprints"]
        n_cmp = min(len(summaries), len(ref_steps))
        mismatches = []
        for j in range(n_cmp):
            micros = summaries[j]["batch_lens"]   # list of per-micro dicts
            ref_micros = ref_steps[f"step_{a.skip + j}"]
            for mi, (mm, rm) in enumerate(zip(micros, ref_micros)):
                if set(mm["groups"].keys()) != set(rm.keys()):
                    mismatches.append(f"step{j} micro{mi}: group KEY SET differs")
                    continue
                for g in sorted(rm.keys()):
                    mine_g, ref_g = mm["groups"][g], rm[g]
                    if mine_g.get("text_hash16") != ref_g.get("text_hash16") or \
                       mine_g["padded_positions"] != ref_g["padded_positions"] or \
                       mine_g["valid_tokens"] != ref_g["valid_tokens"]:
                        mismatches.append(f"step{j} micro{mi} {g}: hash/tokens differ mine={mine_g} ref={ref_g}")
        pairing = {"reference": ref_path, "micros_compared": n_cmp * accum,
                    "mismatches": mismatches[:8], "asserted": len(mismatches) == 0}
        if not pairing["asserted"]:
            print("FATAL: fingerprint pairing FAILED:", mismatches[:8], file=sys.stderr)
            sys.exit(3)
    except FileNotFoundError:
        pairing = {"reference": ref_path, "error": "reference summary missing"}
        print("FATAL: reference fingerprint file missing:", ref_path, file=sys.stderr)
        sys.exit(2)
    tokens_per_step = [sum(m["valid_tokens"] for m in s["batch_lens"]) for s in summaries]
    tsorted = sorted(timings)
    p95 = max(timings) if timings else None   # 3 samples: p95 == sample max, NOT stable p95
    out = {"device": torch.cuda.get_device_name(0),
           "window": f"steps {a.skip}..{a.skip + a.steps - 1} (same seed/loader window as trace)",
           "fresh_model_loader": True,
           "step_ms": [round(x, 1) for x in timings],
           "median_ms": round(statistics.median(timings), 1),
           "p95_ms_label": "sample max (only 3 runs)",
           "p95_ms": round(p95, 1) if p95 else None,
           "valid_tokens_per_step": tokens_per_step,
           "valid_tokens_per_s": [round(t / (ms / 1000), 0) for t, ms in zip(tokens_per_step, timings)],
           "pairing_assertions": pairing,
           "batch_summaries": summaries}
    json.dump(out, open(a.out, "w"), indent=1)
    print("same-window no-profiler steps:", [round(x,1) for x in timings], "median:", round(statistics.median(timings),1))

if __name__ == "__main__":
    main()
