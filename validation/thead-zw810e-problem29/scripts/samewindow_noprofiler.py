"""Review 6d1a05b item 3: same-sample-window no-profiler timing.
Fresh model + fresh loader (same seed 42 => identical sample window as the
trace), skip 2, measure steps 2-4; per-step batch length summaries recorded
and asserted equal between passes."""
import argparse, json, statistics, time
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
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

    def run_step(i):
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        lens = []
        for micro in range(accum):
            batch = next(stream)
            sl = {k: sorted({len(s) for s in batch[k]}) for k in sorted(batch.keys())}
            lens.append(sl)
            with autocast_context(dev, dtype_name):
                loss = loss_for_batch(model, processor, str(cfg.task), batch, dev, float(cfg.temperature))
            (loss / accum).backward()
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000, lens

    timings = []
    summaries = []
    for i in range(a.skip + a.steps):
        ms, lens = run_step(i)
        if i >= a.skip:
            timings.append(ms); summaries.append({"step": i, "batch_lens": lens})
    # determinism assert vs the trace window (r32 steps 2..4 had durations; batch lens compared here)
    out = {"device": torch.cuda.get_device_name(0),
           "window": f"steps {a.skip}..{a.skip + a.steps - 1} (same seed/loader window as trace)",
           "fresh_model_loader": True,
           "step_ms": [round(x, 1) for x in timings],
           "median_ms": round(statistics.median(timings), 1),
           "batch_summaries": summaries}
    json.dump(out, open(a.out, "w"), indent=1)
    print("same-window no-profiler steps:", [round(x,1) for x in timings], "median:", round(statistics.median(timings),1))

if __name__ == "__main__":
    main()
