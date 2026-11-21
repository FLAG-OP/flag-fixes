"""Issue #7 review (2026-11-21): natural training trace on ZW810E.
Real BGE-M3 entry, native/stock, bf16, batch 10, accum 4, mask/dropout kept.
Warmup -> no-profiler timings -> ONE continuous profiler window (3 steps,
record_function markers per phase) -> export_chrome_trace.
Analysis (in parse_training_trace.py): kernel inventory per phase via
correlation ids, GEMM FLOPs vs 139.2 TFLOP/s peak, HBM logical-byte estimate,
critical-path busy union."""
import argparse, json, os, statistics, time
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--trace-steps", type=int, default=3)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    import random
    random.seed(42); torch.manual_seed(42)
    os.environ.setdefault("FLAGSCALE_DEVICE", a.device)
    os.environ["FL_FLAGGEMS"] = ""
    os.environ["FL_ADAMW_FUSED"] = "0"

    from omegaconf import OmegaConf
    from flagscale.train.datasets.retrieval import build_split_loaders
    from flagscale.train.losses.retrieval import loss_for_batch
    from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
    from flagscale.models.retrieval import build_retrieval_model
    from torch.profiler import profile, ProfilerActivity, record_function

    cfg = OmegaConf.load(a.conf)
    accum = int(cfg.gradient_accumulation_steps)
    batch_rows = int(cfg.batch_size)
    dtype_name = str(cfg.model.get("torch_dtype", "bf16"))
    _r, _w, dev, _p = init_distributed(OmegaConf.create({"device_type": a.device}))
    model, processor = build_retrieval_model(str(cfg.model.model_type), cfg.model, dev)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(cfg.optimizer, trainable)
    loaders = build_split_loaders(str(cfg.task), cfg.data, batch_rows, pin_memory=True, seed=42)
    stream = iter(loaders[0])

    def run_step(i, mark=False):
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accum):
            batch = next(stream)
            if mark:
                with record_function(f"rf_s{i}_data_m{micro}"):
                    pass
            with (record_function(f"rf_s{i}_forward_m{micro}") if mark else _NULL()):
                with autocast_context(dev, dtype_name):
                    loss = loss_for_batch(model, processor, str(cfg.task), batch, dev, float(cfg.temperature))
            with (record_function(f"rf_s{i}_backward_m{micro}") if mark else _NULL()):
                (loss / accum).backward()
        with (record_function(f"rf_s{i}_clip_optimizer") if mark else _NULL()):
            gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000
    class _NULL:
        def __enter__(self): return self
        def __exit__(self, *ex): return False

    # per-step padded-position counts (batch shapes; same seed as token_stats)
    step_tokens = []
    total_steps = a.skip + a.trace_steps
    timings = {"warmup": [], "noprofiler": [], "profiled": []}
    trace_path = a.out_prefix + "_trace.json.gz"
    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    for i in range(total_steps):
        if i < a.skip:
            timings["warmup"].append(run_step(i))
            continue
        if i == a.skip:
            prof.__enter__()
        phase = "profiled"
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        padded = 0
        for micro in range(accum):
            batch = next(stream)
            for k in sorted(batch.keys()):
                for s in batch[k]:
                    pass
            with record_function(f"rf_s{i}_forward_m{micro}"):
                with autocast_context(dev, dtype_name):
                    loss = loss_for_batch(model, processor, str(cfg.task), batch, dev, float(cfg.temperature))
            with record_function(f"rf_s{i}_backward_m{micro}"):
                (loss / accum).backward()
        with record_function(f"rf_s{i}_clip_optimizer"):
            gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        timings[phase].append(dt)
        print(f"  step {i}: {dt:.0f} ms (profiled)", flush=True)
        if i == total_steps - 1:
            prof.__exit__(None, None, None)
    # no-profiler re-run of the same window (capture overhead comparison)
    for i in range(a.trace_steps):
        timings["noprofiler"].append(run_step(100 + i))
        print(f"  noprofiler re-run {i}: {timings['noprofiler'][-1]:.0f} ms", flush=True)
    import gzip
    prof.export_chrome_trace(trace_path.replace(".gz", ""))
    with open(trace_path.replace(".gz", ""), "rb") as fin, gzip.open(trace_path, "wb") as fout:
        fout.write(fin.read())
    os.remove(trace_path.replace(".gz", ""))
    out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "config": {"batch_rows": batch_rows, "accum": accum, "dtype": dtype_name,
                       "env": {"FL_FLAGGEMS": "", "FL_ADAMW_FUSED": "0"}, "seed": 42},
           "timings_ms": {k: [round(x, 1) for x in v] for k, v in timings.items()},
           "capture_overhead_pct": round((statistics.median(timings["profiled"]) / statistics.median(timings["noprofiler"]) - 1) * 100, 1),
           "trace_path": os.path.basename(trace_path),
           "commands": {"capture": f"python3 examples/retrieval/capture_training_trace.py --conf {a.conf} --skip {a.skip} --trace-steps {a.trace_steps} --out-prefix {a.out_prefix}",
                         "analysis": "python3 scripts/parse_training_trace.py --trace <trace.json.gz> --out results/training_trace_analysis_r32.json"}}
    json.dump(out, open(a.out_prefix + "_meta.json", "w"), indent=1)
    print("median noprofiler:", round(statistics.median(timings["noprofiler"]), 1), "ms | profiled:",
          round(statistics.median(timings["profiled"]), 1), "ms | overhead:",
          out["capture_overhead_pct"], "%")
    print("trace saved:", trace_path, os.path.getsize(trace_path), "bytes")

if __name__ == "__main__":
    main()
