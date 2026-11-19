"""Issue #7 exploration (comment 7011219007) item 1: capture REAL linear layers
of BGE-M3 training. Wraps every nn.Linear.forward with CUDA event pairs
(single sync at end of step -> exact per-module fwd totals), records
shape/stride/bias/calls, plus torch.profiler aten::mm/addmm hit counts
(registration-mismatch question). Native/stock config, seed 42, --out required."""
import argparse, json, statistics, time
import torch
import torch.nn as nn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--capture-steps", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    import random, os
    random.seed(42); torch.manual_seed(42)
    os.environ.setdefault("FLAGSCALE_DEVICE", a.device)
    os.environ["FL_FLAGGEMS"] = ""            # explicit native/stock
    os.environ["FL_ADAMW_FUSED"] = "0"

    from omegaconf import OmegaConf
    from flagscale.train.datasets.retrieval import build_split_loaders
    from flagscale.train.losses.retrieval import loss_for_batch
    from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
    from flagscale.models.retrieval import build_retrieval_model

    cfg = OmegaConf.load(a.conf)
    batch_rows = int(cfg.batch_size); accum = int(cfg.gradient_accumulation_steps)
    dtype_name = str(cfg.model.get("torch_dtype", "bf16"))
    _r, _w, dev, _p = init_distributed(OmegaConf.create({"device_type": a.device}))
    model, processor = build_retrieval_model(str(cfg.model.model_type), cfg.model, dev)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(cfg.optimizer, trainable, allow_adafactor=False)
    loaders = build_split_loaders(str(cfg.task), cfg.data, batch_rows, pin_memory=True, seed=42)
    stream = iter(loaders[0])

    # ---- wrap every nn.Linear with event-bracketed forward ----
    stats = {}
    def wrap(module, key):
        orig = module.forward
        st = stats[key]
        def forward(*args, **kw):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            x = args[0] if args else kw.get("input")
            if torch.is_tensor(x):
                st["shapes"].add(tuple(x.shape)); st["strides"].add(tuple(x.stride()))
                st["dtype"] = str(x.dtype)
            s.record()
            out = orig(*args, **kw)
            e.record()
            st["events"].append((s, e))
            st["calls"] += 1
            return out
        module.forward = forward
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            stats[name] = {"calls": 0, "fwd_ms": 0.0, "events": [],
                           "shapes": set(), "strides": set(), "dtype": "",
                           "w_shape": tuple(mod.weight.shape),
                           "w_stride": tuple(mod.weight.stride()),
                           "bias": mod.bias is not None}
            wrap(mod, name)

    total_steps = a.skip + a.capture_steps
    for i in range(total_steps):
        capture = i >= a.skip
        if capture:
            for st in stats.values():
                st["calls"] = 0; st["events"] = []; st["fwd_ms"] = 0.0
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accum):
            batch = next(stream)
            with autocast_context(dev, dtype_name):
                loss = loss_for_batch(model, processor, str(cfg.task), batch, dev, float(cfg.temperature))
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        if capture:
            for st in stats.values():
                st["fwd_ms"] += sum(s.elapsed_time(e) for s, e in st["events"])
            print(f"  capture step {i}: done", flush=True)
    # reset wrapped forwards before profiler run
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name in stats:
            pass  # keep wraps; profiler run is separate process concern
    # ---- aten op hit counts from a profiled step (counts only) ----
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accum):
            batch = next(stream)
            with autocast_context(dev, dtype_name):
                loss = loss_for_batch(model, processor, str(cfg.task), batch, dev, float(cfg.temperature))
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        torch.cuda.synchronize()
    hits = {}
    for ev in prof.key_averages():
        if ev.key in ("aten::mm", "aten::addmm", "aten::matmul", "aten::linear",
                      "aten::bmm", "aten::matmul_backward", "aten::sumproduct_pair",
                      "aten::mm_backward", "aten::addmm_backward"):
            hits[ev.key] = {"count": ev.count,
                            "device_time_us": round(getattr(ev, "device_time_total", 0), 1),
                            "cpu_time_us": round(getattr(ev, "cpu_time_total", 0), 1)}
    out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "batch_rows": batch_rows, "accum": accum, "capture_steps": a.capture_steps,
           "aten_hits": hits, "modules": {}}
    for k, st in stats.items():
        out["modules"][k] = {"calls_per_capture_step": st["calls"] // a.capture_steps,
                             "fwd_ms_per_step": round(st["fwd_ms"] / a.capture_steps, 3),
                             "in_shapes": [list(s) for s in sorted(st["shapes"])],
                             "in_strides": [list(s) for s in sorted(st["strides"])[:2]],
                             "w_shape": list(st["w_shape"]),
                             "w_stride": list(st["w_stride"]),
                             "bias": st["bias"], "dtype": st["dtype"]}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"aten hits: {json.dumps({k: v['count'] for k, v in hits.items()})}")
    for k, v in sorted(out["modules"].items(), key=lambda kv: -kv[1]["fwd_ms_per_step"])[:15]:
        print(f"{v['fwd_ms_per_step']:8.2f} ms/step {v['calls_per_capture_step']:3d}x {k} w={v['w_shape']} in={v['in_shapes'][:1]} bias={v['bias']}")

if __name__ == "__main__":
    main()
