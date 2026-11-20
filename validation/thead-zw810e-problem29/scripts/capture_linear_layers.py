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
    orig_forwards = {}
    def wrap(module, key):
        orig = module.forward
        orig_forwards[key] = orig
        st = stats[key]
        def forward(*args, **kw):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            x = args[0] if args else kw.get("input")
            shape_key = None
            if torch.is_tensor(x):
                shape_key = (tuple(x.shape), tuple(x.stride()), str(x.dtype))
                st["shape_counts"][shape_key] = st["shape_counts"].get(shape_key, 0) + 1  # round-28: per-shape COUNT
                st["dtype"] = str(x.dtype)
            s.record()
            out = orig(*args, **kw)
            e.record()
            st["events"].append((s, e, shape_key))   # timing associated to shape
            st["calls"] += 1
            return out
        module.forward = forward
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            stats[name] = {"calls": 0, "fwd_ms": 0.0, "events": [],
                           "shape_counts": {}, "strides": set(), "dtype": "",
                           "w_shape": tuple(mod.weight.shape),
                           "w_stride": tuple(mod.weight.stride()),
                           "bias": mod.bias is not None}
            wrap(mod, name)

    total_steps = a.skip + a.capture_steps
    per_capture_step = []   # independent per-step totals (round-24 fix: profiler step must not mix in)
    for i in range(total_steps):
        capture = i >= a.skip
        if capture:
            for st in stats.values():
                st["calls"] = 0; st["events"] = []; st["shape_counts"] = {}
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
            step_tot = {}
            for k, st in stats.items():
                per_shape_ms = {}; per_shape_calls = {}
                total_ms = 0.0
                for s, e, sk in st["events"]:
                    ms = s.elapsed_time(e); total_ms += ms
                    if sk is not None:
                        per_shape_ms[sk] = per_shape_ms.get(sk, 0.0) + ms
                        per_shape_calls[sk] = per_shape_calls.get(sk, 0) + 1
                shape_ms_ser = {json.dumps({"shape": sh, "stride": st2, "dtype": dt}): round(v, 4)
                                for (sh, st2, dt), v in per_shape_ms.items()}
                shape_calls_ser = {json.dumps({"shape": sh, "stride": st2, "dtype": dt}): c
                                   for (sh, st2, dt), c in per_shape_calls.items()}
                step_tot[k] = {"fwd_ms": round(total_ms, 4), "calls": st["calls"],
                               "per_shape_ms": shape_ms_ser,
                               "per_shape_calls": shape_calls_ser}
            per_capture_step.append(step_tot)
            print(f"  capture step {i}: done", flush=True)
    # round-25: snapshot wrapper counters before profiler; after profiler they must be unchanged
    pre_prof_calls = {k: st["calls"] for k, st in stats.items()}
    # round-24 fix: RESTORE original forwards BEFORE the profiler step so the
    # profiler run cannot append events into capture stats
    for name, mod in model.named_modules():
        if name in orig_forwards:
            mod.forward = orig_forwards[name]
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
    assert all(st["calls"] == pre_prof_calls[k] for k, st in stats.items()), \
        "wrapper counts changed during profiler step (restore failed)"
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
           "aten_hits": hits, "modules": {},
           "note": "calls/timing from capture steps ONLY (wrappers restored before profiler step); fwd_ms_per_step = mean of per-capture-step totals"}
    for k, st in stats.items():
        per_calls = [tot[k]["calls"] for tot in per_capture_step]
        per_ms = [tot[k]["fwd_ms"] for tot in per_capture_step]
        # per-shape aggregation across capture steps
        agg_calls = {}; agg_ms = {}
        for tot in per_capture_step:
            for sj, c in tot[k]["per_shape_calls"].items():
                agg_calls[sj] = agg_calls.get(sj, 0) + c
            for sj, msv in tot[k]["per_shape_ms"].items():
                agg_ms[sj] = agg_ms.get(sj, 0.0) + msv
        shape_rows = []
        for sj in sorted(agg_ms, key=lambda x: -agg_ms[x]):
            meta = json.loads(sj)
            shape_rows.append({"shape": meta["shape"], "stride": meta["stride"],
                               "dtype": meta["dtype"],
                               "calls_sum": agg_calls[sj],
                               "calls_per_step": agg_calls[sj] / a.capture_steps,
                               "fwd_ms_per_step": round(agg_ms[sj] / a.capture_steps, 3)})
        out["modules"][k] = {"calls_per_capture_step": round(sum(per_calls) / len(per_calls), 2),
                             "calls_per_capture_step_list": per_calls,
                             "fwd_ms_per_step": round(sum(per_ms) / len(per_ms), 3),
                             "fwd_ms_per_capture_step": [round(v, 3) for v in per_ms],
                             "per_shape": shape_rows,
                             "reconcile": {"shape_calls_sum": sum(r["calls_sum"] for r in shape_rows),
                                           "expected_calls_sum": sum(per_calls),
                                           "shape_ms_sum": round(sum(r["fwd_ms_per_step"] for r in shape_rows), 3)},
                             "w_shape": list(st["w_shape"]),
                             "w_stride": list(st["w_stride"]),
                             "bias": st["bias"], "dtype": st["dtype"]}
    out["wrapper_restore_verified"] = True
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"aten hits: {json.dumps({k: v['count'] for k, v in hits.items()})}")
    for k, v in sorted(out["modules"].items(), key=lambda kv: -kv[1]["fwd_ms_per_step"])[:15]:
        top = v['per_shape'][0]['shape'] if v['per_shape'] else None
        print(f"{v['fwd_ms_per_step']:8.2f} ms/step {v['calls_per_capture_step']:3g}x {k} w={v['w_shape']} in={top} bias={v['bias']}")

if __name__ == "__main__":
    main()
