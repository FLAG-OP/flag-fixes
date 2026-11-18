#!/usr/bin/env python3
"""FSDP2 2-card, one profiled optimizer step on the REAL BGE-M3 config.

Replicates the trainer's FSDP2 setup exactly (parallel_utils.apply_fsdp2 +
master-dtype move + stock/fused AdamW via build_optimizer) and profiles one
full optimizer step under torch.profiler, saving the raw kernel table so
collective traffic (XCCL all-gather / reduce-scatter, cluster_cast_kl3) can be
split from compute per rank (problem29 §5.5 requirements).

Run under torchrun --nproc_per_node=2. Output per rank: --out rank{rank}.json.
"""
import argparse, json, os, statistics, sys, time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--flagscale-root", default="/workspace/FlagScale-stock")
    args = ap.parse_args()

    sys.path.insert(0, args.flagscale_root)
    os.environ.setdefault("FLAGSCALE_DEVICE", "cuda")
    from omegaconf import OmegaConf
    from flagscale.train.datasets.retrieval import build_split_loaders
    from flagscale.train.losses.retrieval import loss_for_batch
    from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
    from flagscale.train.utils import parallel as parallel_utils
    from flagscale.models.retrieval import build_retrieval_model

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    _r, _w, dev, platform = init_distributed(OmegaConf.create({"device_type": "cuda"}))

    repo = args.flagscale_root
    cfg = OmegaConf.load(repo + "/examples/retrieval/conf/train/bge_m3.yaml")
    # bound the run like the trainer smoke
    OmegaConf.set_struct(cfg.data, False)
    cfg.data.max_samples = int(os.environ.get("MAX_SAMPLES", "600"))

    task = str(cfg.task)
    model, processor = build_retrieval_model(str(cfg.model.model_type), cfg.model, torch.device("cpu"))
    model.train()
    compute_dtype = torch.bfloat16
    model = parallel_utils.apply_fsdp2(
        model, device_type=dev.type, world_size=max(world, 1),
        compute_dtype=compute_dtype, reduce_dtype=compute_dtype,
        log=lambda m: print(f"[fsdp2] {m}", flush=True))
    master_dtype = {"fp32": torch.float32}.get(os.environ.get("MASTER_DTYPE", "bf16"), torch.bfloat16)
    model.to(device=dev, dtype=master_dtype)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(cfg.optimizer, trainable, allow_adafactor=False)
    loaders = build_split_loaders(task, cfg.data, int(cfg.batch_size), pin_memory=True, seed=42)
    stream = iter(loaders[0])
    accum = int(cfg.gradient_accumulation_steps)
    dtype_name = str(cfg.model.get("torch_dtype", "bf16"))

    # unprofiled warmup steps (compile + allocator steady state)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            batch = next(stream)
            with autocast_context(dev, dtype_name):
                loss = loss_for_batch(model, processor, task, batch, dev, float(cfg.temperature))
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(cfg.optimizer.max_grad_norm))
        optimizer.step()
    torch.cuda.synchronize()
    if rank == 0:
        print("warmup done, entering profiled steps", flush=True)

    from torch.profiler import profile, ProfilerActivity
    walls = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], acc_events=True) as prof:
        for step in range(args.steps):
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accum):
                batch = next(stream)
                with autocast_context(dev, dtype_name):
                    loss = loss_for_batch(model, processor, task, batch, dev, float(cfg.temperature))
                (loss / accum).backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(cfg.optimizer.max_grad_norm))
            optimizer.step()
            torch.cuda.synchronize()
            walls.append(time.perf_counter() - t0)
            print(f"rank{rank} step {step}: {walls[-1]*1000:.0f}ms loss={float(loss.detach()):.6f}", flush=True)

    ops = {}
    for e in prof.key_averages():
        if e.device_time_total <= 0:
            continue
        ops[e.key] = {"calls": e.count, "ms": e.device_time_total / 1000.0}
    aten = {k: v for k, v in ops.items() if k.startswith("aten::")}
    level = aten if aten else ops
    out = {
        "rank": rank, "world": world, "master_dtype": str(master_dtype),
        "steps": args.steps,
        "wall_ms_per_step": [round(w * 1000, 1) for w in walls],
        "wall_ms_median": round(statistics.median(walls) * 1000, 1),
        "kernel_level": "aten" if aten else "kernel",
        "kernel_ms_per_step": round(sum(v["ms"] for v in level.values()) / args.steps, 1),
        "ops": {k: {"calls_per_step": round(v["calls"] / args.steps, 1),
                    "ms_per_step": round(v["ms"] / args.steps, 1)} for k, v in ops.items()},
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"rank{rank}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
