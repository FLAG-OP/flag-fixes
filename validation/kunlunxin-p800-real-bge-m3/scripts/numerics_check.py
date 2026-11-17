#!/usr/bin/env python3
"""Numerics trajectory: fixed-seed training steps on the REAL BGE-M3 entry
blocks, recording per-step loss, parameter and gradient checksums, and memory.

Built from the same entry building blocks as the repo's own
profile_step_segments.py (init_distributed / build_retrieval_model /
build_optimizer / build_split_loaders / loss_for_batch / autocast_context) so
the objects and code paths are the trainer's. FlagGems is activated through the
real integration (flagscale.train.utils.flaggems.enable_flaggems with
flaggems_include), exactly as the patched trainer does.

Output JSON: --out. Same seed + same data order per run; compare across runs.
"""
import argparse, hashlib, json, os, statistics, time

import torch


def tensor_digest(t: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", default="/workspace/FlagScale/examples/retrieval/conf/train/bge_m3.yaml")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--flagscale-root", default="/workspace/FlagScale")
    ap.add_argument("--flaggems-include", default=None)
    ap.add_argument("--label", default="run")
    a = ap.parse_args()

    sys_path = a.flagscale_root
    import sys
    sys.path.insert(0, sys_path)
    os.environ.setdefault("FLAGSCALE_DEVICE", a.device)
    os.environ.setdefault("GEMS_VENDOR", "kunlunxin")

    from omegaconf import OmegaConf
    from flagscale.train.datasets.retrieval import build_split_loaders
    from flagscale.train.losses.retrieval import loss_for_batch
    from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
    from flagscale.models.retrieval import build_retrieval_model

    cfg = OmegaConf.load(a.conf)
    task = str(cfg.task)
    batch_rows = int(cfg.batch_size)
    accum = int(cfg.gradient_accumulation_steps)

    _rank, _world, dev, platform = init_distributed(OmegaConf.create({"device_type": a.device}))
    hw = {"use_flaggems": False}
    if a.flaggems_include:
        hw = {"use_flaggems": True,
              "flaggems_include": [s.strip() for s in a.flaggems_include.split(",") if s.strip()],
              "flaggems_required": True}
    from flagscale.train.utils.flaggems import enable_flaggems
    gems_enabled = enable_flaggems(OmegaConf.create(hw), dev, platform)
    torch.manual_seed(42)

    model, processor = build_retrieval_model(str(cfg.model.model_type), cfg.model, dev)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(cfg.optimizer, trainable, allow_adafactor=False)
    loaders = build_split_loaders(task, cfg.data, batch_rows, pin_memory=True, seed=42)
    stream = iter(loaders[0])
    dtype_name = str(cfg.model.get("torch_dtype", "bf16"))

    steps = []
    t_first = None
    for step in range(a.steps):
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        t0 = time.perf_counter()
        for _ in range(accum):
            batch = next(stream)
            with autocast_context(dev, dtype_name):
                loss = loss_for_batch(model, processor, task, batch, dev, float(cfg.temperature))
            (loss / accum).backward()
            micro_losses.append(float(loss.detach()))
        if float(cfg.optimizer.get("max_grad_norm", 0) or 0) > 0:
            torch.nn.utils.clip_grad_norm_(trainable, float(cfg.optimizer.max_grad_norm))
        optimizer.step()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        if t_first is None:
            t_first = wall
        grad_digest = hashlib.sha256(
            b"".join(bytes.fromhex(tensor_digest(p.grad)) if p.grad is not None else b"none"
                     for p in trainable)).hexdigest()[:16]
        param_digest = hashlib.sha256(
            b"".join(bytes.fromhex(tensor_digest(p)) for p in trainable)).hexdigest()[:16]
        nan_params = sum(1 for p in trainable if not torch.isfinite(p).all())
        steps.append({
            "step": step, "wall_ms": round(wall * 1000, 1),
            "micro_losses": [round(v, 6) for v in micro_losses],
            "step_loss_mean": round(statistics.fmean(micro_losses), 6),
            "grad_sha_16": grad_digest, "param_sha_16": param_digest,
            "params_nonfinite": nan_params,
        })
        print(f"step {step}: loss={steps[-1]['step_loss_mean']:.6f} wall={wall*1000:.0f}ms "
              f"param={param_digest} grad={grad_digest}", flush=True)

    out = {
        "label": a.label, "conf": a.conf, "steps": a.steps, "seed": 42,
        "batch_rows": batch_rows, "accum": accum,
        "flaggems_include": a.flaggems_include, "gems_enabled": bool(gems_enabled),
        "flagscale_root": sys_path,
        "adamw_fused_env": os.environ.get("FLAGSCALE_ADAMW_FUSED", "auto"),
        "max_memory_allocated_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "first_step_wall_ms": round(t_first * 1000, 1),
        "steps": steps,
        "param_sha_16_final": steps[-1]["param_sha_16"],
        "grad_sha_16_final": steps[-1]["grad_sha_16"],
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
