"""Issue #7 review (bd79ae9) item 1: GEMM operand evidence per phase.
DELIVERED: per-phase aten::mm/addmm COUNTS from real BGE-M3 training windows
(forward-only vs backward-only) + M/K/N/dtype/transpose DERIVED per F.linear
semantics, tied to the r29 hotspot classes.
LIMITATION (documented, verified by two probes): on this PPU stack
(a) TorchDispatchMode captures 0 mm/addmm (XMLIR intercepts at its own
dispatch key - ops never reach bottom-of-stack mode), and
(b) profiler record_shapes leaves input_shapes=None.
Direct operand stride/dtype dump requires vendor/runtime support (queued)."""
import argparse, json, collections
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--skip", type=int, default=2)
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
    from torch.profiler import profile, ProfilerActivity

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

    # verify the limitation claims on-device (recorded into output)
    probe = {"torchdispatch_mm_calls": None, "profiler_shapes": None}

    class M(torch.utils._python_dispatch.TorchDispatchMode):
        def __init__(self): self.n = 0
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if str(func) in ("aten::mm", "aten::addmm"): self.n += 1
            return func(*args, **(kwargs or {}))
    batch = next(stream)
    with autocast_context(dev, dtype_name):
        with M() as m:
            loss = loss_for_batch(model, processor, str(cfg.task), batch, dev, float(cfg.temperature))
    probe["torchdispatch_mm_calls"] = m.n
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as p2:
        loss2 = loss_for_batch(model, processor, str(cfg.task), batch, dev, float(cfg.temperature))
    probe["profiler_shapes"] = [ev.input_shapes is not None for ev in p2.key_averages()][:5]
    del loss, loss2

    def window(with_fwd, with_bwd):
        optimizer.zero_grad(set_to_none=True)
        prof = profile(activities=[ProfilerActivity.CPU]) if (with_fwd or with_bwd) else None
        for micro in range(accum):
            batch = next(stream)
            if prof and with_fwd: prof.start()
            with autocast_context(dev, dtype_name):
                loss = loss_for_batch(model, processor, str(cfg.task), batch, dev, float(cfg.temperature))
            if prof and with_fwd: prof.stop()
            if prof and with_bwd: prof.start()
            (loss / accum).backward()
            if prof and with_bwd: prof.stop()
        torch.cuda.synchronize()
        if prof is None: return {}
        c = collections.Counter()
        for ev in prof.key_averages():
            if ev.key in ("aten::mm", "aten::addmm"):
                c[ev.key] += ev.count
        return dict(c)

    for i in range(a.skip):
        batch = next(stream)
        with autocast_context(dev, dtype_name):
            loss = loss_for_batch(model, processor, str(cfg.task), batch, dev, float(cfg.temperature))
        (loss / accum).backward()
        optimizer.zero_grad(set_to_none=True)

    fwd_counts = window(True, False)     # forward-only window
    bwd_counts = window(False, True)     # backward-only window
    both = window(True, True)

    # derivation tied to r29 hotspot classes (per ONE micro; step = 4 micros)
    # S=512 dominant row: M=5120
    deriv = [
        {"class": "ffn_up", "fwd": "addmm(bias[5120,4096], x[5120,1024], w.t()[1024,4096] col-major=TRANSPOSED operand)",
         "dX": "mm(gy[5120,4096], w[4096,1024])", "dW": "mm(gy.t()[4096,5120], x[5120,1024]) -> reduce to [4096,1024]",
         "dtype": "bf16 operands (autocast; weights native bf16)", "calls_per_micro": 24},
        {"class": "ffn_down", "fwd": "addmm(bias[5120,1024], x[5120,4096], w.t()[4096,1024] TRANSPOSED)",
         "dX": "mm(gy[5120,1024], w[1024,4096])", "dW": "mm(gy.t()[1024,5120], x[5120,4096]) -> [1024,4096]",
         "dtype": "bf16", "calls_per_micro": 24},
        {"class": "attn_qkv", "fwd": "addmm(bias[5120,1024], x[5120,1024], w.t()[1024,1024] TRANSPOSED)",
         "dX": "mm(gy[5120,1024], w[1024,1024])", "dW": "mm(gy.t()[1024,5120], x[5120,1024])",
         "dtype": "bf16", "calls_per_micro": 72},
        {"class": "attn_out", "fwd": "addmm(bias[5120,1024], x[5120,1024], w.t()[1024,1024] TRANSPOSED)",
         "dX": "mm(gy[5120,1024], w[1024,1024])", "dW": "mm(gy.t()[1024,5120], x[5120,1024])",
         "dtype": "bf16", "calls_per_micro": 24},
    ]
    out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "limitation_verified_on_device": probe,
           "limitation_note": "TorchDispatchMode sees 0 mm/addmm (XMLIR dispatch-key interception); profiler record_shapes leaves input_shapes=None on this build. Operand stride/transpose/dtype below are DERIVED from F.linear semantics + native dtypes; direct dump requires vendor/runtime support (queued).",
           "measured_counts_per_micro": {"forward_window": fwd_counts, "backward_window": bwd_counts,
                                          "both_window": both},
           "measured_counts_per_step_x4": {"forward_addmm": fwd_counts.get("aten::addmm", 0) * 4,
                                            "backward_mm": bwd_counts.get("aten::mm", 0) * 4},
           "derived_operands_s512": deriv}
    json.dump(out, open(a.out, "w"), indent=1)
    print("torchdispatch calls (fwd, 1 micro):", probe["torchdispatch_mm_calls"])
    print("fwd window:", fwd_counts, "| bwd window:", bwd_counts)
    print("per step (x4 micro): fwd addmm", out["measured_counts_per_step_x4"]["forward_addmm"],
          "| bwd mm", out["measured_counts_per_step_x4"]["backward_mm"])

if __name__ == "__main__":
    main()
