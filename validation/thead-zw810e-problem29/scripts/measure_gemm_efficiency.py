"""Issue #7 review (6d1a05b) item 2+4: SAME-WINDOW GEMM efficiency.
One full training step (4 micros, same seed/loader); each aten mm/addmm is
wrapped with CUDA event pairs INSIDE a fixed-matching TorchDispatchMode ->
per (phase, operand shapes) calls + measured device ms -> per-shape TFLOP/s
(2*MKN). Classifies backward mm into dX vs dW by operand shapes.
NOTE: CPU-side dispatch-mode wrapping adds submission overhead; kernel device
times are unaffected (device timeline), absolute step ms NOT comparable to the
clean trace. Use for per-kernel efficiency only."""
import argparse, json, collections
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    import random, os, statistics
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

    MM = torch.ops.aten.mm.default; ADDMM = torch.ops.aten.addmm.default
    GEMM_SET = {MM, ADDMM}
    recs = []  # (phase, shapes+strides+dtypes, ev_pair)
    phase = {"name": "warmup"}
    class EffProbe(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in GEMM_SET:
                opd = []
                for t in args:
                    if isinstance(t, torch.Tensor):
                        cm = (t.dim() == 2 and t.shape[0] > 1 and t.stride(0) == 1 and t.stride(1) == t.shape[0])
                        opd.append({"shape": list(t.shape), "stride": list(t.stride()),
                                     "dtype": str(t.dtype).replace("torch.", ""),
                                     "col_major_transposed": bool(cm)})
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record()
                out = func(*args, **(kwargs or {}))
                e.record()
                recs.append({"phase": phase["name"], "op": str(func).replace("aten.", "").replace(".default", ""),
                              "operands": opd, "ev": (s, e)})
                return out
            return func(*args, **(kwargs or {}))

    for i in range(a.skip):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accum):
            batch = next(stream)
            with autocast_context(dev, dtype_name):
                loss = loss_for_batch(model, processor, str(cfg.task), batch, dev, float(cfg.temperature))
            (loss / accum).backward()
        optimizer.zero_grad(set_to_none=True)

    # measured step
    optimizer.zero_grad(set_to_none=True)
    for micro in range(accum):
        batch = next(stream)
        with autocast_context(dev, dtype_name):
            phase["name"] = f"forward_m{micro}"
            with EffProbe():
                loss = loss_for_batch(model, processor, str(cfg.task), batch, dev, float(cfg.temperature))
        phase["name"] = f"backward_m{micro}"
        with EffProbe():
            (loss / accum).backward()
    torch.cuda.synchronize()
    for r in recs:
        s, e = r.pop("ev")
        r["device_ms"] = round(s.elapsed_time(e), 4)

    # aggregate per (phase_kind, op, operand signature)
    agg = collections.defaultdict(lambda: {"count": 0, "ms": 0.0})
    for r in recs:
        ph = "forward" if r["phase"].startswith("forward") else "backward"
        sig = "|".join(f"{o['shape']}:{o['dtype']}{'T' if o['col_major_transposed'] else ''}" for o in r["operands"])
        key = (ph, r["op"], sig)
        agg[key]["count"] += 1
        agg[key]["ms"] += r["device_ms"]
    rows = []
    for (ph, op, sig), v in sorted(agg.items(), key=lambda kv: -kv[1]["ms"]):
        opd = [json.loads("[" + seg.split("|", 0)[0] + "]") for seg in []]  # placeholder
        shapes = []
        for seg in sig.split("|"):
            sh = seg.split(":")[0]
            shapes.append(json.loads(sh.replace("x", ",").strip("[]")) if False else seg)
        rows.append({"phase": ph, "op": op, "operand_sig": sig,
                      "count_per_step": v["count"], "device_ms_per_step": round(v["ms"], 3)})
    # derive TFLOP/s per row from first two 2D operand shapes
    def parse_shape(s):
        return json.loads(s)
    for r in rows:
        parts = []
        for seg in r["operand_sig"].split("|"):
            shp, rest = seg.split(":", 1)
            parts.append(json.loads(shp))
        m2 = [p for p in parts if len(p) == 2]
        if len(m2) >= 2:
            A, B = m2[0], m2[1]
            K = A[1] if A[1] == B[0] else A[0]
            flops = 2 * A[0] * A[1] * B[1]
            ms = r["device_ms_per_step"] / max(1, r["count_per_step"])
            r["derived_MKN"] = [A[0], A[1], B[1]]
            r["tflops_per_call"] = round((flops / (ms * 1e-3)) / 1e12, 1) if ms > 0 else None
    out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "window": "ONE full training step (4 micros), same seed/loader as r28/r32",
           "note": "dispatch-mode wrapping adds CPU submission overhead - kernel device times valid, step wall NOT comparable to clean trace; TFLOP/s per (phase,shape) same-window",
           "rows": rows,
           "peak_tflops": 139.2, "vendor_reference_tflops": 135.7}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{'phase':<9}{'op':<7}{'calls':>6}{'ms':>9}{'MKN':>18}{'TFLOP/s':>9}")
    for r in rows[:14]:
        print(f"{r['phase']:<9}{r['op']:<7}{r['count_per_step']:>6}{r['device_ms_per_step']:>9.3f}"
              f"{str(r.get('derived_MKN','')):>18}{str(r.get('tflops_per_call','')):>9}")

if __name__ == "__main__":
    main()
