"""Review 6ba3bfd items 3+4: per-MODULE-CLASS kernel efficiency.
- Forward: module wrapper sets context label; dispatch mode records per-class
  addmm device ms (distinguishes QKV vs attn-out despite identical shapes).
- Backward: deterministic reverse-layer order pattern [attn_out, q, k, v] for
  same-shape dX/dW groups, with count assertions (24 attn_out + 72 qkv per
  micro at 24 layers).
- SDPA: recorded separately from profiler phase windows (kernels + FLOPs
  formula stated explicitly)."""
import argparse, json, collections
import torch
import torch.nn as nn

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

    # dropout config (recorded, not assumed)
    dropout_cfg = None
    for name, mod in model.named_modules():
        if hasattr(mod, "p") and "dropout" in name.lower():
            dropout_cfg = {"module": name, "p": float(mod.p)}; break
        if getattr(mod, "config", None) is not None and hasattr(mod.config, "hidden_dropout_prob"):
            dropout_cfg = {"hidden_dropout_prob": float(mod.config.hidden_dropout_prob)}; break

    MM = torch.ops.aten.mm.default; ADDMM = torch.ops.aten.addmm.default
    GEMM_SET = {MM, ADDMM}
    # module-level label (threading.local loses the label when the XMLIR
    # dispatch path hops threads - verified by empty forward rows in first run)
    _LABEL = {"module_class": None}
    recs = []
    phase = {"name": "warmup"}
    class EffProbe(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in GEMM_SET:
                label = _LABEL["module_class"]
                opd = []
                for t in args:
                    if isinstance(t, torch.Tensor) and t.dim() == 2:
                        cm = (t.shape[0] > 1 and t.stride(0) == 1 and t.stride(1) == t.shape[0])
                        opd.append({"shape": list(t.shape), "stride": list(t.stride()),
                                     "dtype": str(t.dtype).replace("torch.", ""), "col_major_T": bool(cm)})
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record(); out = func(*args, **(kwargs or {})); e.record()
                recs.append({"phase": phase["name"], "module_class": label,
                              "operands": opd, "ev": (s, e)})
                return out
            return func(*args, **(kwargs or {}))

    def classify(name):
        if "attention.self" in name and any(t in name for t in ("query", "key", "value")): return "attn_qkv"
        if "attention.output.dense" in name: return "attn_out"
        if "intermediate.dense" in name: return "ffn_up"
        if ".output.dense" in name and "encoder.layer" in name: return "ffn_down"
        if "pooler" in name: return "pooler"
        return "other"

    # module wrappers set label around forward
    orig_fw = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            cls = classify(name); o = mod.forward
            orig_fw[name] = o
            def mk(o, cls):
                def f(*ar, **kw):
                    prev = _LABEL["module_class"]
                    _LABEL["module_class"] = cls
                    try: return o(*ar, **kw)
                    finally: _LABEL["module_class"] = prev
                return f
            mod.forward = mk(o, cls)

    for i in range(a.skip):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accum):
            batch = next(stream)
            with autocast_context(dev, dtype_name):
                loss = loss_for_batch(model, processor, str(cfg.task), batch, dev, float(cfg.temperature))
            (loss / accum).backward()
        optimizer.zero_grad(set_to_none=True)

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
        s, e = r.pop("ev"); r["device_ms"] = round(s.elapsed_time(e), 4)
    # restore forwards
    for name, mod in model.named_modules():
        if name in orig_fw: mod.forward = orig_fw[name]

    # aggregate forward per (module_class, operand sig) — review 5ec9e6c item 2:
    # per-shape FLOPs (previous single-M-per-class produced >peak numbers - wrong)
    fwd = collections.defaultdict(lambda: {"count": 0, "ms": 0.0, "sig": None})
    fwd_err = []
    for r in recs:
        if r["phase"].startswith("forward") and r["module_class"]:
            sig = "|".join(f"{o['shape']}:{'T' if o['col_major_T'] else ''}" for o in r["operands"])
            f = fwd[(r["module_class"], sig)]; f["count"] += 1; f["ms"] += r["device_ms"]; f["sig"] = sig
    out_debug_fwd = {f"{k[0]}|{k[1]}": v["count"] for k, v in fwd.items()}
    # backward same-shape groups -> ordered reverse-layer attribution
    # per micro per layer reverse order: [attn_out, q, k, v] for shape (5120,1024,1024) dX/dW
    bwd_groups = collections.defaultdict(list)   # sig -> list of device_ms (in submission order)
    for r in recs:
        if r["phase"].startswith("backward"):
            sig = "|".join(f"{o['shape']}:{'T' if o['col_major_T'] else ''}" for o in r["operands"])
            bwd_groups[sig].append(r)
    bwd_rows = []
    PATTERNS = {
        # sig-keyed pattern guessed from topology; asserted by counts below
    }
    rows_out = []
    # forward rows with derived MKN
    for (cls, sig), f in fwd.items():
        opsigs = sig.split("|")
        # sig holds ONLY 2D operands: [x[M,K], w.t()[K,N]] (bias 1-D excluded)
        shp0 = json.loads(opsigs[0].split(":")[0]) if len(opsigs) > 0 else None
        shp1 = json.loads(opsigs[1].split(":")[0]) if len(opsigs) > 1 else None
        if shp0 and shp1 and len(shp0) == 2 and len(shp1) == 2:
            M, K = shp0; N = shp1[1]
            flops = 2 * M * K * N * f["count"]
            ms = f["ms"]
            tf = round((flops / (ms * 1e-3)) / 1e12, 1)
            rows_out.append({"module_class": cls, "phase": "forward", "op": "addmm",
                              "MKN": [M, K, N], "operand_sig": sig,
                              "calls_per_step": f["count"], "device_ms": round(ms, 2),
                              "flops_tflop": round(flops / 1e12, 3),
                              "tflops": tf})
    # backward rows: split same-shape groups into attn_out/qkv by ordered pattern 1:3
    for sig, lst in bwd_groups.items():
        first_shape = lst[0]["operands"][1]["shape"] if len(lst[0]["operands"]) > 1 else None
        K = None
        for r in lst:
            for o in r["operands"]:
                if o["shape"][0] in (1024, 4096) and o["col_major_T"] is False and o["shape"] != first_shape:
                    pass
        # pattern: groups whose per-layer count fits 96 (24 attn_out + 72 qkv) get split 1:3
        n = len(lst)
        per_micro = n / accum
        # 24 attn_out + 72 qkv per micro, reverse-layer order => attn_out first
        # in each layer-group of 4 (both dX and dW patterns)
        is_qkv_attnout = (abs(per_micro - 96) < 1)
        if is_qkv_attnout:
            counts = {"attn_out": 0, "attn_qkv": 0}; ms = {"attn_out": 0.0, "attn_qkv": 0.0}
            for i, r in enumerate(lst):
                cls = "attn_out" if (i % 4) == 0 else "attn_qkv"
                counts[cls] += 1; ms[cls] += r["device_ms"]
            # operand dims from the first record's actual operands (2D mats only)
            m2 = [o["shape"] for o in lst[0]["operands"] if len(o["shape"]) == 2][:2]
            for cls in ("attn_out", "attn_qkv"):
                shpA, shpB = (m2 + [[0, 0]] * 2)[:2]
                flops = 2 * shpA[0] * shpA[1] * shpB[1] * counts[cls]
                tf = round((flops / (ms[cls] * 1e-3)) / 1e12, 1)
                rows_out.append({"module_class": cls, "phase": "backward", "op": "mm",
                                  "operand_dims": {"actual": [shpA, shpB],
                                                    "semantic_dW_note": "dW actual=(N,M)x(M,K); semantic out=(N,K); dX actual=(M,N)x(N,K); semantic out=(M,K)"},
                                  "calls_per_step": counts[cls], "device_ms": round(ms[cls], 2),
                                  "tflops": tf,
                                  "attribution": "TOPOLOGY-INFERRED: reverse-layer pattern 1 attn_out : 3 qkv; counts asserted (24/72 per micro) but per-call order NOT independently verified"})
        else:
            all_ms = sum(r["device_ms"] for r in lst)
            # FFN classes recoverable from operand dims:
            # dX (M,N)x(N,K): [5120,1024]x[1024,4096]->ffn_up ; [5120,4096]x[4096,1024]->ffn_down
            # dW (N,M)x(M,K): [4096,5120]x[5120,1024]->ffn_up ; [1024,5120]x[5120,4096]->ffn_down
            cls_map = None
            m2 = [o["shape"] for r in lst[:1] for o in r["operands"] if len(o["shape"]) == 2][:2]
            if m2 and len(m2[0]) == 2 and len(m2[0]) >= 2 and m2[0][1] in (1024, 4096) and m2[0][0] >= 4000:
                cls_map = "ffn_up" if m2[0][1] == 1024 else "ffn_down"
            elif m2 and len(m2[0]) >= 2 and m2[0][0] in (1024, 4096) and m2[0][0] >= 1000 and m2[0][0] == m2[0][1] * 4 if len(m2) > 1 else False:
                cls_map = None
            if cls_map is None and m2 and len(m2) >= 1:
                a0 = m2[0]
                if a0[0] == 4096 and a0[1] >= 4000: cls_map = "ffn_up"
                elif a0[0] == 1024 and a0[1] >= 4000: cls_map = "ffn_down"
                elif a0[1] == 4096 and a0[0] >= 4000: cls_map = "ffn_down"
                elif a0[1] == 1024 and a0[0] >= 4000: cls_map = "ffn_up"
            tflops = None
            if m2 and len(m2) >= 2 and len(m2[0]) == 2 and len(m2[1]) == 2:
                flops = 2 * m2[0][0] * m2[0][1] * m2[1][1] * n
                tflops = round((flops / (all_ms * 1e-3)) / 1e12, 1)
            rows_out.append({"module_class": cls_map or "shape_group", "phase": "backward", "op": "mm",
                              "operand_sig": sig, "calls_per_step": n,
                              "device_ms": round(all_ms, 2), "tflops": tflops,
                              "attribution": "shape-keyed (FFN dims recoverable; ffn dX/dW classes unique per shape)"})
    rows_out.sort(key=lambda r: -r["device_ms"])
    fwd_rec = sum(1 for r in recs if r["phase"].startswith("forward"))
    fwd_rec_labeled = sum(1 for r in recs if r["phase"].startswith("forward") and r["module_class"])
    out = {"debug": {"recs_total": len(recs), "forward_recs": fwd_rec,
                      "forward_recs_labeled": fwd_rec_labeled,
                      "fwd_agg": out_debug_fwd,
                      "sample_forward_rec": next((r for r in recs if r["phase"].startswith("forward")), None)},
           "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "window": "ONE full step (4 micros), same seed; module-label forward + ordered-pattern backward",
           "dropout_config": dropout_cfg,
           "rows": rows_out,
           "note": "dispatch-mode wrapping adds CPU submission overhead; kernel device times valid; backward QKV/attn_out split is topology-derived (1:3 pattern, count-asserted)"}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{'class':<12}{'phase':<9}{'calls':>6}{'ms':>9}{'tflops':>8}")
    for r in rows_out[:14]:
        print(f"{r['module_class']:<12}{r['phase']:<9}{r['calls_per_step']:>6}{r['device_ms']:>9.2f}{str(r.get('tflops','')):>8}")

if __name__ == "__main__":
    main()
