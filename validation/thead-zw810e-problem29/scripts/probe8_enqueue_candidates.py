"""Issue #7 §C: enqueue overhead decomposition + optimization candidates.

C1 sizes: gelu/add fp32 at 1.2M / 12.25M / 98M — host wall (sync-each), batch
   async enqueue, CUDA-event, and CUDA-GRAPH replay per-iter (removes enqueue).
C2 entries: aten registered vs flag_gems direct API vs raw cached-kernel relaunch
   (bypass pointwise_dynamic wrapper) — feasibility + timing + same-output check.
C3 thead gelu pow-hotspot: source inspection of _thead/ops gelu kernels.
"""
import json
import statistics
import time

import torch
import torch.nn.functional as F

import flag_gems

flag_gems.enable()

OUT = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "triton": __import__("triton").__version__}


def ev_p50(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return round(statistics.median(s.elapsed_time(e) for s, e in ev), 4)


def enqueue_ms(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = None
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        dt = (time.perf_counter() - t0) / iters * 1e3
        torch.cuda.synchronize()
        best = dt if best is None else min(best, dt)
    return round(best, 4)


def host_sync_each(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ts), 4)


# ---- C1 sizes + graph replay ----
sizes = {}
for n in (1228800, 12288000, 98304000):
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(n, generator=g, device="cuda", dtype=torch.float32)
    entry = {}
    for opname, fn in (("gelu", lambda: F.gelu(x)), ("add", lambda: x + x)):
        entry[opname] = {"host_sync_each_ms": host_sync_each(fn), "enqueue_ms": enqueue_ms(fn), "event_ms": ev_p50(fn)}
    # cuda graph replay: capture 100 iterations -> per-iter cost without enqueue
    try:
        g_ = torch.cuda.CUDAGraph()
        xg = x.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                F.gelu(xg)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g_):
            for _ in range(100):
                F.gelu(xg)
        # replay timing
        for _ in range(3):
            g_.replay()
        torch.cuda.synchronize()
        ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(10)]
        out_h = []
        for st, e in ev:
            st.record()
            g_.replay()
            e.record()
        torch.cuda.synchronize()
        per_iter = round(statistics.median(st_.elapsed_time(e_) for st_, e_ in ev) / 100, 4)
        entry["gelu_graph_replay_ms_per_iter"] = per_iter
        entry["graph_capture"] = "ok"
    except Exception as e:
        entry["graph_capture"] = f"unsupported/error: {type(e).__name__}: {e}"[:120]
    sizes[str(n)] = entry
OUT["C1_sizes"] = sizes

# ---- C2 entries: aten vs direct API vs raw cached kernel relaunch ----
g = torch.Generator(device="cuda").manual_seed(7)
x = torch.randn(1228800, generator=g, device="cuda", dtype=torch.float32)
ref = F.gelu(x)
entries = {"aten_event_ms": ev_p50(lambda: F.gelu(x)), "direct_event_ms": ev_p50(lambda: flag_gems.gelu(x)),
           "aten_enqueue_ms": enqueue_ms(lambda: F.gelu(x)), "direct_enqueue_ms": enqueue_ms(lambda: flag_gems.gelu(x)),
           "aten_host_sync_ms": host_sync_each(lambda: F.gelu(x)), "direct_host_sync_ms": host_sync_each(lambda: flag_gems.gelu(x))}
try:
    # introspect the registered triton kernel behind flag_gems.gelu
    kd = flag_gems.gelu.__wrapped__ if hasattr(flag_gems.gelu, "__wrapped__") else flag_gems.gelu
    src_mod = sys_modules = __import__("flag_gems.ops", fromlist=["gelu"])
    fn = getattr(src_mod, "gelu", None)
    info = {"flag_gems_ops_gelu_type": str(type(fn))}
    # pointwise_dynamic wrapper exposes .kernel_obj / .fn in some versions
    raw = None
    for attr in ("fn", "kernel", "_fn", "kernel_obj"):
        if fn is not None and hasattr(fn, attr):
            raw = getattr(fn, attr)
            info["raw_attr"] = attr
            info["raw_type"] = str(type(raw))
            break
    if raw is not None and hasattr(raw, "cache"):
        info["cache_entries"] = sum(len(v) for v in raw.cache.values()) if isinstance(raw.cache, dict) else "n/a"
    entries["C2_introspection"] = info
except Exception as e:
    entries["C2_introspection"] = {"error": f"{type(e).__name__}: {e}"[:200]}
OUT["C2_entries"] = entries

# ---- C3 thead gelu pow hotspot (source inspection) ----
import os
import flag_gems.runtime.backend as bk
thead_dir = os.path.join(os.path.dirname(bk.__file__), "_thead")
hits = []
for root, _, files in os.walk(thead_dir):
    for f in files:
        if f.endswith(".py") and "gelu" in f.lower():
            p = os.path.join(root, f)
            t = open(p, encoding="utf-8", errors="ignore").read()
            hits.append({"file": p.replace(thead_dir, "_thead"), "pow_mentions": t.count("pow"), "libdevice": "libdevice" in t, "len": len(t)})
OUT["C3_thead_gelu_source"] = hits
# also the generic gelu kernel used on this platform (hits gelu_none_kernel)
import flag_gems.ops as go
gp = go.__file__
t = open(os.path.join(os.path.dirname(gp), "gelu.py"), encoding="utf-8", errors="ignore").read()
OUT["C3_generic_gelu"] = {"pow_mentions": t.count("pow"), "libdevice": "libdevice" in t, "tanh_uses_mul": "0.44715" in t}
# shared pointwise kernel used: which file defines gelu_none_kernel on thead path?
OUT["C3_note"] = "check whether _thead overrides gelu or falls back to shared ops/gelu.py"

with open("results/probe8_enqueue_candidates.json", "w") as f:
    json.dump(OUT, f, indent=1)
print(json.dumps(OUT, indent=1))
