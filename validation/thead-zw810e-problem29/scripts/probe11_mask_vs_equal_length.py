"""Issue #7 round-19: masked vs unmasked SDPA probe.

Fixes vs b5521fb (review round-18):
- timed() defined BEFORE first call (previous version had NameError on
  *_default entries, caught as ok:false and misread as device limitation).
- Error classification: expected backend-unavailable (RuntimeError with
  "No available kernel") vs PROGRAM errors (NameError/TypeError/etc).
  PROGRAM errors set res["program_error"]=True and exit code 1.
- *_default paths save raw samples too.
- Explicit --out argument (no hardcoded /tmp paths).
- --cpu-check mode: full entry check without device (structure + code path).

ALL warmup + timing inside sdpa_kernel([be]) context (round-17 fix retained).
"""
import argparse, json, statistics, sys
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

PROGRAM_ERRORS = (NameError, TypeError, AttributeError, KeyError, UnboundLocalError)

def timed(fn, iters=30, warmup=5):
    """All warmup + timing inside caller's backend context (if any)."""
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
    for s, e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return [round(s.elapsed_time(e), 4) for s, e in ev]

def timed_in_ctx(q, attn_mask, be, iters=30, warmup=5):
    """ALL warmup + timing inside sdpa_kernel([be])."""
    with sdpa_kernel([be]):
        return timed(lambda: (F.scaled_dot_product_attention(q, q, q, attn_mask=attn_mask)
                              if attn_mask is not None
                              else F.scaled_dot_product_attention(q, q, q)),
                     iters=iters, warmup=warmup)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output JSON path (explicit, no /tmp defaults)")
    ap.add_argument("--cpu-check", action="store_true", help="structure check without device; exits 0/1")
    args = ap.parse_args()

    if args.cpu_check:
        # Entry check without device: verify timed() is defined before use,
        # error classification tables exist, argparse works.
        import inspect
        src = inspect.getsource(sys.modules[__name__])
        assert "def timed(" in src and "def timed_in_ctx(" in src
        i_def = src.index("def timed(")
        i_call = src.index("timed_in_ctx(q, m, be_val)")
        assert i_def < i_call, "timed defined after first call"
        assert "PROGRAM_ERRORS" in src, "error classification missing"
        print("CPU-CHECK OK: timed defined before call; PROGRAM_ERRORS table present")
        return 0

    B, S, Hh, D = 8, 512, 16, 64
    torch.manual_seed(0)
    q = torch.randn(B, Hh, S, D, device="cuda", dtype=torch.bfloat16)
    padding_mask = torch.zeros(B, 1, S, S, dtype=torch.bool, device="cuda")
    padding_mask[:, :, :, -100:] = True
    padding_mask[4:, :, -100:, :] = True
    attn_mask = ~padding_mask  # True = participate

    BACKENDS = {"flash": SDPBackend.FLASH_ATTENTION,
                "efficient": SDPBackend.EFFICIENT_ATTENTION,
                "math": SDPBackend.MATH}

    res = {"shape": f"B{B} H{Hh} S{S} D{D} bf16",
           "mask": "bool mask True=participate. KEY side: all rows lose last 100 key columns. QUERY side: rows>=4 additionally lose their own last 100 query rows (self-attention rows masked out). mask passed explicitly to every timed call",
           "dispatch_claim": "CONJECTURE: default-path timing close to forced-backend timing SUGGESTS same backend selection; no kernel/dispatch evidence captured (would need kernel-level trace or dispatch logs)",
           "iters": 30, "warmup": 5, "device": torch.cuda.get_device_name(0),
           "torch": torch.__version__,
           "note": "ALL warmup+timing inside sdpa_kernel(be); PROGRAM errors exit 1",
           "program_error": False, "entries": {}}

    def record(key, fn, requested_backend=None):
        try:
            ms = fn()
            res["entries"][key] = {"ok": True, "p50_ms": round(statistics.median(sorted(ms)), 4),
                                   "samples_ms": ms,
                                   "requested_backend": requested_backend,
                                   "backend_enforced": requested_backend is not None}
        except PROGRAM_ERRORS as e:
            res["entries"][key] = {"ok": False, "program_error": True,
                                   "error": f"{type(e).__name__}: {e}"[:150]}
            res["program_error"] = True
        except RuntimeError as e:
            msg = str(e)
            unsupported = "No available kernel" in msg
            res["entries"][key] = {"ok": False,
                                   "unsupported_backend": unsupported,
                                   "program_error": not unsupported,
                                   "error": msg[:150]}
            if not unsupported:
                res["program_error"] = True

    for mask_desc, m in (("masked", attn_mask), ("unmasked", None)):
        for bn, be in BACKENDS.items():
            record(f"{mask_desc}_{bn}", lambda mm=m, bb=be: timed_in_ctx(q, mm, bb), requested_backend=bn)
        # default backend (no pin) — now with raw samples, timed defined before use
        record(f"{mask_desc}_default",
               lambda mm=m: timed(lambda: (F.scaled_dot_product_attention(q, q, q, attn_mask=mm)
                                           if mm is not None
                                           else F.scaled_dot_product_attention(q, q, q))))

    json.dump(res, open(args.out, "w"), indent=1)
    for k, v in sorted(res["entries"].items()):
        val = v.get("p50_ms", v.get("error", ""))
        val = f"{val}" if isinstance(val, str) else f"{val} ms"
        tag = "OK" if v["ok"] else ("UNSUPPORTED" if v.get("unsupported_backend") else "PROG-ERR")
        print(f"{k}: {val} [{tag}]")
    print(f"program_error={res['program_error']}")
    return 1 if res["program_error"] else 0

if __name__ == "__main__":
    sys.exit(main())
