"""Review (2026-11-22) experiment 1: FFN dW per-shape stock vs candidates.
Real shapes from r36 trace (S=512 dominant): dW = gy.t()[N,M] @ x[M,K].
Variants: stock transposed-view; stock pre-contiguous (conversion timed
separately); candidate flag_gems.mm. Reports device ms, TFLOP/s, grad error
vs fp32, layout-conversion cost, and per-step saving estimate."""
import argparse, json, statistics, sys
import torch
import torch.nn.functional as F

def timed_raw(fn, iters=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev=[(torch.cuda.Event(True),torch.cuda.Event(True)) for _ in range(iters)]
    for s,e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return [round(s.elapsed_time(e),4) for s,e in ev]

ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); a=ap.parse_args()
import sys
sys.path.insert(0,"/tmp/opencode/FlagGems-base/src")
import flag_gems
FG_MM = getattr(flag_gems, "mm", None)

torch.manual_seed(0)
DEV, DT = "cuda", torch.bfloat16
M = 5120   # B*S flattened (S=512 dominant)
shapes = {"ffn_up_dW": (4096, M, 1024), "ffn_down_dW": (1024, M, 4096)}  # (N, M, K)
CALLS_PER_STEP = 96  # per class (24 modules x 4 micros)
out = {"device": torch.cuda.get_device_name(0), "dtype": "bf16",
        "M": M, "calls_per_step_per_class": CALLS_PER_STEP,
        "candidates": {"stock_transposed_view": "gy.t() @ x (as in training)",
                        "stock_precontig": "gy.t().contiguous() @ x (conversion timed separately)",
                        "flaggems_mm": "flag_gems.mm(gy.t(), x)"},
        "shapes": {}}

for name, (N, Mm, K) in shapes.items():
    gy = torch.randn(Mm, N, device=DEV, dtype=DT) * 0.02   # [M,N]
    x = torch.randn(Mm, K, device=DEV, dtype=DT) * 0.02    # [M,K]
    gy_t = gy.t()                                           # [N,M] transposed view
    e = {}
    e["stock_ms"] = statistics.median(timed_raw(lambda: gy_t @ x))
    def precontig():
        g = gy.t().contiguous()
        return g @ x
    def conv_only():
        return gy.t().contiguous()
    e["stock_precontig_ms"] = statistics.median(timed_raw(precontig))
    e["layout_conversion_ms"] = statistics.median(timed_raw(conv_only))
    if FG_MM is not None:
        try:
            e["fg_mm_ms"] = statistics.median(timed_raw(lambda: FG_MM(gy_t, x)))
            y_fg = FG_MM(gy_t, x)
        except Exception as ex:
            e["fg_mm_error"] = str(ex)[:120]
    # correctness vs fp32
    ref = (gy.float().t() @ x.float())
    e["stock_vs_ref_reldiff"] = round(((gy_t @ x).float()-ref).abs().max().item()/ref.abs().max().item(), 6)
    if FG_MM is not None and "fg_mm_ms" in e:
        e["fg_mm_vs_ref_reldiff"] = round((y_fg.float()-ref).abs().max().item()/ref.abs().max().item(), 6)
    flops_call = 2 * N * Mm * K / 1e12          # TFLOP per call
    e["flops_tflop_per_step"] = round(flops_call * CALLS_PER_STEP, 3)
    e["stock_tflops"] = round(flops_call / (e["stock_ms"] * 1e-3), 1)
    e["stock_precontig_tflops_excl_conv"] = round(flops_call / (e["stock_precontig_ms"] * 1e-3), 1)
    if "fg_mm_ms" in e:
        e["fg_mm_tflops"] = round(flops_call / (e["fg_mm_ms"] * 1e-3), 1)
        e["per_step_saving_ms_vs_stock"] = round((e["stock_ms"] - e["fg_mm_ms"]) * CALLS_PER_STEP, 1)
    e["per_step_saving_ms_precontig"] = round((e["stock_ms"] - e["stock_precontig_ms"]) * CALLS_PER_STEP, 1)
    out["shapes"][name] = e
    print(f"{name}: stock {e['stock_ms']}ms ({e['stock_tflops']} TF/s) | precontig {e['stock_precontig_ms']}ms | "
          f"fg_mm {e.get('fg_mm_ms','ERR')}ms ({e.get('fg_mm_tflops','-')} TF/s) | "
          f"saving(fg) {e.get('per_step_saving_ms_vs_stock','-')}ms/step | reldiff fg {e.get('fg_mm_vs_ref_reldiff','-')}")
json.dump(out, open(a.out, "w"), indent=1)
print("saved", a.out)
