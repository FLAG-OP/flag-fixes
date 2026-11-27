"""Review (2026-11-22) experiment 1b: mid-sequence FFN dW comparison.
Top mid-seq dW shapes (S=167-443, from archived r36 per-shape calls), real
BF16 + transposed-view operand as in training. stock vs precontig vs
FlagGems mm per shape; per-shape device ms, TFLOP/s, reldiff, non-finite;
per-step savings weighted by actual calls; coverage of dW time reported."""
import argparse, json, statistics, sys
import torch

def timed_raw(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev=[(torch.cuda.Event(True),torch.cuda.Event(True)) for _ in range(iters)]
    for s,e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return [round(s.elapsed_time(e),4) for s,e in ev]

ap=argparse.ArgumentParser(); ap.add_argument("--shapes",required=True); ap.add_argument("--out",required=True)
a=ap.parse_args()
sys.path.insert(0,"/tmp/opencode/FlagGems-base/src")
import flag_gems
FG_MM = getattr(flag_gems, "mm", None)
sel=json.load(open(a.shapes))
torch.manual_seed(0)
rows=[]; tot_stock=0.0; tot_best=0.0; tot_ms_coverage=0.0
for i,s in enumerate(sel):
    N,M,K,calls,ms_arch = s["N"],s["M"],s["K"],s["calls_per_step"],s["ms_per_step"]
    gy=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)*0.02
    x=torch.randn(M,K,device="cuda",dtype=torch.bfloat16)*0.02
    gy_t=gy.t()
    e={"module_class":s["module_class"],"N":N,"M":M,"K":K,"calls_per_step":calls,"arch_ms_per_step":ms_arch}
    e["stock_ms"]=round(statistics.median(timed_raw(lambda: gy_t@x)),4)
    e["precontig_ms"]=round(statistics.median(timed_raw(lambda: gy.t().contiguous()@x)),4)
    e["layout_conv_ms"]=round(statistics.median(timed_raw(lambda: gy.t().contiguous())),4)
    if FG_MM is not None:
        try:
            e["fg_mm_ms"]=round(statistics.median(timed_raw(lambda: FG_MM(gy_t,x))),4)
            y=FG_MM(gy_t,x)
        except Exception as ex:
            e["fg_mm_error"]=str(ex)[:100]; e["fg_mm_ms"]=None; y=None
    ref=(gy.float().t()@x.float())
    e["stock_reldiff"]=round(((gy_t@x).float()-ref).abs().max().item()/max(ref.abs().max().item(),1e-9),6)
    e["stock_nonfinite"]=int((~torch.isfinite((gy_t@x).float())).sum())
    if y is not None:
        e["fg_reldiff"]=round((y.float()-ref).abs().max().item()/max(ref.abs().max().item(),1e-9),6)
        e["fg_nonfinite"]=int((~torch.isfinite(y.float())).sum())
        fl=2*N*M*K/1e12
        e["stock_tflops"]=round(fl/(e["stock_ms"]*1e-3),1)
        e["fg_tflops"]=round(fl/(e["fg_mm_ms"]*1e-3),1)
        e["saving_per_step_ms"]=round((e["stock_ms"]-e["fg_mm_ms"])*calls,2)
    rows.append(e)
    tot_stock+=e["stock_ms"]*calls; tot_ms_coverage+=ms_arch
    if e.get("fg_mm_ms") is not None:
        tot_best+=min(e["stock_ms"],e["fg_mm_ms"])*calls
    print(f"{s['module_class']} M{M}: stock {e['stock_ms']} fg {e.get('fg_mm_ms','-')} saving/step {e.get('saving_per_step_ms','-')}ms")
out={"shapes":rows,"coverage_note":"per-step ms from r36 archived calls for these shapes",
      "stock_total_ms_selected":round(tot_stock,2),
      "best_total_ms_selected":round(tot_best,2),
      "dW_time_coverage_ms":round(tot_ms_coverage,2),
      "saving_total_ms":round(tot_stock-tot_best,2)}
json.dump(out,open(a.out,"w"),indent=1)
print(f"selected stock {tot_stock:.2f}ms -> best {tot_best:.2f}ms | saving {out['saving_total_ms']}ms/step | dW coverage {out['dW_time_coverage_ms']}ms")
