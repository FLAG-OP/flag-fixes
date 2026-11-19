"""Issue #7 exploration (comment 7011219007) item 2: single-layer replay
native F.linear vs FlagOS-OP PR#35 candidate (matmul_triton + explicit bias).
Real captured layer: FFN intermediate w=[4096,1024], in=[10,S,1024] bf16,
20 calls/optimizer step (4 micro x 5 groups). Reports fwd/dX/dW/bias timing,
correctness vs fp32 ref with declared tolerance, and registration hit-counts.
NOTE: candidate computes x @ w.t() only; bias add is a separate op (counted).
"""
import argparse, json, statistics, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/tmp/opencode/FlagOS-OP")
import ops.matmul.kernel.triton_level as _tl
# import-path shim: candidate expects flag_gems.runtime.backend.device.DeviceDetector
# (newer layout); local flag_gems v5.3.5 exposes DeviceDetector at the top level.
# Patch returns the TRUE vendor for this device via the available API.
from flag_gems.runtime import DeviceDetector as _DD
_tl._runtime_vendor = lambda: _DD().vendor_name
from ops.matmul.kernel.triton_level import matmul_triton, matmul_backward_triton

def timed_raw(fn, iters=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev=[(torch.cuda.Event(True),torch.cuda.Event(True)) for _ in range(iters)]
    for s,e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return [round(s.elapsed_time(e),4) for s,e in ev]

ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True)
ap.add_argument("--seqs", default="512,256,128")  # captured S values
a=ap.parse_args()

torch.manual_seed(0)
DEV="cuda"; DT=torch.bfloat16
RES={"device":torch.cuda.get_device_name(0),"dtype":"bf16",
     "layer":"encoder.intermediate.dense","w":[4096,1024],
     "calls_per_step":20,"note":"candidate has no fused bias; bias add counted separately",
     "seq_shapes":{}}
TOL={"rule":"per-position |err| <= 4*ulp_bf16(max(|ref|, floor)); floor=0.01*max|ref|",
     "floor_factor":0.01,"ulp_factor":4}

import math
def ulp_bf16(r):
    """ulp of bf16 grid at fp32 values r (cast to bf16, spacing to next bf16)."""
    rb = r.to(torch.bfloat16)
    bits = (rb.view(torch.int16).to(torch.int32) + 1).to(torch.int16)
    up = bits.view(torch.bfloat16).float()
    ulp = (up - rb.float()).abs()
    return torch.where(rb.float() == 0, torch.full_like(ulp, 2.0**-133), ulp)

def tol_check(out, ref):
    r=ref.float(); x=out.float()
    floor=TOL["floor_factor"]*r.abs().max()
    tau=4*torch.maximum(ulp_bf16(torch.maximum(r.abs(),torch.full_like(r,floor))),
                        ulp_bf16(torch.full_like(r,floor)))
    err=(x-r).abs()
    return {"max_abs_err":round(err.max().item(),6),
            "violations":int((err>tau).sum()),"positions":err.numel()}

for S in [int(s) for s in a.seqs.split(",")]:
    B=10
    x=torch.randn(B,S,1024,device=DEV,dtype=DT)*0.02
    w=torch.randn(4096,1024,device=DEV,dtype=DT)*0.02
    b=torch.randn(4096,device=DEV,dtype=DT)*0.02
    gy=torch.randn(B,S,4096,device=DEV,dtype=DT)*0.02
    entry={}
    # ---- native ----
    xn=x.detach().requires_grad_(); wn=w.detach().requires_grad_(); bn=b.detach().requires_grad_()
    def native_fwd(): return F.linear(x, w, b)
    y_native= native_fwd()
    def native_bwd():
        g=torch.autograd.grad(F.linear(xn,wn,bn), (xn,wn), gy, retain_graph=True)
        return g
    g_native=native_bwd()
    entry["native_fwd_ms"]=statistics.median(timed_raw(native_fwd))
    entry["native_bwd_ms"]=statistics.median(timed_raw(native_bwd))
    # ---- candidate ----
    wt=w.t().contiguous()   # candidate needs b as [K,N]; layout conversion counted
    def cand_fwd():
        y=matmul_triton(x, w.t()) + b   # x[B,S,K] @ w.t()[K,N]
        return y
    y_cand=cand_fwd()
    def cand_bwd():
        # dX = gy @ w  -> matmul_triton(gy, w.t().t()=w) ; dW = sum_b gy^T @ x handled by backward impl
        gxa, gwb = matmul_backward_triton(x, w.t(), gy, True, True)
        return gxa, gwb
    gxa_c, gwb_c = cand_bwd()   # gwb_c is grad wrt w.t() i.e. [K,N] -> dW = gwb_c.t()
    entry["cand_fwd_ms"]=statistics.median(timed_raw(cand_fwd))
    entry["cand_bwd_ms"]=statistics.median(timed_raw(cand_bwd))
    def cand_bias(): return gy.sum(dim=(0,1),dtype=torch.bfloat16)
    entry["cand_bias_grad_ms"]=statistics.median(timed_raw(cand_bias))
    # layout conversion cost (w.t().contiguous) — amortized but counted
    entry["cand_layout_precontig_ms"]=statistics.median(timed_raw(lambda: w.t().contiguous()))
    # ---- correctness vs fp32 ----
    ref32=(x.float()@w.float().t()+b.float())
    dx_ref=(gy.float()@w.float())
    dw_ref=(gy.float().reshape(-1,4096).t()@x.float().reshape(-1,1024))
    entry["y_vs_ref"]=tol_check(y_cand, ref32)
    entry["y_native_vs_ref"]=tol_check(y_native, ref32)
    entry["dx_cand_vs_ref"]=tol_check(gxa_c, dx_ref)
    entry["dx_native_vs_ref"]=tol_check(g_native[0], dx_ref)
    entry["dw_cand_vs_ref"]=tol_check(gwb_c.t(), dw_ref)
    entry["dw_native_vs_ref"]=tol_check(g_native[1], dw_ref)
    entry["fwd_total_cand"]=round(entry["cand_fwd_ms"]+entry["cand_bias_grad_ms"]*0,4)
    RES["seq_shapes"][f"B{B}_S{S}"]=entry
    print(f"B{B} S{S}: native fwd {entry['native_fwd_ms']} bwd {entry['native_bwd_ms']} | "
          f"cand fwd {entry['cand_fwd_ms']} bwd {entry['cand_bwd_ms']} bias-g {entry['cand_bias_grad_ms']} "
          f"layout {entry['cand_layout_precontig_ms']} | viol y/dx/dw: "
          f"{entry['y_vs_ref']['violations']}/{entry['dx_cand_vs_ref']['violations']}/{entry['dw_cand_vs_ref']['violations']}")

# registration mismatch evidence (from linear_capture_r23.json)
try:
    cap=json.load(open("/tmp/opencode/zw810e-problem29/results/linear_capture_r23.json"))
    RES["aten_hits_one_step"]=cap["aten_hits"]
except Exception: pass
json.dump(RES, open(a.out,"w"), indent=1)
print("saved", a.out)
