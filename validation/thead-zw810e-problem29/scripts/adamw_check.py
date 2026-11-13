"""ZW810E problem29 AdamW: stock vs fused (torch.optim.AdamW fused=True).

torch fused AdamW requires a CUDA-capable device with supported dtypes.
Vendor guard context (PR3 / adamw-fused fix): fused selection is KunlunXIN-
gated in FlagScale; on other platforms stock is default. This probe measures
whether fused AdamW even works on ZW810E and matches stock trajectories.
"""
import argparse
import json

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
args = ap.parse_args()

OUT = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0)}


def run(opt_kind, dtype, steps=5):
    g = torch.Generator(device="cuda").manual_seed(1234)
    p = torch.nn.Parameter(torch.randn([1024, 1024], generator=g, device="cuda", dtype=dtype))
    x0 = torch.randn([4096, 1024], generator=g, device="cuda", dtype=dtype)
    wref = torch.randn([1, 1024], generator=g, device="cuda", dtype=dtype)
    torch.cuda.synchronize()
    snapshots = []
    kw = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    if opt_kind == "fused":
        kw["fused"] = True
    elif opt_kind == "foreach":
        kw["foreach"] = True
    opt = torch.optim.AdamW([p], **kw)
    try:
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = ((x0 @ p - wref.expand(4096, -1)) ** 2).mean()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            snapshots.append((float(loss.detach()), p.detach().clone()))
        err = None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        torch.cuda.synchronize()
        snapshots = None
    return snapshots, err


for dtype_name in ("float32", "bfloat16"):
    dt = getattr(torch, dtype_name)
    res = {}
    snaps, err_stock = run("stock", dt)
    res["stock_error"] = err_stock
    if snaps:
        res["stock_losses"] = [s[0] for s in snaps]
        res["stock_final_param_sha_finite"] = bool(torch.isfinite(snaps[-1][1]).all().item())
    snaps_f, err_fused = run("fused", dt)
    res["fused_error"] = err_fused
    if snaps and snaps_f:
        res["fused_losses"] = [s[0] for s in snaps_f]
        res["final_param_bitwise_equal"] = bool(torch.equal(snaps[-1][1], snaps_f[-1][1]))
        res["final_param_max_abs_diff"] = float((snaps[-1][1].float() - snaps_f[-1][1].float()).abs().max())
        res["loss_final_rel_diff"] = abs(snaps[-1][0] - snaps_f[-1][0]) / max(abs(snaps[-1][0]), 1e-300)
    OUT[dtype_name] = res

with open(args.out, "w") as f:
    json.dump(OUT, f, indent=1)
print(json.dumps(OUT, indent=1))
