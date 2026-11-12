"""Regression: fused AdamW must match stock AdamW within bf16 rounding.

Runs the same 5-step gradient sequence through
``flagscale.train.train_retrieval.build_optimizer`` (the candidate) and a
directly-constructed stock AdamW, comparing parameters after every step, for
bf16 and fp32 parameter sets.  Measured on the KunlunXIN stack: max |diff|
7.6e-5 for bf16 (one ulp at parameter scale), ~1e-8 for fp32.
"""

import sys

import torch

STEPS = 5
BF16_TOL = dict(rtol=1e-2, atol=1e-5)
FP32_TOL = dict(rtol=1e-5, atol=1e-8)


def tiny_stack(device, dtype):
    torch.manual_seed(7)
    layers = []
    for _ in range(2):
        block = torch.nn.Sequential(
            torch.nn.Linear(32, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 32),
        )
        block.to(device=device, dtype=dtype)
        layers.append(block)
    return torch.nn.Sequential(*layers)


def run(device, dtype, fused):
    model = tiny_stack(device, dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01,
                            fused=fused)
    snapshots = []
    for step in range(STEPS):
        g = torch.Generator(device=device).manual_seed(100 + step)
        for p in model.parameters():
            p.grad = torch.randn(
                p.shape, generator=g, device=device, dtype=p.dtype,
                layout=p.layout) * 0.01
        opt.step()
        snapshots.append(
            torch.cat([p.detach().float().flatten() for p in model.parameters()]))
    return snapshots


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device; exit 2 (skip is not a pass)", file=sys.stderr)
        return 2

    from omegaconf import OmegaConf
    from flagscale.train.train_retrieval import build_optimizer

    cfg = OmegaConf.create({"name": "adamw", "lr": 2e-5, "weight_decay": 0.01})

    failures = 0
    for dtype, tol, name in ((torch.bfloat16, BF16_TOL, "bf16"),
                             (torch.float32, FP32_TOL, "fp32")):
        device = "cuda"
        ref = run(device, dtype, fused=False)
        model = tiny_stack(device, dtype)
        cand = build_optimizer(cfg, list(model.parameters()), allow_adafactor=False)
        fused_flag = bool(cand.defaults.get("fused"))
        cand_snap = []
        for step in range(STEPS):
            g = torch.Generator(device=device).manual_seed(100 + step)
            for p in model.parameters():
                p.grad = torch.randn(
                    p.shape, generator=g, device=device, dtype=p.dtype,
                    layout=p.layout) * 0.01
            cand.step()
            cand_snap.append(
                torch.cat([p.detach().float().flatten() for p in model.parameters()]))
        worst = max((a - b).abs().max().item()
                    for a, b in zip(ref, cand_snap))
        close = all(torch.allclose(a, b, **tol) for a, b in zip(ref, cand_snap))
        print(f"{name}: fused={fused_flag} max|diff|={worst:.3e} allclose={close}")
        if not fused_flag:
            print(f"FAIL({name}): build_optimizer did not choose fused AdamW",
                  file=sys.stderr)
            failures += 1
        if not close:
            print(f"FAIL({name}): fused AdamW diverged from stock beyond {tol}",
                  file=sys.stderr)
            failures += 1

    if failures:
        print(f"FAILED: {failures} case(s)", file=sys.stderr)
        return 1
    print("PASS: fused AdamW numerically equivalent to stock")
    return 0


if __name__ == "__main__":
    sys.exit(main())
