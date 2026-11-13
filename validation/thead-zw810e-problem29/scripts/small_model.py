"""ZW810E problem29 small-model comparison: fixed-seed gated MLP + Transformer
block stack, forward/loss/grads + 5 AdamW steps, three configs (separate
processes): native | gems_full | gems_selective.

No external weights. Records loss trajectory, per-parameter grad max-abs-diff
vs native dumps, grad presence/finiteness, and wall time per step.
"""
import argparse
import hashlib
import json

import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, choices=["native", "gems_full", "gems_selective"])
ap.add_argument("--out", required=True)
args = ap.parse_args()

if args.config != "native":
    import flag_gems
    if args.config == "gems_selective":
        flag_gems.only_enable(include=["gelu", "gelu_backward", "gelu_and_mul"])
    else:
        flag_gems.enable()


class Block(torch.nn.Module):
    def __init__(self, d, h, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w1 = torch.nn.Parameter(torch.randn(d, h, generator=g) * 0.02)
        self.w3 = torch.nn.Parameter(torch.randn(d, h, generator=g) * 0.02)  # gated MLP
        self.w2 = torch.nn.Parameter(torch.randn(h, d, generator=g) * 0.02)
        self.ln_g = torch.nn.Parameter(torch.ones(d))
        self.ln_b = torch.nn.Parameter(torch.zeros(d))
        self.qkv = torch.nn.Parameter(torch.randn(d, 3 * d, generator=g) * 0.02)
        self.wo = torch.nn.Parameter(torch.randn(d, d, generator=g) * 0.02)

    def forward(self, x):
        u = x * self.ln_g + self.ln_b
        q, k, v = (u @ self.qkv).chunk(3, dim=-1)
        att = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0), is_causal=False
        ).squeeze(0).transpose(0, 1)
        x = x + att @ self.wo
        h = self.w1[u] if False else u @ self.w1
        x = x + (F.gelu(h, approximate="none") * (u @ self.w3)) @ self.w2
        return x


def main():
    torch.manual_seed(0)
    steps = 5
    B, S, D, H, L = 8, 128, 256, 512, 2
    blocks = torch.nn.ModuleList([Block(D, H, 100 + i) for i in range(L)])
    head = torch.nn.Parameter(torch.randn(D, D) * 0.02)
    params = list(blocks.parameters()) + [head]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)

    g = torch.Generator().manual_seed(42)
    tok = torch.randint(0, 64, (B, S), generator=g)
    emb = torch.nn.Embedding(64, D)
    with torch.no_grad():
        emb.weight.copy_(torch.randn(64, D, generator=g) * 0.02)

    rec = {"config": args.config, "losses": [], "step_ms": [], "grad_stats": {}}
    for st in range(steps):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        x = emb(tok)
        for blk in blocks:
            x = blk(x)
        logits = x @ head
        loss = logits.float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        t1.record()
        torch.cuda.synchronize()
        rec["losses"].append(float(loss.detach()))
        rec["step_ms"].append(round(t0.elapsed_time(t1), 2))

    rec["param_names"] = []
    for i, p in enumerate(params):
        n = f"param_{i}"
        rec["param_names"].append(n)
        rec["grad_stats"][n] = {
            "grad_sha": hashlib.sha256(p.grad.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16] if p.grad is not None else None,
            "grad_finite": bool(torch.isfinite(p.grad).all().item()) if p.grad is not None else None,
            "grad_max_abs": float(p.grad.abs().max()) if p.grad is not None and p.grad.numel() else 0.0,
            "param_sha": hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16],
        }
    rec["all_grads_present"] = all(v["grad_sha"] for v in rec["grad_stats"].values())
    rec["all_grads_finite"] = all(v["grad_finite"] for v in rec["grad_stats"].values())
    rec["loss_step1"] = rec["losses"][0]

    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: rec[k] for k in ("config", "losses", "step_ms", "all_grads_present", "all_grads_finite")}))


if __name__ == "__main__":
    main()
