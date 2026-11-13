"""ZW810E problem29 small-model training comparison v2 (review round 2).

v1 correction: v1/v1-probe4 built model/params/embedding/tokens on DEFAULT
device (CPU) — CUDA events and flag_gems CUDA registration do not affect CPU
execution, so those results could not validate the platform training path.
They are SUPERSEDED.

v2:
  - device = torch.device("cuda"); model blocks/head/embedding moved to device,
    tokens moved to device, ALL BEFORE optimizer construction;
  - asserts (recorded): every param on cuda, inputs on cuda, grads on cuda;
  - profiler hit evidence for forward+backward kernels per config (gelu triton
    names under gems configs / aten names under native);
  - unified 8 steps; per-step losses, final per-param grad sha256 + max_abs +
    device, final param sha256 — full evidence for cross-config comparison;
  - per-step CUDA-event wall times.
"""
import argparse
import hashlib
import json

import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, choices=["native", "gems_full", "gems_selective"])
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--out", required=True)
args = ap.parse_args()

DEVICE = torch.device("cuda")

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
        # explicit attention (matmul+softmax) instead of F.scaled_dot_product_attention:
        # FlagGems sdpa (flash_attention registration) HANGS on ZW810E for this
        # training shape (see probe7_sdpa_hang.py evidence), so the model uses
        # well-supported ops; sdpa itself is separately probed.
        att = torch.softmax(q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5), dim=-1) @ v
        x = x + att @ self.wo
        h = u @ self.w1
        x = x + (F.gelu(h, approximate="none") * (u @ self.w3)) @ self.w2
        return x


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def main():
    torch.manual_seed(0)
    B, S, D, H, L = 8, 128, 256, 512, 2
    blocks = torch.nn.ModuleList([Block(D, H, 100 + i) for i in range(L)]).to(DEVICE)
    head = (torch.randn(D, D) * 0.02).to(DEVICE)
    head = torch.nn.Parameter(head)
    params = list(blocks.parameters()) + [head]
    rec = {"config": args.config, "device": str(DEVICE), "device_name": torch.cuda.get_device_name(0)}

    # embedding created BEFORE optimizer and included in params (review round 3:
    # trainable embedding, not a frozen operator-path-only test)
    g = torch.Generator().manual_seed(42)
    tok = torch.randint(0, 64, (B, S), generator=g).to(DEVICE)
    emb = torch.nn.Embedding(64, D).to(DEVICE)
    with torch.no_grad():
        emb.weight.copy_((torch.randn(64, D, generator=g) * 0.02).to(DEVICE))
    params = params + list(emb.parameters())

    # device assertions BEFORE optimizer construction
    rec["params_all_cuda_before_optim"] = all(p.device.type == "cuda" for p in params)

    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)

    rec["tokens_cuda"] = tok.device.type == "cuda"
    rec["embedding_trainable_in_params"] = any(p is emb.weight for p in params)
    rec["steps"] = args.steps
    losses, step_ms = [], []
    for st in range(args.steps):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        x = emb(tok)
        for blk in blocks:
            x = blk(x)
        logits = x @ head
        loss = logits.float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        e.record()
        torch.cuda.synchronize()
        losses.append(float(loss.detach()))
        step_ms.append(round(s.elapsed_time(e), 2))

    rec["losses"] = losses
    rec["step_ms"] = step_ms

    # device + evidence asserts AFTER training
    rec["grads_all_on_cuda"] = all(p.grad is not None and p.grad.device.type == "cuda" for p in params)
    rec["all_grads_present"] = all(p.grad is not None for p in params)
    rec["grad_stats"] = {}
    for i, p in enumerate(params):
        n = f"param_{i}" + ("_embedding_weight" if p is emb.weight else "")
        rec["grad_stats"][n] = {
            "device": p.grad.device.type if p.grad is not None else None,
            "grad_sha": sha(p.grad) if p.grad is not None else None,
            "grad_max_abs": float(p.grad.abs().max()) if p.grad is not None and p.grad.numel() else 0.0,
            "grad_finite": bool(torch.isfinite(p.grad).all().item()) if p.grad is not None else None,
            "param_sha": sha(p),
        }
    rec["all_grads_finite"] = all(v["grad_finite"] for v in rec["grad_stats"].values())

    # profiler hit evidence: one extra fwd+bwd step (no optimizer)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        x = emb(tok)
        for blk in blocks:
            x = blk(x)
        loss = (x @ head).float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
    ks = [ev.key for ev in prof.key_averages() if ev.device_time > 0]
    rec["fwd_bwd_kernels_gelu"] = [k[:80] for k in ks if "gelu" in k.lower()]
    rec["fwd_bwd_kernels_sample"] = [k[:60] for k in ks[:6]]

    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: rec[k] for k in ("config", "params_all_cuda_before_optim", "grads_all_on_cuda", "losses", "fwd_bwd_kernels_gelu")})[:400])


if __name__ == "__main__":
    main()
