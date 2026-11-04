"""Caller-level / end-to-end regression for issue #5379.

1. Vision-style block: Conv2d output transposed to NHWC, then an MLP - the
   linear input is a strided view, which is what broke with input.view(M, K).
2. Transformer-style block: (B, S, D) hidden states transposed to (S, B, D)
   before the MLP, the layout issue #5379 reports for every training step.
3. Real model: a randomly initialised SigLIP vision tower (no download) run on
   non-contiguous pixel_values with FlagGems linear registered.

Native references are computed before FlagGems registers anything; comparisons
happen on CPU so the oracle is never routed through the code under test.

Run:  PYTHONPATH=<FlagGems>/src python test_linear_callers.py
Exit: 0 all cases pass, 1 any case fails, 2 no device.
"""
import sys

import torch
from torch import nn

if not torch.cuda.is_available():
    print("SKIP: no CUDA device; a skip is not a pass", file=sys.stderr)
    raise SystemExit(2)

device = "cuda"
torch.manual_seed(0)

import flag_gems  # noqa: E402

passed = 0
failed = 0


def check(name, got, want, rtol=1e-3, atol=1e-3):
    global passed, failed
    got, want = got.detach().float().cpu(), want.detach().float().cpu()
    if got.shape == want.shape and torch.allclose(got, want, rtol=rtol, atol=atol):
        passed += 1
        print(f"  [PASS] {name} mse={((got - want) ** 2).mean().item():.3e}")
    else:
        failed += 1
        diff = (got - want).abs().max().item() if got.shape == want.shape else float("nan")
        print(f"  [FAIL] {name} max_diff={diff:.3e}")


# ---------------------------------------------------------------- native oracles
conv = nn.Conv2d(3, 16, kernel_size=3, stride=2).to(device).eval()
head = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 10)).to(device).eval()
mlp = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 10)).to(device).eval()

image = torch.randn(2, 3, 32, 32, device=device)
nhwc = conv(image).permute(0, 2, 3, 1)  # (2, H, W, 16), strided, last dim = in_features
print(f"nhwc input {tuple(nhwc.shape)} stride={nhwc.stride()} contiguous={nhwc.is_contiguous()}")
assert not nhwc.is_contiguous()
ref_nhwc = head(nhwc)

hidden = torch.randn(8, 16, 32, device=device)  # (B, S, D)
seq_first = hidden.transpose(0, 1)  # (S, B, D), strided, last dim = in_features
print(f"seq-first input {tuple(seq_first.shape)} stride={seq_first.stride()} "
      f"contiguous={seq_first.is_contiguous()}")
assert not seq_first.is_contiguous()
ref_seq = mlp(seq_first)

vision = None
ref_pooled = None
pixels = None
try:
    from transformers import SiglipVisionConfig, SiglipVisionModel

    config = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        image_size=32,
        patch_size=8,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        projection_dim=32,
    )
    vision = SiglipVisionModel(config).to(device).eval()
    pixels = torch.randn(2, 3, 32, 32, device=device).transpose(2, 3)  # strided copy
    print(f"pixel_values contiguous={pixels.is_contiguous()}")
    assert not pixels.is_contiguous()
    with torch.no_grad():
        ref_pooled = vision(pixel_values=pixels).pooler_output
    print("native SigLIP vision forward ok", tuple(ref_pooled.shape))
except ImportError:
    print("  [SKIP] transformers unavailable - case 3 not run (a skip is not a pass)")

# ---------------------------------------------------------------- under test
flag_gems.only_enable(include=["linear"])

print("=== 1. vision NHWC block (conv -> transpose -> MLP) ===")
with torch.no_grad():
    check("nhwc-mlp", head(nhwc), ref_nhwc)

print("=== 2. transformer seq-first block (B,S,D -> S,B,D -> MLP) ===")
with torch.no_grad():
    check("seq-first-mlp", mlp(seq_first), ref_seq)

print("=== 3. SigLIP vision tower, non-contiguous pixel_values ===")
if vision is None:
    print("  [SKIP] no transformers")
else:
    with torch.no_grad():
        out = vision(pixel_values=pixels).pooler_output
    check("siglip-vision", out, ref_pooled, rtol=1e-3, atol=1e-3)

print(f"TOTAL: {passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
