# Minimal reproduction for the round-4 f_all model-A failure family:
# torch.empty_like / conj / sgn on torch.complex(p, q)-ASSEMBLED complex
# tensors under full FlagGems registration -> fill_scalar_ ->
# triton canonicalisation KeyError ('complex64' / 'complex128').
#
# Key trigger (found in round-5 probing): tensors produced by CAST
# (randn(...).to(complex64)) do NOT trigger it; tensors ASSEMBLED by
# torch.complex(p, q) do. On the pre-fix tree the same probes crash at the
# same fill node (abs(z) forward crashes too there), so this doubles as the
# isolated same-path baseline control for the pre-existing attribution.
#
# Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<tree>/src python3 tests/repro_complex_fill.py
import sys

import torch

import flag_gems  # noqa: F401

flag_gems.enable()
torch.use_deterministic_algorithms(True)

p = torch.randn(2, 16, 8, device="cuda", requires_grad=True)
q = torch.randn(2, 16, 8, device="cuda")
z = torch.complex(p, q)
zng = z.detach()
results = []


def probe(name, fn):
    try:
        fn()
        status = "OK"
    except KeyError as e:
        status = f"KeyError {e}"
    except Exception as e:
        status = f"{type(e).__name__}: {str(e)[:60]}"
    results.append((name, status))
    print(f"{name}: {status}", flush=True)
    return status


failures = 0
for name, fn in [
    ("zeros(c128)", lambda: torch.zeros((8, 8), dtype=torch.complex128,
                                        device="cuda")),
    ("empty_like(assembled c64)", lambda: torch.empty_like(z)),
    ("conj(assembled c64)", lambda: zng.conj()),
    ("sgn(assembled c64)", lambda: torch.ops.aten.sgn(zng)),
    ("abs(sum(assembled)) backward", lambda: (
        (lambda zz: torch.abs(torch.sum(zz, dim=1)).backward(
            torch.randn(2, 8, device="cuda")))(
            z.detach().clone().requires_grad_(True)))),
]:
    if "KeyError" in probe(name, fn):
        failures += 1

print(f"SUMMARY: {failures}/5 probes crashed with triton KeyError",
      flush=True)
sys.exit(1 if failures else 0)
