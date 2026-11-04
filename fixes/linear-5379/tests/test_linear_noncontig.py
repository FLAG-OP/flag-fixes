"""Regression matrix for issue #5379: flag_gems linear must accept any layout.

Covers the direct kernel entry (flag_gems.linear) and Torch dispatch
(F.linear under only_enable(include=["linear"])), across dtypes, batch ranks,
the GEMV fast path (fp32, M <= 5) and contiguous control cases.

Run:  PYTHONPATH=<FlagGems>/src python test_linear_noncontig.py
Exit: 0 all cases pass, 1 any case fails, 2 no device.
"""
import sys

import torch

if not torch.cuda.is_available():
    print("SKIP: no CUDA device; a skip is not a pass", file=sys.stderr)
    raise SystemExit(2)

device = "cuda"
K, N = 64, 32

import flag_gems  # noqa: E402

DTYPES = (torch.float32, torch.float16, torch.bfloat16)

TOLERANCE = {
    torch.float32: (1e-3, 1e-3),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (2e-2, 2e-2),
}

passed = 0
failed = 0


def make_cases(dtype):
    """name -> (input tensor, with bias)"""
    def t(*shape):
        return torch.randn(*shape).to(device=device, dtype=dtype)

    cases = {
        "control-contig-2d": (t(32, K), True),
        "control-contig-1d": (t(K), False),
        "transpose-3d": (t(2, K, 16).transpose(1, 2), True),  # (2,16,K)
        "slice-3d": (t(4, 16, 2 * K)[..., :K], True),  # last-dim stride 2K
        "permute-3d": (t(K, 16, 4).permute(2, 1, 0), True),  # (4,16,K)
        "expand-3d": (t(1, 16, K).expand(2, 16, K), True),  # stride-0 view
        "transpose-4d": (t(2, 3, 16, K).transpose(1, 2), True),  # (2,16,3,K)
        "transpose-no-bias": (t(3, K, 8).transpose(1, 2), False),
    }
    if dtype == torch.float32:
        # M=5 <= _GEMV_MAX_M, only reachable on the fp32 GEMV path. Half
        # dtypes with M in 5..8 return garbage on this device with or without
        # the patch (pre-existing defect, see reports/diagnosis.md), so they
        # are deliberately not asserted here.
        cases["gemv-3d-fp32"] = (t(1, 5, 2 * K)[..., :K], True)
    return cases


def build_oracles():
    """Native references for every case, computed before any registration."""
    plans = []
    for dtype in DTYPES:
        entries = []
        for name, (x, with_bias) in make_cases(dtype).items():
            w = torch.randn(N, K, device=device, dtype=dtype)
            b = torch.randn(N, device=device, dtype=dtype) if with_bias else None
            entries.append((name, x, w, b, torch.nn.functional.linear(x, w, b)))
        plans.append((str(dtype).replace("torch.", ""), entries, TOLERANCE[dtype]))
    return plans


def check(label, name, got, want, tol):
    global passed, failed
    got, want = got.cpu(), want.cpu()
    rtol, atol = tol
    if got.shape == want.shape and torch.allclose(got, want, rtol=rtol, atol=atol):
        passed += 1
        print(f"  [PASS] {label} {name} mse={((got.float() - want.float()) ** 2).mean().item():.3e}")
    else:
        failed += 1
        diff = (got.float() - want.float()).abs().max().item() if got.shape == want.shape else float("nan")
        print(f"  [FAIL] {label} {name} max_diff={diff:.3e} (rtol={rtol}, atol={atol})")
        print(f"         got[0:4]={got.flatten()[:4].tolist()} want[0:4]={want.flatten()[:4].tolist()}")


def main():
    global passed, failed
    plans = build_oracles()  # native torch, nothing registered yet
    flag_gems.only_enable(include=["linear"])

    for dtype_label, entries, tol in plans:
        print(f"=== {dtype_label} via Torch dispatch (F.linear) ===")
        for name, x, w, b, ref in entries:
            try:
                out = torch.nn.functional.linear(x, w, b)
            except Exception as error:  # noqa: BLE001 - the failure itself is the signal
                failed += 1
                print(f"  [FAIL] dispatch {name}: {type(error).__name__}: {error}")
                continue
            check("dispatch", name, out, ref, tol)

        print(f"=== {dtype_label} via direct flag_gems.linear ===")
        for name, x, w, b, ref in entries:
            try:
                out = flag_gems.linear(x, w, b)
            except Exception as error:  # noqa: BLE001
                failed += 1
                print(f"  [FAIL] direct {name}: {type(error).__name__}: {error}")
                continue
            check("direct", name, out, ref, tol)

    print(f"TOTAL: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
