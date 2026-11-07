"""Issue #7181 / problem.md minimal repro: complex dtype through flag_gems.

Pre-fix : the first complex op routed through flag_gems raises
          ``KeyError: 'complex64'`` from ``triton/_utils.py
          type_canonicalisation_dict`` (observed verbatim in the m2-bert
          container run, problem.md) and exits non-zero with the traceback.
Post-fix: flag_gems.zeros preserves the complex dtype, flag_gems.abs returns
          native-comparable real magnitudes, torch.sum over complex matches a
          native oracle captured before enable() -> exit 0.

Run : PYTHONPATH=<FlagGems>/src python repro_complex64.py
Exit: 0 = fixed | 1 = issue reproduced/failed | 2 = missing CUDA
A skip is not a pass: no CUDA device -> exit 2.
"""
import os
import sys
import traceback

import torch

if not torch.cuda.is_available():
    print("MISSING CUDA DEVICE: complex64 repro needs a CUDA device -> exit 2", flush=True)
    os._exit(2)

DEV = "cuda"

# Native oracle captured BEFORE flag_gems registers any op.
DTYPE_CASES = (torch.complex64, torch.complex128)
_native = {}
for dt in DTYPE_CASES:
    x = torch.tensor([[1 + 2j, 3 - 4j], [5 + 0.5j, -2 + 7j]], dtype=dt, device=DEV)
    _native[dt] = {
        "zeros": torch.zeros((2,), dtype=dt, device=DEV),
        "x": x,
        "sum": torch.sum(x),
        "sum_dim": torch.sum(x, dim=1),
        "abs": torch.abs(x),
    }

_src = os.environ.get("FLAG_GEMS_SRC")
if _src:
    sys.path.insert(0, _src)
import flag_gems  # noqa: E402

print("flag_gems from:", flag_gems.__file__, flush=True)
flag_gems.enable()

import flag_gems.ops as gops  # noqa: E402


def _eq(got, want, name):
    """dtype + value compare; complex compared through the real view (eq is
    not complex-capable in flag_gems, a known pre-existing gap)."""
    assert got.dtype == want.dtype, f"{name}: dtype {got.dtype} != {want.dtype}"
    if got.is_complex() or want.is_complex():
        g, w = torch.view_as_real(got), torch.view_as_real(want)
    else:
        g, w = got, want
    if g.numel():
        torch.testing.assert_close(g, w, rtol=1e-5, atol=1e-5)
    else:
        assert g.shape == w.shape, f"{name}: shape {g.shape} != {w.shape}"
    print(f"  ok {name}", flush=True)


try:
    # Exact issue #7181 call style: flag_gems.zeros((2,), dtype=<cplx>).
    for dt in DTYPE_CASES:
        z = gops.zeros((2,), dtype=dt, device=DEV)
        _eq(z, _native[dt]["zeros"], f"flag_gems.zeros {dt}")

        a = gops.abs(_native[dt]["x"])
        _eq(a, _native[dt]["abs"], f"flag_gems.abs {dt}")

        _eq(torch.zeros((2,), dtype=dt, device=DEV), _native[dt]["zeros"], f"torch.zeros {dt}")
        _eq(torch.sum(_native[dt]["x"]), _native[dt]["sum"], f"torch.sum full {dt}")
        _eq(torch.sum(_native[dt]["x"], dim=1), _native[dt]["sum_dim"], f"torch.sum dim1 {dt}")
except Exception:
    print("\nISSUE REPRODUCED (expected pre-fix, problem.md / issue #7181):", flush=True)
    traceback.print_exc()
    os._exit(1)

print("\nREPRO OK (pre-fix KeyError complex64/128 not raised)", flush=True)
os._exit(0)
