# Round-11 (review 6788210724 section 3): dedicated repro/verification for
# FIX-ITEM-4 — FlagGems sum on uint8 returns uint8 dtype and wraps the
# accumulator. Direct value+dtype comparison native vs registered across
# the reviewer's checklist: 1-D/multi-D, default vs explicit dtype,
# >255 accumulation, empty input, different reduction dims, contiguous vs
# non-contiguous. Run against BOTH trees for attribution.
#
# Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<tree>/src python3 benchmarks/u8sum_repro.py
import sys
import torch

import flag_gems  # noqa: F401

flag_gems.only_enable(include=["sum", "sum_dim", "sum_out", "sum_dim_out"])

torch.manual_seed(7)
fails = []
results = []


def check(name, got, want, want_dtype):
    """want: scalar or (cpu long tensor / list). Compares values and dtype."""
    if isinstance(want, (int, float)):
        got_v = got.item() if got.numel() == 1 else None
        ok = (got_v == want) and (got.dtype == want_dtype)
        want_v = want
    else:
        wt = torch.tensor(want, dtype=torch.long)
        got_v = got.cpu().long()
        ok = torch.equal(got_v, wt) and (got.dtype == want_dtype)
        want_v = f"<{wt.numel()} elems>"
    results.append({"case": name,
                    "got_value": (got_v if isinstance(got_v, int)
                                  else got_v.tolist()),
                    "got_dtype": str(got.dtype),
                    "want": str(want_v), "want_dtype": str(want_dtype),
                    "ok": ok})
    if not ok:
        fails.append(name)
    gv = got_v if isinstance(got_v, int) else got_v.tolist()
    print(f"{'OK ' if ok else 'BAD'} {name}: got={gv} {got.dtype} "
          f"(want {want_v} {want_dtype})", flush=True)


# round-12: explicit dtype=torch.uint8 (the previous tensor was int64, so
# the two "passing" cases never exercised uint8)
u8 = torch.tensor([200, 100, 50, 7], dtype=torch.uint8, device="cuda")
assert u8.dtype == torch.uint8
check("1d/default", u8.sum(), 357, torch.int64)
check("1d/explicit-i64", u8.sum(dtype=torch.int64), 357, torch.int64)
big = torch.randint(0, 255, (1 << 20,), dtype=torch.uint8, device="cuda")
want_big = int(big.cpu().long().sum().item())                 # >255 for sure
check("1d/large-accum", big.sum(), want_big, torch.int64)
m2d = torch.randint(0, 255, (16, 16), dtype=torch.uint8, device="cuda")
check("2d/full", m2d.sum(), int(m2d.cpu().long().sum().item()),
      torch.int64)
check("2d/dim0", m2d.sum(dim=0), m2d.cpu().long().sum(dim=0).tolist(),
      torch.int64)
check("2d/dim1", m2d.sum(dim=1), m2d.cpu().long().sum(dim=1).tolist(),
      torch.int64)
# round-12: build the uint8 base FIRST, then slice (slicing float then
# .to(uint8) produced a fresh contiguous tensor, not a strided view)
base_u8 = torch.randint(0, 255, (4, 8), dtype=torch.uint8, device="cuda")
v = base_u8[:, ::2]
assert v.dtype == torch.uint8 and not v.is_contiguous()
wv = int(v.cpu().long().sum().item())
check("noncontig/view", v.sum(), wv, torch.int64)
e = torch.zeros(0, dtype=torch.uint8, device="cuda")
check("empty", e.sum(), 0, torch.int64)
results.append({"case": "_inputs", "u8_dtype": str(u8.dtype),
                "v_dtype": str(v.dtype), "v_contiguous": v.is_contiguous(),
                "v_stride": list(v.stride())})
print(f"SUMMARY: {len(fails)} failures / {len(results)-1} cases "
      f"({'PASS' if not fails else 'FAIL'})", flush=True)
if fails:
    print("failing cases:", fails, flush=True)
sys.exit(1 if fails else 0)
