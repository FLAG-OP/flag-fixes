"""Regression battery: complex64/128 support across sum, the zeros family,
abs/absolute, plus real-dtype regressions (torch-vs-native oracle).

Pre-fix : every complex case fails with KeyError 'complex64'/'complex128'
          from triton/_utils.py type_canonicalisation_dict (baseline); real
          cases pass -> non-zero exit, TOTAL shows failures.
Post-fix: all complex paths run in Triton (real-view lowering where needed),
          all real paths unchanged -> ``TOTAL n passed, 0 failed`` exit 0.

Comparisons on complex tensors go through torch.view_as_real: flag_gems eq/
nonzero are not complex-capable (pre-existing gap outside this fix).

Run : PYTHONPATH=<FlagGems>/src python test_complex_ops.py
Exit: 0 = all pass | 1 = failures | 2 = missing CUDA
A skip is not a pass: no CUDA device -> exit 2.
"""
import os
import sys
import traceback

import torch

if not torch.cuda.is_available():
    print("MISSING CUDA DEVICE: needs a CUDA device -> exit 2", flush=True)
    os._exit(2)

DEV = "cuda"
passed = failed = 0


def check(name, fn, oracle):
    global passed, failed
    try:
        got = fn()
        assert got.dtype == oracle.dtype, f"dtype {got.dtype} != {oracle.dtype}"
        g = torch.view_as_real(got) if got.is_complex() else got
        w = torch.view_as_real(oracle) if oracle.is_complex() else oracle
        if g.numel():
            torch.testing.assert_close(g, w, rtol=1e-5, atol=1e-5)
        else:
            assert g.shape == w.shape
        passed += 1
        print(f"[PASS] {name}", flush=True)
    except Exception as e:
        failed += 1
        print(f"[FAIL] {name}: {e!r}", flush=True)
        if os.environ.get("VERBOSE"):
            traceback.print_exc()


# --- native oracles captured BEFORE flag_gems registers ---
native = {}
for dt in (torch.complex64, torch.complex128):
    rdt = torch.float32 if dt == torch.complex64 else torch.float64
    c = torch.tensor([[1 + 2j, 3 - 4j, 0 - 0.5j], [-2 + 7j, 0 + 0j, 4 - 1j]], dtype=dt, device=DEV)
    native[("c", dt, "base")] = c
    native[("c", dt, "sumfull")] = torch.sum(c)
    native[("c", dt, "sum0")] = torch.sum(c, dim=0)
    native[("c", dt, "sum-1keep")] = torch.sum(c, dim=-1, keepdim=True)
    native[("c", dt, "sum(0,1)")] = torch.sum(c, dim=(0, 1))
    native[("c", dt, "sumupcast")] = torch.sum(c, dtype=torch.complex128)
    native[("c", dt, "realout")] = torch.sum(c, dtype=torch.float64 if rdt == torch.float32 else torch.float32)
    native[("c", dt, "zeros")] = torch.zeros((3, 4), dtype=dt, device=DEV)
    native[("c", dt, "zeroslike")] = torch.zeros_like(c)
    native[("c", dt, "zeroslikedt")] = torch.zeros_like(c, dtype=torch.complex128)
    native[("c", dt, "emptyzeros")] = torch.zeros((0, 3), dtype=dt, device=DEV)
    native[("c", dt, "zeros22")] = torch.zeros((2, 2), dtype=dt, device=DEV)
    native[("c", dt, "rej")] = torch.tensor([1 + 2j, -3 + 4j], dtype=dt, device=DEV)
    native[("c", dt, "sumT")] = torch.sum(c.t())
    native[("c", dt, "abs")] = torch.abs(c)
    native[("c", dt, "absT")] = torch.abs(c.t())
    native[("c", dt, "absempty")] = torch.abs(torch.empty((0, 3), dtype=dt, device=DEV))
    native[("c", dt, "emptyfull")] = torch.sum(torch.empty((0, 3), dtype=dt, device=DEV))

_r32 = torch.randn(4, 5, device=DEV)
_r64 = _r32.double()
_i64 = torch.randint(-30, 30, (4, 5), device=DEV)
native[("r", "f32", "sumfull")] = torch.sum(_r32)
native[("r", "f32", "sum0")] = torch.sum(_r32, dim=0)
native[("r", "f64", "sumfull")] = torch.sum(_r64)
native[("r", "i64", "sumfull")] = torch.sum(_i64)
native[("r", "i64", "sum0keep")] = torch.sum(_i64, dim=0, keepdim=True)
native[("r", "f32", "zeros")] = torch.zeros((3, 4), dtype=torch.float32, device=DEV)
native[("r", "f32", "zeroslike")] = torch.zeros_like(_r32)
native[("r", "i64", "zeroslikedt")] = torch.zeros_like(_i64, dtype=torch.float64)
native[("r", "f32", "emptyzeros")] = torch.zeros((0, 3), dtype=torch.float32, device=DEV)
native[("r", "f32", "abs")] = torch.abs(_r32)
native[("r", "i64", "abs")] = torch.abs(_i64)

# native in-place rejection messages, captured pre-enable (native dispatch there)
_native_abs_reject = {}
for _dt in (torch.complex64, torch.complex128):
    _x = torch.tensor([1 + 2j, -3 + 4j], dtype=_dt, device=DEV)
    try:
        _x.clone().abs_()
        _native_abs_reject[_dt] = None
    except RuntimeError as e:
        _native_abs_reject[_dt] = str(e)

_src = os.environ.get("FLAG_GEMS_SRC")
if _src:
    sys.path.insert(0, _src)
import flag_gems  # noqa: E402

print("flag_gems from:", flag_gems.__file__, flush=True)
flag_gems.enable()
import flag_gems.ops as gops  # noqa: E402

for dt in (torch.complex64, torch.complex128):
    c = native[("c", dt, "base")]
    ct = c.t()
    check(f"sum full {dt}", lambda c=c: torch.sum(c), native[("c", dt, "sumfull")])
    check(f"sum dim0 {dt}", lambda c=c: torch.sum(c, dim=0), native[("c", dt, "sum0")])
    check(f"sum dim-1 keepdim {dt}", lambda c=c: torch.sum(c, dim=-1, keepdim=True), native[("c", dt, "sum-1keep")])
    check(f"sum dim(0,1) {dt}", lambda c=c: torch.sum(c, dim=(0, 1)), native[("c", dt, "sum(0,1)")])
    check(f"sum upcast c128 {dt}", lambda c=c: torch.sum(c, dtype=torch.complex128), native[("c", dt, "sumupcast")])
    check(f"sum out real {dt}", lambda c=c: torch.sum(c, dtype=native[("c", dt, "realout")].dtype), native[("c", dt, "realout")])
    check(f"sum empty {dt}", lambda: torch.sum(torch.empty((0, 3), dtype=dt, device=DEV)), native[("c", dt, "emptyfull")])
    check(f"sum noncontig {dt}", lambda ct=ct: torch.sum(ct), native[("c", dt, "sumT")])
    check(f"zeros {dt}", lambda dt=dt: torch.zeros((3, 4), dtype=dt, device=DEV), native[("c", dt, "zeros")])
    check(f"zeros_like {dt}", lambda c=c: torch.zeros_like(c), native[("c", dt, "zeroslike")])
    check(f"zeros_like dtype {dt}", lambda c=c: torch.zeros_like(c, dtype=torch.complex128), native[("c", dt, "zeroslikedt")])
    check(f"zeros empty {dt}", lambda dt=dt: torch.zeros((0, 3), dtype=dt, device=DEV), native[("c", dt, "emptyzeros")])
    def _zero_direct(dt=dt, name=None):
        return gops.zeros((2, 2), dtype=dt, device=DEV)
    check(f"flag_gems.zeros direct {dt}", _zero_direct, native[("c", dt, "zeros22")])
    def _zero_(dt=dt):
        t = native[("c", dt, "base")].clone()
        r = gops.zero_(t)
        assert r is t, "zero_ must return the input tensor"
        return t
    check(f"flag_gems.zero_ {dt}", _zero_, native[("c", dt, "zeroslike")])
    def _zero_out(dt=dt):
        t = native[("c", dt, "base")].clone()
        torch.ops.aten.zero_(t)
        return t
    check(f"aten.zero_ {dt}", _zero_out, native[("c", dt, "zeroslike")])
    check(f"abs {dt}", lambda c=c: torch.abs(c), native[("c", dt, "abs")])
    check(f"abs noncontig {dt}", lambda ct=ct: torch.abs(ct), native[("c", dt, "absT")])
    check(f"abs empty {dt}", lambda dt=dt: torch.abs(torch.empty((0, 3), dtype=dt, device=DEV)), native[("c", dt, "absempty")])  # oracle pre-captured
    check(f"flag_gems.abs direct {dt}", lambda c=c: gops.abs(c), native[("c", dt, "abs")])
    check(f"torch.absolute {dt}", lambda c=c: torch.absolute(c), native[("c", dt, "abs")])  # absolute oracle == abs oracle (pre-captured)

check("sum f32", lambda: torch.sum(_r32), native[("r", "f32", "sumfull")])
check("sum f32 dim0", lambda: torch.sum(_r32, dim=0), native[("r", "f32", "sum0")])
check("sum f64", lambda: torch.sum(_r64), native[("r", "f64", "sumfull")])
check("sum i64", lambda: torch.sum(_i64), native[("r", "i64", "sumfull")])
check("sum i64 dim0 keepdim", lambda: torch.sum(_i64, dim=0, keepdim=True), native[("r", "i64", "sum0keep")])
check("zeros f32", lambda: torch.zeros((3, 4), dtype=torch.float32, device=DEV), native[("r", "f32", "zeros")])
check("zeros_like f32", lambda: torch.zeros_like(_r32), native[("r", "f32", "zeroslike")])
check("zeros_like i64->f64", lambda: torch.zeros_like(_i64, dtype=torch.float64), native[("r", "i64", "zeroslikedt")])
check("zeros f32 empty", lambda: torch.zeros((0, 3), dtype=torch.float32, device=DEV), native[("r", "f32", "emptyzeros")])
check("abs f32", lambda: torch.abs(_r32), native[("r", "f32", "abs")])
check("abs i64", lambda: torch.abs(_i64), native[("r", "i64", "abs")])
def _abs_inplace_real():
    t = _r32.clone()
    t.abs_()
    return t
check("abs_ real in-place", _abs_inplace_real, torch.abs(_r32))

# in-place complex rejection: native (pre-enable) raises RuntimeError; flag_gems
# must raise the identical message. Broad catch so pre-fix KeyError still counts
# as a recorded failure instead of killing the run before TOTAL.
for dt in (torch.complex64, torch.complex128):
    native_err = _native_abs_reject[dt]
    if native_err is None:
        failed += 1
        print(f"[FAIL] abs_ {dt}: native oracle expected RuntimeError", flush=True)
        continue
    x = native[("c", dt, "rej")]
    try:
        x.clone().abs_()
        failed += 1
        print(f"[FAIL] abs_ {dt}: expected RuntimeError (native: {native_err!r})", flush=True)
    except Exception as e:
        if type(e) is RuntimeError and str(e) == native_err:
            passed += 1
            print(f"[PASS] abs_ {dt} rejected identical to native: {e}", flush=True)
        else:
            failed += 1
            print(f"[FAIL] abs_ {dt}: got {type(e).__name__}: {e!r}, native {native_err!r}", flush=True)

print(f"\nTOTAL: {passed + failed} passed, {failed} failed", flush=True)
if failed:
    os._exit(1)
print("ALL COMPLEX OPS REGRESSION PASSED", flush=True)
os._exit(0)
