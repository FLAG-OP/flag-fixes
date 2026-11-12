"""Regression battery: complex64/128 support across sum, the zeros family,
abs/absolute, plus real-dtype regressions (torch-vs-native oracle).

Pre-fix : every complex case fails with KeyError 'complex64'/'complex128'
          from triton/_utils.py type_canonicalisation_dict (baseline); real
          cases pass -> non-zero exit, TOTAL shows failures.
Post-fix: all complex paths run in Triton (real-view lowering where needed),
          all real paths unchanged -> ``TOTAL n passed, 0 failed, n total``
          exit 0.

Suites (review round 1):
  * zero_ strided views  -- slice[::2]/[1::2]/view-col/narrow/transpose
    cases must zero exactly the selected storage elements of the underlying
    tensor (the old view_as_real + dense-kernel path corrupted neighbors),
    via flag_gems.zero_, flag_gems.zero (zero.py) and aten.zero_; identity
    (return is input) asserted.
  * abs magnitude edges  -- 1e20+1e20j / 1e-30+1e-30j class values must match
    native hypot (naive sqrt over/underflows float32); rtol=1e-6/atol=0 so a
    wrong 0.0 does not pass, plus inf/nan special values (equal_nan).
  * sum.out cross-dtype  -- dim-form out-dtype adoption vs full-form dtype
    requirement, wrong-dtype RuntimeError identity (aten message), resize of
    a wrong-shape out: every case oracle-driven from native, ok and err
    categories must match bidirectionally.
  * autograd parity      -- local complex chains abs().sum(), sum().abs(),
    (x*2).abs().sum() must produce finite grads equal to native within
    1e-4 (end in a real scalar; native rejects complex-scalar backward).
  * review round 2       -- dim-sum with an out adopts out.dtype (bool out
    works for bool/real inputs, full sum.out does not adopt and raises),
    complex->bool uses the any-component nonzero rule (sum([0+1j],
    dtype=bool) is True), empty inputs yield zeros without launching a
    reduction (no ZeroDivisionError), and abs/sum resolve an unresolved
    conj bit before reading storage (conj views are built pre-enable).

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


def check(name, fn, oracle, *, rtol=1e-5, atol=1e-5, equal_nan=False):
    global passed, failed
    try:
        got = fn()
        assert got.dtype == oracle.dtype, f"dtype {got.dtype} != {oracle.dtype}"
        g = torch.view_as_real(got) if got.is_complex() else got
        w = torch.view_as_real(oracle) if oracle.is_complex() else oracle
        if g.numel():
            torch.testing.assert_close(
                g, w, rtol=rtol, atol=atol, equal_nan=equal_nan
            )
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

# --- review P1a: zero_ on strided complex views (native oracle per case) ---
# For every (dtype, view-kind) pair: pattern input, a view-builder applied to
# a fresh clone of the pattern, and the underlying base tensor after native
# (pre-enable) zeroing of the view -- flag_gems must reproduce the base
# exactly, including the elements the view does NOT cover.
_zero_stride = {}
for _dt in (torch.complex64, torch.complex128):
    _pat = torch.tensor(
        [1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j, 9 + 10j, 11 + 12j, 13 + 14j, 15 + 16j],
        dtype=_dt,
        device=DEV,
    )

    def _view_even(b):
        return b[::2]

    def _view_odd(b):
        return b[1::2]

    def _view_col(b):
        return b.view(4, 2)[:, 1]

    def _view_narrow(b):
        return b.narrow(0, 2, 4)

    def _view_transpose(b):
        return b.view(4, 2).t()

    for _cname, _mk in (
        ("slice[::2]", _view_even),
        ("slice[1::2]", _view_odd),
        ("view-col", _view_col),
        ("narrow-offset", _view_narrow),
        ("transpose", _view_transpose),
    ):
        _ob = _pat.clone()
        _mk(_ob).zero_()  # native oracle base, pre-enable
        _zero_stride[(_dt, _cname)] = (_pat, _mk, _ob)
    _p0 = torch.tensor(5 - 6j, dtype=_dt, device=DEV)
    _b0 = _p0.clone()
    _b0.zero_()
    _zero_stride[(_dt, "0-dim")] = (_p0, lambda b: b, _b0)

# --- review P1b: abs magnitude edges (native hypot oracle, pre-enable) ---
_mag_vals = {
    torch.complex64: [
        1e20 + 1e20j,          # naive sqrt overflows float32 -> inf
        1e-30 - 1e-30j,        # naive sqrt underflows to 0
        1e20 + 1e-20j,         # mixed scale
        3.0 + 4.0j,
        -3.0 - 4.0j,
        0.0 + 0.0j,
        -0.0 - 0.0j,
        1e-38 + 1e-38j,
        3.4e38 + 1.0j,         # near float32 max
        float("inf") + 0.0j,
        0.0 + float("inf") * 1j,
        float("inf") + float("nan") * 1j,  # hypot: inf wins over nan
        float("nan") + 1.0j,
    ],
    torch.complex128: [
        1e200 + 1e200j,
        1e-200 - 1e-200j,
        1e200 + 1e-100j,
        3.0 + 4.0j,
        0.0 + 0.0j,
        -0.0 - 0.0j,
        1e-320 + 1e-320j,      # subnormal
        1.7e308 + 1.0j,
        float("inf") + 0.0j,
        float("inf") + float("nan") * 1j,
        float("nan") + 1.0j,
    ],
}
_mag = {}
for _dt, _vals in _mag_vals.items():
    _tin = torch.tensor(_vals, dtype=_dt, device=DEV)
    _mag[_dt] = (_tin, torch.abs(_tin))  # native oracle, pre-enable

# --- review P2: sum.out / sum.dim_IntList.out cross-dtype oracle matrix ---
# Each case is a zero-arg runner that builds fresh input+out and performs one
# call; the native outcome (ok value or exception type+message) is captured
# pre-enable and the same runner is re-run under flag_gems.
_sumout_cases = []


def _so(label, run):
    _sumout_cases.append((label, run))


for _dt in (torch.complex64, torch.complex128):
    _ci = torch.tensor(
        [[1 + 2j, 3 - 4j, 0 - 0.5j], [-2 + 7j, 0 + 0j, 4 - 1j]],
        dtype=_dt,
        device=DEV,
    )
    _co = torch.complex128 if _dt == torch.complex64 else torch.complex64
    _so(f"{_dt} dim0 out=same",
        lambda _ci=_ci, _dt=_dt: torch.sum(_ci, 0, out=torch.empty(3, dtype=_dt, device=DEV)))
    _so(f"{_dt} dim0 out=other-complex adopt",
        lambda _ci=_ci, _co=_co: torch.sum(_ci, 0, out=torch.empty(3, dtype=_co, device=DEV)))
    _so(f"{_dt} dim0 dtype=c128 out=c128",
        lambda _ci=_ci: torch.sum(_ci, 0, dtype=torch.complex128,
                                  out=torch.empty(3, dtype=torch.complex128, device=DEV)))
    _so(f"{_dt} dim0 dtype=c128 out=c64 ERR",
        lambda _ci=_ci: torch.sum(_ci, 0, dtype=torch.complex128,
                                  out=torch.empty(3, dtype=torch.complex64, device=DEV)))
    _so(f"{_dt} dim0 dtype=f32 out=f32",
        lambda _ci=_ci: torch.sum(_ci, 0, dtype=torch.float32,
                                  out=torch.empty(3, dtype=torch.float32, device=DEV)))
    _so(f"{_dt} dim0 dtype=f64 out=f32 ERR",
        lambda _ci=_ci: torch.sum(_ci, 0, dtype=torch.float64,
                                  out=torch.empty(3, dtype=torch.float32, device=DEV)))
    _so(f"{_dt} dim0 out=f32 adopt",
        lambda _ci=_ci: torch.sum(_ci, 0, out=torch.empty(3, dtype=torch.float32, device=DEV)))
    _so(f"{_dt} dim0 out=f64 adopt",
        lambda _ci=_ci: torch.sum(_ci, 0, out=torch.empty(3, dtype=torch.float64, device=DEV)))
    _so(f"{_dt} dim0 out=i64 adopt",
        lambda _ci=_ci: torch.sum(_ci, 0, out=torch.empty(3, dtype=torch.int64, device=DEV)))
    _so(f"{_dt} dim(0,1) out=same",
        lambda _ci=_ci, _dt=_dt: torch.sum(_ci, (0, 1), out=torch.empty((), dtype=_dt, device=DEV)))
    _so(f"{_dt} dim(0,1) out=c128 adopt",
        lambda _ci=_ci: torch.sum(_ci, (0, 1),
                                  out=torch.empty((), dtype=torch.complex128, device=DEV)))
    _so(f"{_dt} dim0 keepdim out=(1,3)",
        lambda _ci=_ci, _dt=_dt: torch.sum(_ci, 0, keepdim=True,
                                           out=torch.empty(1, 3, dtype=_dt, device=DEV)))
    _so(f"{_dt} dim0 out=(2,1) resize",
        lambda _ci=_ci, _dt=_dt: torch.sum(_ci, 0,
                                           out=torch.empty(2, 1, dtype=_dt, device=DEV)))
    _so(f"{_dt} empty dim0 out",
        lambda _dt=_dt: torch.sum(torch.empty(0, 3, dtype=_dt, device=DEV), 0,
                                  out=torch.empty(3, dtype=_dt, device=DEV)))
    _so(f"{_dt} aten full out=same",
        lambda _ci=_ci, _dt=_dt: torch.ops.aten.sum.out(_ci, out=torch.empty((), dtype=_dt, device=DEV)))
    _so(f"{_dt} aten full out=other-complex ERR",
        lambda _ci=_ci, _co=_co: torch.ops.aten.sum.out(_ci, out=torch.empty((), dtype=_co, device=DEV)))
    _so(f"{_dt} aten full dtype=c128 out=c64 ERR",
        lambda _ci=_ci: torch.ops.aten.sum.out(_ci, dtype=torch.complex128,
                                               out=torch.empty((), dtype=torch.complex64, device=DEV)))
    _so(f"{_dt} aten full dtype=c128 out=c128",
        lambda _ci=_ci: torch.ops.aten.sum.out(_ci, dtype=torch.complex128,
                                               out=torch.empty((), dtype=torch.complex128, device=DEV)))
    _so(f"{_dt} aten full out=i64 ERR",
        lambda _ci=_ci: torch.ops.aten.sum.out(_ci, out=torch.empty((), dtype=torch.int64, device=DEV)))

_rf = torch.randn(2, 3, device=DEV)
_so("f32 dim0 out=i64 adopt",
    lambda: torch.sum(_rf, 0, out=torch.empty(3, dtype=torch.int64, device=DEV)))
_so("f32 dim0 out=f64 adopt",
    lambda: torch.sum(_rf, 0, out=torch.empty(3, dtype=torch.float64, device=DEV)))
_so("f32 dim0 out=c64 adopt",
    lambda: torch.sum(_rf, 0, out=torch.empty(3, dtype=torch.complex64, device=DEV)))
_so("f32 dim0 dtype=f64 out=f32 ERR",
    lambda: torch.sum(_rf, 0, dtype=torch.float64,
                      out=torch.empty(3, dtype=torch.float32, device=DEV)))
_so("f32 aten full out=f64 ERR",
    lambda: torch.ops.aten.sum.out(_rf, out=torch.empty((), dtype=torch.float64, device=DEV)))
_so("f32 aten full out=f32",
    lambda: torch.ops.aten.sum.out(_rf, out=torch.empty((), dtype=torch.float32, device=DEV)))
_so("f32 aten full out=i64 ERR",
    lambda: torch.ops.aten.sum.out(_rf, out=torch.empty((), dtype=torch.int64, device=DEV)))

# --- review round 2: bool out adoption / complex->bool / empty inputs ---
# Dim-form sum adopts out.dtype (including bool); full-form sum.out does not
# adopt and raises. Complex->bool follows c10 TypeCast (any component
# nonzero -> True). Empty inputs must produce zeros of the reduced shape
# without launching a reduction.
_xb = torch.tensor([[True, False, True], [False, False, True]], device=DEV)
_xb2 = torch.tensor([True, True], device=DEV)
_xb3 = torch.tensor([True, True, True], device=DEV)
_xe = torch.zeros(0, dtype=torch.bool, device=DEV)
_xe3 = torch.zeros(0, 3, dtype=torch.bool, device=DEV)
_re = torch.empty(0, dtype=torch.float32, device=DEV)
_pure = torch.tensor([0 + 1j, 0 + 1j], device=DEV)
_pure128 = torch.tensor([0 + 1j], dtype=torch.complex128, device=DEV)
_mixed = torch.tensor([0 + 1j, 1 + 1j, 0 + 0j, 2 + 0j], device=DEV)
_c22 = torch.tensor([[0 + 1j, 0 + 2j], [0 + 3j, 0 + 4j]], device=DEV)
_cz = torch.zeros(2, dtype=torch.complex64, device=DEV)
_ce = torch.empty(0, dtype=torch.complex64, device=DEV)
# cast-first nonzero-rule probes: sum-then-cast would give False/0 where
# ATen casts each element to True first (0.5 -> True, not truncation).
_cancel = torch.tensor([[0.5, 10.0], [-0.5, 10.0]], device=DEV)
_cancel_full = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], device=DEV)
_frac = torch.tensor([0.5 + 0.5j], device=DEV)

_so("f32 cancel dim0 out=bool",
    lambda: torch.sum(_cancel, 0, out=torch.empty(2, dtype=torch.bool, device=DEV)))
_so("f32 cancel dim0 dtype=bool",
    lambda: torch.sum(_cancel, 0, dtype=torch.bool))
_so("f32 cancel full dtype=bool",
    lambda: torch.sum(_cancel_full, dtype=torch.bool))
_so("f32 cancel aten full dtype=bool out=bool",
    lambda: torch.ops.aten.sum.out(_cancel_full, dtype=torch.bool,
                                   out=torch.empty((), dtype=torch.bool, device=DEV)))
_so("c64 frac dim0 dtype=bool",
    lambda: torch.sum(_frac, 0, dtype=torch.bool))
_so("c64 frac all dtype=bool out bool",
    lambda: torch.sum(_frac, dtype=torch.bool,
                      out=torch.empty((), dtype=torch.bool, device=DEV)))
_so("f32 empty dim0 dtype=bool",
    lambda: torch.sum(_re, 0, dtype=torch.bool))

_so("bool dim0 out=bool adopt",
    lambda: torch.sum(_xb, 0, out=torch.empty(3, dtype=torch.bool, device=DEV)))
_so("bool dim0 dtype=bool out=bool",
    lambda: torch.sum(_xb, 0, dtype=torch.bool,
                      out=torch.empty(3, dtype=torch.bool, device=DEV)))
_so("bool dim0 out=i64 adopt",
    lambda: torch.sum(_xb, 0, out=torch.empty(3, dtype=torch.int64, device=DEV)))
_so("bool dim0 out=f32 adopt",
    lambda: torch.sum(_xb, 0, out=torch.empty(3, dtype=torch.float32, device=DEV)))
_so("bool dim0 dtype=f32 out=f32",
    lambda: torch.sum(_xb, 0, dtype=torch.float32,
                      out=torch.empty(3, dtype=torch.float32, device=DEV)))
_so("bool dim0 dtype=f32 out=bool ERR",
    lambda: torch.sum(_xb, 0, dtype=torch.float32,
                      out=torch.empty(3, dtype=torch.bool, device=DEV)))
_so("bool dim0 dtype=bool out=i64 ERR",
    lambda: torch.sum(_xb, 0, dtype=torch.bool,
                      out=torch.empty(3, dtype=torch.int64, device=DEV)))
_so("bool dim(0,1) out=bool",
    lambda: torch.sum(_xb, (0, 1), out=torch.empty((), dtype=torch.bool, device=DEV)))
_so("bool dim0 keepdim out bool",
    lambda: torch.sum(_xb, 0, keepdim=True,
                      out=torch.empty(1, 3, dtype=torch.bool, device=DEV)))
_so("bool count2 out=bool",
    lambda: torch.sum(_xb2, 0, out=torch.empty((), dtype=torch.bool, device=DEV)))
_so("bool count3 out=bool",
    lambda: torch.sum(_xb3, 0, out=torch.empty((), dtype=torch.bool, device=DEV)))
_so("bool aten full out=bool ERR",
    lambda: torch.ops.aten.sum.out(_xb, out=torch.empty((), dtype=torch.bool, device=DEV)))
_so("bool aten full dtype=bool out=bool",
    lambda: torch.ops.aten.sum.out(_xb2, dtype=torch.bool,
                                   out=torch.empty((), dtype=torch.bool, device=DEV)))
_so("bool aten full out=i64",
    lambda: torch.ops.aten.sum.out(_xb2, out=torch.empty((), dtype=torch.int64, device=DEV)))
_so("f32 dim0 out=bool adopt",
    lambda: torch.sum(_rf, 0, out=torch.empty(3, dtype=torch.bool, device=DEV)))
_so("bool empty dim0 out=bool",
    lambda: torch.sum(_xe3, 0, out=torch.empty(3, dtype=torch.bool, device=DEV)))
_so("bool empty dim0 out=i64",
    lambda: torch.sum(_xe, 0, out=torch.empty((), dtype=torch.int64, device=DEV)))
_so("bool empty dim0 out=f32",
    lambda: torch.sum(_xe, 0, out=torch.empty((), dtype=torch.float32, device=DEV)))
_so("bool empty dim0 dtype=bool out=bool",
    lambda: torch.sum(_xe, 0, dtype=torch.bool,
                      out=torch.empty((), dtype=torch.bool, device=DEV)))
_so("bool empty dim0 dtype=f32 out=i64 ERR",
    lambda: torch.sum(_xe, 0, dtype=torch.float32,
                      out=torch.empty((), dtype=torch.int64, device=DEV)))
_so("bool empty dim1 out=bool",
    lambda: torch.sum(_xe3, 1, out=torch.empty(0, dtype=torch.bool, device=DEV)))
_so("bool empty dim0 keepdim out bool",
    lambda: torch.sum(_xe, 0, keepdim=True,
                      out=torch.empty(1, dtype=torch.bool, device=DEV)))
_so("bool empty full out=i64",
    lambda: torch.ops.aten.sum.out(_xe, out=torch.empty((), dtype=torch.int64, device=DEV)))
_so("bool empty full out=bool ERR",
    lambda: torch.ops.aten.sum.out(_xe, out=torch.empty((), dtype=torch.bool, device=DEV)))
_so("bool empty full dtype=bool out=bool",
    lambda: torch.ops.aten.sum.out(_xe, dtype=torch.bool,
                                   out=torch.empty((), dtype=torch.bool, device=DEV)))
_so("bool empty no-dim",
    lambda: torch.sum(_xe))
_so("bool empty no-dim dtype=bool",
    lambda: torch.sum(_xe, dtype=torch.bool))
_so("bool empty dim0 dtype=bool",
    lambda: torch.sum(_xe, 0, dtype=torch.bool))
_so("f32 empty dim0 out=f32",
    lambda: torch.sum(_re, 0, out=torch.empty((), dtype=torch.float32, device=DEV)))
_so("f32 empty dim0 out=i64",
    lambda: torch.sum(_re, 0, out=torch.empty((), dtype=torch.int64, device=DEV)))
_so("c64 pureimag dim0 out=bool adopt",
    lambda: torch.sum(_pure, 0, out=torch.empty((), dtype=torch.bool, device=DEV)))
_so("c64 pureimag dim0 dtype=bool out=bool",
    lambda: torch.sum(_pure, 0, dtype=torch.bool,
                      out=torch.empty((), dtype=torch.bool, device=DEV)))
_so("c64 pureimag dim0 dtype=bool",
    lambda: torch.sum(_pure, 0, dtype=torch.bool))
_so("c64 pureimag all dtype=bool",
    lambda: torch.sum(_pure, dtype=torch.bool))
_so("c128 pureimag dim0 dtype=bool out=bool",
    lambda: torch.sum(_pure128, 0, dtype=torch.bool,
                      out=torch.empty((), dtype=torch.bool, device=DEV)))
_so("c64 mixed dim0 dtype=bool out=bool",
    lambda: torch.sum(_mixed, 0, dtype=torch.bool,
                      out=torch.empty((), dtype=torch.bool, device=DEV)))
_so("c64 mixed all dtype=bool",
    lambda: torch.sum(_mixed, dtype=torch.bool))
_so("c64 c22 dim0 dtype=bool out bool",
    lambda: torch.sum(_c22, 0, dtype=torch.bool,
                      out=torch.empty(2, dtype=torch.bool, device=DEV)))
_so("c64 c22 dim1 dtype=bool out bool",
    lambda: torch.sum(_c22, 1, dtype=torch.bool,
                      out=torch.empty(2, dtype=torch.bool, device=DEV)))
_so("c64 allzero dim0 dtype=bool out bool",
    lambda: torch.sum(_cz, 0, dtype=torch.bool,
                      out=torch.empty((), dtype=torch.bool, device=DEV)))
_so("c64 dim0 dtype=bool out=i64 ERR",
    lambda: torch.sum(_mixed, 0, dtype=torch.bool,
                      out=torch.empty((), dtype=torch.int64, device=DEV)))
_so("c64 empty dim0 dtype=bool out bool",
    lambda: torch.sum(_ce, 0, dtype=torch.bool,
                      out=torch.empty((), dtype=torch.bool, device=DEV)))
_so("c64 empty full dtype=bool out bool",
    lambda: torch.ops.aten.sum.out(_ce, dtype=torch.bool,
                                   out=torch.empty((), dtype=torch.bool, device=DEV)))
_so("c64 empty full out=bool ERR",
    lambda: torch.ops.aten.sum.out(_ce, out=torch.empty((), dtype=torch.bool, device=DEV)))

_sumout_oracles = {}
for _label, _run in _sumout_cases:
    try:
        _sumout_oracles[_label] = ("ok", _run().clone())
    except Exception as _e:
        _sumout_oracles[_label] = ("err", type(_e), str(_e))

# --- review P2: local complex autograd parity (native grads, pre-enable) ---
_ag_chains = {
    "abs().sum()": lambda t: t.abs().sum(),
    "sum().abs()": lambda t: t.sum().abs(),
    "(x*2).abs().sum()": lambda t: (t * 2).abs().sum(),
}
_ag = {}
for _dt in (torch.complex64, torch.complex128):
    _x0 = torch.complex(torch.randn(4, 5, device=DEV), torch.randn(4, 5, device=DEV))
    _x0 = _x0.to(_dt)
    for _cname, _chain in _ag_chains.items():
        _xn = _x0.clone().requires_grad_()
        _y = _chain(_xn)
        _y.backward()
        _ag[(_dt, _cname)] = (_x0, _y.detach().clone(), _xn.grad.clone())

import flag_gems  # noqa: E402

# --- review round 2: unresolved conj views (built pre-enable) ---
# Post-enable .conj() materializes under flag_gems, so the conj bit must be
# captured here; abs and sum must resolve it before reading storage.
_conj = {}
for _dt in (torch.complex64, torch.complex128):
    _cbase = torch.tensor(
        [[1 + 2j, 5 + 6j], [3 + 4j, 7 + 8j]], dtype=_dt, device=DEV
    )
    _pc = _cbase.conj()
    # round 3: _cbase[::2] leaves one row, which torch still counts as
    # contiguous; a column slice is genuinely non-contiguous, so assert
    # both strided-sample premises before enable.
    _pcs = _cbase[:, ::2].conj()
    assert _pc.is_conj()
    assert not _pcs.is_contiguous()
    assert _pcs.is_conj()
    _conj[(_dt, "abs")] = (_pc, torch.abs(_pc))
    _conj[(_dt, "abs_strided")] = (_pcs, torch.abs(_pcs))
    _conj[(_dt, "sum")] = (_pc, torch.sum(_pc))
    _conj[(_dt, "sum_dim0")] = (_pc, torch.sum(_pc, 0))
    _conj[(_dt, "sum_strided")] = (_pcs, torch.sum(_pcs))

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

# --- review P1a: zero_ must be stride-correct for complex views ---
for (_dt, _cname), (_pat, _mk, _ob) in _zero_stride.items():
    def _run_zero(_pat=_pat, _mk=_mk):
        b = _pat.clone()
        v = _mk(b)
        r = gops.zero_(v)
        assert r is v, "zero_ must return its input"
        return b
    check(f"zero_ strided {_dt} {_cname}", _run_zero, _ob)

    def _run_zero_py(_pat=_pat, _mk=_mk):
        b = _pat.clone()
        v = _mk(b)
        r = gops.zero(v)
        assert r is v, "zero must return its input"
        return b
    check(f"zero (zero.py) strided {_dt} {_cname}", _run_zero_py, _ob)

    if _cname == "slice[::2]":
        def _run_aten_zero(_pat=_pat, _mk=_mk):
            b = _pat.clone()
            torch.ops.aten.zero_(_mk(b))
            return b
        check(f"aten.zero_ strided {_dt} {_cname}", _run_aten_zero, _ob)

# --- review P1b: abs magnitude edges (hypot-stable, no overflow/underflow) ---
for _dt, (_tin, _oracle) in _mag.items():
    _mag_rtol = 1e-6 if _dt == torch.complex64 else 1e-10
    check(
        f"abs magnitude edges {_dt}",
        lambda _tin=_tin: torch.abs(_tin),
        _oracle,
        rtol=_mag_rtol,
        atol=0.0,
        equal_nan=True,
    )


# --- review P2: sum.out matrix, oracle-driven in both directions ---
def sumout_case(label, run, orc):
    global passed, failed
    try:
        got = run()
        got_err = None
    except Exception as e:
        got, got_err = None, e
    try:
        if orc[0] == "err":
            _want_type, _want_msg = orc[1], orc[2]
            assert got_err is not None, f"expected {_want_type.__name__}, got ok result {got!r}"
            assert type(got_err) is _want_type, (
                f"error type {type(got_err).__name__} != native {_want_type.__name__}: {got_err!r}"
            )
            assert str(got_err) == _want_msg, (
                f"error message {str(got_err)!r} != native {_want_msg!r}"
            )
        else:
            want = orc[1]
            assert got_err is None, f"unexpected {type(got_err).__name__}: {got_err!r}"
            assert got.dtype == want.dtype, f"dtype {got.dtype} != {want.dtype}"
            assert got.shape == want.shape, f"shape {tuple(got.shape)} != {tuple(want.shape)}"
            g = torch.view_as_real(got) if got.is_complex() else got
            w = torch.view_as_real(want) if want.is_complex() else want
            if g.numel():
                torch.testing.assert_close(g, w, rtol=1e-5, atol=1e-5)
        passed += 1
        print(f"[PASS] sum.out matrix {label}", flush=True)
    except Exception as e:
        failed += 1
        print(f"[FAIL] sum.out matrix {label}: {e!r}", flush=True)
        if os.environ.get("VERBOSE"):
            traceback.print_exc()


for _label, _run in _sumout_cases:
    sumout_case(_label, _run, _sumout_oracles[_label])

# --- review round 2: conj materialization in abs and sum ---
for (_dt, _cname), (_tin, _oracle) in _conj.items():
    if _cname in ("abs", "abs_strided"):
        check(f"conj {_cname} {_dt} direct",
              lambda _tin=_tin: gops.abs(_tin), _oracle, rtol=1e-6, atol=0.0)
        check(f"conj {_cname} {_dt} dispatch",
              lambda _tin=_tin: torch.abs(_tin), _oracle, rtol=1e-6, atol=0.0)
    elif _cname == "sum_dim0":
        check(f"conj {_cname} {_dt} direct",
              lambda _tin=_tin: gops.sum_dim(_tin, 0), _oracle)
        check(f"conj {_cname} {_dt} dispatch",
              lambda _tin=_tin: torch.sum(_tin, 0), _oracle)
    else:  # sum, sum_strided
        check(f"conj {_cname} {_dt} direct",
              lambda _tin=_tin: gops.sum(_tin), _oracle)
        check(f"conj {_cname} {_dt} dispatch",
              lambda _tin=_tin: torch.sum(_tin), _oracle)

# --- review P2: local complex autograd parity vs native ---
for (_dt, _cname), (_x0, _y_nat, _g_nat) in _ag.items():
    def _ag_grad(_x0=_x0, _cname=_cname):
        xn = _x0.clone().requires_grad_()
        _ag_chains[_cname](xn).backward()
        assert xn.grad is not None, "gradient missing"
        return xn.grad
    check(f"autograd grad {_cname} {_dt}", _ag_grad, _g_nat, rtol=1e-4, atol=1e-6)

    def _ag_fwd(_x0=_x0, _cname=_cname):
        return _ag_chains[_cname](_x0).detach()
    check(f"autograd fwd {_cname} {_dt}", _ag_fwd, _y_nat, rtol=1e-4, atol=1e-6)

print(f"\nTOTAL: {passed} passed, {failed} failed, {passed + failed} total", flush=True)
if failed:
    os._exit(1)
print("ALL COMPLEX OPS REGRESSION PASSED", flush=True)
os._exit(0)
