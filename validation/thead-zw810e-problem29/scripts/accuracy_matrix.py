"""ZW810E problem29 accuracy matrix v3: gelu(none/tanh) fwd+bwd, gelu_and_mul.

Phases (single process, deterministic inputs built up-front so both modes see
bit-identical inputs):
  P0  build all inputs (no flag_gems import yet -> native RNG)
  P1  native aten reference outputs/grads -> CPU  (reference first, per issue #5)
      + fp64 oracle (out & grad) computed on-device
  P2  import flag_gems + enable, recompute THROUGH ATEN (autograd-aware,
      profiler-proven to hit flag_gems triton kernels) and compare.

Verdict (documented gates, per issue #5 "明确精度门槛后执行"):
  A specials : NaN/Inf masks must equal native; grad presence must match.
  B accuracy : gems must not be meaningfully less accurate than native:
      rel_err(gems, oracle) <= max(4*eps(dtype), 1.5 * rel_err(native, oracle)),
      evaluated on finite positions with |oracle| >= denormal floor of dtype.
  C FTZ zone : positions whose oracle output is subnormal (< dtype min normal):
      native and gems must agree bitwise (both flush or both keep).
  D grads    : same B/C rule on x.grad.
  zero-sign mismatch is recorded as a note (IEEE 0 == -0).
  gelu_and_mul has no autograd path: forward gates only, grad = N/A.
"""
import argparse
import hashlib
import json
import math
import zlib

import torch
import torch.nn.functional as F

MIN_NORMAL = {"float32": 1.1754943508222875e-38, "bfloat16": 1.1754943508222875e-38, "float16": 6.103515625e-05}
EPS = {"float32": 1.1920929e-07, "bfloat16": 0.0078125, "float16": 0.0009765625}
DTYPE_BITS = {
    "float16": {"max_sub": 0x03FF, "min_norm": 0x0400, "min_sub": 0x0001, "inf": 0x7C00, "nan": 0x7E00},
    "bfloat16": {"max_sub": 0x007F, "min_norm": 0x0080, "min_sub": 0x0001, "inf": 0x7F80, "nan": 0x7FC0},
    "float32": {"max_sub": 0x007FFFFF, "min_norm": 0x00800000, "min_sub": 0x00000001, "inf": 0x7F800000, "nan": 0x7FC00000},
}


def bits_tensor(pairs, dtype):
    vals = []
    for bits, neg in pairs:
        if dtype == "float16":
            u = torch.tensor([bits], dtype=torch.int16).view(torch.float16).item()
        elif dtype == "bfloat16":
            u = torch.tensor([bits], dtype=torch.int16).view(torch.bfloat16).item()
        else:
            u = torch.tensor([bits], dtype=torch.int32).view(torch.float32).item()
        vals.append(-u if neg else u)
    return torch.tensor(vals, dtype=getattr(torch, dtype)).repeat(1, 8)


def boundary_pairs(dtype):
    b = DTYPE_BITS[dtype]
    return [
        (b["max_sub"], False), (b["max_sub"], True),
        (b["min_norm"], False), (b["min_norm"], True),
        (b["min_sub"], False), (b["min_sub"], True),
        (0x0000, False), (0x0000, True),
        (b["inf"], False), (b["inf"], True), (b["nan"], False),
    ]


def make_value_class(cls, shape, dtype):
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cls.encode()))
    dt = getattr(torch, dtype)
    if cls == "empty":
        return torch.empty(*shape, device="cuda", dtype=dt)
    if cls == "boundary":
        t = bits_tensor(boundary_pairs(dtype), dtype)
        flat = t.reshape(1, -1).repeat(1, math.ceil(math.prod(shape) / t.numel()) + 1)
        return flat[:, : math.prod(shape)].reshape(shape).cuda()
    if cls == "small":
        return torch.randn(shape, generator=g, device="cuda", dtype=dt) * 0.01
    if cls == "large":
        return torch.randn(shape, generator=g, device="cuda", dtype=dt) * 10
    return torch.randn(shape, generator=g, device="cuda", dtype=dt)


def apply_layout(x, layout):
    if layout == "contig":
        return x
    if layout == "transposed":
        return x.clone().t().contiguous().t()
    if layout == "sliced":
        base = torch.cat([x, x], dim=1).contiguous()
        return base[:, ::2]
    raise ValueError(layout)


def sha256_tensor(t):
    if t.numel() == 0:
        return "empty"
    return hashlib.sha256(t.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def mask(t, kind):
    f = t.float()
    if kind == "nan":
        return torch.isnan(f)
    if kind == "inf":
        return torch.isinf(f)
    if kind == "posinf":
        return torch.isposinf(f)
    if kind == "neginf":
        return torch.isinf(f) & torch.signbit(f)
    if kind == "negzero":
        return (f == 0) & torch.signbit(f)
    if kind == "poszero":
        return (f == 0) & ~torch.signbit(f)
    raise ValueError(kind)


def oracle_out(x, op, b=None):
    xx = x.double()
    if op == "gelu_none":
        y = xx * 0.5 * (1.0 + torch.erf(xx / math.sqrt(2.0)))
    elif op == "gelu_tanh":
        u = math.sqrt(2.0 / math.pi) * (xx + 0.044715 * xx**3)
        y = xx * 0.5 * (1.0 + torch.tanh(u))
    else:
        raise ValueError(op)
    if b is not None:
        y = y * b.double()
    return y


def oracle_grad(x, op, b=None):
    xx = x.double()
    if op == "gelu_none":
        phi = 0.5 * (1.0 + torch.erf(xx / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * xx * xx) / math.sqrt(2.0 * math.pi)
        g = phi + xx * pdf
    else:
        u = math.sqrt(2.0 / math.pi) * (xx + 0.044715 * xx**3)
        t = torch.tanh(u)
        du = math.sqrt(2.0 / math.pi) * (1.0 + 3 * 0.044715 * xx * xx)
        g = 0.5 * (1.0 + t) + xx * 0.5 * (1.0 - t * t) * du
    if b is not None:
        g = g * b.double()
    return g


def rel_err(a_cpu_f, oracle):
    """a: float cpu tensor; oracle: fp64 cpu tensor, same shape."""
    if a_cpu_f.numel() == 0:
        return 0.0
    af = a_cpu_f.double()
    fin = torch.isfinite(af) & torch.isfinite(oracle)
    if not fin.any():
        return 0.0
    denom = oracle[fin].abs().clamp_min(1e-300)
    return float(((af[fin] - oracle[fin]).abs() / denom).max())


def dt_ulp(a, b, dtype):
    """exact ulp distance in the ORIGINAL dtype via ordinal mapping."""
    import numpy as np

    if a.numel() == 0:
        return 0
    nbits = 16 if dtype != "float32" else 32
    ai = a.view(torch.uint16 if nbits == 16 else torch.uint32).numpy().astype(np.int64)
    bi = b.view(torch.uint16 if nbits == 16 else torch.uint32).numpy().astype(np.int64)
    sign_hi = 1 << (nbits - 1)
    def to_ord(x):
        pos = (x & (sign_hi - 1))
        return np.where(x & sign_hi == 0, pos + sign_hi, ~pos & (2 * sign_hi - 1))
    fa, fb = a.float(), b.float()
    m = ~(torch.isnan(fa) | torch.isnan(fb) | torch.isinf(fa) | torch.isinf(fb)).numpy()
    if not m.any():
        return 0
    return int(np.abs(to_ord(ai) - to_ord(bi))[m].max())


CASES = []


def build_cases():
    shapes = {"small": (17, 33), "typical": (2048, 1280), "empty": (0, 128), "boundary": (64, 128), "large": (2048, 1280)}
    for op in ("gelu_none", "gelu_tanh", "gelu_and_mul"):
        for dtype in ("float32", "bfloat16", "float16"):
            for layout in ("contig", "transposed", "sliced"):
                for cls in ("typical", "boundary", "small", "large", "empty"):
                    if op == "gelu_and_mul" and cls == "empty":
                        continue
                    CASES.append((f"{op}|{dtype}|{layout}|{cls}", (op, dtype, layout, cls), shapes[cls]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print("accuracy_matrix v5 (adds sign-aware Inf comparison to hard gate; supersedes v4): gates identical to v3; additive per-case evidence fields "
          "(worst position/bits, signed-zero count, FTZ definition, special counts). "
          "v1 (autograd bug) and v2 (ULP walker + NaN-in-diff) batches are VOID; "
          "this run is the final full re-run on identical deterministic inputs.", flush=True)

    build_cases()

    inputs = {}
    for name, (op, dtype, layout, cls), shape in CASES:
        if op == "gelu_and_mul":
            a = apply_layout(make_value_class(cls, shape, dtype), layout)
            b = apply_layout(make_value_class(cls + "_mul", shape, dtype), layout)
            inputs[name] = (a, b)
        else:
            inputs[name] = (apply_layout(make_value_class(cls, shape, dtype), layout),)

    # ---- P1: native reference + fp64 oracle ----
    native, oracles = {}, {}
    for name, (op, dtype, layout, cls), _ in CASES:
        x = inputs[name][0]
        try:
            if op == "gelu_and_mul":
                b = inputs[name][1]
                y = F.gelu(x) * b
                torch.cuda.synchronize()
                native[name] = {"out": y.detach().cpu().clone(), "grad": None}
                oracles[name] = {"out": (oracle_out(x, "gelu_none") * b.double()).cpu()}
            else:
                xv = x.clone().requires_grad_(True)
                y = F.gelu(xv, approximate="none" if op == "gelu_none" else "tanh")
                y.sum().backward()
                torch.cuda.synchronize()
                native[name] = {"out": y.detach().cpu().clone(), "grad": xv.grad.detach().cpu().clone()}
                oracles[name] = {"out": oracle_out(x, op).cpu(), "grad": oracle_grad(x, op).cpu()}
        except Exception as e:
            native[name] = {"error": f"{type(e).__name__}: {e}"}

    # ---- P2: flag_gems (aten path) ----
    import flag_gems

    flag_gems.enable()
    gems = {}
    for name, (op, dtype, layout, cls), _ in CASES:
        rec = {}
        try:
            x = inputs[name][0].clone()
            if op == "gelu_and_mul":
                b = inputs[name][1]
                y = flag_gems.gelu_and_mul(x, b)
                torch.cuda.synchronize()
                rec["out"] = y.detach().cpu().clone()
                rec["grad"] = None
            else:
                xv = x.requires_grad_(True)
                y = F.gelu(xv, approximate="none" if op == "gelu_none" else "tanh")
                y.sum().backward()
                torch.cuda.synchronize()
                rec["out"] = y.detach().cpu().clone()
                rec["grad"] = xv.grad.detach().cpu().clone()
            rec["out_sha"] = sha256_tensor(rec["out"])
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        gems[name] = rec

    # ---- compare ----
    rows = []
    for name, (op, dtype, layout, cls), _ in CASES:
        n, gm = native.get(name, {}), gems.get(name, {})
        row = {"case": name, "op": op, "dtype": dtype, "layout": layout, "value_class": cls}
        if "error" in n or "error" in gm:
            row["native_error"] = n.get("error")
            row["gems_error"] = gm.get("error")
            row["verdict"] = "FAIL"
            rows.append(row)
            continue
        yo, yg = n["out"], gm["out"]
        orc = oracles[name]["out"]
        mn = MIN_NORMAL[dtype]
        row["bitwise_equal_out"] = bool(torch.equal(yo, yg))
        row["nan_mask_equal"] = bool(torch.equal(mask(yo, "nan"), mask(yg, "nan")))
        row["inf_mask_equal"] = bool(torch.equal(mask(yo, "inf"), mask(yg, "inf")))
        row["inf_sign_equal"] = bool(
            torch.equal(mask(yo, "posinf"), mask(yg, "posinf")) and torch.equal(mask(yo, "neginf"), mask(yg, "neginf"))
        )
        row["zero_sign_equal"] = bool(torch.equal(mask(yo, "negzero"), mask(yg, "negzero")) and torch.equal(mask(yo, "poszero"), mask(yg, "poszero")))
        row["ulp_out"] = dt_ulp(yo, yg, dtype)
        row["out_nan_count"] = int(mask(yo, "nan").sum())
        row["out_inf_count"] = int(mask(yo, "inf").sum())
        if yo.numel():
            d = (yo.float() - yg.float()).abs()
            d[~torch.isfinite(d)] = -1  # specials compared via masks
            wi = int(d.argmax())
            row["worst_position"] = wi
            sc = torch.int16 if dtype != "float32" else torch.int32
            row["worst_native_hex"] = hex(int(yo.contiguous().view(-1)[wi].view(sc)) & 0xFFFFFFFF)
            row["worst_gems_hex"] = hex(int(yg.contiguous().view(-1)[wi].view(sc)) & 0xFFFFFFFF)
            row["worst_abs_diff"] = float(d.max())
        row["signed_zero_diff_count"] = int((mask(yo, "negzero") != mask(yg, "negzero")).sum() + (mask(yo, "poszero") != mask(yg, "poszero")).sum())
        row["ftz_definition"] = "oracle(|out|) < dtype min normal (fp16 6.103515625e-05, bf16/fp32 1.1754943508222875e-38)"
        row["native_rel_vs_oracle"] = rel_err(yo.float(), orc)
        row["gems_rel_vs_oracle"] = rel_err(yg.float(), orc)
        # FTZ zone: oracle output subnormal-or-zero -> bitwise agreement required
        of = orc.clone()
        ftz = torch.isfinite(of) & (of.abs() < mn)
        if ftz.any() and yo.numel():
            row["ftz_bitwise_equal"] = bool(torch.equal(yo[ftz], yg[ftz]))
            row["ftz_positions"] = int(ftz.sum())
            if not row["ftz_bitwise_equal"]:
                fd = yo[ftz] != yg[ftz]
                row["ftz_diff_count"] = int(fd.sum())
                row["ftz_diff_max_abs"] = float((yo[ftz][fd].float() - yg[ftz][fd].float()).abs().max())
        # normal zone rel gate
        tol = max(4 * EPS[dtype], 1.5 * row["native_rel_vs_oracle"])
        row["acc_gate_tol"] = tol
        row["acc_gate_pass"] = row["gems_rel_vs_oracle"] <= tol
        if n.get("grad") is not None and gm.get("grad") is not None:
            go, gg = n["grad"], gm["grad"]
            gorc = oracles[name]["grad"]
            row["ulp_grad"] = dt_ulp(go, gg, dtype)
            row["grad_nan_equal"] = bool(torch.equal(mask(go, "nan"), mask(gg, "nan")))
            row["grad_inf_equal"] = bool(
                torch.equal(mask(go, "posinf"), mask(gg, "posinf")) and torch.equal(mask(go, "neginf"), mask(gg, "neginf"))
            )
            row["grad_native_rel_vs_oracle"] = rel_err(go.float(), gorc)
            row["grad_gems_rel_vs_oracle"] = rel_err(gg.float(), gorc)
            gftz = torch.isfinite(gorc) & (gorc.abs() < mn)
            if gftz.any() and go.numel():
                row["grad_ftz_bitwise_equal"] = bool(torch.equal(go[gftz], gg[gftz]))
                row["grad_ftz_positions"] = int(gftz.sum())
                if not row["grad_ftz_bitwise_equal"]:
                    fd = go[gftz] != gg[gftz]
                    row["grad_ftz_diff_count"] = int(fd.sum())
                    row["grad_ftz_diff_max_abs"] = float((go[gftz][fd].float() - gg[gftz][fd].float()).abs().max())
            gtol = max(4 * EPS[dtype], 1.5 * row["grad_native_rel_vs_oracle"])
            row["grad_gate_tol"] = gtol
            row["grad_gate_pass"] = row["grad_gems_rel_vs_oracle"] <= gtol
        elif n.get("grad") is None and gm.get("grad") is None:
            row["grad"] = "N/A"
            row["grad_nan_equal"] = True
            row["grad_inf_equal"] = True
            row["grad_gate_pass"] = True
        else:
            row["grad_presence_mismatch"] = True

        hard_fail = not (
            row["nan_mask_equal"]
            and row["inf_mask_equal"]
            and row.get("inf_sign_equal", False)
            and row.get("acc_gate_pass", False)
            and row.get("grad_gate_pass", False)
            and row.get("grad_nan_equal", False)
            and row.get("grad_inf_equal", False)
            and not row.get("grad_presence_mismatch", False)
        )
        soft_diff = not (
            row.get("ftz_bitwise_equal", True)
            and row.get("grad_ftz_bitwise_equal", True)
        )
        if hard_fail:
            row["verdict"] = "FAIL"
        elif soft_diff:
            row["verdict"] = "DIFF"
        else:
            row["verdict"] = "PASS"
        if not row["zero_sign_equal"]:
            row["note"] = "zero-sign differs (recorded)"
        rows.append(row)

    total = len(rows)
    passed = sum(1 for r in rows if r["verdict"] == "PASS")
    difft = sum(1 for r in rows if r["verdict"] == "DIFF")
    notes = sum(1 for r in rows if r.get("note"))
    summary = {"total": total, "passed": passed, "diff": difft, "failed": total - passed, "zero_sign_notes": notes}
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    print(json.dumps(summary))
    for r in rows:
        if r["verdict"] != "PASS":
            print(r["verdict"] + ":", json.dumps(r)[:260])


if __name__ == "__main__":
    main()
