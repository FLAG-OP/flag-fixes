"""ZW810E problem29 accuracy matrix v6 (issue #5, review round 2).

Uses the PRODUCTION comparator (scripts/comparator.py) — single gate source,
also exercised by test_comparator_negative.py.

v6 changes vs v5 (per review comment 6842724980):
  - normal-zone gate is PER-POSITION (rel <= max(4*eps, 1.5*native_rel) at the
    same position); native errors at denormal positions can no longer relax
    normal positions (review counter-example now FAILs);
  - FTZ zone compared on RAW BIT patterns (torch.equal treats +0==-0);
  - signed zero compared via raw bits; zero-sign-only -> DIFF (production rule,
    consistent with the self-test);
  - ULP ordinal mapping fixed (unsigned bits; ±0 distance 1); selftest asserts;
  - normal-zone and FTZ-zone bit differences REPORTED SEPARATELY (no claim
    that all differences are FTZ-only);
  - summary counts passed/diff/hard_failed separately; exit code is
    non-zero iff hard failures exist (aggregated, with diagnostic JSON always
    written first);
  - --inject-fail: deterministic selftest that corrupts one gems output
    (x2) -> that case must be FAIL and the process must exit non-zero.

Batch note: v1/v2/v5 batches VOID / superseded; this file is the current one.
"""
import argparse
import hashlib
import json
import math
import sys
import zlib

import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import comparator as C  # production gate

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
    ap.add_argument("--inject-fail", action="store_true", help="selftest: corrupt first case gems output (x2); that case must FAIL and exit code must be non-zero")
    args = ap.parse_args()

    print("accuracy_matrix v6: production comparator (per-position normal gate, "
          "raw-bit FTZ/±0, fixed ULP ordinals), separate passed/diff/hard counts, "
          "non-zero exit on hard failures. v1/v2/v5 batches superseded; this is the "
          "current full run." + (" [INJECT-FAIL SELFTEST]" if args.inject_fail else ""), flush=True)

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

    # ---- P2: flag_gems (aten path, autograd-aware; hit evidence in probe1/probe4) ----
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
            if args.inject_fail and name == CASES[0][0]:
                rec["out"] = rec["out"] * 2  # deterministic corruption -> must FAIL
            rec["out_sha"] = sha256_tensor(rec["out"])
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        gems[name] = rec

    # ---- compare via production comparator ----
    rows = []
    for name, (op, dtype, layout, cls), _ in CASES:
        n, gm = native.get(name, {}), gems.get(name, {})
        row = {"case": name, "op": op, "dtype": dtype, "layout": layout, "value_class": cls}
        if "error" in n or "error" in gm:
            row["native_error"] = n.get("error")
            row["gems_error"] = gm.get("error")
            row["verdict"] = "FAIL"
            row["fail_reasons"] = ["exception"]
            rows.append(row)
            continue
        try:
            ev = C.compare_case(
                n["out"], gm["out"], n.get("grad"), gm.get("grad"),
                oracles[name]["out"], oracles[name].get("grad"), dtype,
            )
            row.update(ev)
            row["gems_out_sha"] = gm.get("out_sha")
            row["ftz_definition"] = "oracle(|y|) < dtype min normal; raw-bit comparison"
            row["verdict"] = ev["verdict"]
        except Exception as e:
            import traceback
            row["verdict"] = "FAIL"
            row["fail_reasons"] = [f"comparator_exception:{type(e).__name__}: {e}"]
            row["comparator_traceback"] = traceback.format_exc()[-800:]
        rows.append(row)

    hard = sum(1 for r in rows if r["verdict"] == "FAIL")
    difft = sum(1 for r in rows if r["verdict"] == "DIFF")
    passed = sum(1 for r in rows if r["verdict"] == "PASS")
    summary = {"total": len(rows), "passed": passed, "diff": difft, "hard_failed": hard,
               "note": "hard_failed excludes DIFF; exit code is 1 iff hard_failed > 0",
               "inject_fail": bool(args.inject_fail)}
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    print(json.dumps(summary))
    for r in rows:
        if r["verdict"] != "PASS":
            print(r["verdict"] + ":", json.dumps(r)[:240])
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
