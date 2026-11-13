"""Production comparator for ZW810E accuracy matrices (issue #5, v6).

Single source of truth for gates — imported by accuracy_matrix.py and
test_comparator_negative.py so the self-test exercises the PRODUCTION logic.

Terminology / policy (review round 2, comment 6842724980):
- NORMAL zone: |oracle| >= MIN_NORMAL[dtype]. Per-position gate:
    gems_rel <= max(4*eps(dtype), 1.5 * native_rel)   evaluated PER POSITION,
    so a native error at a denormal position can never relax another position.
- FTZ/near-zero zone: |oracle| < MIN_NORMAL (includes flushed-to-zero).
    Policy: native and gems must agree BIT-FOR-BIT (raw bit patterns).
    Deviation -> DIFF (recorded), with per-case counts and max abs error.
- signed zero: compared via RAW BITS (uint view), not torch.equal
    (which treats +0 == -0). Zero-sign-only mismatch -> DIFF (declared rule).
- ULP: exact dtype ordinal mapping computed on FULL bit patterns
    (positive: bits + sign_hi; negative: ~bits & mask). Guarantees:
    adjacent values distance 1, +0/-0 distance 1, ±min_subnormal to 0
    distance 1, max_subnormal to min_normal distance 1 (unit-tested).
- verdicts: PASS / DIFF / FAIL; hard failures counted separately from DIFF;
    callers aggregate hard failures and MUST exit non-zero if any.
"""
import torch

EPS = {"float32": 1.1920929e-07, "bfloat16": 0.0078125, "float16": 0.0009765625}
MIN_NORMAL = {"float32": 1.1754943508222875e-38, "bfloat16": 1.1754943508222875e-38, "float16": 6.103515625e-05}
INT_VIEW = {"float32": torch.int32, "float16": torch.int16, "bfloat16": torch.int16}


def bits_uint(t, dtype):
    return t.contiguous().view(INT_VIEW[dtype])


def m_nan(t):
    return torch.isnan(t.float())


def m_posinf(t):
    return torch.isposinf(t.float())


def m_neginf(t):
    return torch.isinf(t.float()) & torch.signbit(t.float())


def zero_bits(t, dtype):
    """boolean masks of +0 and -0 by RAW BITS."""
    b = bits_uint(t, dtype)
    zero = b == 0
    neg = (b.view(torch.int16 if dtype != "float32" else torch.int32) < 0) if dtype != "float32" else (b.view(torch.int32) < 0)
    return zero & ~neg, zero & neg


UINT_VIEW = {"float32": torch.uint32, "float16": torch.uint16, "bfloat16": torch.uint16}


def dt_ulp(a, b, dtype):
    """exact ulp distance in the ORIGINAL dtype; specials excluded (caller masks).

    Ordinal mapping on UNSIGNED full bit patterns (standard total order):
      positive: bits + sign_hi ; negative: full - bits (= ~bits)
      => +0/-0 distance 1; adjacent values distance 1 across the whole range.
    """
    import numpy as np

    if a.numel() == 0:
        return 0
    nbits = 16 if dtype != "float32" else 32
    sign_hi = 1 << (nbits - 1)
    full = (1 << nbits) - 1
    ai = a.contiguous().view(UINT_VIEW[dtype]).numpy().astype(np.int64)
    bi = b.contiguous().view(UINT_VIEW[dtype]).numpy().astype(np.int64)

    def to_ord(x):
        return np.where(x < sign_hi, x + sign_hi, full - x)  # negatives: ~bits

    fa, fb = a.float(), b.float()
    m = ~(torch.isnan(fa) | torch.isnan(fb) | torch.isinf(fa) | torch.isinf(fb)).numpy()
    if not m.any():
        return 0
    return int(np.abs(to_ord(ai) - to_ord(bi))[m].max())


def rel_err_positions(out, oracle):
    """per-position relative error; non-finite positions -> nan (excluded downstream)."""
    af = out.float().double()
    fin = torch.isfinite(af) & torch.isfinite(oracle)
    denom = torch.where(fin, oracle.abs().clamp_min(1e-300), torch.ones_like(oracle))
    r = (af - oracle).abs() / denom
    r[~fin] = float("nan")
    return r


def zone_masks(oracle, dtype):
    """returns (normal_mask, ftz_mask) over finite oracle positions (oracle made contiguous)."""
    oracle = oracle.contiguous()
    fin = torch.isfinite(oracle)
    normal = fin & (oracle.abs() >= MIN_NORMAL[dtype])
    ftz = fin & ~normal
    return normal, ftz


def compare_case(native_out, gems_out, native_grad, gems_grad, oracle_out, oracle_grad, dtype):
    """Production verdict for one case. Returns dict of all evidence fields.

    oracle_*: fp64 tensors (device-independent numeric truth, NOT infinite
    precision; see README note on fp64 analytic reference limits on the
    negative tail where gelu underflows).
    native_grad/gems_grad may be None (op without autograd path) -> N/A.
    """
    row = {}
    eps = EPS[dtype]

    # ---- specials (sign-aware) ----
    row["nan_mask_equal"] = bool(torch.equal(m_nan(native_out), m_nan(gems_out)))
    row["inf_mask_equal"] = bool(torch.equal(m_posinf(native_out) | m_neginf(native_out), m_posinf(gems_out) | m_neginf(gems_out)))
    row["inf_sign_equal"] = bool(torch.equal(m_posinf(native_out), m_posinf(gems_out)) and torch.equal(m_neginf(native_out), m_neginf(gems_out)))
    row["out_nan_count"] = int(m_nan(native_out).sum())
    row["out_inf_count"] = int((m_posinf(native_out) | m_neginf(native_out)).sum())

    # ---- signed zero by RAW BITS ----
    nz_n, pz_n = zero_bits(native_out, dtype)
    nz_g, pz_g = zero_bits(gems_out, dtype)
    row["zero_sign_equal"] = bool(torch.equal(nz_n, nz_g) and torch.equal(pz_n, pz_g))
    row["signed_zero_diff_count"] = int((nz_n != nz_g).sum() + (pz_n != pz_g).sum())

    # ---- zones ----
    normal, ftz = zone_masks(oracle_out, dtype)
    row["normal_positions"] = int(normal.sum())
    row["ftz_positions"] = int(ftz.sum())

    # normal zone: PER-POSITION gate
    rn = rel_err_positions(native_out, oracle_out)[normal]
    rg = rel_err_positions(gems_out, oracle_out)[normal]
    if rn.numel():
        tol_pos = torch.maximum(torch.full_like(rn, 4 * eps), 1.5 * rn)
        viol = rg > tol_pos
        row["normal_gate_violations"] = int(viol.sum())
        row["normal_worst_gems_rel"] = float(rg.max())
        row["normal_worst_native_rel"] = float(rn.max())
        row["normal_worst_tol"] = float(tol_pos.max())
        wi_local = int(rg.argmax())
        w = torch.nonzero(normal.reshape(-1), as_tuple=False)[wi_local].item()
        row["normal_worst_flat_index"] = w
        sc = INT_VIEW[dtype]
        row["normal_worst_native_hex"] = hex(int(bits_uint(native_out, dtype).view(-1)[w]))
        row["normal_worst_gems_hex"] = hex(int(bits_uint(gems_out, dtype).view(-1)[w]))
        row["normal_worst_abs_diff"] = float((native_out.float() - gems_out.float()).abs().reshape(-1)[w])
        if viol.any():
            fidx = torch.nonzero(normal.reshape(-1), as_tuple=False)[torch.topk((rg - tol_pos).clamp_min(0), k=min(5, int(viol.sum()))).indices]
            row["fwd_violation_examples"] = [
                {"flat_index": int(i), "native": float(native_out.float().reshape(-1)[i]), "gems": float(gems_out.float().reshape(-1)[i]), "oracle": float(oracle_out.reshape(-1)[i])}
                for i in fidx.reshape(-1).tolist()
            ]
    else:
        row["normal_gate_violations"] = 0
        row["normal_worst_gems_rel"] = 0.0

    # normal zone bit differences (reported separately from FTZ; may be within tolerance)
    row["normal_zone_bit_diff_count"] = int((bits_uint(native_out, dtype) != bits_uint(gems_out, dtype))[normal].sum()) if normal.any() else 0
    # FTZ zone: bitwise policy
    if ftz.any():
        fb = bits_uint(native_out, dtype)[ftz] != bits_uint(gems_out, dtype)[ftz]
        row["ftz_bit_diff_count"] = int(fb.sum())
        row["ftz_bitwise_equal"] = not bool(fb.any())
        row["ftz_max_abs_diff"] = float((native_out.float()[ftz] - gems_out.float()[ftz]).abs().max()) if fb.any() else 0.0
    else:
        row["ftz_bit_diff_count"] = 0
        row["ftz_bitwise_equal"] = True

    # ---- ULP (dtype-exact; informational) ----
    row["ulp_out"] = dt_ulp(native_out, gems_out, dtype)
    row["bits_equal_out"] = bool(torch.equal(bits_uint(native_out, dtype), bits_uint(gems_out, dtype)))

    # ---- grads ----
    if native_grad is None and gems_grad is None:
        row["grad"] = "N/A"
        grad_ok = True
    elif native_grad is None or gems_grad is None:
        row["grad_presence_mismatch"] = True
        grad_ok = False
    else:
        gn, gg = native_grad, gems_grad
        row["grad_nan_equal"] = bool(torch.equal(m_nan(gn), m_nan(gg)))
        row["grad_inf_sign_equal"] = bool(torch.equal(m_posinf(gn), m_posinf(gg)) and torch.equal(m_neginf(gn), m_neginf(gg)))
        if oracle_grad is not None:
            grn = rel_err_positions(gn, oracle_grad)
            grg = rel_err_positions(gg, oracle_grad)
            gnorm, gftz = zone_masks(oracle_grad, dtype)
            rn2, rg2 = grn[gnorm], grg[gnorm]
            if rn2.numel():
                gt = torch.maximum(torch.full_like(rn2, 4 * eps), 1.5 * rn2)
                gv = rg2 > gt
                row["grad_normal_gate_violations"] = int(gv.sum())
                row["grad_worst_gems_rel"] = float(rg2.max())
                row["grad_worst_tol"] = float(gt.max())
                if gv.any():
                    gidx = torch.nonzero(gnorm.reshape(-1), as_tuple=False)[torch.topk((rg2 - gt).clamp_min(0), k=min(5, gv.sum())).indices]
                    row["grad_violation_examples"] = [
                        {"flat_index": int(i), "native_grad": float(native_grad.float().reshape(-1)[i]), "gems_grad": float(gems_grad.float().reshape(-1)[i]), "oracle_grad": float(oracle_grad.reshape(-1)[i])}
                        for i in gidx.reshape(-1).tolist()
                    ]
            else:
                row["grad_normal_gate_violations"] = 0
            if gftz.any():
                gb = bits_uint(gn, dtype)[gftz] != bits_uint(gg, dtype)[gftz]
                row["grad_ftz_bit_diff_count"] = int(gb.sum())
                row["grad_ftz_bitwise_equal"] = not bool(gb.any())
            else:
                row["grad_ftz_bit_diff_count"] = 0
                row["grad_ftz_bitwise_equal"] = True
        row["ulp_grad"] = dt_ulp(gn, gg, dtype)
        g_nzn, g_pzn = zero_bits(gn, dtype)
        g_nzg, g_pzg = zero_bits(gg, dtype)
        row["grad_signed_zero_diff_count"] = int((g_nzn != g_nzg).sum() + (g_pzn != g_pzg).sum())
        grad_ok = (
            row["grad_nan_equal"]
            and row["grad_inf_sign_equal"]
            and not row.get("grad_presence_mismatch", False)
            and row.get("grad_normal_gate_violations", 0) == 0
        )

    # ---- verdict ----
    hard_fail = not (
        row["nan_mask_equal"]
        and row["inf_mask_equal"]
        and row["inf_sign_equal"]
        and row.get("normal_gate_violations", 0) == 0
        and grad_ok
    )
    soft_diff = (
        row["ftz_bit_diff_count"] > 0
        or row["signed_zero_diff_count"] > 0
        or row.get("grad_ftz_bit_diff_count", 0) > 0
        or row.get("grad_signed_zero_diff_count", 0) > 0
    )
    if hard_fail:
        row["verdict"] = "FAIL"
        row["fail_reasons"] = [k for k in ("nan_mask_equal", "inf_sign_equal", "normal_gate_violations") if not row.get(k, True) or (k == "normal_gate_violations" and row[k] > 0)] + (["grad"] if not grad_ok else [])
    elif soft_diff:
        row["verdict"] = "DIFF"
        row["diff_reasons"] = [k for k in ("ftz_bit_diff_count", "signed_zero_diff_count", "grad_ftz_bit_diff_count", "grad_signed_zero_diff_count") if row.get(k, 0) > 0]
    else:
        row["verdict"] = "PASS"
    return row


def selftest():
    """unit asserts for the ordinal mapping and zones (device-independent)."""
    ok = []

    def chk(name, cond):
        ok.append((name, bool(cond)))

    a = torch.tensor([6.0975552e-05, 6.103515625e-05], dtype=torch.float16)  # max_sub, min_normal
    chk("fp16 max_sub<->min_normal ulp == 1", dt_ulp(a[:1], a[1:], "float16") == 1)
    z = torch.tensor([0.0, -0.0], dtype=torch.float16)
    chk("fp16 +0/-0 ulp == 1", dt_ulp(z[:1], z[1:], "float16") == 1)
    p = torch.tensor([6.103515625e-05], dtype=torch.float16)
    m = torch.tensor([-6.103515625e-05], dtype=torch.float16)
    chk("fp16 +min_normal/-min_normal ulp == 2049 (1024 steps to each zero +1)", dt_ulp(p, m, "float16") == 2049)
    s = torch.tensor([5.960464477539063e-08, 0.0], dtype=torch.float16)  # min_sub, +0
    chk("fp16 min_sub->+0 ulp == 1", dt_ulp(s[:1], s[1:], "float16") == 1)
    f32z = torch.tensor([0.0, -0.0], dtype=torch.float32)
    chk("fp32 +0/-0 ulp == 1", dt_ulp(f32z[:1], f32z[1:], "float32") == 1)
    n1 = torch.tensor([-1.0], dtype=torch.float16)
    n2 = torch.tensor([-1.0009765625], dtype=torch.float16)
    chk("fp16 negative adjacent ulp == 1", dt_ulp(n1, n2, "float16") == 1)
    xb16 = torch.tensor([0x007F], dtype=torch.int16).view(torch.bfloat16).item()
    chk("bf16 0x007F max subnormal", 0 < xb16 < MIN_NORMAL["bfloat16"])
    # review counter-example: native flushes denormal (rel=1 at FTZ) must NOT
    # relax the normal position where gems is 100% wrong -> FAIL
    oracle = torch.tensor([1e-40, 1.0], dtype=torch.float64)
    native = torch.tensor([0.0, 1.0])
    gems = torch.tensor([0.0, 2.0])
    r = compare_case(native, gems, None, None, oracle, None, "float32")
    chk("review counter-example -> FAIL (no cross-position dilution)", r["verdict"] == "FAIL" and r["normal_gate_violations"] == 1)
    # zero-sign-only -> DIFF via production path (bit-level, not torch.equal)
    n4 = torch.tensor([0.0, 1.0])
    g4 = torch.tensor([-0.0, 1.0])
    r = compare_case(n4, g4, None, None, torch.tensor([0.0, 1.0], dtype=torch.float64), None, "float32")
    chk("zero-sign-only -> DIFF (raw bits)", r["verdict"] == "DIFF" and r["signed_zero_diff_count"] == 1)
    # bitwise: bits_equal sees +0 vs -0 as different even though torch.equal doesn't
    chk("bits_equal distinguishes +0/-0", not compare_case(n4, g4, None, None, torch.tensor([0.0, 1.0], dtype=torch.float64), None, "float32")["bits_equal_out"])
    return ok


if __name__ == "__main__":
    fails = [n for n, c in selftest() if not c]
    for n, c in selftest():
        print(("PASS " if c else "FAIL ") + n)
    raise SystemExit(1 if fails else 0)
