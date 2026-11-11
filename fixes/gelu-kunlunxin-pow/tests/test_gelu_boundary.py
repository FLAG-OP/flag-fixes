"""Dump FlagGems gelu boundary outputs (bit-exact, for the two-tree comparator).

Dumps, per dtype and boundary input:
  * the post-cast DEVICE inputs (float values + raw bit patterns) -- proves
    what actually reached the kernel (subnormal/FTZ behaviour is recorded,
    not assumed);
  * gelu fwd (none/tanh) and gelu backward (none/tanh) outputs
    (float values + raw bit patterns).

Raw bit patterns are saved so the comparator can do TRUE bit-pattern
comparisons (including NaN payloads and signed zero), separately from
numeric equality.

Usage (run twice, then compare):
    PYTHONPATH=<FlagGems-base>/src  python test_gelu_boundary.py --out base.json
    PYTHONPATH=<FlagGems-fixed>/src python test_gelu_boundary.py --out fixed.json
    python compare_gelu_boundary.py --base base.json --fixed fixed.json --report report.json
"""

import argparse
import json

import torch
import torch.nn.functional as F

SCALE = 0.7071067811865476


def bits(t):
    """Raw bit patterns of a tensor as hex strings (dtype-preserving)."""
    t = t.detach().contiguous()
    if t.dtype == torch.bfloat16:
        raw = t.view(torch.int16)
        width = 16
    elif t.dtype == torch.float16:
        raw = t.view(torch.int16)
        width = 16
    elif t.dtype == torch.float32:
        raw = t.view(torch.int32)
        width = 32
    else:
        raise ValueError(f"unsupported dtype {t.dtype}")
    return [f"0x{int(v) & ((1 << width) - 1):0{width // 4}x}" for v in raw.tolist()]


def build_inputs(dtype):
    """Boundary inputs constructed at the requested dtype (bit-exact), with
    post-cast verification: any value that was non-zero on CPU but zero on
    device is recorded as flush-to-zero (FTZ), not silently assumed present.

    Subnormals are built from integer bit patterns (finfo.smallest_subnormal is
    unavailable for bf16/fp16 in this torch build).
    """
    finfo = torch.finfo(dtype)
    candidates = [0.0, -0.0, float("inf"), float("-inf"), float("nan")]

    # subnormals via bit patterns: mantissa bits set, exponent = 0
    if dtype == torch.float32:
        width, min_sub_int = 32, 1          # 0x00000001
    elif dtype == torch.bfloat16:
        width, min_sub_int = 16, 1          # 0x0001
    elif dtype == torch.float16:
        width, min_sub_int = 16, 1
    else:
        width, min_sub_int = 32, 1
    if width == 32:
        def int_to_float(v):
            return torch.tensor([v], dtype=torch.int32).view(torch.float32).item()
    else:
        def int_to_float(v):
            t = torch.tensor([v if v < (1 << 15) else v - (1 << width)],
                             dtype=torch.int16)
            return t.view(dtype).item()

    min_sub = int_to_float(min_sub_int)
    subs = [min_sub_int, min_sub_int * 2, min_sub_int * 3, min_sub_int * 8]
    for v in subs:
        if width == 32:
            candidates.append(int_to_float(v))
            candidates.append(-int_to_float(v))
        else:
            candidates.append(int_to_float(v))
            candidates.append(-int_to_float(v))
    candidates += [float(finfo.tiny), -float(finfo.tiny)]   # min normal ±
    # nextafter(min_normal, 0)：最大次正规数（位型按 dtype 精确给出）
    #   fp32 0x007fffff / bf16 0x007f / fp16 0x03ff
    nm1_bits = {32: 0x007FFFFF, 16: 0x007F}[width]
    nm1 = int_to_float(nm1_bits)
    assert 0.0 < nm1 < float(finfo.tiny), f"nm1 construction broken: {nm1!r}"
    candidates += [nm1]
    candidates += [1e-30, -1e-30]                            # may FTZ in fp16/bf16
    candidates += [1e30, -1e30, 3.0, -3.0, 0.5, -0.5]

    cpu = torch.tensor(candidates, dtype=torch.float32)
    dev = cpu.to(dtype)
    ftz = [i for i, (c, d) in enumerate(zip(candidates, dev.float().tolist()))
           if c != 0.0 and float(d) == 0.0]
    sub_idx = [i for i, c in enumerate(candidates)
               if c != 0.0 and abs(c) < float(finfo.tiny)]
    x = dev.requires_grad_(True)
    return x, dev, ftz, sub_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import flag_gems
    flag_gems.only_enable(include=["gelu", "gelu_backward"])

    payload = {"inputs": {}, "outputs": {}, "ftz": {}, "sanity": {}}
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        x_dev, _dev_vals, ftz, sub_idx = build_inputs(dtype)
        payload["subnormal_idx"] = payload.get("subnormal_idx", {})
        payload["subnormal_idx"][str(dtype)] = sub_idx
        dy = torch.ones_like(x_dev)
        payload["inputs"][str(dtype)] = {"values": x_dev.float().tolist(), "bits": bits(x_dev)}
        payload["ftz"][str(dtype)] = ftz

        def capture(fwd, bwd):
            row = {}
            fwd()
            row["gelu_none_fwd"] = {"values": bwd["y"].float().tolist(), "bits": bits(bwd["y"])}
            row["gelu_none_bwd"] = {"values": bwd["dx"].float().tolist(), "bits": bits(bwd["dx"])}
            fwd(tanh=True)
            row["gelu_tanh_fwd"] = {"values": bwd["y"].float().tolist(), "bits": bits(bwd["y"])}
            row["gelu_tanh_bwd"] = {"values": bwd["dx"].float().tolist(), "bits": bits(bwd["dx"])}
            return row

        # run via closures that reuse the saved-tensor machinery
        x_dev.requires_grad_(True)
        row = {}
        y = F.gelu(x_dev)
        y.backward(dy)
        row["gelu_none_fwd"] = {"values": y.detach().float().tolist(), "bits": bits(y)}
        row["gelu_none_bwd"] = {"values": x_dev.grad.float().tolist(), "bits": bits(x_dev.grad)}
        x_dev.grad = None
        y = F.gelu(x_dev, approximate="tanh")
        y.backward(dy)
        row["gelu_tanh_fwd"] = {"values": y.detach().float().tolist(), "bits": bits(y)}
        row["gelu_tanh_bwd"] = {"values": x_dev.grad.float().tolist(), "bits": bits(x_dev.grad)}
        row["bits_x"] = bits(x_dev.detach())
        payload["outputs"][str(dtype)] = row

        nan_in = torch.tensor([float("nan")], device="cuda", dtype=dtype)
        nan_out = F.gelu(nan_in).detach()
        zz = torch.tensor([0.0, -0.0], device="cuda", dtype=dtype)
        o = F.gelu(zz).detach()
        payload["sanity"][str(dtype)] = {
            "nan_propagates": bool(torch.isnan(nan_out.float()[0])),
            # signed-zero: compare SIGN BITS, not +0 == -0
            "pm_zero_sign_bits": bits(o),
        }
        print(f"  {dtype}: ftz={ftz} nan_ok={payload['sanity'][str(dtype)]['nan_propagates']}")

    with open(a.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
