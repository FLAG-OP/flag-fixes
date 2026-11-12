# Shared small-model definitions for the round-4 evaluation (review plan
# "随机权重小模型的数值一致性与性能评估"). This module must be importable
# WITHOUT flag_gems on the path (native group N), so it only imports torch.
import json
import math

import torch
import torch.nn.functional as F

TEACHER_SEED = 7777
A_VARIANTS = ("contig", "noncontig", "conj")
B_VARIANTS = ("default",)

A_CFGS = {
    "min": {"B": 2, "L": 16, "D": 8, "H": 8, "out": 4},
    "mid": {"B": 4, "L": 64, "D": 16, "H": 16, "out": 8},
    "large": {"B": 8, "L": 256, "D": 64, "H": 64, "out": 8},
}
B_CFGS = {
    "min": {"N": 2, "Cin": 3, "H": 8, "W": 8, "Cout": 4, "lin": 5},
    "mid": {"N": 4, "Cin": 3, "H": 16, "W": 16, "Cout": 8, "lin": 5},
    "large": {"N": 8, "Cin": 3, "H": 32, "W": 32, "Cout": 32, "lin": 5},
}

# Frozen acceptance thresholds (review table). The driver freezes these into
# a JSON only after the native-noise estimate is below them; they are never
# loosened after a fix-group failure.
THRESHOLDS = {
    "fp32": {
        "inter": {"atol": 1e-5, "rtol": 1e-4},
        "grad": {"atol": 1e-5, "rtol": 1e-3},
    },
    "fp64": {
        "inter": {"atol": 1e-10, "rtol": 1e-8},
        "grad": {"atol": 1e-9, "rtol": 1e-7},
    },
}

DTYPE_MAP = {
    "fp32": {"real": torch.float32, "complex": torch.complex64},
    "fp64": {"real": torch.float64, "complex": torch.complex128},
}


def cfg_of(model, cfg_name):
    return (A_CFGS if model == "a" else B_CFGS)[cfg_name]


def variants_of(model):
    return A_VARIANTS if model == "a" else B_VARIANTS


def set_determinism(seed, dtype):
    """Disable TF32 / dropout-equivalent randomness and pin seeds; record
    anything the backend does not support instead of silently proceeding."""
    rec = {
        "seed": seed,
        "deterministic_algorithms": None,
        "deterministic_error": None,
        "tf32_matmul_before": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn_before": torch.backends.cudnn.allow_tf32,
    }
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True)
        rec["deterministic_algorithms"] = True
    except Exception as e:  # unsupported capability must be recorded
        rec["deterministic_algorithms"] = False
        rec["deterministic_error"] = f"{type(e).__name__}: {e}"
    return rec


def unset_determinism(rec):
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = rec.get(
        "tf32_matmul_before", False
    )
    torch.backends.cudnn.allow_tf32 = rec.get("tf32_cudnn_before", True)


def _init_tensor(shape, fan_in, dtype, gen):
    t = torch.randn(shape, generator=gen) if len(shape) else torch.tensor(0.0)
    return (t / math.sqrt(max(fan_in, 1))).to(dtype)


def _teacher_params(model, cfg_name, dtype):
    """Frozen teacher with its own fixed seed; same arch as the student."""
    torch.manual_seed(TEACHER_SEED)
    g = torch.Generator().manual_seed(TEACHER_SEED)
    if model == "a":
        c = A_CFGS[cfg_name]
        rt = DTYPE_MAP[dtype]["real"]
        return {
            "Wr": _init_tensor((c["D"], c["H"]), c["D"], rt, g),
            "br": torch.zeros(c["H"], dtype=rt),
            "Wi": _init_tensor((c["D"], c["H"]), c["D"], rt, g),
            "bi": torch.zeros(c["H"], dtype=rt),
            "W2": _init_tensor((c["out"], 2 * c["H"]), 2 * c["H"], rt, g),
            "b2": torch.zeros(c["out"], dtype=rt),
        }
    c = B_CFGS[cfg_name]
    rt = DTYPE_MAP[dtype]["real"]
    return {
        "Wc": _init_tensor((c["Cout"], c["Cin"], 3, 3), c["Cin"] * 9, rt, g),
        "bc": torch.zeros(c["Cout"], dtype=rt),
        "Wl": _init_tensor((c["lin"], c["Cout"]), c["Cout"], rt, g),
        "bl": torch.zeros(c["lin"], dtype=rt),
    }


def _student_params(model, cfg_name, dtype, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    if model == "a":
        c = A_CFGS[cfg_name]
        rt = DTYPE_MAP[dtype]["real"]
        return {
            "Wr": _init_tensor((c["D"], c["H"]), c["D"], rt, g),
            "br": torch.zeros(c["H"], dtype=rt),
            "Wi": _init_tensor((c["D"], c["H"]), c["D"], rt, g),
            "bi": torch.zeros(c["H"], dtype=rt),
            "W2": _init_tensor((c["out"], 2 * c["H"]), 2 * c["H"], rt, g),
            "b2": torch.zeros(c["out"], dtype=rt),
        }
    c = B_CFGS[cfg_name]
    rt = DTYPE_MAP[dtype]["real"]
    return {
        "Wc": _init_tensor((c["Cout"], c["Cin"], 3, 3), c["Cin"] * 9, rt, g),
        "bc": torch.zeros(c["Cout"], dtype=rt),
        "Wl": _init_tensor((c["lin"], c["Cout"]), c["Cout"], rt, g),
        "bl": torch.zeros(c["lin"], dtype=rt),
    }


def _teacher_forward(model, cfg_name, x, params):
    if model == "a":
        p = x @ params["Wr"] + params["br"]
        q = x @ params["Wi"] + params["bi"]
        z = torch.complex(p, q)
        u = torch.abs(torch.sum(z, dim=1)) / z.shape[1]
        v = torch.sum(torch.abs(z), dim=1) / z.shape[1]
        h = torch.cat([u, v], dim=-1)
        return F.linear(h, params["W2"], params["b2"])
    y = F.conv2d(x, params["Wc"], params["bc"], padding=1)
    y = torch.tanh(y)
    yh = y.permute(0, 2, 3, 1)
    return F.linear(yh, params["Wl"], params["bl"])


def gen_task(model, cfg_name, seed, dtype):
    """Deterministic CPU task data: serialized input / init params / target.
    Generated once natively; every group loads the identical bytes."""
    rt = DTYPE_MAP[dtype]["real"]
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    if model == "a":
        c = A_CFGS[cfg_name]
        x = torch.randn((c["B"], c["L"], c["D"]), generator=g).to(rt)
        params = _student_params(model, cfg_name, dtype, seed)
        with torch.no_grad():
            tparams = _teacher_params(model, cfg_name, dtype)
            target = _teacher_forward(model, cfg_name, x, tparams)
    else:
        c = B_CFGS[cfg_name]
        x = torch.randn((c["N"], c["Cin"], c["H"], c["W"]), generator=g).to(rt)
        params = _student_params(model, cfg_name, dtype, seed)
        with torch.no_grad():
            tparams = _teacher_params(model, cfg_name, dtype)
            target = _teacher_forward(model, cfg_name, x, tparams)
    return {
        "model": model,
        "cfg": cfg_name,
        "seed": seed,
        "dtype": dtype,
        "X": x,
        "params": params,
        "target": target,
        "teacher_seed": TEACHER_SEED,
    }


def apply_variant(z, variant):
    """Logical value identical to z for contig/noncontig; conj(z) for conj.
    Premises are asserted (review: 断言 stride/conj 前提)."""
    if variant == "contig":
        assert z.is_contiguous(), "contig variant premise failed"
        return z
    if variant == "noncontig":
        if z.dim() < 2:
            raise ValueError("noncontig variant needs ndim >= 2")
        perm = list(range(z.dim()))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        znc = z.permute(*perm).contiguous().permute(*perm)
        assert not znc.is_contiguous(), "noncontig variant is contiguous"
        assert znc.shape == z.shape and torch.equal(znc.real, z.real)
        assert torch.equal(znc.imag, z.imag)
        return znc
    if variant == "conj":
        zc = z.conj()
        assert zc.is_conj(), "conj variant lost its conj bit"
        assert zc.shape == z.shape
        return zc
    raise ValueError(variant)


def forward(model, cfg_name, x, params, variant):
    """One forward pass; returns outputs, loss and key intermediates."""
    if model == "a":
        c = A_CFGS[cfg_name]
        p = x @ params["Wr"] + params["br"]
        q = x @ params["Wi"] + params["bi"]
        z = torch.complex(p, q)
        layout = {
            "z_contig": bool(z.is_contiguous()),
            "z_is_conj": bool(z.is_conj()),
            "z_stride": list(z.stride()),
        }
        zv = apply_variant(z, variant)
        layout.update(
            {
                "variant_contig": bool(zv.is_contiguous()),
                "variant_is_conj": bool(zv.is_conj()),
                "variant_stride": list(zv.stride()),
            }
        )
        u = torch.abs(torch.sum(zv, dim=1)) / c["L"]
        v = torch.sum(torch.abs(zv), dim=1) / c["L"]
        h = torch.cat([u, v], dim=-1)
        out = F.linear(h, params["W2"], params["b2"])
        loss = F.mse_loss(out, _target_cache)
        return out, loss, {"u": u, "v": v, "h": h, "layout": layout}
    y = F.conv2d(x, params["Wc"], params["bc"], padding=1)
    act = torch.tanh(y)
    yh = act.permute(0, 2, 3, 1)
    assert yh.shape[-1] == params["Wc"].shape[0]
    assert not yh.is_contiguous(), "NHWC view must be non-contiguous"
    out = F.linear(yh, params["Wl"], params["bl"])
    loss = F.mse_loss(out, _target_cache)
    return out, loss, {
        "conv": y,
        "act": act,
        "nhwc_stride": list(yh.stride()),
        "nhwc_contig": bool(yh.is_contiguous()),
    }


# target is bound per-run (kept simple; forward is never called concurrently
# inside one process).
_target_cache = None


def set_target(target):
    global _target_cache
    _target_cache = target


def param_names(model):
    if model == "a":
        return ["Wr", "br", "Wi", "bi", "W2", "b2"]
    return ["Wc", "bc", "Wl", "bl"]


def to_device(task, device):
    x = task["X"].to(device)
    x.requires_grad_(True)
    params = {k: v.to(device).clone().detach().requires_grad_(True)
              for k, v in task["params"].items()}
    target = task["target"].to(device)
    set_target(target)
    return x, params, target


def fresh_params(task, device):
    return {k: v.to(device).clone().detach().requires_grad_(True)
            for k, v in task["params"].items()}


def zero_grads(x, params):
    x.grad = None
    for p in params.values():
        p.grad = None


def snapshot(t):
    return t.detach().cpu().clone()


def grad_stats(x, params):
    """Zero-grad ratio / norm / finite flags per tensor (degeneracy checks)."""
    out = {}
    for name, t in [("x", x)] + list(params.items()):
        g = t.grad
        if g is None:
            out[name] = {"present": False}
            continue
        g64 = g.detach().to(torch.float64)
        nz = (g64 != 0).sum().item()
        out[name] = {
            "present": True,
            "finite": bool(torch.isfinite(g).all().item()),
            "zero_ratio": 1.0 - nz / max(g64.numel(), 1),
            "norm": float(g64.norm().item()),
            "shape": list(g.shape),
            "dtype": str(g.dtype),
        }
    return out


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def read_json(path):
    with open(path) as f:
        return json.load(f)
