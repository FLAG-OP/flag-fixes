"""Selection matrix for adamw_impl_kwargs (PR #3 review round 2).

Every case EXPLICITLY sets the env vars it depends on
(FLAGSCALE_ADAMW_FUSED / GEMS_VENDOR / FLAGSCALE_VENDOR), restores them
afterwards, asserts the returned kwargs AND the fallback log line where the
review requires one, and separates device-free cases from CUDA cases.

No CUDA device -> CUDA cases are skipped with exit 2 (skip is not a pass);
CPU / DTensor-like / env-parsing cases always run (device-free).
"""

import contextlib
import importlib.util
import io
import os
import sys

import torch

from flagscale.train.train_retrieval import adamw_impl_kwargs

ENV_KEY = "FLAGSCALE_ADAMW_FUSED"
ENV_KEYS = (ENV_KEY, "GEMS_VENDOR", "FLAGSCALE_VENDOR")
FAILURES = 0


def run_case(label, env_overrides, params, expected, expect_log=None):
    """Set env, call adamw_impl_kwargs, assert kwargs + optional log substring."""
    global FAILURES
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    passed = False
    got = None
    try:
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(env_overrides)
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            got = adamw_impl_kwargs(params)
        ok = got == expected
        log_ok = expect_log is None or (expect_log in buf.getvalue()
                                        or expect_log in err.getvalue())
        passed = ok and log_ok
        if not ok:
            print(f"    (kwargs mismatch: got={got} expected={expected})", file=sys.stderr)
        if not log_ok:
            print(f"    (expected log {expect_log!r} not found; "
                  f"stdout={buf.getvalue()[:200]!r})", file=sys.stderr)
    except Exception as error:  # noqa: BLE001
        print(f"FAIL {label}: raised {type(error).__name__}: {error}", file=sys.stderr)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    if passed:
        print(f"ok   {label}: kwargs={got}")
    else:
        print(f"FAIL {label}: kwargs={got} expected={expected} "
              f"log_assert={expect_log is not None}")
        FAILURES += 1


def main():
    global FAILURES
    from omegaconf import OmegaConf
    from flagscale.train.train_retrieval import build_optimizer

    dtensor_like = type("DTensor", (), {})
    dtensor_like.__module__ = "torch.distributed.tensor"
    cpu_params = [torch.nn.Parameter(torch.randn(4, 4))]
    cfg = OmegaConf.create({"name": "adamw", "lr": 2e-5, "weight_decay": 0.01})

    has_cuda = torch.cuda.is_available()
    cuda_params = mixed_params = None
    if has_cuda:
        cuda_params = [
            torch.nn.Parameter(torch.randn(64, 64, device="cuda", dtype=torch.bfloat16))
            for _ in range(8)
        ]
        mixed_params = cuda_params + [
            torch.nn.Parameter(torch.randn(4, 4, device="cuda", dtype=torch.float32))
        ]

    shim = importlib.util.find_spec("torch_xmlir") is not None
    print(f"host: cuda={has_cuda} torch_xmlir_shim={shim}")

    # ---------- device-free cases（始终运行） ----------
    run_case("auto + DTensor（kunlunxin，FSDP2 护栏）",
             {"GEMS_VENDOR": "kunlunxin"}, [dtensor_like()], {})
    run_case("forced 1 + DTensor（kunlunxin）：回退并记录原因",
             {"GEMS_VENDOR": "kunlunxin", ENV_KEY: "1"}, [dtensor_like()], {},
             expect_log="parameters are DTensors; keeping stock AdamW")
    run_case("auto, 空参数表", {"GEMS_VENDOR": "kunlunxin"}, [], {})
    run_case("auto, cpu params（kunlunxin）", {"GEMS_VENDOR": "kunlunxin"}, cpu_params, {})
    run_case("forced 1 + cpu：回退并记录原因",
             {"GEMS_VENDOR": "kunlunxin", ENV_KEY: "1"}, cpu_params, {},
             expect_log="not CUDA fp32/bf16/fp16; keeping stock AdamW")
    run_case("env=0 -> stock", {ENV_KEY: "0"}, cpu_params, {})
    run_case("无法识别的 env 值 -> stock + log",
             {ENV_KEY: "nonsense"}, cpu_params, {},
             expect_log="is not recognised; keeping stock AdamW")

    # ---------- vendor/platform guard（无设备形态：平台护栏是绝对护栏） ----------
    run_case("vendor nvidia, auto（平台护栏）", {"GEMS_VENDOR": "nvidia"}, [], {})
    run_case("vendor nvidia, forced 1 -> 回退并记录原因",
             {"GEMS_VENDOR": "nvidia", ENV_KEY: "1"}, [], {},
             expect_log="not validated for fused AdamW")
    run_case("vendor thead, forced 1 -> 回退并记录原因",
             {"GEMS_VENDOR": "thead", ENV_KEY: "1"}, [], {},
             expect_log="not validated for fused AdamW")
    run_case("FLAGSCALE_VENDOR=nvidia 覆盖 shim 探测 -> 回退",
             {"FLAGSCALE_VENDOR": "nvidia", ENV_KEY: "1"}, [], {},
             expect_log="not validated for fused AdamW")

    # ---------- CUDA cases ----------
    if not has_cuda:
        if FAILURES:
            print(f"FAILED: {FAILURES} case(s) before device skip", file=sys.stderr)
            return 1
        print("SKIP: no CUDA device; exit 2 (skip is not a pass)", file=sys.stderr)
        return 2

    run_case("vendor kunlunxin, auto -> fused",
             {"GEMS_VENDOR": "kunlunxin"}, cuda_params, {"fused": True})
    run_case("vendor kunlunxin, forced 1 -> fused",
             {"GEMS_VENDOR": "kunlunxin", ENV_KEY: "1"}, cuda_params, {"fused": True})
    run_case("无 GEMS_VENDOR、torch_xmlir shim 存在 -> fused",
             {}, cuda_params, {"fused": True})
    run_case("混合 dtype（bf16+fp32，均在支持集合）-> fused",
             {"GEMS_VENDOR": "kunlunxin"}, mixed_params, {"fused": True})

    # build_optimizer 集成（auto / env=0 / 非 kunlunxin vendor）
    os.environ.pop(ENV_KEY, None)
    os.environ["GEMS_VENDOR"] = "kunlunxin"
    opt = build_optimizer(cfg, cuda_params, allow_adafactor=False)
    fused = bool(opt.defaults.get("fused"))
    print(f"{'ok  ' if fused else 'FAIL'} build_optimizer auto (kunlunxin): fused={fused}")
    if not fused:
        FAILURES += 1
    os.environ[ENV_KEY] = "0"
    opt = build_optimizer(cfg, cuda_params, allow_adafactor=False)
    off = not opt.defaults.get("fused")
    print(f"{'ok  ' if off else 'FAIL'} build_optimizer env=0 -> stock: "
          f"fused={opt.defaults.get('fused')}")
    if not off:
        FAILURES += 1
    os.environ["GEMS_VENDOR"] = "nvidia"
    opt = build_optimizer(cfg, cuda_params, allow_adafactor=False)
    stock = not opt.defaults.get("fused")
    print(f"{'ok  ' if stock else 'FAIL'} build_optimizer vendor=nvidia -> stock: "
          f"fused={opt.defaults.get('fused')}")
    if not stock:
        FAILURES += 1
    os.environ.pop(ENV_KEY, None)
    os.environ.pop("GEMS_VENDOR", None)

    if FAILURES:
        print(f"FAILED: {FAILURES} case(s)", file=sys.stderr)
        return 1
    print("PASS: selection matrix matches the documented behaviour")
    return 0


if __name__ == "__main__":
    sys.exit(main())
