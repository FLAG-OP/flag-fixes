"""Selection matrix: which AdamW implementation does build_optimizer pick?

Covers the FLAGSCALE_ADAMW_FUSED env override, the DTensor (FSDP2) guard, the
CPU-parameter guard, garbage env values, and the construction-time fallback.
The CUDA bf16 case requires a device; without one the script exits 2 (a skip
is not a pass) after running all CPU-side cases.
"""

import os
import sys

import torch


def main():
    failures = 0

    from omegaconf import OmegaConf
    from flagscale.train.train_retrieval import adamw_impl_kwargs, build_optimizer

    cfg = OmegaConf.create({"name": "adamw", "lr": 2e-5, "weight_decay": 0.01})
    env_key = "FLAGSCALE_ADAMW_FUSED"

    # DTensor-like object: the guard inspects type(param).__module__ first and
    # must return {} before touching tensor attributes (FSDP2 keeps stock).
    dtensor_like = type("DTensor", (), {})
    dtensor_like.__module__ = "torch.distributed.tensor"

    cuda_params = None
    if torch.cuda.is_available():
        cuda_params = [
            torch.nn.Parameter(torch.randn(64, 64, device="cuda", dtype=torch.bfloat16))
            for _ in range(8)
        ]

    cases = []

    def check(label, fn, expected):
        nonlocal failures
        try:
            got = fn()
        except Exception as error:  # noqa: BLE001 - report and continue
            print(f"FAIL {label}: raised {type(error).__name__}: {error}",
                  file=sys.stderr)
            failures += 1
            return
        ok = got == expected
        print(f"{'ok  ' if ok else 'FAIL'} {label}: kwargs={got} expected={expected}")
        if not ok:
            failures += 1

    os.environ.pop(env_key, None)
    check("auto, dtensor-like (FSDP2 guard)",
          lambda: adamw_impl_kwargs([dtensor_like()]), {})
    check("auto, empty parameter list", lambda: adamw_impl_kwargs([]), {})

    cpu_params = [torch.nn.Parameter(torch.randn(4, 4))]
    check("auto, cpu params", lambda: adamw_impl_kwargs(cpu_params), {})

    os.environ[env_key] = "1"
    check("forced 1, dtensor-like (still guarded)",
          lambda: adamw_impl_kwargs([dtensor_like()]), {})
    check("forced 1, cpu params", lambda: adamw_impl_kwargs(cpu_params), {})

    os.environ[env_key] = "0"
    check("forced 0", lambda: adamw_impl_kwargs(cpu_params), {})
    check("forced 0, empty list", lambda: adamw_impl_kwargs([]), {})

    os.environ[env_key] = "nonsense"
    check("unrecognised value", lambda: adamw_impl_kwargs(cpu_params), {})

    if cuda_params is not None:
        os.environ[env_key] = "1"
        check("forced 1, cuda bf16", lambda: adamw_impl_kwargs(cuda_params),
              {"fused": True})
        os.environ[env_key] = "auto"
        check("auto, cuda bf16", lambda: adamw_impl_kwargs(cuda_params),
              {"fused": True})
        os.environ[env_key] = "0"
        opt = build_optimizer(cfg, cuda_params, allow_adafactor=False)
        print(f"{'ok  ' if not opt.defaults.get('fused') else 'FAIL'} "
              f"build_optimizer honours 0 on cuda: fused={opt.defaults.get('fused')}")
        if opt.defaults.get("fused"):
            failures += 1
        os.environ[env_key] = "auto"
        opt = build_optimizer(cfg, cuda_params, allow_adafactor=False)
        fused = bool(opt.defaults.get("fused"))
        print(f"{'ok  ' if fused else 'FAIL'} build_optimizer auto on cuda: "
              f"fused={fused}")
        if not fused:
            failures += 1
    else:
        print("cuda cases skipped: no CUDA device (exit 2; skip is not a pass)")
        return 2

    os.environ.pop(env_key, None)
    if failures:
        print(f"FAILED: {failures} case(s)", file=sys.stderr)
        return 1
    print("PASS: selection matrix matches the documented behaviour")
    return 0


if __name__ == "__main__":
    sys.exit(main())
