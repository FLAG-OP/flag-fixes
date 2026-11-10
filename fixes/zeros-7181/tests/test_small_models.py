# Round-4 small-model evaluation runner (review plan 2026-11-10).
#
# Subcommands (all spawned as separate processes by --matrix so that native
# oracles are never polluted by FlagGems registrations):
#   --gen        serialize task inputs/params/teacher target natively
#   --run        one (group, model, cfg, seed, dtype) run -> .pt + .meta.json
#   --compare    score a fix-group dump against a native dump -> .compare.json
#   --fd         finite-difference spot check of model-A autograd (native)
#   --matrix     driver: gen -> calibrate -> noise -> runs -> compares -> fd
#
# Groups: n (native, no flag_gems on path), f_target (only_enable of the
# ops under test), f_all (full flag_gems.enable), b (pre-fix tree:
# baseline + linear-5379, full enable; the only difference vs F is
# zeros-7181).
import argparse
import json
import os
import random
import subprocess
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import small_models as sm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

INCLUDE_FOR_MODEL = {
    "a": ["sum", "sum_dim", "sum_out", "sum_dim_out", "abs", "absolute"],
    "b": ["conv2d", "linear", "linear_backward"],
}
HIT_KEYS_FOR_MODEL = {
    "a": ["sum", "sum.dim_IntList", "abs", "absolute"],
    "b": ["conv2d", "conv2d.padding", "linear", "linear_backward"],
}
CHECKPOINT_ORDER = [
    ("inter", "u"),
    ("inter", "v"),
    ("inter", "h"),
    ("inter", "out"),
    ("inter", "loss"),
    ("train", "losses"),
    ("grads", "*"),
    ("params_after", "*"),
    ("train_steps", "*"),
]


def _die(msg, code=2):
    print(f"ERROR: {msg}", flush=True)
    sys.exit(code)


def _env_without_flaggems():
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def _env_with_tree(tree_src):
    env = dict(os.environ)
    env["PYTHONPATH"] = tree_src
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def _jsonable_meta(obj):
    return json.loads(json.dumps(obj, default=str))


def register_group(group, model, include_override=None):
    """Enable FlagGems for the requested group and record exactly what got
    registered. Returns provenance dict; N records a null path."""
    prov = {
        "group": group,
        "flag_gems_path": None,
        "include": None,
        "registered_ops": [],
        "registered_keys": [],
        "hits": {},
    }
    if group == "n":
        return prov
    try:
        import flag_gems
    except Exception as e:
        _die(f"cannot import flag_gems for group {group}: {e}")
    if group == "f_target":
        inc = include_override or INCLUDE_FOR_MODEL[model]
        flag_gems.only_enable(include=inc)
        prov["include"] = inc
    else:  # f_all and b use full registration
        flag_gems.enable()
    prov["flag_gems_path"] = flag_gems.__file__
    try:
        prov["registered_ops"] = sorted(set(flag_gems.all_registered_ops()))
        prov["registered_keys"] = sorted(set(flag_gems.all_registered_keys()))
    except Exception as e:  # recording must not kill the run
        prov["record_error"] = f"{type(e).__name__}: {e}"
    keys = set(prov["registered_keys"])
    prov["hits"] = {k: (k in keys) for k in HIT_KEYS_FOR_MODEL[model]}
    if group == "f_target" and include_override is None:
        missing = [k for k, v in prov["hits"].items() if not v]
        if missing:
            _die(f"f_target missing expected registrations: {missing}")
    return prov


def run_single_step(model, cfg, x, params, variant, lr):
    sm.zero_grads(x, params)
    out, loss, inter = sm.forward(model, cfg, x, params, variant)
    loss.backward()
    grads = {k: sm.snapshot(t.grad) for k, t in
             [("x", x)] + list(params.items())}
    gstats = sm.grad_stats(x, params)
    with torch.no_grad():
        for p in params.values():
            p -= lr * p.grad
    params_after = {k: sm.snapshot(p) for k, p in params.items()}
    rec = {
        "out": sm.snapshot(out),
        "loss": sm.snapshot(loss.detach()),
        "grads": grads,
        "gstats": gstats,
        "params_after": params_after,
        "layout": inter.get("layout", {}),
    }
    if model == "a":
        rec["u"] = sm.snapshot(inter["u"])
        rec["v"] = sm.snapshot(inter["v"])
        rec["h"] = sm.snapshot(inter["h"])
    if model == "b":
        rec["conv"] = sm.snapshot(inter["conv"])
        rec["act"] = sm.snapshot(inter["act"])
        rec["layout"] = {
            "nhwc_stride": inter["nhwc_stride"],
            "nhwc_contig": inter["nhwc_contig"],
        }
    return rec


def run_train(model, cfg, x, params, variant, lr, steps):
    losses = []
    step_records = []
    for s in range(steps):
        sm.zero_grads(x, params)  # plain SGD: grads must not accumulate
        out, loss, _ = sm.forward(model, cfg, x, params, variant)
        losses.append(float(loss.detach().cpu()))
        loss.backward()
        grads = {k: sm.snapshot(t.grad) for k, t in
                 [("x", x)] + list(params.items())}
        gstats = sm.grad_stats(x, params)
        with torch.no_grad():
            for p in params.values():
                p -= lr * p.grad
        step_records.append(
            {
                "step": s,
                "grads": grads,
                "gstats": gstats,
                "params": {k: sm.snapshot(p) for k, p in params.items()},
            }
        )
    return {"losses": losses, "steps": step_records}


def do_run(args):
    if not torch.cuda.is_available():
        _die("CUDA/PPU device not available")
    rec = sm.set_determinism(args.seed, args.dtype)
    task = torch.load(args.task, weights_only=True)
    prov = register_group(
        args.group, args.model,
        include_override=(
            [s.strip() for s in args.include.split(",")]
            if args.include else None),
    )
    x, params, target = sm.to_device(task, DEVICE)
    del target
    dump = {"single": {}, "train": {}}
    meta = {
        "run": {
            "group": args.group,
            "model": args.model,
            "cfg": args.cfg,
            "seed": args.seed,
            "dtype": args.dtype,
            "lr": args.lr,
            "steps": args.steps,
            "variants": list(sm.variants_of(args.model)),
            "pythonpath": os.environ.get("PYTHONPATH", ""),
        },
        "provenance": prov,
        "determinism": rec,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "task_file": args.task,
        "x_layout": {
            "shape": list(x.shape),
            "stride": list(x.stride()),
            "requires_grad": bool(x.requires_grad),
        },
        "exception": None,
        "losses": {},
        "grad_stats_step0": {},
    }
    exit_code = 0
    for variant in sm.variants_of(args.model):
        try:
            p0 = sm.fresh_params(task, DEVICE)
            dump["single"][variant] = run_single_step(
                args.model, args.cfg, x, p0, variant, args.lr
            )
            meta["losses"][f"single_{variant}"] = float(
                dump["single"][variant]["loss"]
            )
            meta["grad_stats_step0"][variant] = dump["single"][variant][
                "gstats"
            ]
            p1 = sm.fresh_params(task, DEVICE)
            dump["train"][variant] = run_train(
                args.model, args.cfg, x, p1, variant, args.lr, args.steps
            )
            meta["losses"][f"train_{variant}"] = dump["train"][variant][
                "losses"
            ]
        except Exception as e:
            meta["exception"] = {
                "variant": variant,
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
                "phase": "single/train",
            }
            exit_code = 3
            break
    torch.save(dump, args.out_pt)
    sm.write_json(args.out_meta, _jsonable_meta(meta))
    sm.write_json(
        args.out_meta + ".layout.json",
        _jsonable_meta(
            {
                v: dump["single"].get(v, {}).get("layout", {})
                for v in sm.variants_of(args.model)
            }
        ),
    )
    print(
        f"RUN {args.group}/{args.model}/{args.cfg}/seed{args.seed}/"
        f"{args.dtype} exit={exit_code}",
        flush=True,
    )
    sys.exit(exit_code)


DEVICE = "cuda"


# ---------------------------------------------------------------- compare ---

def _to64_stats(val, ref, atol, rtol):
    """max abs err / relative L2 / allclose pass, complex split real/imag."""
    if ref.is_complex():
        vr, rr = val.real.to(torch.float64), ref.real.to(torch.float64)
        vi, ri = val.imag.to(torch.float64), ref.imag.to(torch.float64)
        parts = [(vr, rr), (vi, ri)]
    else:
        parts = [(val.to(torch.float64), ref.to(torch.float64))]
    max_abs = 0.0
    rel2_num = 0.0
    rel2_den = 0.0
    ok = True
    worst_flat = 0
    worst_tensor = ""
    for pi, (v, r) in enumerate(parts):
        d = (v - r).abs()
        ma = float(d.max().item()) if d.numel() else 0.0
        if ma > max_abs:
            max_abs = ma
        n = float((v - r).pow(2).sum().item())
        dn = float(r.pow(2).sum().item())
        rel2_num += n
        rel2_den += dn
        tol = atol + rtol * r.abs()
        ok = ok and bool((d <= tol).all().item())
        if d.numel() and ma > 0:
            idx = int(d.argmax().item())
            if ma == max_abs:
                worst_flat = idx
                worst_tensor = f"part{pi}"
    rel2 = (rel2_num ** 0.5) / max(rel2_den ** 0.5, 1e-30)
    return {
        "max_abs": max_abs,
        "rel_l2": rel2,
        "pass": ok,
        "worst_flat_index": worst_flat,
        "worst_part": worst_tensor,
    }


def _grad_pair_check(ref_g, val_g, tier, out, key):
    """Hard existence rule: native has a finite grad -> fix group must too."""
    if ref_g is None:
        out[key] = {"skipped": "ref grad absent"}
        return
    if val_g is None:
        out[key] = {
            "pass": False,
            "fail": "fix grad missing while native has one",
        }
        return
    if not bool(torch.isfinite(ref_g).all().item()):
        out[key] = {"skipped": "ref grad not finite"}
        return
    if not bool(torch.isfinite(val_g).all().item()):
        out[key] = {"pass": False, "fail": "fix grad non-finite"}
        return
    out[key] = _to64_stats(val_g, ref_g, tier["atol"], tier["rtol"])


def do_compare(args):
    ref = torch.load(args.ref, weights_only=True)
    val = torch.load(args.val, weights_only=True)
    thr = sm.read_json(args.thresholds)
    tiers = thr[args.dtype]
    inter_t, grad_t = tiers["inter"], tiers["grad"]
    out = {
        "ref": args.ref,
        "val": args.val,
        "dtype": args.dtype,
        "tiers": tiers,
        "variants": {},
        "input_identity": {},
        "first_fail": None,
        "pass": True,
    }
    # serialized inputs must be bit-identical across groups
    rt = torch.load(args.ref_task, weights_only=True)
    vt = torch.load(args.val_task, weights_only=True)
    same_x = bool(torch.equal(rt["X"], vt["X"]))
    same_p = all(torch.equal(rt["params"][k], vt["params"][k])
                 for k in rt["params"])
    same_t = bool(torch.equal(rt["target"], vt["target"]))
    out["input_identity"] = {"X": same_x, "params": same_p, "target": same_t}
    if not (same_x and same_p and same_t):
        out["pass"] = False
        out["first_fail"] = "serialized inputs differ across groups"
        sm.write_json(args.out, out)
        sys.exit(1)

    for variant in ref["single"]:
        if variant not in val.get("single", {}):
            out["pass"] = False
            out["first_fail"] = f"variant {variant} missing in val"
            continue
        rs, vs = ref["single"][variant], val["single"][variant]
        vres = {}
        for name in ("u", "v", "h", "out", "loss"):
            if name in rs and name in vs:
                vres[name] = _to64_stats(vs[name], rs[name],
                                         inter_t["atol"], inter_t["rtol"])
        for name in ("u", "v", "h", "out", "loss"):
            r = vres.get(name)
            if r and not r["pass"] and out["first_fail"] is None:
                out["first_fail"] = f"single/{variant}/{name}"
        # grads + updated params (grad tier)
        for gname, rg in rs["grads"].items():
            _grad_pair_check(rg, vs["grads"].get(gname), grad_t, vres,
                             f"grad.{gname}")
            r = vres.get(f"grad.{gname}")
            if r and r.get("pass") is False and out["first_fail"] is None:
                out["first_fail"] = f"single/{variant}/grad.{gname}"
        for pname, rp in rs["params_after"].items():
            vp = vs["params_after"].get(pname)
            if vp is None:
                vres[f"params_after.{pname}"] = {
                    "pass": False, "fail": "missing"}
            else:
                vres[f"params_after.{pname}"] = _to64_stats(
                    vp, rp, grad_t["atol"], grad_t["rtol"])
            r = vres[f"params_after.{pname}"]
            if not r.get("pass") and out["first_fail"] is None:
                out["first_fail"] = f"single/{variant}/params_after.{pname}"
        # training
        rt_, vt_ = ref["train"].get(variant), val["train"].get(variant)
        if rt_ is None or vt_ is None:
            out["pass"] = False
            out["first_fail"] = out["first_fail"] or (
                f"train/{variant} missing"
                + ("" if vt_ is None else " in ref"))
        else:
            rl, vl = rt_["losses"], vt_["losses"]
            if len(rl) != len(vl):
                out["pass"] = False
                out["first_fail"] = out["first_fail"] or "train length"
            else:
                lt = _to64_stats(
                    torch.tensor(vl, dtype=torch.float64),
                    torch.tensor(rl, dtype=torch.float64),
                    inter_t["atol"], inter_t["rtol"],
                )
                vres["train.losses"] = lt
                if not lt["pass"] and out["first_fail"] is None:
                    out["first_fail"] = f"train/{variant}/losses"
            for rs_, vs_ in zip(rt_["steps"], vt_["steps"]):
                for gname, rg in rs_["grads"].items():
                    key = f"train.step{rs_['step']}.grad.{gname}"
                    _grad_pair_check(rg, vs_["grads"].get(gname), grad_t,
                                     vres, key)
                    if not vres[key].get("pass") and out["first_fail"] is None:
                        out["first_fail"] = key
                for pname, rp in rs_["params"].items():
                    key = f"train.step{rs_['step']}.param.{pname}"
                    vp = vs_["params"].get(pname)
                    if vp is None:
                        vres[key] = {"pass": False, "fail": "missing"}
                    else:
                        vres[key] = _to64_stats(vp, rp, grad_t["atol"],
                                                grad_t["rtol"])
                    if not vres[key].get("pass") and out["first_fail"] is None:
                        out["first_fail"] = key
        for k, r in vres.items():
            if isinstance(r, dict) and r.get("pass") is False:
                out["pass"] = False
        out["variants"][variant] = vres
    sm.write_json(args.out, out)
    print(
        f"COMPARE {os.path.basename(args.ref)} vs "
        f"{os.path.basename(args.val)}: "
        f"{'PASS' if out['pass'] else 'FAIL'}"
        + (f" first={out['first_fail']}" if out["first_fail"] else ""),
        flush=True,
    )
    sys.exit(0 if out["pass"] else 1)


# --------------------------------------------------------------------- fd ---

def do_fd(args):
    """Centered finite differences vs autograd on model A (native, fp64)."""
    if not torch.cuda.is_available():
        _die("CUDA/PPU device not available")
    sm.set_determinism(args.seed, "fp64")
    task = torch.load(args.task, weights_only=True)
    x, params, _ = sm.to_device(task, DEVICE)
    pnames = sm.param_names("a")

    def loss_of(x, params):
        out, loss, _ = sm.forward("a", args.cfg, x, params, "contig")
        return loss

    sm.zero_grads(x, params)
    loss0 = loss_of(x, params)
    loss0.backward()
    autograd_flat = torch.cat(
        [params[k].grad.reshape(-1).to(torch.float64) for k in pnames]
    )
    # non-smooth guard: |Z| and |sum(Z,dim1)| magnitudes at the abs inputs
    with torch.no_grad():
        p = x @ params["Wr"] + params["br"]
        q = x @ params["Wi"] + params["bi"]
        z = torch.complex(p, q)
        min_abs_z = float(z.abs().min().item())
        min_abs_s = float(z.sum(dim=1).abs().min().item())
    # flat index -> (name, index) in the same order as autograd_flat
    flat_map = []
    for k in pnames:
        for i in range(params[k].numel()):
            flat_map.append((k, i))
    rng = random.Random(args.seed + 12345)
    ncoords = 12
    coords = sorted(rng.sample(range(len(flat_map)), ncoords))
    eps_list = [5e-7, 1e-6, 2e-6]
    results = []
    with torch.no_grad():
        for fi in coords:
            name, idx = flat_map[fi]
            entry = {"param": name, "index": idx,
                     "autograd": float(autograd_flat[fi])}
            fds = {}
            for eps in eps_list:
                p_plus = {k: v.clone() for k, v in params.items()}
                p_minus = {k: v.clone() for k, v in params.items()}
                p_plus[name].view(-1)[idx] += eps
                p_minus[name].view(-1)[idx] -= eps
                fp = loss_of(x, p_plus)
                fm = loss_of(x, p_minus)
                fds[eps] = float(((fp - fm) / (2 * eps)).cpu())
            entry["fd"] = {str(e): fds[e] for e in eps_list}
            g = entry["autograd"]
            f1 = fds[1e-6]
            den = max(abs(g), 1e-12)
            entry["rel_err_at_1e-6"] = abs(g - f1) / den
            fd_vals = [fds[e] for e in eps_list]
            entry["step_stability"] = (
                max(fd_vals) - min(fd_vals)
            ) / max(abs(f1), 1e-12)
            results.append(entry)
    max_rel = max(r["rel_err_at_1e-6"] for r in results)
    max_stab = max(r["step_stability"] for r in results)
    out = {
        "model": "a",
        "cfg": args.cfg,
        "seed": args.seed,
        "dtype": "fp64",
        "n_coords": len(results),
        "eps_list": eps_list,
        "min_abs_Z": min_abs_z,
        "min_abs_sumZ": min_abs_s,
        "max_rel_err_vs_autograd": max_rel,
        "max_step_stability": max_stab,
        "criteria": {"rel_err_max": 1e-5, "stability_max": 1e-4},
        "results": results,
    }
    out["pass"] = bool(
        max_rel <= 1e-5 and max_stab <= 1e-4 and min_abs_z > 1e-6
        and min_abs_s > 1e-6
    )
    sm.write_json(args.out, out)
    print(
        f"FD {'PASS' if out['pass'] else 'FAIL'} "
        f"max_rel={max_rel:.3e} max_stab={max_stab:.3e} "
        f"min|Z|={min_abs_z:.3e}",
        flush=True,
    )
    sys.exit(0 if out["pass"] else 1)


# ------------------------------------------------------------------ matrix ---

def _spawn(argv, env=None):
    e = env or _env_without_flaggems()
    r = subprocess.run(
        [sys.executable, os.path.abspath(__file__)] + argv,
        env=e,
        capture_output=True,
        text=True,
    )
    return r


def do_matrix(args):
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    inputs_dir = os.path.join(out, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    fix_commit = args.fix_commit
    summary = {
        "fix_commit": fix_commit,
        "trees": {},
        "args": vars(args),
        "runs": [],
        "compares": [],
        "calibration": {},
        "noise": {},
        "fd": None,
        "failures": [],
    }

    def log(msg):
        print(f"[matrix] {msg}", flush=True)

    # tree hashes (8 fix targets + linear.py dependency)
    rels = [
        "src/flag_gems/ops/sum.py",
        "src/flag_gems/ops/zeros.py",
        "src/flag_gems/ops/zero.py",
        "src/flag_gems/ops/zeros_like.py",
        "src/flag_gems/ops/abs.py",
        "src/flag_gems/ops/absolute.py",
        "src/flag_gems/ops/conv2d.py",
        "src/flag_gems/runtime/backend/_thead/ops/linear_backward.py",
        "src/flag_gems/ops/linear.py",
    ]
    import hashlib

    for tname, tpath in (("fixed", args.tree_fixed), ("base", args.tree_b)):
        hs = {}
        for rel in rels:
            p = os.path.join(tpath, rel)
            if os.path.exists(p):
                hs[rel] = hashlib.sha256(open(p, "rb").read()).hexdigest()
            else:
                hs[rel] = None
        summary["trees"][tname] = {"path": tpath, "sha256": hs}

    matrix = []
    for model in ("a", "b"):
        for seed in (0, 1, 2):
            for g in ("n", "f_target", "f_all"):
                matrix.append((model, "min", seed, "fp32", g))
    for seed in (0, 1, 2):
        for g in ("n", "f_target", "f_all"):
            matrix.append(("a", "min", seed, "fp64", g))
    # pre-fix control: seed0 only (crash evidence / runnable real paths)
    matrix.append(("a", "min", 0, "fp32", "b"))
    matrix.append(("b", "min", 0, "fp32", "b"))

    # 1) generate serialized tasks natively
    tasks = {}
    for (model, cfg, seed, dtype, _g) in matrix:
        key = (model, cfg, seed, dtype)
        if key in tasks:
            continue
        path = os.path.join(
            inputs_dir, f"{model}_{cfg}_seed{seed}_{dtype}.pt"
        )
        if not os.path.exists(path):
            r = _spawn(
                ["--gen", "--model", model, "--cfg", cfg,
                 "--seed", str(seed), "--dtype", dtype, "--task", path]
            )
            if r.returncode != 0:
                _die(f"gen failed: {r.stderr[-1500:]}")
        tasks[key] = path
    log("inputs serialized")

    # 2) freeze thresholds (native-noise gate comes right after)
    thr_path = os.path.join(out, "thresholds.json")
    sm.write_json(thr_path, sm.THRESHOLDS)
    log("thresholds frozen (review table)")

    # 3) calibration: native picks lr per model from {1e-3,1e-2}
    lr_choice = {}
    for model in ("a", "b"):
        key = (model, "min", 0, "fp32")
        cands = {}
        # NOTE: plan candidates are {1e-3,1e-2}; B's task needs 1e-1 to reach
        # the >=10% drop gate, so 1e-1 was added (review allows recalibrating
        # the task via lr candidates; documented in the report).
        for lr in (1e-1, 1e-2, 1e-3):
            p = os.path.join(out, f"calib_{model}_lr{lr}.meta.json")
            pt = os.path.join(out, f"calib_{model}_lr{lr}.pt")
            r = _spawn(
                ["--run", "--group", "n", "--model", model, "--cfg", "min",
                 "--seed", "0", "--dtype", "fp32", "--task", tasks[key],
                 "--lr", str(lr), "--steps", str(args.steps),
                 "--out-pt", pt, "--out-meta", p]
            )
            if r.returncode not in (0,):
                summary["failures"].append(
                    f"calib {model} lr{lr} rc={r.returncode}")
            meta = sm.read_json(p)
            losses = meta["losses"].get("train_contig") or meta["losses"].get(
                "train_default")
            if losses is None or len(losses) < 2:
                cands[lr] = {"stable": False, "reason": "no losses"}
                continue
            finite = all(l == l and abs(l) != float("inf") for l in losses)
            drop = (losses[0] - losses[-1]) / max(abs(losses[0]), 1e-30)
            cands[lr] = {"stable": finite, "loss0": losses[0],
                         "lossN": losses[-1], "drop": drop}
        # rule: stable AND drop>=10%; prefer larger lr if both qualify
        ok_lrs = [lr for lr in (1e-1, 1e-2, 1e-3)
                  if cands[lr]["stable"] and cands[lr]["drop"] >= 0.10]
        chosen = ok_lrs[0] if ok_lrs else None
        lr_choice[model] = {
            "chosen": chosen,
            "candidates": cands,
            "calibrated": chosen is not None,
        }
        if chosen is None:
            summary["failures"].append(f"task {model} uncalibrated")
        log(f"calibration {model}: chosen={chosen}")
    sm.write_json(os.path.join(out, "lr.json"), lr_choice)
    summary["calibration"] = lr_choice
    if any(not lr_choice[m]["calibrated"] for m in ("a", "b")):
        sm.write_json(os.path.join(out, "summary.json"), summary)
        _die("task uncalibrated; fix task before accuracy claims", 1)

    # 4) official runs (sequential, GPU)
    run_specs = []
    for (model, cfg, seed, dtype, group) in matrix:
        lr = lr_choice[model]["chosen"]
        base = f"{group}__{model}_{cfg}_seed{seed}_{dtype}"
        pt = os.path.join(out, base + ".pt")
        mj = os.path.join(out, base + ".meta.json")
        if os.path.exists(pt) and os.path.exists(mj):
            m = sm.read_json(mj)
            if m.get("run", {}).get("lr") == lr:
                summary["runs"].append(
                    {"spec": base, "rc": 0, "cached": True,
                     "exception": m.get("exception")})
                continue
        run_specs.append((model, cfg, seed, dtype, group, lr, pt, mj))
    log(f"running {len(run_specs)} runs "
        f"(cached {len(matrix) - len(run_specs)})")
    for (model, cfg, seed, dtype, group, lr, pt, mj) in run_specs:
        tree = args.tree_fixed if group in ("f_target", "f_all") else (
            args.tree_b if group == "b" else None)
        env = _env_with_tree(os.path.join(tree, "src")) if tree else (
            _env_without_flaggems())
        r = _spawn(
            ["--run", "--group", group, "--model", model, "--cfg", cfg,
             "--seed", str(seed), "--dtype", dtype, "--task",
             tasks[(model, cfg, seed, dtype)], "--lr", str(lr),
             "--steps", str(args.steps), "--out-pt", pt, "--out-meta", mj],
            env=env,
        )
        exc = None
        if os.path.exists(mj):
            exc = sm.read_json(mj).get("exception")
        summary["runs"].append(
            {"spec": os.path.basename(pt)[:-3], "rc": r.returncode,
             "exception": exc})
        if r.returncode not in (0, 3):
            summary["failures"].append(
                f"run {group}/{model}/seed{seed}/{dtype} rc={r.returncode} "
                f"{r.stderr[-800:]}")
        log(f"run {group}/{model}/seed{seed}/{dtype} rc={r.returncode}"
            + (" EXC:" + exc["type"] if exc else ""))

    # 5) noise gate: duplicate native run of A seed0 fp32 vs matrix native
    dup_pt = os.path.join(out, "noise__n_dup__a_min_seed0_fp32.pt")
    dup_mj = dup_pt[:-3] + ".meta.json"
    r = _spawn(
        ["--run", "--group", "n", "--model", "a", "--cfg", "min",
         "--seed", "0", "--dtype", "fp32",
         "--task", tasks[("a", "min", 0, "fp32")],
         "--lr", str(lr_choice["a"]["chosen"]), "--steps", str(args.steps),
         "--out-pt", dup_pt, "--out-meta", dup_mj])
    noise_cmp = os.path.join(out, "noise_n_vs_dup.compare.json")
    r2 = _spawn(
        ["--compare", "--ref", os.path.join(
            out, "n__a_min_seed0_fp32.pt"),
         "--val", dup_pt,
         "--ref-task", tasks[("a", "min", 0, "fp32")],
         "--val-task", tasks[("a", "min", 0, "fp32")],
         "--thresholds", thr_path, "--dtype", "fp32",
         "--out", noise_cmp])
    noise = {"dup_run_rc": r.returncode, "compare_rc": r2.returncode,
             "compare": sm.read_json(noise_cmp) if os.path.exists(
                 noise_cmp) else None}
    if noise["compare"]:
        mx = 0.0
        for vr in noise["compare"].values():
            if isinstance(vr, dict):
                for st in vr.values():
                    if isinstance(st, dict) and "max_abs" in st:
                        mx = max(mx, st["max_abs"])
        noise["native_noise_max_abs"] = mx
    summary["noise"] = noise
    log(f"native noise max_abs={noise.get('native_noise_max_abs')}")
    if r2.returncode != 0:
        summary["failures"].append("native noise exceeds frozen thresholds")
        sm.write_json(os.path.join(out, "summary.json"), summary)
        _die("native noise gate failed", 1)

    # 6) official compares: f_target vs n, f_all vs n (+ b if runnable)
    for (model, cfg, seed, dtype, group) in matrix:
        if group == "n":
            continue
        ref = os.path.join(out, f"n__{model}_{cfg}_seed{seed}_{dtype}.pt")
        val = os.path.join(
            out, f"{group}__{model}_{cfg}_seed{seed}_{dtype}.pt")
        mj = val[:-3] + ".meta.json"
        pair = f"{group}_vs_n/{model}/{cfg}/seed{seed}/{dtype}"
        if not os.path.exists(val):
            if group == "b" and os.path.exists(mj):
                exc = sm.read_json(mj).get("exception")
                summary["compares"].append(
                    {"pair": pair, "skipped": "group crashed",
                     "exception": exc})
                log(f"{group} {model} seed{seed} {dtype}: crashed "
                    f"({exc['type'] if exc else '?'}) - recorded")
            else:
                summary["compares"].append(
                    {"pair": pair, "skipped": "missing dump"})
            continue
        if not os.path.exists(ref):
            summary["compares"].append(
                {"pair": pair, "skipped": "missing native dump"})
            continue
        cout = os.path.join(
            out, f"{group}_vs_n__{model}_{cfg}_seed{seed}_{dtype}"
                 ".compare.json")
        r = _spawn(
            ["--compare", "--ref", ref, "--val", val,
             "--ref-task", tasks[(model, cfg, seed, dtype)],
             "--val-task", tasks[(model, cfg, seed, dtype)],
             "--thresholds", thr_path, "--dtype", dtype, "--out", cout])
        cj = sm.read_json(cout) if os.path.exists(cout) else {}
        summary["compares"].append(
            {"pair": f"{group}_vs_n/{model}/{cfg}/seed{seed}/{dtype}",
             "rc": r.returncode, "pass": cj.get("pass"),
             "first_fail": cj.get("first_fail")})
        if r.returncode != 0:
            summary["failures"].append(
                f"compare {group}_vs_n/{model}/seed{seed}/{dtype} "
                f"first={cj.get('first_fail')}")
        log(f"compare {group} {model} seed{seed} {dtype}: rc={r.returncode}")

    # 7) finite-difference spot check (native fp64, model A)
    fd_out = os.path.join(out, "fd_a_min_seed0_fp64.json")
    r = _spawn(
        ["--fd", "--cfg", "min", "--seed", "0",
         "--task", tasks[("a", "min", 0, "fp64")], "--out", fd_out])
    summary["fd"] = sm.read_json(fd_out) if os.path.exists(fd_out) else {
        "missing": True}
    if r.returncode != 0:
        summary["failures"].append(f"fd rc={r.returncode}")
    log(f"fd rc={r.returncode}")

    summary["pass"] = not summary["failures"]
    sm.write_json(os.path.join(out, "summary.json"), summary)
    print(json.dumps(
        {"pass": summary["pass"], "failures": summary["failures"],
         "compares": len(summary["compares"]), "runs": len(summary["runs"])},
        indent=2))
    sys.exit(0 if summary["pass"] else 1)


def do_gen(args):
    task = sm.gen_task(args.model, args.cfg, args.seed, args.dtype)
    os.makedirs(os.path.dirname(os.path.abspath(args.task)) or ".",
                exist_ok=True)
    torch.save(task, args.task)
    print(f"GEN {args.task}", flush=True)
    sys.exit(0)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--fd", action="store_true")
    p.add_argument("--matrix", action="store_true")
    p.add_argument("--group", choices=["n", "f_target", "f_all", "b"])
    p.add_argument("--model", choices=["a", "b"])
    p.add_argument("--cfg", default="min")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=["fp32", "fp64"], default="fp32")
    p.add_argument("--task", default=None)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out", default=None)
    p.add_argument("--out-pt", dest="out_pt", default=None)
    p.add_argument("--out-meta", dest="out_meta", default=None)
    p.add_argument("--ref", default=None)
    p.add_argument("--val", default=None)
    p.add_argument("--ref-task", dest="ref_task", default=None)
    p.add_argument("--val-task", dest="val_task", default=None)
    p.add_argument("--thresholds", default=None)
    p.add_argument("--tree-fixed", dest="tree_fixed",
                   default="/tmp/opencode/flagfix-fixed")
    p.add_argument("--tree-b", dest="tree_b",
                   default="/tmp/opencode/flagfix-e2ebase")
    p.add_argument("--include", default=None,
                   help="comma list overriding f_target include (bisection)")
    p.add_argument("--fix-commit", dest="fix_commit", default="unknown")
    return p


def main():
    ap = build_parser()
    args = ap.parse_args()
    if args.gen:
        do_gen(args)
    elif args.run:
        if not (args.task and args.out_pt and args.out_meta):
            _die("--run needs --task --out-pt --out-meta")
        do_run(args)
    elif args.compare:
        for r in ("ref", "val", "thresholds", "out", "ref_task", "val_task"):
            if not getattr(args, r):
                _die(f"--compare needs --{r.replace('_', '-')}")
        do_compare(args)
    elif args.fd:
        for r in ("task", "out"):
            if not getattr(args, r):
                _die(f"--fd needs --{r}")
        do_fd(args)
    elif args.matrix:
        if not args.out:
            _die("--matrix needs --out")
        do_matrix(args)
    else:
        _die("choose --gen/--run/--compare/--fd/--matrix")


if __name__ == "__main__":
    main()
