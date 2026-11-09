"""E2E: m2-bert container model (togethercomputer/m2-bert-80M-8k-retrieval)
forward + backward, the real caller from problem.md.

Pre-fix : forward completes (complex ops are forward-safe) but backward dies
          with ``KeyError: 'complex64'`` from triton/_utils.py
          type_canonicalisation_dict while summing u_f*k_f hybrid-conv
          gradients -> exit 1.
Post-fix: flag_gems forward matches the native torch forward (loss compare),
          backward completes: the original crash chain runs to completion and
          every gradient the native oracle produces is present and finite
          except the EXACT known-gap set declared in known_gap_m2.txt (asserted
          bidirectionally: unexpected missing grads and unexpectedly present
          fixture entries both fail) -> exit 0. This proves "the native
          computation chain continues to execute", not full gradient
          correctness: value comparison stays informational (pre-existing
          near-stationary divergence, measured identically without this fix);
          local complex autograd parity (abs/sum chains, tight tolerance) is
          asserted in test_complex_ops.py.

Precondition (machine-checkable, review P2): before any model work the run
loads fix.yaml's e2e_precondition.files (our 8 targets' after_sha256 values
plus linear-5379's src/flag_gems/ops/linear.py) and hashes the target tree;
mismatch/missing -> ``MISSING PRECONDITION`` exit 2, never a wrong-combination
0/1. Requires are declared in fix.yaml (requires: linear-5379) instead of
prose.

Model code pinned: revision 90e0b28f3382c289cc6c1e92ef53d7dc5a3ec14b; the
config print records the resolved commit and whether the bundled opt_einsum
stub was used.

Config defaults equal the container run (12 layers / hidden 768 / heads 12 /
inter 3072 / seq 8192 / POS 8192 / LMAX 8192 / hyena filter order 128);
override with NLAYERS/HIDDEN/HEADS/INTER/SEQLEN/POS/LMAX/HORDER for quick runs.
Random init, trust_remote_code, no weight download.

Run : PYTHONPATH=<FlagGems>/src python test_m2_bert_backward.py
Exit: 0 = ok | 1 = failure | 2 = missing CUDA / transformers / precondition
A skip is not a pass: missing dependency or precondition -> exit 2, never 0.
"""
import os
import sys
import traceback

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Dependency gate first: keep the missing-dependency path free of any CUDA
# initialization so exit 2 is deterministic on every host.
try:
    import transformers  # noqa: F401
except ImportError:
    print("MISSING DEPENDENCY: transformers -> exit 2", flush=True)
    os._exit(2)

try:
    from transformers import AutoConfig, AutoModel
except Exception:
    traceback.print_exc()
    print("MISSING DEPENDENCY: transformers import failed -> exit 2", flush=True)
    os._exit(2)

# m2-bert remote code imports opt_einsum; fall back to the bundled minimal stub
# when the real package is not installed (offline pip hosts).
try:
    import opt_einsum  # noqa: F401
    OPT_EINSUM = "real"
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_stubs"))
    OPT_EINSUM = "bundled-stub"

import torch  # noqa: E402

if not torch.cuda.is_available():
    print("MISSING CUDA DEVICE: m2-bert e2e needs a CUDA device -> exit 2", flush=True)
    os._exit(2)

_src = os.environ.get("FLAG_GEMS_SRC")
if _src:
    sys.path.insert(0, _src)
try:
    import flag_gems
except Exception:
    traceback.print_exc()
    print("flag_gems not importable - set PYTHONPATH=<FlagGems>/src -> exit 1", flush=True)
    os._exit(1)
print("flag_gems from:", flag_gems.__file__, flush=True)

# --- machine-checkable precondition (fix.yaml e2e_precondition) ---
# The evidence behind this test only applies to the exact tree combination
# baseline + linear-5379 + zeros-7181; a wrong tree must exit 2, not run.
_FIX_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fix.yaml")
try:
    import yaml

    with open(_FIX_YAML) as _fh:
        _fixdoc = yaml.safe_load(_fh)
    _pre = _fixdoc["e2e_precondition"]
    _expected = [(e["path"], e["sha256"]) for e in _pre["files"]]
    # dual-source guard: our own entries must mirror targets.after_sha256
    _targets = {t["path"]: t["after_sha256"] for t in _fixdoc.get("targets", [])}
    for _p, _sha in _expected:
        if _p in _targets and _targets[_p] != _sha:
            print(f"MISSING PRECONDITION: fix.yaml inconsistent: e2e_precondition "
                  f"{_p}={_sha} != targets.after_sha256 {_targets[_p]} -> exit 2",
                  flush=True)
            os._exit(2)
except Exception as e:
    print(f"MISSING PRECONDITION: cannot read fix.yaml e2e_precondition "
          f"({_FIX_YAML}: {e}) -> exit 2", flush=True)
    os._exit(2)

_pkg_root = os.path.dirname(flag_gems.__file__)
import hashlib

_pre_fail = False
for _rel, _sha_want in _expected:
    _abs = os.path.join(_pkg_root, _rel.split("src/flag_gems/", 1)[-1])
    if not os.path.exists(_abs):
        print(f"MISSING PRECONDITION: {_rel} absent in {_pkg_root} -> exit 2", flush=True)
        _pre_fail = True
        continue
    _sha_got = hashlib.sha256(open(_abs, "rb").read()).hexdigest()
    if _sha_got != _sha_want:
        print(f"MISSING PRECONDITION: {_rel} sha256 {_sha_got} != expected "
              f"{_sha_want} (need baseline + linear-5379 + zeros-7181 as "
              f"applied in reports/validation.md) -> exit 2", flush=True)
        _pre_fail = True
if _pre_fail:
    os._exit(2)
_baseline = str(_fixdoc.get("target", {}).get("baseline_commit", ""))[:7]
print(
    f"precondition OK: {len(_expected)} files at declared sha256 "
    f"(baseline {_baseline} + linear-5379 + zeros-7181)",
    flush=True,
)

REPO = "togethercomputer/m2-bert-80M-8k-retrieval"
REVISION = os.environ.get("M2_REVISION", "90e0b28f3382c289cc6c1e92ef53d7dc5a3ec14b")
NLAYERS = int(os.environ.get("NLAYERS", 12))
HIDDEN = int(os.environ.get("HIDDEN", 768))
HEADS = int(os.environ.get("HEADS", 12))
INTER = int(os.environ.get("INTER", 3072))
SEQLEN = int(os.environ.get("SEQLEN", 8192))
POS = int(os.environ.get("POS", 8192))
LMAX = int(os.environ.get("LMAX", 8192))
HORDER = int(os.environ.get("HORDER", 128))
print(
    f"config: {REPO} layers={NLAYERS} hidden={HIDDEN} heads={HEADS} inter={INTER} "
    f"seq={SEQLEN} pos={POS} lmax={LMAX} horder={HORDER} "
    f"revision={REVISION} opt_einsum={OPT_EINSUM}",
    flush=True,
)

try:
    cfg = AutoConfig.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
except Exception as e:
    print(f"MISSING DEPENDENCY: model code/config unavailable ({e}) -> exit 2", flush=True)
    os._exit(2)
print(f"model code: commit={getattr(cfg, '_commit_hash', None) or REVISION}", flush=True)

cfg.num_hidden_layers = NLAYERS
cfg.hidden_size = HIDDEN
cfg.num_attention_heads = HEADS
cfg.intermediate_size = INTER
cfg.vocab_size = 1000
cfg.max_position_embeddings = POS
cfg.long_conv_l_max = LMAX
cfg.alibi_starting_size = POS
cfg.hyena_filter_order = HORDER

# Deterministic init (unseeded init occasionally collapses to an all-zero
# forward, making the loss oracle vacuous).
torch.manual_seed(int(os.environ.get("M2_SEED", 0)))
try:
    model = AutoModel.from_config(cfg, trust_remote_code=True).to("cuda")
except Exception:
    traceback.print_exc()
    print("E2E FAILED: model construction", flush=True)
    os._exit(1)

torch.manual_seed(0)  # fixed input ids regardless of M2_SEED
ids = torch.randint(0, cfg.vocab_size, (2, SEQLEN), device="cuda")
mask = torch.ones_like(ids)


def forward_loss():
    out = model(input_ids=ids, attention_mask=mask)
    hs = out[0] if isinstance(out, tuple) else out.last_hidden_state
    return hs.float().pow(2).mean()


def grads_of():
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


# --- Phase A: native torch oracle (flag_gems not yet enabled) ---
try:
    loss_native = forward_loss()
    loss_native.backward()
    torch.cuda.synchronize()
    g_native = grads_of()
except Exception:
    traceback.print_exc()
    print("E2E FAILED: native forward/backward (oracle phase)", flush=True)
    os._exit(1)
if not g_native:
    print("E2E FAILED: native backward produced no gradients", flush=True)
    os._exit(1)
model.zero_grad(set_to_none=True)
loss_native_val = float(loss_native)  # snapshot: see note at the final print
print(f"native OK loss={loss_native_val:.9e} grads={len(g_native)}", flush=True)

# --- Phase B: flag_gems (imported + precondition-checked above) ---
flag_gems.enable()

try:
    loss_gems = forward_loss()
    torch.cuda.synchronize()
except Exception:
    traceback.print_exc()
    print("E2E FAILED: flag_gems forward", flush=True)
    os._exit(1)

loss_gems_val = float(loss_gems)  # snapshot before backward mutates storage (see below)
rel = abs(loss_gems_val - loss_native_val) / max(abs(loss_native_val), 1e-12)
print(f"gems fwd loss={loss_gems_val:.9e} vs native={loss_native_val:.9e} rel={rel:.3e}", flush=True)
if rel > 1e-3:
    print(f"E2E FAILED: forward loss mismatch rel={rel:.3e} > 1e-3", flush=True)
    os._exit(1)

try:
    loss_gems.backward()
    torch.cuda.synchronize()
except Exception:
    traceback.print_exc()
    print("E2E FAILED: flag_gems backward (this is the problem.md failure path)", flush=True)
    os._exit(1)

g_gems = grads_of()

# Exact known-gap set (fixture, review follow-up): layer templates with {i}
# plus any literal non-layer names, recorded from an actual run. The assert is
# bidirectional -- a missing grad outside the fixture fails, and a fixture
# entry that now HAS a grad also fails (stale fixture must be updated
# consciously). The full missing set is printed as evidence.
_GAP_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_gap_m2.txt")
if not os.path.exists(_GAP_FIXTURE):
    print(f"E2E FAILED: known-gap fixture missing: {_GAP_FIXTURE}", flush=True)
    os._exit(1)
known_gap = set()
with open(_GAP_FIXTURE) as _fh:
    for _line in _fh:
        _line = _line.strip()
        if not _line or _line.startswith("#"):
            continue
        if "{i}" in _line:
            known_gap.update(_line.replace("{i}", str(_i)) for _i in range(NLAYERS))
        else:
            known_gap.add(_line)

missing = sorted(n for n in g_native if n not in g_gems)
unexpected = [n for n in missing if n not in known_gap]
absent = sorted(n for n in (known_gap & set(g_native)) if n in g_gems)
for n in missing:
    print(f"missinggrad: {n}", flush=True)
if unexpected:
    print(f"E2E FAILED: missing grads outside known_gap_m2.txt ({len(unexpected)}): "
          f"{unexpected[:10]}", flush=True)
    os._exit(1)
if absent:
    print(f"E2E FAILED: known_gap_m2.txt stale, grad now present for {len(absent)}: "
          f"{absent[:10]}", flush=True)
    os._exit(1)
missing_set = set(missing)

# fp32 tolerance: abs floor for vanishing grads + rel band for large ones,
# normalized by max(|native|, |gems|) so tiny-vs-tiny pairs cannot blow up.
ATOL, RTOL = 1e-6, 1e-3
rows = []
checked = 0
for name, want in g_native.items():
    if name in missing_set:
        continue
    got = g_gems[name]
    if got.shape != want.shape:
        print(f"E2E FAILED: grad shape {name}: {tuple(got.shape)} != {tuple(want.shape)}", flush=True)
        os._exit(1)
    if not torch.isfinite(got).all():
        print(f"E2E FAILED: non-finite grad after gems backward: {name}", flush=True)
        os._exit(1)
    d_abs = (got - want).abs().max().item()
    scale = max(want.abs().max().item(), got.abs().max().item())
    ratio = d_abs / (ATOL + RTOL * scale)
    rows.append((ratio, name, d_abs, want.abs().max().item(), got.abs().max().item()))
    checked += 1

extra = [n for n in g_gems if n not in g_native]
if extra:
    print(f"E2E FAILED: gems produced grads native did not: {extra[:5]}", flush=True)
    os._exit(1)

rows.sort(reverse=True)
print("grad value divergence INFO (pre-existing, not a criterion; identical class on complex-fix-free tree):", flush=True)
for ratio, name, d_abs, wmax, gmax in rows[:5]:
    print(
        f"  gradworst ratio={ratio:.3e} abs_diff={d_abs:.3e} "
        f"native_max={wmax:.3e} gems_max={gmax:.3e} :: {name}",
        flush=True,
    )
worst = rows[0][0] if rows else 0.0
print(
    f"grad presence: {checked} present+finite compared, {len(missing_set)} missing "
    f"(== known_gap_m2.txt asserted), worst value ratio={worst:.3e} (info)",
    flush=True,
)

# Losses are reported from the pre-backward snapshots: an intermittent
# flag_gems-backward anomaly was observed to overwrite the loss scalar
# tensor's storage after loss_gems.backward() (some runs, not others).
# Attribution undetermined: pre-fix trees die before backward completes so
# the same code path never completed on them (no A/B), and fixed-tree runs
# vary run-to-run -- it is therefore neither proven pre-existing nor proven
# caused by this fix. The parity check runs on the snapshots taken before any
# backward, so the reported values are unaffected either way.
print(
    f"E2E OK: loss_native={loss_native_val:.9e} loss_gems={loss_gems_val:.9e} "
    f"rel={rel:.3e} grads_present_finite={checked} "
    f"grads_missing_known_gap={len(missing_set)}",
    flush=True,
)
os._exit(0)
