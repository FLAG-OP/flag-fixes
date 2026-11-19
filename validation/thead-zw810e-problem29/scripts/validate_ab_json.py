"""Strict A/B JSON validator (review bd1eea8 P1 fix).

Required evidence (no defaults-through):
- correctness.losses: list, len>=20, all finite
- correctness.grad_norms: list, len>=3, all finite
- correctness.losses_finite: exactly boolean True
- correctness.grad_norms_finite: exactly boolean True
- correctness.nonfinite_param_count: exactly int 0
- wall_ms present and finite; segments or segments_ms_median present

Exit 0 = valid; exit 1 = invalid with reason lines on stderr.
"""
import json, math, sys

def fail(msgs):
    for m in msgs:
        print("INVALID: " + m, file=sys.stderr)
    sys.exit(1)

def main(path):
    msgs = []
    try:
        d = json.load(open(path))
    except Exception as e:
        fail([f"unparseable JSON: {e}"])
    c = d.get("correctness")
    if not isinstance(c, dict):
        fail(["missing correctness object"])

    losses = c.get("losses")
    if not isinstance(losses, list) or len(losses) < 20:
        msgs.append(f"losses must be list len>=20, got {type(losses).__name__} len={len(losses) if isinstance(losses,list) else 'n/a'}")
    elif not all(isinstance(v, (int, float)) and math.isfinite(v) for v in losses):
        msgs.append("losses contain non-finite or non-numeric values")

    grads = c.get("grad_norms")
    if not isinstance(grads, list) or len(grads) < 3:
        msgs.append(f"grad_norms must be list len>=3, got {type(grads).__name__} len={len(grads) if isinstance(grads,list) else 'n/a'}")
    elif not all(isinstance(v, (int, float)) and math.isfinite(v) for v in grads):
        msgs.append("grad_norms contain non-finite or non-numeric values")

    if c.get("losses_finite") is not True:  # strict bool True (null/false/missing all fail)
        msgs.append(f"losses_finite must be literal true, got {c.get('losses_finite')!r}")
    if c.get("grad_norms_finite") is not True:
        msgs.append(f"grad_norms_finite must be literal true, got {c.get('grad_norms_finite')!r}")

    npc = c.get("nonfinite_param_count")
    if isinstance(npc, bool) or not isinstance(npc, int):
        msgs.append(f"nonfinite_param_count must be int, got {type(npc).__name__} {npc!r}")
    elif npc != 0:
        msgs.append(f"nonfinite_param_count must be 0, got {npc}")

    w = d.get("wall_ms")
    if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0:
        msgs.append(f"wall_ms must be finite positive number (bool/none/negative rejected), got {w!r}")
    seg = d.get("segments_ms_median")
    segd = d.get("segments")
    seg_ok = False
    if isinstance(seg, dict) and seg and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in seg.values()):
        seg_ok = True
    if isinstance(segd, dict) and segd:
        seg_ok = True
    if not seg_ok:
        msgs.append(f"segments_ms_median must be dict of positive finite numbers (or non-empty segments), got {seg!r}")

    if msgs:
        fail(msgs)
    print(f"VALID {path}")
    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv[1])
