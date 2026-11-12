# Round-11 (review 6788210724): minimal repro for FIX-ITEM-5 — under
# FlagGems zero_ registration, zero_ on a STRIDED REAL (float) view writes
# densely: the whole underlying buffer is zeroed instead of only the
# selected elements (unselected storage corrupted). Native ATen is
# stride-correct. Reproduces on the pre-fix tree too (pre-existing gap;
# same corruption pattern that motivated the round-2 view_as_real
# abandonment inside zeros-7181's complex path).
#
# Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<tree>/src python3 benchmarks/strided_zero_probe.py
import sys
import torch

import flag_gems  # noqa: F401

flag_gems.only_enable(include=["zeros", "zero_", "zeros_like"])

torch.manual_seed(0)
buf = torch.randn(4, 8, device="cuda")
v = buf[:, ::2]          # strided float view (4,4), stride (8,2)
before = buf.clone()
v.zero_()
after = buf
sel_ok = bool((after[:, ::2] == 0).all().item())
uns = ((after != before) & (before != 0))[:, 1::2]
unsel_ok = bool((uns == 0).all().item())
print(f"selected_all_zero={sel_ok} unselected_untouched={unsel_ok}")
print("buf row0:", [round(x, 3) for x in after[0].tolist()])
ok = sel_ok and unsel_ok
sys.exit(0 if ok else 1)
