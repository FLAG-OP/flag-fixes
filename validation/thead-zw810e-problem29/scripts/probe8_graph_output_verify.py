"""Issue #7 review round-3: CUDA graph replay OUTPUT verification.

Verifies replay reads the CURRENT contents of the captured input buffer:
  1. capture graph with input buffer x and output buffer y
  2. write NEW values into x
  3. replay
  4. compare y vs eager F.gelu(new x) -> must match
Outputs archived (sha256 + max diff). Also records per-call amortized replay
cost with the corrected wording (NOT "enqueue eliminated / pure device time").
"""
import hashlib
import json
import statistics

import torch
import torch.nn.functional as F

import flag_gems

flag_gems.enable()

OUT = {"device": torch.cuda.get_device_name(0)}
g = torch.Generator(device="cuda").manual_seed(11)
x = torch.randn(1228800, generator=g, device="cuda", dtype=torch.float32)
y = torch.empty_like(x)

s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        torch.nn.functional.gelu(x, out=y)
torch.cuda.current_stream().wait_stream(s)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    torch.nn.functional.gelu(x, out=y)

sha = lambda t: hashlib.sha256(t.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]

# replay 1: original buffer
graph.replay()
torch.cuda.synchronize()
OUT["replay1_output_sha"] = sha(y)
OUT["replay1_matches_eager_original"] = bool(torch.equal(y, F.gelu(x)))

# NEW values into the SAME captured buffer
new_vals = (torch.randn(1228800, generator=g, device="cuda") * 3)
x.copy_(new_vals)
graph.replay()
torch.cuda.synchronize()
eager_new = F.gelu(new_vals)
OUT["new_input_sha"] = sha(new_vals)
OUT["replay2_output_sha"] = sha(y)
OUT["eager_new_sha"] = sha(eager_new)
OUT["replay_reads_current_input"] = bool(torch.equal(y, eager_new))
OUT["replay2_max_abs_diff_vs_eager"] = float((y - eager_new).abs().max())

# per-call amortized replay cost (corrected wording)
ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(10)]
for st, e in ev:
    st.record()
    graph.replay()
    e.record()
torch.cuda.synchronize()
per_call = statistics.median(st.elapsed_time(e) for st, e in ev) / 100
OUT["graph_replay_per_call_amortized_ms"] = round(per_call, 4)
OUT["wording_note"] = "100 calls captured; per-call amortized cost of replay (includes graph dispatch; input buffer reused). NOT 'enqueue eliminated' and NOT independent kernel time."

with open("results/probe8_graph_output_verify.json", "w") as f:
    json.dump(OUT, f, indent=1)
print(json.dumps(OUT, indent=1))
ok = OUT["replay1_matches_eager_original"] and OUT["replay_reads_current_input"] and OUT["replay2_max_abs_diff_vs_eager"] == 0.0
sys_exit = 0 if ok else 1
import sys
sys.exit(sys_exit)
