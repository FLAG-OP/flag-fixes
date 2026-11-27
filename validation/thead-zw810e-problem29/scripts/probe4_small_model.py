"""Probe 4: small-model step-time attribution + 3-process replication."""
import json
import statistics
import subprocess
import sys

PY = sys.executable
ENV = {"PYTHONPATH": "/tmp/opencode/FlagGems-base/src", "CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin:/bin:/usr/local/bin", "PPU_SDK": "/usr/local/PPU_SDK", "HOME": "/root"}

CODE_TIMED = r'''
import json, sys, torch, torch.nn.functional as F
sys.path.insert(0, "/tmp/opencode/zw810e-problem29/scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("sm", "/tmp/opencode/zw810e-problem29/scripts/small_model.py")
mode = sys.argv[1]
if mode != "native":
    import flag_gems
    if mode == "gems_selective":
        flag_gems.only_enable(include=["gelu", "gelu_backward", "gelu_and_mul"])
    else:
        flag_gems.enable()

torch.manual_seed(0)
import hashlib
class Block(torch.nn.Module):
    def __init__(self, d, h, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w1 = torch.nn.Parameter(torch.randn(d, h, generator=g) * 0.02)
        self.w3 = torch.nn.Parameter(torch.randn(d, h, generator=g) * 0.02)
        self.w2 = torch.nn.Parameter(torch.randn(h, d, generator=g) * 0.02)
        self.ln_g = torch.nn.Parameter(torch.ones(d)); self.ln_b = torch.nn.Parameter(torch.zeros(d))
        self.qkv = torch.nn.Parameter(torch.randn(d, 3 * d, generator=g) * 0.02)
        self.wo = torch.nn.Parameter(torch.randn(d, d, generator=g) * 0.02)
    def forward(self, x):
        u = x * self.ln_g + self.ln_b
        q, k, v = (u @ self.qkv).chunk(3, dim=-1)
        att = F.scaled_dot_product_attention(q.transpose(0,1).unsqueeze(0), k.transpose(0,1).unsqueeze(0), v.transpose(0,1).unsqueeze(0)).squeeze(0).transpose(0,1)
        x = x + att @ self.wo
        h = u @ self.w1
        x = x + (F.gelu(h, approximate="none") * (u @ self.w3)) @ self.w2
        return x

B,S,D,H,L = 8,128,256,512,2
blocks = torch.nn.ModuleList([Block(D,H,100+i) for i in range(L)])
head = torch.nn.Parameter(torch.randn(D,D)*0.02)
params = list(blocks.parameters())+[head]
opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
g = torch.Generator().manual_seed(42)
tok = torch.randint(0,64,(B,S),generator=g)
emb = torch.nn.Embedding(64,D)
with torch.no_grad(): emb.weight.copy_(torch.randn(64,D,generator=g)*0.02)

losses=[]; times=[]
def step():
    x = emb(tok)
    for blk in blocks: x = blk(x)
    loss = (x @ head).float().pow(2).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return loss

for st in range(8):
    e0=torch.cuda.Event(True); e1=torch.cuda.Event(True); e0.record()
    l = step(); torch.cuda.synchronize(); e1.record(); torch.cuda.synchronize()
    losses.append(float(l.detach())); times.append(e0.elapsed_time(e1))

# bitwise param check
psha = [hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16] for p in params]
print(json.dumps({"mode": mode, "losses": losses, "times": times, "psha": psha}))
'''

res = {}
all_data = {}
for mode in ("native", "gems_full", "gems_selective"):
    runs = []
    for r in range(3):
        p = subprocess.run([PY, "-c", CODE_TIMED, mode], env=ENV, capture_output=True, text=True, timeout=600)
        runs.append(json.loads(p.stdout.strip().splitlines()[-1]))
    all_data[mode] = runs
    steady = [sorted(t[2:])[len(t[2:]) // 2] for t in [r["times"] for r in runs]]
    losses0 = set(round(r["losses"][0], 12) for r in runs)
    res[mode] = {"steady_step_ms_med": steady, "loss_step1": sorted(losses0), "psha_r1": runs[0]["psha"]}

nat = all_data["native"][0]
for mode in ("gems_full", "gems_selective"):
    res[mode]["losses_bitwise_equal_native"] = all(r["losses"] == nat["losses"] for r in all_data[mode])
    res[mode]["params_bitwise_equal_native_r1"] = all(r["psha"] == nat["psha"] for r in all_data[mode])

json.dump(res, open("results/small_model_3rep.json", "w"), indent=1)
print(json.dumps(res, indent=1))
