"""Build PR branch via GitHub Git Data API (git clone is unstable on this network)."""
import base64
import json
import subprocess
import sys
from pathlib import Path

REPO = "FLAG-OP/flag-fixes"
BASE_SHA = "3c5a211001d2d6c5c543245bf0f9c155baa2d4d9"
BASE_TREE = "a66dc42a060dad2b6f8d4e0e4202698fa71d8ddd"
BRANCH = "issue5-zw810e-evidence"
WORK = Path("/tmp/opencode/zw810e-problem29")
REMOTE_DOCS = Path("/tmp/opencode/remote-files")


def gh(endpoint, method="GET", body=None):
    cmd = ["gh", "api", f"repos/{REPO}/{endpoint}"]
    if method != "GET":
        cmd += ["-X", method, "--input", "-"]
    data = json.dumps(body) if body is not None else None
    p = subprocess.run(cmd, capture_output=True, text=True, input=data, timeout=300)
    if p.returncode != 0:
        raise RuntimeError(f"{endpoint}: {p.stderr[:500]}")
    return json.loads(p.stdout) if p.stdout.strip() else {}


def blob(path):
    content = base64.b64encode(Path(path).read_bytes()).decode()
    r = gh("git/blobs", "POST", {"content": content, "encoding": "base64"})
    return r["sha"]


entries = []

# validation/README.md (edited remote copy)
vr = REMOTE_DOCS / "validation-readme.md"
s = vr.read_text()
s = s.replace(
    "管理工具和来源检查的现有结果见",
    "现有记录：\n\n- [thead-zw810e-problem29](thead-zw810e-problem29/README.md)（2026-11-12 首轮 / 2026-11-13 修订，issue #5）：平头哥 PPU-ZW810E 单卡平台基线，判定\"需优化\"（FlagGems `29e14822` 全量 enable 为净负收益，归因=host 侧 per-call enqueue 固定成本；selective 有效但本轮无有益算子），gelu 族 126 用例精度矩阵 0 硬失败 + 比较器负例自测 10/10。\n\n管理工具和来源检查的现有结果见",
)
vr.write_text(s)
entries.append({"path": "validation/README.md", "mode": "100644", "type": "blob", "sha": blob(vr)})

# validation/thead-zw810e-problem29/**
base = WORK
mapping = {"README_v2.md": "validation/thead-zw810e-problem29/README.md"}
for f in sorted((base / "scripts").glob("*.py")):
    mapping[str(f.relative_to(base))] = f"validation/thead-zw810e-problem29/{f.parent.name}/{f.name}"
for f in sorted((base / "results").glob("*.json")):
    mapping[str(f.relative_to(base))] = f"validation/thead-zw810e-problem29/{f.parent.name}/{f.name}"
for f in sorted((base / "logs").glob("*.log")):
    mapping[str(f.relative_to(base))] = f"validation/thead-zw810e-problem29/{f.parent.name}/{f.name}"

for i, (src, dst) in enumerate(sorted(mapping.items())):
    entries.append({"path": dst, "mode": "100644", "type": "blob", "sha": blob(base / src)})
    print(f"blob {i+1}/{len(mapping)}: {dst}")

entries.append({"path": "docs/current-status.md", "mode": "100644", "type": "blob", "sha": blob(REMOTE_DOCS / "current-status.md")})
entries.append({"path": "bundles/kunlunxin-bge-perf/README.md", "mode": "100644", "type": "blob", "sha": blob(REMOTE_DOCS / "bundle-readme.md")})

tree = gh("git/trees", "POST", {"base_tree": BASE_TREE, "tree": entries})
print("tree:", tree["sha"])

commit = gh(
    "git/commits",
    "POST",
    {
        "message": "issue #5: T-Head ZW810E problem29 platform baseline (accuracy + perf evidence, review-revised)\n\n"
        "- validation/thead-zw810e-problem29/: scripts, raw JSON (per-iteration perf samples, 126-case accuracy matrix v5), logs, exit codes, env record\n"
        "- attribution: linear-fit intercept 60.4/70.0us (gems) vs 4.7/4.6us (native), host enqueue 57-69us vs ~5us; direct entry == aten entry (same triton kernel, ~5us)\n"
        "- accuracy gates v3->v5 unchanged; comparator negative selftest 10/10; batches v1/v2 void, archived run is the full v5 re-run\n"
        "- AdamW fused: 5-step parity + perf (336 tensors: 26.8->11.7 ms/step, launches 2688->39)\n"
        "- scoped conclusions; native/full/selective are integration configs, not optimization candidates\n"
        "- docs: current-status.md platform note + history row; kunlunxin-bge-perf bundle matrix; validation/README.md index\n\n"
        "Covers review round on issue #5 (comment 6841909645). No fix-entry status changes: all remain draft/deployable=false/hardware_status=unverified.",
        "tree": tree["sha"],
        "parents": [BASE_SHA],
    },
)
print("commit:", commit["sha"])

ref = gh("git/refs", "POST", {"ref": f"refs/heads/{BRANCH}", "sha": commit["sha"]})
print("ref:", ref["ref"])
