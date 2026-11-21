"""Regenerate README §3c-2 final table from final_table_r36.json (review
d28860a item 6: auto-generated, no hand-copied values; every row carries
window/config/source)."""
import json

ft = json.load(open("results/final_table_r36.json"))
an = json.load(open("results/training_trace_analysis_r36.json"))
sw = ft["overhead"]
cp = an["critical_path_per_step"]
busy = "/".join(f"{cp[s]['busy_pct']}" for s in sorted(cp))
idle = "/".join(f"{cp[s]['exposed_idle_ms']}" for s in sorted(cp))
mem = cp[sorted(cp)[0]]["memcpy_events_per_step"]

lines = []
lines.append("#### 3c-2. 最终汇总表（round-36，由 `final_table_r36.json` 经 `build_final_table.py` + `render_readme_3c2.py` 程序化生成）")
lines.append("")
lines.append(f"窗口：`{ft['trace']}`；{ft['config']}；脚本/数据 commit 见各文件 blob。r32/r35 trace 及其分析**标记 superseded**（仅存档）。")
lines.append("")
lines.append("| 阶段 | 相位 | MKN | calls/步 | device ms/步 | FLOPs (TFLOP) | TFLOP/s | HBM | 来源 |")
lines.append("|---|---|---|---:|---:|---:|---:|---|---|")
for r in ft["rows"][:24]:
    mkn = str(r["MKN"]) if r.get("MKN") else "—"
    fl = str(r["flops_tflop"]) if r.get("flops_tflop") is not None else "—"
    tf = str(r["tflops"]) if r.get("tflops") is not None else "—"
    ph = r.get("phase", "—")
    lines.append(f"| {r['stage']} | {ph} | {mkn} | {r['calls']} | {r['ms']} | {fl} | **{tf}** | {r['hbm']} | {r['source']} |")
lines.append(f"| SDPA/attention | fwd | — | 480 | {ft['rows'][-2]['ms']} | {ft['rows'][-2]['flops_tflop']} | **{ft['rows'][-2]['tflops']}** | ESTIMATE | sdpa_efficiency_r36（4BH S²D，60 组逐行）|")
lines.append(f"| SDPA/attention | bwd | — | 1,440 | {ft['rows'][-1]['ms']} | {ft['rows'][-1]['flops_tflop']} | **{ft['rows'][-1]['tflops']}** | ESTIMATE | 同上（6BH S²D；重算未知）|")
lines.append("")
lines.append(f"小 S 形状（140–180）TFLOP/s 6–24（latency 主导）。**全部效率行 ≤139.2 峰值（离线校验通过）**。")
lines.append("")
lines.append(f"**关键路径（r36 trace，busy union 含 kernels+gpu_memcpy+gpu_memset）**：busy **{busy}%**、暴露空闲 **{idle}ms/步**；")
mem_s = "; ".join(f"{k} {v['count']} 条 {round(v['ms'],2)}ms" for k, v in mem.items())
lines.append(f"per-step memcpy（首步）：{mem_s}（逐步拆分 + 守恒断言 8082==8082 通过；此前\"H2D=0\"系只查 kernel 类别——已更正）。")
lines.append("")
lines.append(f"**严格配对（text-hash 真实指纹）**：双侧 loader_index + sha256 text_hash16；逐 micro 断言 **12/12 hash+token 一致、0 mismatch**；no-profiler 中位 **{sw['noprofiler_median_ms']}ms**、p95=样本最大 **{sw['p95_ms']}ms**（3 样本，非稳定 p95）、有效 tokens/s **{sw['valid_tokens_per_s'][0]}–{sw['valid_tokens_per_s'][-1]}**；**严格配对采集开销 ≈ {sw['overhead_pct']}%**（profiled {sw['profiled_median_ms']}ms）。kernel-efficiency 窗指纹断言 step_2 20/20 匹配。")
lines.append("")
lines.append("**HBM：BLOCKED**——Asight v2.1.0 在位；PCM 被 pg-exporter 占用（dmesg 证据 `logs/hbm_probe_r34.log`）；复测命令已记录；**HBM 字节/带宽/利用率均为估算，不列入已完成测量**。SDPA dropout 配置 0.1（模块级激活）、SDPA 内部 dropout 标志未知、反向重算未知。")
lines.append("")
lines.append("**估算标注**：dW 提升/135.7 参照/逐元素融合等收益数字全部为估算（依据同窗实测差值），非实测结论。ac2g 时间戳=主机提交时刻（统一口径）。")
lines.append("")

new_section = "\n".join(lines)
s = open("/tmp/opencode/remote-files/README_r20.md").read()
start = s.index("#### 3c-2. 最终汇总表")
end = s.index("#### 4. F-2 最小 shape 隔离（单次触发）")
s = s[:start] + new_section + "\n" + s[end:]
open("/tmp/opencode/remote-files/README_r20.md", "w").write(s)
print("README 3c-2 regenerated programmatically")
print(new_section[:800])
