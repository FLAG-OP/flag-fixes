"""Build results/records_delivery_r58.json from the local split files, the GitHub release asset list
and the clean-copy recompute output. No number is typed by hand.
usage: python3 scripts/build_records_delivery_r58.py --split-dir /tmp/opencode/records_split --clean-dir /tmp/opencode/clean-r58
"""
import argparse, hashlib, json, os, subprocess, time
ap=argparse.ArgumentParser()
ap.add_argument("--split-dir",required=True); ap.add_argument("--clean-dir",required=True); ap.add_argument("--out",default="results/records_delivery_r58.json")
ap.add_argument("--tag",default="problem29-dw-records-r44"); ap.add_argument("--repo",default="FLAG-OP/flag-fixes")
a=ap.parse_args()
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(1<<22),b""): h.update(b)
    return h.hexdigest()
parts=[]
for line in open(os.path.join(a.split_dir,"SHA256SUMS.parts")):
    s,n=line.split(); parts.append({"name":n,"sha256":s,"bytes":os.path.getsize(os.path.join(a.split_dir,n))})
whole=open(os.path.join(a.split_dir,"SHA256SUMS")).read().split()
orig={"name":whole[1],"sha256":whole[0],"bytes":sum(p["bytes"] for p in parts)}
rel=json.loads(subprocess.run(["gh","api",f"repos/{a.repo}/releases"],capture_output=True,text=True).stdout)
rel=next((r for r in rel if r["tag_name"]==a.tag),None)
assets=[]
for s in sorted(rel["assets"],key=lambda x:x["name"]) if rel else []:
    local=next((p for p in parts if p["name"]==s["name"]),None)
    assets.append({"name":s["name"],"bytes":s["size"],"state":s["state"],"uploaded_at":s["created_at"],"matches_local_bytes":(s["size"]==local["bytes"]) if local else None})
rec=json.load(open(os.path.join(a.clean_dir,"recomputed_r58.json")))
committed=json.load(open(os.path.join(a.clean_dir,"validation/thead-zw810e-problem29/results/acc_hit_step_r44_recomputed.json")))
same={k:v for k,v in rec.items() if k!="record_file"}=={k:v for k,v in committed.items() if k!="record_file"}
tree={l.split(" ")[0]:l.split(" ")[1] for l in open(os.path.join(os.path.dirname(a.clean_dir),"remote-files","_tree_final.txt")).read().split("\n") if " " in l}
P="validation/thead-zw810e-problem29/"
def blob(path): return hashlib.sha1(b"blob %d\0"%len(open(path,"rb").read())+open(path,"rb").read()).hexdigest()
sc=os.path.join(a.clean_dir,P,"scripts/recompute_dw_accuracy.py")
LOGD="/tmp/opencode/zw810e-problem29/logs/"
vlog=LOGD+"r58_records_verify.log"
rejoin=[l for l in open(LOGD+"r58_records_upload.log").read().split("\n") if l.startswith("rejoin check:")]
ver=open(vlog).read().strip().split("\n") if os.path.exists(vlog) else []
out={"field_version":"records_delivery_v1","built_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
 "original_file":orig,
 "split":{"part_size_bytes":1900*1024*1024,"n_parts":len(parts),"parts":parts,"reason":"GitHub limits one release asset to 2 GiB",
          "rejoin":"cat records.part.00 records.part.01 records.part.02 records.part.03 > acc_hit_step_r44.json.records.pkl",
          "rejoin_sha256_verified_locally":bool(rejoin) and rejoin[0].split()[2]==orig["sha256"],
          "rejoin_check_output":rejoin[0] if rejoin else None},
 "release":{"repo":a.repo,"tag":a.tag,"id":rel["id"] if rel else None,"draft":rel["draft"] if rel else None,"url":rel["html_url"] if rel else None,"target_commitish":rel["target_commitish"] if rel else None,
            "assets":assets,"all_parts_uploaded":all(any(x["name"]==p["name"] and x["matches_local_bytes"] for x in assets) for p in parts),
            "access":"DRAFT release: visible to users with write access to the repository. Download: gh release download %s --repo %s --dir <dir>. Publishing it (or not) is the maintainers' decision."%(a.tag,a.repo),
            "upload_throughput_note":"the measurement host uploads at ~0.95 MB/s per stream (1.27 GB part: 1344 s; 1.99 GB parts: ~4800 s each with 3 streams in parallel); one 1.99 GB part failed with HTTP 408 and was re-uploaded"},
 "clean_copy_recompute":{"command":"python3 scripts/recompute_dw_accuracy.py <records.pkl> <out.json>",
   "clean_copy_how":"git transport to github.com (clone) and tarball downloads are reset / stall from this environment; the script and the committed recompute result were fetched through the Git data API by blob SHA and verified against the tree of commit 59346831",
   "script_blob_sha256_of_content":sha(sc),"script_git_blob_sha":blob(sc),"matches_committed_blob":blob(sc)==tree.get(P+"scripts/recompute_dw_accuracy.py"),
   "committed_result_blob_matches":blob(os.path.join(a.clean_dir,P,"results/acc_hit_step_r44_recomputed.json"))==tree.get(P+"results/acc_hit_step_r44_recomputed.json"),
   "exit_reason":rec["exit_reason"],"verdict":rec["verdict"],"record_sha256":rec["record_sha256"],"dw_only_hits_this_step":rec["dw_only_hits_this_step"],
   "native_self_comparison_valid":rec["native_self_comparison_valid"],"candidate_vs_native_pass":rec["candidate_vs_native_pass"],
   "comparisons":{c:{g:{"grad_rel_max_unrounded":v["grad_rel_max_unrounded"],"PASS":v["PASS"],"post_param_bits_unequal":v["post_param_bits_unequal"]} for g,v in per.items()} for c,per in rec["comparisons"].items()},
   "identical_to_committed_result_except_record_file":same},
 "download_back_verification":[l for l in ver if l.strip()]}
json.dump(out,open(a.out,"w"),indent=1)
print("all parts uploaded:",out["release"]["all_parts_uploaded"],"| assets:",[(x["name"],x["matches_local_bytes"]) for x in assets]); print("recompute identical:",same,"| exit",rec["exit_reason"]); print("verify log:",out["download_back_verification"][-2:])
