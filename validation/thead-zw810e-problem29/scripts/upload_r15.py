import base64, json, subprocess, time, os
REPO="FLAG-OP/flag-fixes"; BR="issue7-zw810e-opt-round1"
def gh(endpoint, method="GET", body=None, tries=10):
    args=["gh","api","repos/"+REPO+"/"+endpoint]
    if method!="GET": args+=["-X",method]
    payload=None; bf=None
    if body is not None:
        bf=tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False)
        json.dump(body,bf); bf.close()
        args+=["--input",bf.name]
    for t in range(tries):
        p=subprocess.run(args,capture_output=True,text=True,timeout=180)
        if p.returncode==0 and p.stdout.strip():
            try:
                if bf and os.path.exists(bf.name): os.unlink(bf.name)
                return json.loads(p.stdout)
            except Exception: pass
        time.sleep(25)
    if bf and os.path.exists(bf.name): os.unlink(bf.name)
    raise RuntimeError(f"{endpoint}: {p.stderr[:120] or p.stdout[:120]}")
def blob(path):
    body={"content":base64.b64encode(open(path,'rb').read()).decode(),"encoding":"base64"}
    return gh("git/blobs","POST",body)["sha"]

br=gh("git/refs/heads/issue7-zw810e-opt-round1")
parent=br["object"]["sha"]
base_tree=gh("git/commits/"+parent)["tree"]["sha"]
FSCALE="/tmp/opencode/FlagScale-ppu"
W="/tmp/opencode/zw810e-problem29"
pairs=[]
pairs.append((FSCALE+"/examples/retrieval/profile_step_segments_ab.py","validation/thead-zw810e-problem29/scripts/profile_step_segments_ab.py"))
pairs.append((FSCALE+"/examples/retrieval/probe11_mask_vs_equal_length.py","validation/thead-zw810e-problem29/scripts/probe11_mask_vs_equal_length.py"))
for root,dirs,files in os.walk(W):
    for f in sorted(files):
        fp=os.path.join(root,f)
        if not os.path.isfile(fp): continue
        rel=os.path.relpath(fp,W)
        pairs.append((fp,f"validation/thead-zw810e-problem29/{rel}"))
pairs.append((FSCALE+"/examples/retrieval/probe10_gemm_ablation.py","validation/thead-zw810e-problem29/scripts/probe10_gemm_ablation.py"))
# dedup
seen=set(); dedup=[]
for src,dst in pairs:
    if dst not in seen:
        seen.add(dst); dedup.append((src,dst))
entries=[]
for i,(src,dst) in enumerate(dedup):
    for t in range(5):
        try:
            sha=blob(src); entries.append({"path":dst,"mode":"100644","type":"blob","sha":sha})
            if (i+1)%20==0: print(f"  {i+1}/{len(dedup)}",flush=True)
            break
        except Exception as e:
            print(f"  retry {dst}: {str(e)[:50]}",flush=True); time.sleep(20)
    else:
        raise RuntimeError(f"blob failed: {src}")
print(f"total: {len(entries)}")
json.dump({"base_tree":base_tree,"tree":entries},open("/tmp/opencode/remote-files/tree_r15_final.json","w"))
tree=gh("git/trees","POST",bodyfile="/tmp/opencode/remote-files/tree_r15_final.json")
msg="""issue #7 round-15: AB seed+init+accumulation fix + probe11 backend context timing + probe10 pre-conversion + evidence sync

1. AB script: seed 42 restored; losses_run/grad_norms_run initialized OUTSIDE optimizer step loop; JSON includes losses/grad_norms/non-finite/peak-mem
2. probe11_mask_vs_equal_length.py: mask parameter in ALL timed calls; backend context covers warmup+timing
3. probe10_gemm_ablation.py: fp32 pre-conversion outside timed region; raw samples added
4. AdamW stable-reference honest FAIL all impls; fp16 stock/foreach corrupted; fp16 fused bounded but FAILs declared tau
5. README: peak mem 24.3 GiB; no stale claims; GEMM scope limited to measured shape"""
c=gh("git/commits","POST",{"message":msg,"tree":tree["sha"],"parents":[parent]})
gh("git/refs/heads/"+BR,"PATCH",{"sha":c["sha"],"force":False})
print("COMMIT_OK:",c["sha"])
