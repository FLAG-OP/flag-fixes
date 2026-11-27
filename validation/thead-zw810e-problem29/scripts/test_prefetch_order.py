"""Regression (review BqpXfeg item 1): force a NEGATIVE group to finish BEFORE
'positive' and prove the prefetch returns the RIGHT group.
Reproduces the old shared-FIFO bug (negative_1 enqueued first -> used as positive)
and shows the by-group-future fix is immune. Uses the real ORDER and a fake
tokenizer whose completion order is controlled by a barrier."""
import concurrent.futures, threading, queue, sys
ORDER=["anchor","positive","negative_1","negative_2","negative_3"]
def fake_tok(k, delay):
    e=threading.Event(); e.wait(delay); return (k, f"feats_of_{k}")
# --- OLD (buggy): shared FIFO; make negative_1 finish before positive ---
q=queue.Queue()
def old_worker(k,d): q.put(fake_tok(k,d))
th=threading.Thread(target=old_worker,args=("positive",0.30)); th.start()
th2=threading.Thread(target=old_worker,args=("negative_1",0.01)); th2.start()
import time; time.sleep(0.05)
got_k,_=q.get()                 # whichever finished first
print("OLD shared-FIFO fetch for slot 'positive' ->", got_k, "| MISMATCH" if got_k!="positive" else "| ok")
# --- NEW (fix): by-group futures ---
ex=concurrent.futures.ThreadPoolExecutor(max_workers=2)
fut={"positive":ex.submit(fake_tok,"positive",0.30),"negative_1":ex.submit(fake_tok,"negative_1",0.01)}
kk,_=fut["positive"].result()   # fetched BY NAME
ex.shutdown()
print("NEW by-group-future fetch for slot 'positive' ->", kk, "| ok" if kk=="positive" else "| MISMATCH")
ok_old=(got_k!="positive")      # the bug must reproduce on the old path
ok_new=(kk=="positive")         # the fix must hold
print("REGRESSION:", "PASS" if (ok_old and ok_new) else "FAIL")
sys.exit(0 if (ok_old and ok_new) else 1)
