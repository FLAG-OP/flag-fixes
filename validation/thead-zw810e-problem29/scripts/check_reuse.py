"""Manifest-vs-file reuse check (review 9fe0a05 item 2).
A manifest record authorizes reuse of a result JSON only if ALL of:
  exit==0, script_sha matches current profiler, env matches expected config,
  wall_ms matches JSON wall (±0.5), result_sha256 matches file content hash.
Usage: check_reuse.py <manifest_line_json> <json_path> <script_sha> <expected_env_json>
Exit 0=reuse allowed, 1=reject (reason on stderr)."""
import hashlib, json, sys

def sha256f(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 16), b""):
            h.update(b)
    return h.hexdigest()

def main():
    rec = json.loads(sys.argv[1])
    path, script_sha = sys.argv[2], sys.argv[3]
    expected_env = json.loads(sys.argv[4])
    reasons = []
    if rec.get("exit") != 0:
        reasons.append(f"exit={rec.get('exit')}")
    if rec.get("script_sha") != script_sha:
        reasons.append("script_sha mismatch")
    if rec.get("env") != expected_env:
        reasons.append(f"env mismatch: manifest={rec.get('env')} expected={expected_env}")
    try:
        cur = json.load(open(path))
        w = cur.get("wall_ms")
        if not isinstance(w, (int, float)) or abs(float(rec.get("wall_ms", -1)) - float(w)) >= 0.5:
            reasons.append(f"wall mismatch: manifest={rec.get('wall_ms')} file={w}")
    except Exception as e:
        reasons.append(f"unreadable JSON: {e}")
    file_sha = sha256f(path)
    if rec.get("result_sha256") != file_sha:
        reasons.append("result_sha256 mismatch (file content changed since run)")
    if reasons:
        for r in reasons:
            print("REJECT: " + r, file=sys.stderr)
        sys.exit(1)
    print("REUSE-OK")
    sys.exit(0)

if __name__ == "__main__":
    main()
