import base64, time, os, sys, requests, pandas as pd

MANIFEST_NAMES = {"requirements.txt","requirements-dev.txt","pyproject.toml","setup.py",
    "setup.cfg","environment.yml","environment.yaml","Pipfile","conda.yaml"}
MAX_M = 25
TOKEN = os.environ.get("GITHUB_TOKEN")
if not TOKEN:
    sys.exit("ERROR: set GITHUB_TOKEN first (see instructions)")

s = requests.Session()
s.headers.update({"Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})

def gh(url):
    while True:
        r = s.get(url)
        if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
            w = max(int(r.headers.get("X-RateLimit-Reset", time.time()+60)) - int(time.time()) + 1, 1)
            print(f"  rate limit, sleeping {w}s", flush=True); time.sleep(w); continue
        return r

def branch(full):
    r = gh(f"https://api.github.com/repos/{full}")
    return r.json().get("default_branch") if r.status_code == 200 else None

def manifests(full):
    b = branch(full)
    if not b: return None
    r = gh(f"https://api.github.com/repos/{full}/git/trees/{b}?recursive=1")
    if r.status_code != 200: return None
    return [t["path"] for t in r.json().get("tree", [])
            if t.get("type")=="blob" and t["path"].split("/")[-1] in MANIFEST_NAMES][:MAX_M]

def has_mlflow(full, p):
    r = gh(f"https://api.github.com/repos/{full}/contents/{p}")
    if r.status_code != 200: return False
    d = r.json()
    if d.get("encoding") != "base64": return False
    try: return "mlflow" in base64.b64decode(d["content"]).decode("utf-8","ignore").lower()
    except Exception: return False

all_repos = pd.read_csv("candidates.csv")["name"].dropna().tolist()
kept_v2 = set(pd.read_csv("mlflow_repos.csv")["repo"].dropna().tolist())
PROC, OUT = "widened_processed.txt", "mlflow_repos_widened.csv"
done = {l.strip() for l in open(PROC)} if os.path.exists(PROC) else set()
if not os.path.exists(OUT): open(OUT,"w").write("repo,evidence_file,evidence_path\n")
todo = [r for r in all_repos if r not in kept_v2 and r not in done]
kept = sum(1 for _ in open(OUT)) - 1
print(f"Candidates {len(all_repos)} | v2 kept {len(kept_v2)} | done {len(done)} | remaining {len(todo)} | new keeps {kept}", flush=True)

of, pf = open(OUT,"a"), open(PROC,"a")
for i, full in enumerate(todo, 1):
    ps = manifests(full)
    if ps:
        for p in ps:
            if has_mlflow(full, p):
                of.write(f"{full},{p.split('/')[-1]},{p}\n"); of.flush(); kept += 1
                print(f"KEEP {full} ({p})  new keeps: {kept}", flush=True); break
    pf.write(full+"\n"); pf.flush()
    if i % 20 == 0: print(f"[{i}/{len(todo)}] new keeps: {kept}", flush=True)
    time.sleep(0.03)
of.close(); pf.close()
print(f"\nFinished. New MLflow repos found by widened scan: {kept}", flush=True)