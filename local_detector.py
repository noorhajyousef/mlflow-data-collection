import ast, base64, time, os, sys, csv, requests
import pandas as pd

TOKEN = os.environ.get("GITHUB_TOKEN")
if not TOKEN:
    sys.exit("ERROR: set GITHUB_TOKEN first")

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

def code_search(full):
    paths, page = [], 1
    while True:
        r = gh(f"https://api.github.com/search/code?q=mlflow+repo:{full}+language:python&per_page=100&page={page}")
        if r.status_code != 200:
            print(f"  search failed {full}: {r.status_code}"); break
        items = r.json().get("items", [])
        paths += [it["path"] for it in items]
        if len(items) < 100: break
        page += 1; time.sleep(2)
    return paths

def fetch(full, path):
    r = gh(f"https://api.github.com/repos/{full}/contents/{path}")
    if r.status_code != 200: return None
    d = r.json()
    if d.get("encoding") != "base64": return None
    try: return base64.b64decode(d["content"]).decode("utf-8","ignore")
    except Exception: return None

def analyze(src):
    has_import, n_calls = False, 0
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ("import mlflow" in src or "from mlflow" in src), src.count("mlflow.")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(n.name=="mlflow" or n.name.startswith("mlflow.") for n in node.names):
            has_import = True
        elif isinstance(node, ast.ImportFrom) and node.module and (node.module=="mlflow" or node.module.startswith("mlflow.")):
            has_import = True
        elif isinstance(node, ast.Call):
            f = node.func
            while isinstance(f, ast.Attribute):
                if isinstance(f.value, ast.Name) and f.value.id=="mlflow":
                    n_calls += 1; break
                f = f.value
    return has_import, n_calls

widened = pd.read_csv("mlflow_repos_widened.csv")["repo"].dropna().tolist()

PROC = "local_detector_processed.txt"
OUT  = "mlflow_files.csv"

done = set()
if os.path.exists(PROC):
    done = {l.strip() for l in open(PROC) if l.strip()}
if not os.path.exists(OUT):
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerow(["repo","file_path","has_import","n_calls"])

todo = [r for r in widened if r not in done]
print(f"New repos to detect: {len(widened)} | already done: {len(done)} | remaining: {len(todo)}\n", flush=True)

out_f = open(OUT, "a", newline=""); w = csv.writer(out_f)
proc_f = open(PROC, "a")
total_files = 0
for full in todo:
    print(f"\n{full}", flush=True)
    paths = code_search(full)
    print(f"  {len(paths)} candidate file(s)", flush=True)
    for p in paths:
        src = fetch(full, p)
        if not src: continue
        imp, calls = analyze(src)
        if imp or calls:
            w.writerow([full, p, imp, calls]); out_f.flush(); total_files += 1
            print(f"  MLflow file: {p}  (import={imp}, calls={calls})", flush=True)
    proc_f.write(full + "\n"); proc_f.flush()
    time.sleep(2)
out_f.close(); proc_f.close()
print(f"\nFinished. {total_files} new MLflow files found in the widened repos.", flush=True)