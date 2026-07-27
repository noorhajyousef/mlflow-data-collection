import csv
import datetime as dt
import json
import os
import time
from pathlib import Path

import requests

# CONFIG
from getpass import getpass
GITHUB_TOKEN = getpass("Paste your GitHub token (input hidden): ").strip()

# Google Drive paths (assumes drive already mounted in an earlier cell:
#   from google.colab import drive; drive.mount('/content/drive')
BASE_DIR = Path("/content/drive/MyDrive/mlflow_research")
OUT_CSV = BASE_DIR / "results_v2.csv"
DONE_WINDOWS_FILE = BASE_DIR / "v2_done_windows.txt"

DATE_FROM = dt.date(2025, 4, 29)  
DATE_TO = dt.date.today()

QUERY_BASE = "language:python stars:>=10 fork:false"
PER_PAGE = 100
MAX_PAGES = 10                     
INITIAL_WINDOW_DAYS = 6            

API = "https://api.github.com/search/repositories"
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

CSV_FIELDS = [
    "name",           
    "created_at",
    "stargazers",
    "license",         
    "is_fork",
    "default_branch",
    "html_url",
]

# HELPERS

def gh_get(params):
    """One API call with rate-limit handling and retries."""
    for attempt in range(8):
        r = requests.get(API, headers=HEADERS, params=params, timeout=30)
        if r.status_code == 200:
            
            remaining = int(r.headers.get("X-RateLimit-Remaining", 1))
            if remaining <= 1:
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - time.time(), 1) + 1
                print(f"    rate limit reached, sleeping {wait:.0f}s")
                time.sleep(wait)
            else:
                time.sleep(2.1)  
            return r.json()
        if r.status_code in (403, 429):
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - time.time(), 5) + 2
            print(f"    throttled ({r.status_code}), sleeping {wait:.0f}s")
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            time.sleep(10 * (attempt + 1))
            continue
        raise RuntimeError(f"GitHub API error {r.status_code}: {r.text[:200]}")
    raise RuntimeError("Giving up after repeated rate-limit/5xx responses")


def window_query(start: dt.date, end: dt.date) -> str:
    return f"{QUERY_BASE} created:{start.isoformat()}..{end.isoformat()}"


def count_for_window(start, end) -> int:
    data = gh_get({"q": window_query(start, end), "per_page": 1})
    return data.get("total_count", 0)


def fetch_window(start, end):
    """Yield all repos in a window known to have <=1000 results."""
    for page in range(1, MAX_PAGES + 1):
        data = gh_get({
            "q": window_query(start, end),
            "per_page": PER_PAGE,
            "page": page,
            "sort": "updated",
        })
        items = data.get("items", [])
        for it in items:
            yield {
                "name": it["full_name"],
                "created_at": it["created_at"],
                "stargazers": it["stargazers_count"],
                "license": (it.get("license") or {}).get("name", ""),
                "is_fork": it["fork"],
                "default_branch": it.get("default_branch", ""),
                "html_url": it["html_url"],
            }
        if len(items) < PER_PAGE:
            break


def split(start: dt.date, end: dt.date):
    """Split a window in half by days."""
    mid = start + (end - start) / 2
    if isinstance(mid, dt.timedelta):  # py<3.12 guard; compute manually
        mid = start + dt.timedelta(days=(end - start).days // 2)
    if mid <= start:
        mid = start
    return (start, mid), (mid + dt.timedelta(days=1), end)


def load_done() -> set:
    if DONE_WINDOWS_FILE.exists():
        return set(DONE_WINDOWS_FILE.read_text().split())
    return set()


def mark_done(tag: str):
    with DONE_WINDOWS_FILE.open("a") as f:
        f.write(tag + "\n")


# MAIN

BASE_DIR.mkdir(parents=True, exist_ok=True)
done = load_done()
seen_names = set()


write_header = not OUT_CSV.exists()
if OUT_CSV.exists():
    with OUT_CSV.open() as f:
        for row in csv.DictReader(f):
            seen_names.add(row["name"])
    print(f"Resuming: {len(seen_names)} repos already collected, "
          f"{len(done)} windows done")


work = []
cursor = DATE_FROM
while cursor <= DATE_TO:
    wend = min(cursor + dt.timedelta(days=INITIAL_WINDOW_DAYS), DATE_TO)
    work.append((cursor, wend))
    cursor = wend + dt.timedelta(days=1)

total_written = len(seen_names)
with OUT_CSV.open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    while work:
        start, end = work.pop(0)
        tag = f"{start.isoformat()}_{end.isoformat()}"
        if tag in done:
            continue

        n = count_for_window(start, end)
        if n == 0:
            mark_done(tag)
            continue
        if n > 1000 and start != end:
            w1, w2 = split(start, end)
            print(f"[{tag}] {n} results > 1000, splitting")
            work.insert(0, w2)
            work.insert(0, w1)
            continue
        if n > 1000 and start == end:
        
            print(f"[{tag}] WARNING: single day exceeds 1000 ({n}); "
                  f"collecting first 1000 only")

        added = 0
        for rec in fetch_window(start, end):
            if rec["name"] in seen_names:
                continue
            writer.writerow(rec)
            seen_names.add(rec["name"])
            added += 1
        f.flush()
        mark_done(tag)
        total_written += added
        print(f"[{tag}] expected~{n}  added {added}  total {total_written}")

print(f"\nDone. {total_written} unique repos in {OUT_CSV}")
print("Next: run the license-filter cell on results_v2.csv, then dedup "
      "against candidates.csv from v1 before the pre-filter stage.")