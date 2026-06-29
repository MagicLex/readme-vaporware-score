"""Collect a labelled README dataset from GitHub.

Positive class (vaporware=1): repos the owner has archived  -> abandoned.
Negative class (vaporware=0): repos pushed to recently       -> alive.

Both classes are drawn from the SAME repo-creation cohort to reduce the
"README fashion changed over the years" confound. We still note the caveat in
the project README: archived is a proxy for abandonment, not ground-truth
"died within 6 months".

Resumable: every repo is appended as one JSON line to data/repos.jsonl.
Re-running skips full_names already present.
"""
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from readme_features import extract  # noqa: E402

OUT = ROOT / "data" / "repos.jsonl"
API = "https://api.github.com"

# Same creation cohort for both classes. Old enough that survivors had a real
# chance to be abandoned, recent enough that READMEs use modern conventions.
COHORT = "created:2021-01-01..2023-06-30"
# Star buckets let us pull >1000 per class (search caps at 1000 results/query).
STAR_BUCKETS = ["10..30", "30..80", "80..250", "250..1000", "1000..50000"]

TARGET_PER_CLASS = 1800
PER_PAGE = 100


def token():
    t = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if t:
        return t
    return subprocess.check_output(["gh", "auth", "token"], text=True).strip()


SESSION = requests.Session()
SESSION.headers.update({
    "Authorization": f"Bearer {token()}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
})


def _respect_limits(resp):
    """Sleep if we are about to exhaust a rate-limit window."""
    remaining = int(resp.headers.get("X-RateLimit-Remaining", "1"))
    reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
    if remaining <= 1 and reset:
        wait = max(reset - int(time.time()) + 2, 2)
        print(f"  rate limit hit, sleeping {wait}s", flush=True)
        time.sleep(wait)


def get(url, params=None, tries=4):
    for attempt in range(tries):
        resp = SESSION.get(url, params=params, timeout=30)
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            _respect_limits(resp)
            continue
        if resp.status_code in (502, 503, 504):
            time.sleep(2 * (attempt + 1))
            continue
        return resp
    return resp


def search_repos(query):
    """Yield repo metadata dicts for a search query, paginating to the cap."""
    for page in range(1, 11):  # 10 pages * 100 = 1000 max
        resp = get(f"{API}/search/repositories", params={
            "q": query, "per_page": PER_PAGE, "page": page, "sort": "stars",
        })
        if resp.status_code != 200:
            print(f"  search {resp.status_code}: {resp.text[:120]}", flush=True)
            break
        items = resp.json().get("items", [])
        if not items:
            break
        for it in items:
            yield it
        _respect_limits(resp)
        time.sleep(2.2)  # search allows 30/min; stay well under


def fetch_readme(full_name):
    resp = get(f"{API}/repos/{full_name}/readme")
    if resp.status_code != 200:
        return None
    data = resp.json()
    if data.get("encoding") != "base64":
        return None
    try:
        return base64.b64decode(data["content"]).decode("utf-8", "ignore")
    except Exception:
        return None


def load_seen():
    seen = set()
    if OUT.exists():
        with open(OUT) as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["full_name"])
                except Exception:
                    pass
    return seen


def collect_class(label, base_query, seen, sink):
    got = 0
    for bucket in STAR_BUCKETS:
        if got >= TARGET_PER_CLASS:
            break
        query = f"{base_query} {COHORT} stars:{bucket}"
        print(f"[label={label}] {query}", flush=True)
        for repo in search_repos(query):
            if got >= TARGET_PER_CLASS:
                break
            fn = repo["full_name"]
            if fn in seen:
                continue
            seen.add(fn)
            readme = fetch_readme(fn)
            if not readme or len(readme) < 80:
                continue  # need text to score; very short README -> skip
            feats = extract(readme)
            row = {
                "repo_id": repo["id"],
                "full_name": fn,
                "vaporware": label,
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "open_issues": repo.get("open_issues_count", 0),
                "size_kb": repo.get("size", 0),
                "language": repo.get("language") or "unknown",
                "created_at": repo.get("created_at"),
                "pushed_at": repo.get("pushed_at"),
                "archived": repo.get("archived", False),
                **feats,
            }
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            got += 1
            if got % 50 == 0:
                print(f"  label={label}: {got}/{TARGET_PER_CLASS}", flush=True)
    print(f"[label={label}] done: {got}", flush=True)
    return got


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen()
    print(f"resuming: {len(seen)} repos already collected", flush=True)
    with open(OUT, "a") as sink:
        collect_class(1, "archived:true", seen, sink)
        collect_class(0, "archived:false pushed:>2026-01-01", seen, sink)
    print(f"all done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
