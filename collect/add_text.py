"""Augment the existing labelled dataset with the raw README text.

Reads data/repos.jsonl (the 3600 already-labelled repos), re-fetches each
README, and writes data/repos_text.jsonl with every original field plus a
`readme_text` field. Same repos, same labels, so the dataset is identical to
the count-feature run -- only the raw text is added.

Resumable: skips full_names already present in the output.
"""
import base64
import json
import os
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "repos.jsonl"
OUT = ROOT / "data" / "repos_text.jsonl"
API = "https://api.github.com"


def token():
    t = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    return t or subprocess.check_output(["gh", "auth", "token"], text=True).strip()


SESSION = requests.Session()
SESSION.headers.update({
    "Authorization": f"Bearer {token()}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
})


def get(url, tries=4):
    for attempt in range(tries):
        r = SESSION.get(url, timeout=30)
        if r.status_code == 403 and "rate limit" in r.text.lower():
            reset = int(r.headers.get("X-RateLimit-Reset", "0"))
            wait = max(reset - int(time.time()) + 2, 2)
            print(f"  rate limit, sleeping {wait}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code in (502, 503, 504):
            time.sleep(2 * (attempt + 1))
            continue
        return r
    return r


def fetch_readme(full_name):
    r = get(f"{API}/repos/{full_name}/readme")
    if r.status_code != 200:
        return None
    d = r.json()
    if d.get("encoding") != "base64":
        return None
    try:
        return base64.b64decode(d["content"]).decode("utf-8", "ignore")
    except Exception:
        return None


def seen():
    s = set()
    if OUT.exists():
        for line in open(OUT):
            try:
                s.add(json.loads(line)["full_name"])
            except Exception:
                pass
    return s


def main():
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    done = seen()
    print(f"{len(rows)} repos, {len(done)} already have text", flush=True)
    n = 0
    with open(OUT, "a") as sink:
        for row in rows:
            fn = row["full_name"]
            if fn in done:
                continue
            text = fetch_readme(fn)
            if not text:
                continue  # gone / unreadable; drop from the text dataset
            row["readme_text"] = text
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            n += 1
            if n % 100 == 0:
                print(f"  {n} fetched", flush=True)
    print(f"done, wrote {n} new rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
