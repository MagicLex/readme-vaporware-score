"""EDA: profile the README features against the abandonment label and check
for leakage / confounds before training.

Reads the local collected jsonl (fast). Writes a markdown report + a couple of
plots to models/eda/.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from readme_features import FEATURE_NAMES  # noqa: E402

DATA = ROOT / "data" / "repos.jsonl"
OUT = ROOT / "models" / "eda"


def load():
    rows = [json.loads(l) for l in open(DATA) if l.strip()]
    return pd.DataFrame(rows).drop_duplicates(subset=["repo_id"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    y = df["vaporware"]
    lines = ["# README Vaporware — EDA\n"]
    lines.append(f"- rows: **{len(df)}**")
    lines.append(f"- vaporware (archived): **{int(y.sum())}**, "
                 f"alive: **{int((1-y).sum())}**")
    lines.append(f"- languages: {df['language'].nunique()} "
                 f"(top: {', '.join(df['language'].value_counts().head(5).index)})\n")

    # Signed point-biserial correlation of each README feature with the label.
    lines.append("## README signal vs abandonment (correlation with label)\n")
    lines.append("Positive = more of this feature -> more likely archived.\n")
    corr = {f: float(np.corrcoef(df[f], y)[0, 1]) for f in FEATURE_NAMES}
    ranked = sorted(corr.items(), key=lambda kv: abs(kv[1]), reverse=True)
    lines.append("| feature | corr | vaporware mean | alive mean |")
    lines.append("|---|---:|---:|---:|")
    for f, c in ranked:
        vm = df.loc[y == 1, f].mean()
        am = df.loc[y == 0, f].mean()
        lines.append(f"| {f} | {c:+.3f} | {vm:.2f} | {am:.2f} |")

    # Leakage / confound check: are the two classes from the same era?
    df["created_year"] = pd.to_datetime(df["created_at"], utc=True).dt.year
    lines.append("\n## Confound check: creation year by class\n")
    ct = pd.crosstab(df["created_year"], df["vaporware"])
    lines.append(ct.to_markdown())

    # Plot top signed correlations.
    top = ranked[:12]
    plt.figure(figsize=(8, 6))
    names = [t[0] for t in top][::-1]
    vals = [t[1] for t in top][::-1]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals]
    plt.barh(names, vals, color=colors)
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("correlation with abandonment")
    plt.title("README signals (red = vaporware-leaning)")
    plt.savefig(OUT / "correlations.png", bbox_inches="tight", dpi=120)
    plt.close()

    (OUT / "eda_report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {OUT/'eda_report.md'} and correlations.png")


if __name__ == "__main__":
    main()
