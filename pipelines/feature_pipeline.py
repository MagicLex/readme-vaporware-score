"""Feature pipeline (F stage of FTI) — runs as a Hopsworks job.

Loads the collected README dataset into an offline feature group. Offline only:
at inference the features are computed from the pasted README on demand, so
there is no online lookup by repo_id -> this FG only backs training.

Self-contained single file (no local imports) so it deploys cleanly as a job.
Reads the dataset from the local repo when run in the terminal, otherwise
downloads it from the project's Resources dataset (so the job pod can find it).
"""
import json
import os
from pathlib import Path

import pandas as pd
import hopsworks
from hsfs.feature import Feature

FG_NAME = "readme_vaporware_features"
FG_VERSION = 1
LOCAL_DATA = Path(__file__).resolve().parent.parent / "data" / "repos.jsonl"
RESOURCES_DATA = "Resources/vaporware/repos.jsonl"  # project-relative HopsFS path

# Columns whose type we pin; everything else is inferred (ints / float densities).
TIMESTAMPS = ["created_at", "pushed_at"]
BIGINT_COLS = ["repo_id", "vaporware", "stars", "forks", "open_issues", "size_kb"]
STRING_COLS = ["language", "full_name"]
BOOL_COLS = ["archived"]

DESCRIPTIONS = {
    "repo_id": "GitHub numeric repository id (primary key)",
    "full_name": "owner/repo identifier",
    "vaporware": "Label: 1 = owner-archived (abandoned), 0 = recently active",
    "stars": "Stargazer count at collection time",
    "forks": "Fork count at collection time",
    "open_issues": "Open issues + PRs at collection time",
    "size_kb": "Repository size in KB",
    "language": "Primary language reported by GitHub",
    "created_at": "Repository creation timestamp",
    "pushed_at": "Last push timestamp",
    "archived": "Whether the repo is archived on GitHub (== the label)",
    "readme_chars": "README length in characters",
    "readme_words": "README word count",
    "readme_lines": "README line count",
    "emoji_count": "Total emoji in README",
    "emoji_per_1k_chars": "Emoji per 1000 characters",
    "rocket_count": "Count of the rocket emoji",
    "fire_count": "Count of the fire emoji",
    "sparkles_count": "Count of the sparkles emoji",
    "image_count": "Markdown + HTML image count",
    "badge_count": "Shield/badge image count",
    "badge_density": "Badges per 1000 characters",
    "link_count": "Markdown link count",
    "heading_count": "Markdown heading count",
    "list_item_count": "Bulleted list item count",
    "code_fence_blocks": "Fenced code block count",
    "inline_code_count": "Inline code span count",
    "buzzword_count": "Marketing buzzword hits (blazingly fast, production-ready, ...)",
    "buzzword_per_1k_words": "Buzzwords per 1000 words",
    "wip_marker_count": "Unfinished-work markers (TODO, coming soon, WIP, ...)",
    "exclamation_count": "Exclamation mark count",
    "exclamation_per_1k_words": "Exclamation marks per 1000 words",
    "uppercase_word_count": "ALL-CAPS word count",
    "has_install_section": "README mentions installation",
    "has_usage_section": "README has usage / getting-started",
    "has_license_section": "README mentions a license",
    "has_contributing": "README mentions contributing",
    "has_tests_mention": "README mentions tests",
}


def resolve_data(project) -> Path:
    if LOCAL_DATA.exists():
        return LOCAL_DATA
    print(f"local data absent, downloading {RESOURCES_DATA} from Resources", flush=True)
    dl = project.get_dataset_api().download(RESOURCES_DATA, local_path=".", overwrite=True)
    return Path(dl)


def load_df(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows).drop_duplicates(subset=["repo_id"]).reset_index(drop=True)
    for c in TIMESTAMPS:
        df[c] = pd.to_datetime(df[c], utc=True)
    for c in BIGINT_COLS:
        df[c] = df[c].astype("int64")
    for c in BOOL_COLS:
        df[c] = df[c].astype(bool)
    for c in STRING_COLS:
        df[c] = df[c].astype(str)
    return df


def feature_list(df: pd.DataFrame):
    feats = []
    for col in df.columns:
        if col == "repo_id" or col in BIGINT_COLS:
            ftype = "bigint"
        elif col in TIMESTAMPS:
            ftype = "timestamp"
        elif col in BOOL_COLS:
            ftype = "boolean"
        elif col in STRING_COLS:
            ftype = "string"
        else:
            ftype = "double" if df[col].dtype.kind == "f" else "bigint"
        feats.append(Feature(col, ftype, description=DESCRIPTIONS.get(col, col)))
    return feats


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    df = load_df(resolve_data(project))
    pos = int(df["vaporware"].sum())
    print(f"loaded {len(df)} repos | vaporware={pos} alive={len(df)-pos}", flush=True)

    fg = fs.get_or_create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        description="README-derived features for predicting repo abandonment "
                    "(vaporware score). Label from the GitHub archived flag.",
        primary_key=["repo_id"],
        event_time="created_at",
        features=feature_list(df),
        online_enabled=False,
        statistics_config=True,
    )
    fg.insert(df, wait=True)
    print(f"inserted {len(df)} rows into {FG_NAME} v{FG_VERSION}", flush=True)


if __name__ == "__main__":
    main()
