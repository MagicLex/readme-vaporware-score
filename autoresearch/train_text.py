"""Approach 2: raw README text (TF-IDF) + counts.

Loads the text-augmented dataset (data/repos_text.jsonl), builds TF-IDF n-gram
features over the raw README markdown alongside the numeric count features, and
scores with 5-fold stratified CV mean ROC-AUC (same folds/seed as train.py so
the two approaches are comparable).

Edit EXP_DESC and build_model(). TF-IDF is fit inside the pipeline per fold, so
there is no train/serve leakage.
"""
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ============================ EXPERIMENT (edit me) ============================
EXP_DESC = "TF-IDF word(1,2)+char(3,5) + counts -> LogReg C2"


def build_model(text_col, num_cols):
    """Return an unfitted estimator over [text_col] + num_cols."""
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    pre = ColumnTransformer([
        ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                 min_df=5, max_features=30000,
                                 sublinear_tf=True, strip_accents="unicode"),
         text_col),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=5, max_features=30000,
                                 sublinear_tf=True),
         text_col),
        ("num", make_pipeline(StandardScaler()), num_cols),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(C=2.0, max_iter=4000, random_state=42)),
    ])
# ========================== end experiment section ===========================

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "repos_text.jsonl"
LABEL = "vaporware"
NON_FEATURES = {
    "repo_id", "full_name", "vaporware", "archived", "created_at", "pushed_at",
    "stars", "forks", "open_issues", "size_kb", "language", "readme_text",
}
TEXT_COL = "readme_text"


def main():
    t0 = time.time()
    rows = [json.loads(l) for l in open(DATA) if l.strip()]
    df = pd.DataFrame(rows).drop_duplicates(subset=["repo_id"]).reset_index(drop=True)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    num_cols = [c for c in df.columns if c not in NON_FEATURES]
    for c in num_cols:
        df[c] = df[c].astype(float)
    X = df[[TEXT_COL] + num_cols]
    y = df[LABEL].astype(int)
    print(f"rows={len(df)} num_features={len(num_cols)}")

    from sklearn.model_selection import StratifiedKFold, cross_val_score
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(build_model(TEXT_COL, num_cols), X, y,
                             cv=skf, scoring="roc_auc")
    val_metric = scores.mean()
    print(f"cv_folds: {' '.join(f'{s:.4f}' for s in scores)} (std {scores.std():.4f})")

    model = build_model(TEXT_COL, num_cols)
    model.fit(X, y)
    import joblib
    (HERE / "model").mkdir(exist_ok=True)
    joblib.dump(model, HERE / "model" / "model.joblib")
    (HERE / "model" / "meta.json").write_text(json.dumps({
        "exp": EXP_DESC, "val_metric": float(val_metric), "n_rows": len(df),
    }, indent=2))

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    print(f"val_metric: {val_metric:.4f}")
    print(f"peak_memory_gb: {peak_gb:.3f}")
    print(f"training_seconds: {time.time() - t0:.2f}")


if __name__ == "__main__":
    main()
