"""Build, evaluate (with plots), and register the production text model.

The autoresearch models registered with metrics only, so the registry showed no
visuals. This registers vaporware_text with the full plot set (ROC, PR,
confusion) plus a run-progression chart, so the registry renders it like the
original model.
"""
import json
import shutil
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hopsworks

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "repos_text.jsonl"
OUT = ROOT / "models" / "text_artifact"
MODEL_NAME = "vaporware_text"
LABEL = "vaporware"
TEXT_COL = "readme_text"
NON_FEATURES = {
    "repo_id", "full_name", "vaporware", "archived", "created_at", "pushed_at",
    "stars", "forks", "open_issues", "size_kb", "language", "readme_text",
}

# Kept-step progression across the whole run (counts -> text), for the chart.
RUN = [
    ("baseline HGB", 0.5970, "counts"), ("XGBoost", 0.6073, "counts"),
    ("LogReg", 0.6082, "counts"), ("vote", 0.6150, "counts"),
    ("drop feats", 0.6164, "counts"),
    ("TF-IDF word", 0.7462, "text"), ("TF-IDF word+char", 0.7624, "text"),
]


def build_model(num_cols):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    pre = ColumnTransformer([
        ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=5,
                                 max_features=30000, sublinear_tf=True,
                                 strip_accents="unicode"), TEXT_COL),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                                 max_features=30000, sublinear_tf=True), TEXT_COL),
        ("num", make_pipeline(StandardScaler()), num_cols),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(C=2.0, max_iter=4000,
                                                random_state=42))])


def plots(y_test, y_prob, y_pred):
    from sklearn.metrics import (RocCurveDisplay, PrecisionRecallDisplay,
                                 ConfusionMatrixDisplay, confusion_matrix)
    OUT.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC — vaporware text model")
    plt.savefig(OUT / "roc_curve.png", bbox_inches="tight", dpi=120); plt.close()
    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.title("Precision-Recall — vaporware text model")
    plt.savefig(OUT / "pr_curve.png", bbox_inches="tight", dpi=120); plt.close()
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                           display_labels=["alive", "vaporware"]).plot()
    plt.title("Confusion matrix")
    plt.savefig(OUT / "confusion_matrix.png", bbox_inches="tight", dpi=120); plt.close()
    # Run-progression chart (the autoresearch story, counts -> text).
    plt.figure(figsize=(8, 5))
    xs = range(len(RUN))
    ys = [r[1] for r in RUN]
    colors = ["#1f77b4" if r[2] == "counts" else "#d62728" for r in RUN]
    plt.plot(xs, ys, "-", color="#888", zorder=1)
    plt.scatter(xs, ys, c=colors, s=80, zorder=2)
    for x, (name, v, _) in zip(xs, RUN):
        plt.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8)
    plt.xticks(list(xs), [r[0] for r in RUN], rotation=30, ha="right", fontsize=8)
    plt.ylabel("CV ROC-AUC"); plt.title("autoresearch progression (blue=counts, red=text)")
    plt.tight_layout()
    plt.savefig(OUT / "run_progression.png", bbox_inches="tight", dpi=120); plt.close()


def main():
    rows = [json.loads(l) for l in open(DATA) if l.strip()]
    df = pd.DataFrame(rows).drop_duplicates(subset=["repo_id"]).reset_index(drop=True)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    num_cols = [c for c in df.columns if c not in NON_FEATURES]
    for c in num_cols:
        df[c] = df[c].astype(float)
    X = df[[TEXT_COL] + num_cols]
    y = df[LABEL].astype(int)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 accuracy_score, precision_score, recall_score, f1_score)
    # Two fits only (be light on the terminal pod): one holdout fit for the
    # plots/metrics, one final fit for the artifact. The 5-fold CV (0.7624) is
    # already established by the autoresearch run; no need to recompute it here.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = build_model(num_cols); m.fit(Xtr, ytr)
    prob = m.predict_proba(Xte)[:, 1]; pred = (prob >= 0.5).astype(int)
    plots(yte, prob, pred)

    metrics = {
        "roc_auc_cv": 0.7624,  # from the autoresearch run (5-fold CV mean)
        "roc_auc_holdout": round(float(roc_auc_score(yte, prob)), 4),
        "average_precision": round(float(average_precision_score(yte, prob)), 4),
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "precision": round(float(precision_score(yte, pred)), 4),
        "recall": round(float(recall_score(yte, pred)), 4),
        "f1": round(float(f1_score(yte, pred)), 4),
    }
    print("metrics:", json.dumps(metrics), flush=True)

    # Final artifact: fit on ALL data + ship predictor and extractor alongside.
    final = build_model(num_cols); final.fit(X, y)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, OUT / "model.joblib")
    shutil.copy(ROOT / "readme_features.py", OUT / "readme_features.py")
    shutil.copy(ROOT / "serving" / "predict.py", OUT / "predict.py")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    project = hopsworks.login()
    mr = project.get_model_registry()
    fv = project.get_feature_store().get_feature_view("vaporware_fv", version=1)
    model = mr.python.create_model(
        name=MODEL_NAME,
        metrics=metrics,
        description="Vaporware score from README text (TF-IDF word+char + counts). "
                    "Custom predictor takes raw README markdown.",
        input_example=["# my-project\n\nBlazingly fast, production-ready. Coming soon."],
        feature_view=fv,
    )
    model.save(str(OUT))
    print(f"registered {MODEL_NAME} v{model.version} (roc_auc_cv={metrics['roc_auc_cv']})", flush=True)


if __name__ == "__main__":
    main()
