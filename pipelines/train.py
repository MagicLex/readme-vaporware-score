"""Training pipeline (T stage of FTI) — runs as a Hopsworks job.

Feature view over the README features -> gradient-boosted classifier ->
evaluate -> register in the model registry with provenance.

Honesty rules baked in:
- The model sees ONLY README-derived features. The feature view is built to
  exclude `archived` (that IS the label) and stars/forks/popularity (the premise
  is "from the README alone").
- The classifier is fit on a named DataFrame, so `model.feature_names_in_` is
  persisted. Inference orders its features by that, so there is one source of
  truth for feature order and no train/serve skew.

Self-contained single file so it deploys cleanly as a job.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay,
)

import hopsworks

FG_NAME = "readme_vaporware_features"
FV_NAME = "vaporware_fv"
MODEL_NAME = "vaporware_score"
LABEL = "vaporware"
OUT = Path("artifact").resolve()

# README columns only. Excludes the label, the leaky `archived` flag, and
# popularity/metadata that are not part of the README itself.
NON_FEATURES = {
    "repo_id", "full_name", "vaporware", "archived", "created_at", "pushed_at",
    "stars", "forks", "open_issues", "size_kb", "language",
}


def get_feature_view(fs):
    fg = fs.get_feature_group(FG_NAME, version=1)
    feature_cols = [f.name for f in fg.features if f.name not in NON_FEATURES]
    query = fg.select(feature_cols + [LABEL])
    fv = fs.get_or_create_feature_view(
        name=FV_NAME,
        version=1,
        description="README-only features for the vaporware abandonment model",
        query=query,
        labels=[LABEL],
    )
    return fv, feature_cols


def make_plots(model, X_test, y_test, y_prob, y_pred, feature_cols):
    OUT.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC — vaporware score")
    plt.savefig(OUT / "roc_curve.png", bbox_inches="tight", dpi=120); plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.title("Precision-Recall — vaporware score")
    plt.savefig(OUT / "pr_curve.png", bbox_inches="tight", dpi=120); plt.close()

    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                           display_labels=["alive", "vaporware"]).plot()
    plt.title("Confusion matrix")
    plt.savefig(OUT / "confusion_matrix.png", bbox_inches="tight", dpi=120); plt.close()

    imp = permutation_importance(model, X_test, y_test, n_repeats=10,
                                 random_state=42, scoring="roc_auc")
    order = np.argsort(imp.importances_mean)[::-1][:15]
    names = [feature_cols[i] for i in order]
    vals = imp.importances_mean[order]
    plt.figure(figsize=(8, 6))
    plt.barh(names[::-1], vals[::-1])
    plt.xlabel("drop in ROC-AUC when shuffled")
    plt.title("Top README signals of vaporware")
    plt.savefig(OUT / "feature_importance.png", bbox_inches="tight", dpi=120); plt.close()
    return {names[i]: round(float(vals[i]), 4) for i in range(len(names))}


def main():
    import sklearn
    print(f"sklearn={sklearn.__version__}", flush=True)  # pin app env to match
    project = hopsworks.login()
    fs = project.get_feature_store()
    fv, feature_cols = get_feature_view(fs)

    X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)
    X_train, X_test = X_train[feature_cols], X_test[feature_cols]
    y_train = y_train[LABEL].astype(int)
    y_test = y_test[LABEL].astype(int)
    print(f"train={len(X_train)} test={len(X_test)} "
          f"pos_rate={y_train.mean():.3f} features={len(feature_cols)}", flush=True)

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, random_state=42,
    )
    model.fit(X_train, y_train)  # named DataFrame -> model.feature_names_in_ set

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "average_precision": round(average_precision_score(y_test, y_prob), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    print("metrics:", json.dumps(metrics), flush=True)

    top = make_plots(model, X_test, y_test, y_prob, y_pred, feature_cols)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT / "model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "feature_names.json").write_text(json.dumps(list(model.feature_names_in_), indent=2))
    (OUT / "top_signals.json").write_text(json.dumps(top, indent=2))

    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    mr = project.get_model_registry()
    model_schema = ModelSchema(Schema(X_train), Schema(y_train.to_frame()))
    hops_model = mr.sklearn.create_model(
        name=MODEL_NAME,
        metrics=metrics,
        description="Predicts repo abandonment from README features alone.",
        input_example=X_train.head(1),
        model_schema=model_schema,
        feature_view=fv,
    )
    hops_model.save(str(OUT))
    print(f"registered model {MODEL_NAME} v{hops_model.version}", flush=True)


if __name__ == "__main__":
    main()
