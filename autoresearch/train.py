"""autoresearch train.py — the single file the loop edits.

Intent: MAXIMIZE ROC-AUC for vaporware (repo abandonment) from README-only
features. Direction: max. Baseline ~0.622.

Fixed seeded split (random_state=42, stratified) so every experiment is judged
on identical train/test data. Data is cached to parquet on first run.

Edit only EXP_DESC, engineer_features(), and build_model(). Everything below
the marker stays put.
"""
import os
import resource
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================ EXPERIMENT (edit me) ============================
EXP_DESC = "soft-vote XGB + RF + LogReg (log1p)"


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Transform the raw README feature frame. Return a numeric DataFrame."""
    return np.log1p(X)  # harmless for trees (monotonic), helps the linear member


def build_model():
    """Return an unfitted sklearn-compatible estimator with predict_proba."""
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                        eval_metric="logloss", random_state=42, n_jobs=-1)
    rf = RandomForestClassifier(n_estimators=500, max_depth=None,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(C=0.5, max_iter=2000, random_state=42))
    return VotingClassifier([("xgb", xgb), ("rf", rf), ("lr", lr)],
                            voting="soft", weights=[2, 1, 2], n_jobs=-1)
# ========================== end experiment section ===========================


HERE = Path(__file__).resolve().parent
CACHE = HERE / "data_cache.parquet"
FG_NAME = "readme_vaporware_features"
LABEL = "vaporware"
# README-only columns (exclude label, leaky archived flag, popularity/metadata).
NON_FEATURES = {
    "repo_id", "full_name", "vaporware", "archived", "created_at", "pushed_at",
    "stars", "forks", "open_issues", "size_kb", "language",
}


def load_data():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    import hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FG_NAME, version=1)
    df = fg.read(dataframe_type="pandas")
    df.to_parquet(CACHE)
    return df


def main():
    t0 = time.time()
    df = load_data()
    feat_cols = [c for c in df.columns if c not in NON_FEATURES]
    X = engineer_features(df[feat_cols].astype(float).copy())
    y = df[LABEL].astype(int)

    # Stable metric: 5-fold stratified CV mean ROC-AUC (same folds every run).
    # engineer_features must be stateless (row-wise); put any fit-stateful step
    # (scaling, selection) inside build_model()'s Pipeline so it fits per fold.
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(build_model(), X, y, cv=skf, scoring="roc_auc")
    val_metric = scores.mean()
    print(f"cv_folds: {' '.join(f'{s:.4f}' for s in scores)} (std {scores.std():.4f})")

    model = build_model()
    model.fit(X, y)

    import joblib
    (HERE / "model").mkdir(exist_ok=True)
    joblib.dump(model, HERE / "model" / "model.joblib")
    (HERE / "model" / "meta.json").write_text(json.dumps({
        "exp": EXP_DESC, "val_metric": float(val_metric),
        "features": feat_cols, "n_features_in": X.shape[1],
    }, indent=2))

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    print(f"val_metric: {val_metric:.4f}")
    print(f"peak_memory_gb: {peak_gb:.3f}")
    print(f"training_seconds: {time.time() - t0:.2f}")


if __name__ == "__main__":
    main()
