# Model optimization run

This folder is an autonomous model-search run over the vaporware dataset, in the
style of [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
(Hopsworks adaptation: [MagicLex/hopsworks-autoresearch](https://github.com/MagicLex/hopsworks-autoresearch)).
The loop edits a single `train.py`, runs a fixed-budget experiment, keeps the
change if the metric improves and reverts it otherwise, and records every run in
two Hopsworks surfaces:

- **Leaderboard**: feature group `autoresearch_experiments_jun29`, one row per
  experiment (`commit`, `val_metric`, `status`, `description`, `ts`).
- **Versions**: model registry `autoresearch_jun29`, one version per kept model,
  so the registry charts the improvement curve.

The metric is **5-fold stratified CV mean ROC-AUC** (same folds every run), not a
single split. A single 80/20 split on 720 test rows has a standard error around
0.02 here, so a single number is noise. The fold scores ranged 0.55 to 0.64.

## Two approaches to the features

### Approach 1: README structure as counts

The model on `main`. The README is reduced to 27 numeric features (length,
heading count, badge count, emoji count, buzzword count, and so on; see
[`readme_features.py`](../readme_features.py)). The text itself is discarded.

Cheap, deterministic, interpretable, and the same `extract()` runs at training
and serving so there is no skew. The autoresearch loop optimized within this
feature set:

| step | val_metric (CV) | decision |
|---|---:|---|
| baseline HGB | 0.5970 | keep |
| XGBoost (n400, depth3, lr0.03) | 0.6073 | keep |
| LogReg + log1p + scaling | 0.6082 | keep |
| soft-vote XGB + RF + LogReg | 0.6150 | keep |
| ensemble + ratio/structure features | 0.6136 | discard |
| stacking meta-learner | 0.6131 | discard |
| drop 8 near-zero-signal features | **0.6164** | keep (best) |
| 4-model vote (+HGB) | 0.6134 | discard |

Best count model: a soft-voting ensemble of XGBoost, RandomForest, and a
log1p + scaled LogisticRegression, after dropping the eight features that carried
no signal (fire, sparkles, rocket, WIP markers, and the per-1k densities). CV
ROC-AUC **0.6164**, up from the **0.5970** baseline measured the same way.

### Approach 2: raw README text (TF-IDF)

The lever Approach 1 cannot reach: the actual words. The dataset is re-fetched
with the raw README markdown stored ([`collect/add_text.py`](../collect/add_text.py)),
then the model adds TF-IDF n-gram features over the text alongside the counts,
fit per fold inside the sklearn pipeline (no leakage).

Status: running. Results land here when the run completes.

| step | val_metric (CV) | decision |
|---|---:|---|
| TF-IDF (1,2)-gram + counts | _pending_ | |

## Head to head, same protocol

The original `0.622` everyone remembers was one lucky unseeded split. Measured
identically, the optimized model wins both ways:

| protocol | original HGB | best ensemble |
|---|---:|---:|
| 5-fold CV mean | 0.5970 | 0.6164 |
| single seeded 80/20 | 0.5914 | 0.6125 |

The gain is real but small (about +0.02). With count features alone, ~0.62 is the
ceiling. Approach 2 is where a larger jump, if any, has to come from.

## Files

- `train.py`: the single file the loop edits. Only `EXP_DESC`,
  `engineer_features()`, and `build_model()` change between experiments.
- `log_exp.sh`: run a result into the leaderboard FG and (on keeps) register a
  model version.
- `log_row.py`: SDK insert for the leaderboard (the CLI cannot write a timestamp
  column from JSON).

## Reproduce

From a Hopsworks terminal, after `make features` has populated the feature group:

```bash
python autoresearch/train.py > autoresearch/run.log 2>&1   # one experiment
bash autoresearch/log_exp.sh keep "my change"               # record it
```

Edit the experiment section of `train.py`, rerun, keep or revert. The first run
caches the feature group to `autoresearch/data_cache.parquet` so later runs are
fast.
