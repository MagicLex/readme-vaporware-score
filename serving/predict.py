"""KServe predictor for the vaporware text model.

Input: raw README markdown (a list of strings, or list of {"readme": "..."}).
The predictor computes the 27 count features with the same readme_features.extract
used at training time, assembles the [readme_text + counts] frame the pipeline
expects, and returns a 0-100 vaporware score per README.
"""
import glob
import os
import sys

import joblib
import pandas as pd


def load_model_file(name):
    """Resolve a file saved alongside the model. Model files mount under
    MODEL_FILES_PATH at serving time; ARTIFACT_FILES_PATH holds only this
    script. Fall back to the standard mount roots."""
    for root in (os.environ.get("MODEL_FILES_PATH"),
                 os.environ.get("ARTIFACT_FILES_PATH"),
                 "/mnt/models", "/mnt/artifacts"):
        if root:
            hits = glob.glob(f"{root}/**/{name}", recursive=True)
            if hits:
                return hits[0]
    raise FileNotFoundError(f"{name} not found under model/artifact mounts")


class Predict:
    def __init__(self):
        self.model = joblib.load(load_model_file("model.joblib"))
        sys.path.insert(0, os.path.dirname(load_model_file("readme_features.py")))
        from readme_features import extract
        self.extract = extract
        self.num_cols = list(self.extract("").keys())  # the 27 count features

    def _texts(self, inputs):
        out = []
        for it in inputs:
            if isinstance(it, dict):
                out.append(it.get("readme") or it.get("readme_text") or "")
            else:
                out.append(str(it))
        return out

    def predict(self, inputs):
        texts = self._texts(inputs)
        rows = []
        for t in texts:
            f = self.extract(t)
            row = {"readme_text": t}
            row.update({c: f[c] for c in self.num_cols})
            rows.append(row)
        df = pd.DataFrame(rows)
        scores = self.model.predict_proba(df)[:, 1] * 100.0
        return {"predictions": [round(float(s), 2) for s in scores]}
