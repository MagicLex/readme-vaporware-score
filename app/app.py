"""README Vaporware Score — Streamlit app.

Paste a README (or a GitHub repo URL) and get a 0-100 score: how much the
README *reads* like a project that gets abandoned. Trained on archived-vs-active
repos from the same creation cohort. Score from the README text alone.
"""
import base64
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

# readme_features.py lives next to this file when deployed (vendored at deploy
# time) or one dir up in the repo. Same source as training -> no skew.
HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE.parent):
    if (cand / "readme_features.py").exists():
        sys.path.insert(0, str(cand))
        break
from readme_features import extract  # noqa: E402

MODEL_NAME = "vaporware_score"

st.set_page_config(page_title="README Vaporware Score", page_icon="🚀", layout="centered")


@st.cache_resource
def load_model():
    """Download the model from the Hopsworks registry; fall back to a local
    artifact when running outside the cluster."""
    try:
        import hopsworks
        project = hopsworks.login()
        mr = project.get_model_registry()
        m = mr.get_model(MODEL_NAME)
        d = m.download()
        return joblib.load(Path(d) / "model.joblib")
    except Exception as e:
        local = HERE.parent / "models" / "artifact" / "model.joblib"
        if local.exists():
            return joblib.load(local)
        raise RuntimeError(f"could not load model: {e}")


def fetch_readme(url: str) -> str:
    """Pull README markdown for a github.com/owner/repo URL."""
    parts = url.strip().rstrip("/").replace("https://github.com/", "").split("/")
    if len(parts) < 2:
        raise ValueError("expected a github.com/owner/repo URL")
    owner, repo = parts[0], parts[1]
    headers = {"Accept": "application/vnd.github+json"}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    r = requests.get(f"https://api.github.com/repos/{owner}/{repo}/readme",
                     headers=headers, timeout=20)
    r.raise_for_status()
    return base64.b64decode(r.json()["content"]).decode("utf-8", "ignore")


model = load_model()

st.title("🚀 README Vaporware Score")
st.caption("How much does your README *read* like a project that gets "
           "abandoned? Score from the README text alone. This is a toy. "
           "Do not take it personally (take it a little personally).")

tab_paste, tab_url = st.tabs(["Paste README", "GitHub URL"])
readme = ""
with tab_paste:
    readme = st.text_area("Paste README markdown", height=240,
                          placeholder="# 🚀 my-project\n\nBlazingly fast, "
                                      "production-ready, zero-config...")
with tab_url:
    url = st.text_input("github.com/owner/repo")
    if url:
        try:
            readme = fetch_readme(url)
            st.success(f"fetched {len(readme)} chars")
        except Exception as e:
            st.error(f"fetch failed: {e}")

if st.button("Score it", type="primary") and readme.strip():
    feats = extract(readme)
    cols = list(model.feature_names_in_)  # the model's own feature order
    X = pd.DataFrame([[feats[c] for c in cols]], columns=cols)
    score = float(model.predict_proba(X)[0, 1]) * 100

    st.metric("Vaporware score", f"{score:.0f} / 100")
    st.progress(min(int(score), 100))
    if score >= 70:
        st.error("Reads like vaporware. Lots of promise, archived energy.")
    elif score >= 40:
        st.warning("Mixed signals. Ship something.")
    else:
        st.success("Reads like a project that survives. Boring. Good.")

    st.subheader("What the README is doing")
    smelly = {
        "🚀 rockets": feats["rocket_count"],
        "emoji": feats["emoji_count"],
        "badges": feats["badge_count"],
        "buzzwords": feats["buzzword_count"],
        "exclamations": feats["exclamation_count"],
        "WIP / TODO markers": feats["wip_marker_count"],
    }
    cols = st.columns(3)
    for i, (k, v) in enumerate(smelly.items()):
        cols[i % 3].metric(k, v)

    with st.expander("All extracted features"):
        st.json(feats)

st.divider()
st.caption("Label = GitHub `archived` flag (proxy for abandonment). Model sees "
           "README features only, never stars or the archived flag. Built on "
           "Hopsworks.")
