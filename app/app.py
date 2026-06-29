"""README Vaporware Score — Streamlit app.

Paste a README (or a GitHub repo URL) and get a 0-100 score: how much the
README *reads* like a project that gets abandoned. Trained on archived-vs-active
repos from the same creation cohort. Score from the README text alone.
"""
import base64
import os
import sys
from pathlib import Path

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

DEPLOYMENT_NAME = "vaporwaretext"

st.set_page_config(page_title="README Vaporware Score", page_icon="🚀", layout="centered")


@st.cache_resource
def get_deployment():
    """Handle to the served text model (TF-IDF word+char + counts). The heavy
    model runs on its own KServe predictor; the app just sends the README text."""
    import hopsworks
    project = hopsworks.login()
    return project.get_model_serving().get_deployment(DEPLOYMENT_NAME)


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
    dep = get_deployment()
    if dep is None or not dep.is_running():
        st.warning("Scorer is starting up. Try again in a moment.")
        st.stop()
    with st.spinner("Scoring..."):
        res = dep.predict(inputs=[readme])
    score = float(res["predictions"][0])
    feats = extract(readme)  # local, only for the breakdown below

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
