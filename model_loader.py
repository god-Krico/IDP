"""Auto-download custom model files from HuggingFace Hub at startup.

Usage: call ensure_models() once at the top of dashboard.py.
Models are cached by st.cache_resource so they download only once per session.
Secrets are read from st.secrets["models"]["HF_REPO"] and ["HF_TOKEN"].
"""

import os
import streamlit as st

MODEL_FILENAMES = [
    "finetuned_multi.pt",
    "mould_detection.pt",
    "phase_detection.pt",
    "ppe_best.pt",
    "progress.pt",
    "proximity_best.pt",
]

LOCAL_MODEL_DIR = "model"


@st.cache_resource(show_spinner=False)
def _download_models(hf_repo: str, hf_token: str) -> list[str]:
    from huggingface_hub import hf_hub_download

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    paths = []
    for filename in MODEL_FILENAMES:
        local_path = os.path.join(LOCAL_MODEL_DIR, filename)
        if not os.path.exists(local_path):
            with st.spinner(f"Downloading {filename} …"):
                downloaded = hf_hub_download(
                    repo_id=hf_repo,
                    filename=filename,
                    token=hf_token,
                    local_dir=LOCAL_MODEL_DIR,
                    local_dir_use_symlinks=False,
                )
                # hf_hub_download may place the file in a cache subfolder;
                # move it to the flat model/ directory if needed.
                if os.path.abspath(downloaded) != os.path.abspath(local_path):
                    import shutil
                    shutil.move(downloaded, local_path)
        paths.append(local_path)
    return paths


def ensure_models() -> None:
    """Download missing model files. Safe to call on every rerun."""
    # Skip if all models already exist locally (local dev / Docker with volume mount).
    if all(os.path.exists(os.path.join(LOCAL_MODEL_DIR, f)) for f in MODEL_FILENAMES):
        return

    try:
        hf_repo = st.secrets["models"]["HF_REPO"]
        hf_token = st.secrets["models"]["HF_TOKEN"]
    except (KeyError, FileNotFoundError):
        st.error(
            "Model files not found locally and HuggingFace secrets are not configured. "
            "Add [models] HF_REPO and HF_TOKEN to your Streamlit secrets."
        )
        st.stop()

    _download_models(hf_repo, hf_token)
