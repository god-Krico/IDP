"""Auto-download custom model and video files from HuggingFace Hub at startup.

Usage: call ensure_models() once at the top of dashboard.py.
Assets are cached by st.cache_resource so they download only once per session.
Secrets are read from st.secrets["models"]["HF_REPO"] and ["HF_TOKEN"].
"""

import os
import shutil
import streamlit as st

MODEL_FILENAMES = [
    "finetuned_multi.pt",
    "mould_detection.pt",
    "phase_detection.pt",
    "ppe_best.pt",
    "progress.pt",
    "proximity_best.pt",
]

VIDEO_FILENAMES = [
    "TLC00008.mp4",
    "trial_vid.mp4",
]

LOCAL_MODEL_DIR = "model"
LOCAL_VIDEO_DIR = "video"


def _download_file(hf_hub_download, hf_repo: str, hf_token: str,
                   filename: str, local_dir: str) -> None:
    local_path = os.path.join(local_dir, filename)
    if os.path.exists(local_path):
        return
    with st.spinner(f"Downloading {filename} …"):
        downloaded = hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            token=hf_token,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        if os.path.abspath(downloaded) != os.path.abspath(local_path):
            shutil.move(downloaded, local_path)


@st.cache_resource(show_spinner=False)
def _download_assets(hf_repo: str, hf_token: str) -> None:
    from huggingface_hub import hf_hub_download

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    os.makedirs(LOCAL_VIDEO_DIR, exist_ok=True)

    for filename in MODEL_FILENAMES:
        _download_file(hf_hub_download, hf_repo, hf_token, filename, LOCAL_MODEL_DIR)

    for filename in VIDEO_FILENAMES:
        _download_file(hf_hub_download, hf_repo, hf_token, filename, LOCAL_VIDEO_DIR)


def ensure_models() -> None:
    """Download missing models and videos. Safe to call on every rerun."""
    all_models_present = all(
        os.path.exists(os.path.join(LOCAL_MODEL_DIR, f)) for f in MODEL_FILENAMES
    )
    all_videos_present = all(
        os.path.exists(os.path.join(LOCAL_VIDEO_DIR, f)) for f in VIDEO_FILENAMES
    )
    if all_models_present and all_videos_present:
        return

    try:
        hf_repo = st.secrets["models"]["HF_REPO"]
        hf_token = st.secrets["models"]["HF_TOKEN"]
    except (KeyError, FileNotFoundError):
        st.error(
            "Asset files not found locally and HuggingFace secrets are not configured. "
            "Add [models] HF_REPO and HF_TOKEN to your Streamlit secrets."
        )
        st.stop()

    _download_assets(hf_repo, hf_token)
