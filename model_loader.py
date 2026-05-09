"""Auto-download assets from HuggingFace Hub (models) and Google Drive (videos).

Usage: call ensure_models() once at the top of dashboard.py.
Models: downloaded from HuggingFace — secrets HF_REPO and HF_TOKEN required.
Videos: downloaded from a public Google Drive folder — no secrets needed.
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

GDRIVE_FOLDER_ID = "1NQAlG_hDoZryA7yakeKaoU3qbSBJHctZ"

LOCAL_MODEL_DIR = "model"
LOCAL_VIDEO_DIR = "video"


# ── Models (HuggingFace) ──────────────────────────────────────────────────────

def _hf_download_file(hf_hub_download, hf_repo, hf_token, filename, local_dir):
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
def _download_models(hf_repo: str, hf_token: str) -> None:
    from huggingface_hub import hf_hub_download

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    for filename in MODEL_FILENAMES:
        _hf_download_file(hf_hub_download, hf_repo, hf_token, filename, LOCAL_MODEL_DIR)


# ── Videos (Google Drive) ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _download_videos() -> None:
    import gdown

    os.makedirs(LOCAL_VIDEO_DIR, exist_ok=True)

    missing = [f for f in VIDEO_FILENAMES
               if not os.path.exists(os.path.join(LOCAL_VIDEO_DIR, f))]
    if not missing:
        return

    with st.spinner("Downloading demo videos from Google Drive …"):
        try:
            gdown.download_folder(
                id=GDRIVE_FOLDER_ID,
                output=LOCAL_VIDEO_DIR,
                quiet=False,
                use_cookies=False,
            )
        except Exception as e:
            st.warning(f"Could not download videos from Google Drive: {e}")


# ── Public entry point ────────────────────────────────────────────────────────

def ensure_models() -> None:
    """Download missing models and videos. Safe to call on every rerun."""
    # Models
    all_models_present = all(
        os.path.exists(os.path.join(LOCAL_MODEL_DIR, f)) for f in MODEL_FILENAMES
    )
    if not all_models_present:
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

    # Videos
    all_videos_present = all(
        os.path.exists(os.path.join(LOCAL_VIDEO_DIR, f)) for f in VIDEO_FILENAMES
    )
    if not all_videos_present:
        _download_videos()
