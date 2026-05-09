"""Auto-download assets at startup.

- Models: from HuggingFace Hub (private repo, token required)
- Videos: from Google Drive (public link, no auth needed)

Usage: call ensure_models() once at the top of dashboard.py.
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

# {filename: Google Drive file ID}
VIDEO_FILES = {
    "TLC00008.mp4": "1erSIs3ttONA7U5UwBc_HyZUszaJx0tY8",
}

LOCAL_MODEL_DIR = "model"
LOCAL_VIDEO_DIR = "video"


# ── Models (HuggingFace) ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _download_models(hf_repo: str, hf_token: str) -> None:
    from huggingface_hub import hf_hub_download

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    for filename in MODEL_FILENAMES:
        local_path = os.path.join(LOCAL_MODEL_DIR, filename)
        if os.path.exists(local_path):
            continue
        with st.spinner(f"Downloading {filename} …"):
            downloaded = hf_hub_download(
                repo_id=hf_repo,
                filename=filename,
                token=hf_token,
                local_dir=LOCAL_MODEL_DIR,
                local_dir_use_symlinks=False,
            )
            if os.path.abspath(downloaded) != os.path.abspath(local_path):
                shutil.move(downloaded, local_path)


# ── Videos (Google Drive) ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _download_videos() -> None:
    import gdown

    os.makedirs(LOCAL_VIDEO_DIR, exist_ok=True)
    for filename, file_id in VIDEO_FILES.items():
        local_path = os.path.join(LOCAL_VIDEO_DIR, filename)
        if os.path.exists(local_path):
            continue
        with st.spinner(f"Downloading {filename} from Google Drive …"):
            url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"
            gdown.download(url, local_path, quiet=False)
            # If gdown wrote an HTML warning page instead of the video, remove it
            if os.path.exists(local_path) and os.path.getsize(local_path) < 5 * 1024 * 1024:
                os.remove(local_path)
                st.warning(
                    f"Google Drive returned a warning page instead of {filename}. "
                    "Try opening the Drive link directly in your browser once to accept the "
                    "large-file warning, then redeploy."
                )
                st.stop()


# ── Public entry point ────────────────────────────────────────────────────────

def ensure_models() -> None:
    """Download missing models and videos. Safe to call on every rerun."""
    # Models
    if not all(os.path.exists(os.path.join(LOCAL_MODEL_DIR, f)) for f in MODEL_FILENAMES):
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
    if not all(os.path.exists(os.path.join(LOCAL_VIDEO_DIR, f)) for f in VIDEO_FILES):
        _download_videos()
