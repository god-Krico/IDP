"""Auto-download assets at startup.

- Models: from HuggingFace Hub (private repo, token required)
- HF videos: from HuggingFace Hub (same repo, token required)
- GDrive videos: from Google Drive (public link, no auth needed)

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

# Videos stored on HuggingFace (root of repo, same token as models)
HF_VIDEO_FILENAMES = [
    "trial_vid.mp4",
]

# Videos on Google Drive {filename: file_id}
GDRIVE_VIDEO_FILES = {
    "TLC00008.mp4": "1erSIs3ttONA7U5UwBc_HyZUszaJx0tY8",
}

# Combined list of local filenames — used only for existence checks
ALL_VIDEO_FILENAMES = list(GDRIVE_VIDEO_FILES.keys()) + HF_VIDEO_FILENAMES

LOCAL_MODEL_DIR = "model"
LOCAL_VIDEO_DIR = "video"


# ── Models (HuggingFace) ──────────────────────────────────────────────────────

def _hf_download_file(hf_repo: str, hf_token: str, filename: str, dest_dir: str) -> None:
    from huggingface_hub import hf_hub_download

    os.makedirs(dest_dir, exist_ok=True)
    local_path = os.path.join(dest_dir, filename)
    if os.path.exists(local_path):
        return
    with st.spinner(f"Downloading {filename} …"):
        downloaded = hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            token=hf_token,
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
        )
        if os.path.abspath(downloaded) != os.path.abspath(local_path):
            shutil.move(downloaded, local_path)


@st.cache_resource(show_spinner=False)
def _download_models(hf_repo: str, hf_token: str) -> None:
    for filename in MODEL_FILENAMES:
        _hf_download_file(hf_repo, hf_token, filename, LOCAL_MODEL_DIR)


@st.cache_resource(show_spinner=False)
def _download_hf_videos(hf_repo: str, hf_token: str) -> None:
    for filename in HF_VIDEO_FILENAMES:
        _hf_download_file(hf_repo, hf_token, filename, LOCAL_VIDEO_DIR)



# ── Videos (Google Drive) ─────────────────────────────────────────────────────

def _gdrive_download(file_id: str, output_path: str) -> None:
    """Download a public Google Drive file using the usercontent endpoint (no confirmation page)."""
    import requests

    url = (
        f"https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&authuser=0&confirm=t"
    )
    with requests.Session() as session:
        resp = session.get(url, stream=True, timeout=600)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)

    # Sanity check — Google sometimes returns an HTML error page for restricted files
    size = os.path.getsize(output_path)
    if size < 1024 * 1024:
        os.remove(output_path)
        raise RuntimeError(
            f"Downloaded file is only {size} bytes — likely an HTML error page. "
            "Make sure the Drive file is shared as 'Anyone with the link'."
        )


@st.cache_resource(show_spinner=False)
def _download_gdrive_videos() -> None:
    os.makedirs(LOCAL_VIDEO_DIR, exist_ok=True)
    for filename, file_id in GDRIVE_VIDEO_FILES.items():
        local_path = os.path.join(LOCAL_VIDEO_DIR, filename)
        if os.path.exists(local_path):
            continue
        with st.spinner(f"Downloading {filename} from Google Drive …"):
            try:
                _gdrive_download(file_id, local_path)
            except Exception as e:
                st.error(f"Could not download {filename}: {e}")
                st.stop()


# ── Public entry point ────────────────────────────────────────────────────────

def ensure_models() -> None:
    """Download missing models and videos. Safe to call on every rerun."""
    # Resolve HF credentials (needed for models + HF videos)
    hf_repo = hf_token = None
    needs_hf = (
        not all(os.path.exists(os.path.join(LOCAL_MODEL_DIR, f)) for f in MODEL_FILENAMES)
        or not all(os.path.exists(os.path.join(LOCAL_VIDEO_DIR, f)) for f in HF_VIDEO_FILENAMES)
    )

    if needs_hf:
        try:
            hf_repo = st.secrets["models"]["HF_REPO"]
            hf_token = st.secrets["models"]["HF_TOKEN"]
        except (KeyError, FileNotFoundError):
            st.error(
                "Model files not found locally and HuggingFace secrets are not configured. "
                "Add [models] HF_REPO and HF_TOKEN to your Streamlit secrets."
            )
            st.stop()

    # Models from HuggingFace
    if not all(os.path.exists(os.path.join(LOCAL_MODEL_DIR, f)) for f in MODEL_FILENAMES):
        _download_models(hf_repo, hf_token)

    # Videos from HuggingFace
    if not all(os.path.exists(os.path.join(LOCAL_VIDEO_DIR, f)) for f in HF_VIDEO_FILENAMES):
        _download_hf_videos(hf_repo, hf_token)

    # Videos from Google Drive
    if not all(os.path.exists(os.path.join(LOCAL_VIDEO_DIR, f)) for f in GDRIVE_VIDEO_FILES.keys()):
        _download_gdrive_videos()
