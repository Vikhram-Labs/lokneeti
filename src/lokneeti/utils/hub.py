"""
lokneeti.utils.hub
==================
HuggingFace Hub integration utilities for publishing models,
tokenizers, datasets and model cards.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, login, ModelCard, DatasetCard
from huggingface_hub.utils import RepositoryNotFoundError

from lokneeti.utils.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Authentication                                                                #
# --------------------------------------------------------------------------- #
def hub_login(token: Optional[str] = None) -> None:
    """
    Authenticate with HuggingFace Hub.

    Reads HF_TOKEN from environment if token is not supplied.
    """
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN not found. Set it in .env or pass token= explicitly."
        )
    login(token=token)
    log.info("✅ Authenticated with HuggingFace Hub")


# --------------------------------------------------------------------------- #
# Model Publishing                                                              #
# --------------------------------------------------------------------------- #
def push_model_to_hub(
    local_model_dir: str | Path,
    repo_id: str,
    commit_message: str = "Upload Lokneeti-3B QLoRA adapter",
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """
    Push a model (or adapter) directory to the Hub.

    Args:
        local_model_dir: Path to the saved model/adapter directory.
        repo_id:         Full repo id, e.g. 'vikhram-labs/Lokneeti-3B'.
        commit_message:  Commit message for the upload.
        private:         Whether to create a private repository.
        token:           HF token (falls back to env HF_TOKEN).

    Returns:
        URL of the published model repository.
    """
    token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    local_model_dir = Path(local_model_dir)

    # Ensure repo exists
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        log.info(f"Repository {repo_id!r} already exists — uploading files.")
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private)
        log.info(f"Created new repository: {repo_id!r}")

    api.upload_folder(
        folder_path=str(local_model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    url = f"https://huggingface.co/{repo_id}"
    log.info(f"✅ Model pushed to: {url}")
    return url


# --------------------------------------------------------------------------- #
# Dataset Publishing                                                            #
# --------------------------------------------------------------------------- #
def push_dataset_to_hub(
    local_dataset_dir: str | Path,
    repo_id: str,
    commit_message: str = "Upload Lokneeti governance dataset",
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """
    Push a JSONL/Parquet dataset directory to the Hub.

    Args:
        local_dataset_dir: Path to the dataset directory.
        repo_id:           Full dataset repo id.
        commit_message:    Commit message.
        private:           Whether to create a private repository.
        token:             HF token.

    Returns:
        URL of the published dataset repository.
    """
    token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    local_dataset_dir = Path(local_dataset_dir)

    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        log.info(f"Dataset repo {repo_id!r} already exists — uploading files.")
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private)
        log.info(f"Created dataset repository: {repo_id!r}")

    api.upload_folder(
        folder_path=str(local_dataset_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    log.info(f"✅ Dataset pushed to: {url}")
    return url


# --------------------------------------------------------------------------- #
# Model Card                                                                    #
# --------------------------------------------------------------------------- #
def push_model_card(
    card_path: str | Path,
    repo_id: str,
    token: Optional[str] = None,
) -> None:
    """Push a README.md model card to an existing Hub repo."""
    token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update model card",
    )
    log.info(f"✅ Model card pushed to: https://huggingface.co/{repo_id}")
