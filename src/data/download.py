
"""
Data download and loading utilities.
Downloads IMDb, Twitter Airline, and Amazon review datasets.
"""

import os
import re
import zipfile
import tarfile
import requests
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from src.utils.helpers import get_logger, get_config, ensure_dir, get_project_root

logger = get_logger(__name__)
ROOT = get_project_root()


# ─────────────────────────────────────────────
#  Generic downloader
# ─────────────────────────────────────────────
def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """Stream-download a file with a progress bar."""
    ensure_dir(dest.parent)
    if dest.exists():
        logger.info(f"File already exists, skipping download: {dest.name}")
        return dest
    logger.info(f"Downloading {url} → {dest}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="iB", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


# ─────────────────────────────────────────────
#  IMDb (via HuggingFace datasets)
# ─────────────────────────────────────────────
def load_imdb(max_samples: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load IMDb dataset from HuggingFace.
    Returns: (train_df, val_df, test_df) with columns [text, label, sentiment]
    """
    logger.info("Loading IMDb dataset from HuggingFace …")
    raw = load_dataset("imdb")

    train_df = pd.DataFrame(raw["train"]).rename(columns={"label": "label"})
    test_df  = pd.DataFrame(raw["test"]).rename(columns={"label": "label"})

    # Shuffle and optionally limit
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df  = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    if max_samples:
        train_df = train_df.head(max_samples)
        test_df  = test_df.head(max_samples // 4)

    # Carve out a validation split from train
    val_size = int(0.15 * len(train_df))
    val_df   = train_df[:val_size].reset_index(drop=True)
    train_df = train_df[val_size:].reset_index(drop=True)

    # Human-readable sentiment column
    label_map = {0: "negative", 1: "positive"}
    for df in (train_df, val_df, test_df):
        df["sentiment"] = df["label"].map(label_map)

    logger.info(
        f"IMDb — train:{len(train_df)} | val:{len(val_df)} | test:{len(test_df)}"
    )
    return train_df, val_df, test_df


# ─────────────────────────────────────────────
#  Twitter US Airline Sentiment (via HuggingFace)
# ─────────────────────────────────────────────
def load_twitter(max_samples: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load Twitter Airline Sentiment dataset.
    Returns: (train_df, val_df, test_df) with columns [text, label, sentiment]
    """
    logger.info("Loading Twitter Airline Sentiment dataset …")

    try:
        raw = load_dataset("tweet_eval", "sentiment")
        df = pd.DataFrame(raw["train"])
        test_df_raw = pd.DataFrame(raw["test"])
    except Exception:
        # Fallback: load from local CSV if already downloaded
        csv_path = ROOT / "data" / "raw" / "twitter_airline.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                "Twitter dataset not found. Download from Kaggle: "
                "https://www.kaggle.com/crowdflower/twitter-airline-sentiment"
            )
        full_df = pd.read_csv(csv_path)
        df = full_df[["text", "airline_sentiment"]].copy()
        df["label"] = df["airline_sentiment"].map(
            {"negative": 0, "neutral": 1, "positive": 2}
        )
        df = df[["text", "label"]].dropna()
        split = int(0.85 * len(df))
        train_df = df[:split].reset_index(drop=True)
        test_df_raw = df[split:].reset_index(drop=True)

    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if max_samples:
        df = df.head(max_samples)

    val_size = int(0.15 * len(df))
    val_df   = df[:val_size].reset_index(drop=True)
    train_df = df[val_size:].reset_index(drop=True)

    for frame in (train_df, val_df, test_df_raw):
        frame["sentiment"] = frame["label"].map(label_map)

    logger.info(
        f"Twitter — train:{len(train_df)} | val:{len(val_df)} | test:{len(test_df_raw)}"
    )
    return train_df, val_df, test_df_raw


# ─────────────────────────────────────────────
#  Unified loader
# ─────────────────────────────────────────────
def get_dataset(
    dataset: str = "imdb",
    max_samples: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Unified entry-point: returns (train, val, test) DataFrames.
    Each DataFrame has at minimum: text, label, sentiment columns.
    """
    loaders = {
        "imdb":    load_imdb,
        "twitter": load_twitter,
    }
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(loaders)}")

    train, val, test = loaders[dataset](max_samples=max_samples)

    # Persist raw split to disk for reproducibility
    raw_dir = ensure_dir(ROOT / "data" / "raw" / dataset)
    train.to_csv(raw_dir / "train.csv", index=False)
    val.to_csv(raw_dir  / "val.csv",   index=False)
    test.to_csv(raw_dir / "test.csv",  index=False)
    logger.info(f"Raw splits saved to {raw_dir}")

    return train, val, test


if __name__ == "__main__":
    cfg = get_config()
    get_dataset(
        dataset=cfg["data"]["dataset"],
        max_samples=cfg["data"]["max_samples"],
    )
