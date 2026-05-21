
"""
Text preprocessing pipeline.
Handles cleaning, tokenization, and feature extraction for all model tiers.
"""

import re
import string
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

from src.utils.helpers import get_logger, get_config, get_project_root, ensure_dir

logger = get_logger(__name__)
ROOT = get_project_root()

# Download required NLTK data (idempotent)
for _pkg in ("stopwords", "wordnet", "punkt", "punkt_tab"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

STOP_WORDS  = set(stopwords.words("english"))
LEMMATIZER  = WordNetLemmatizer()


# ──────────────────────────────────────────────────────────
#  Core Cleaning
# ──────────────────────────────────────────────────────────
def clean_text(
    text: str,
    remove_html: bool = True,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    remove_hashtags: bool = False,
    lowercase: bool = True,
    remove_stopwords: bool = False,
    lemmatize: bool = True,
) -> str:
    """
    Full text cleaning pipeline.
    Apply steps sequentially; each step is independently toggleable.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. HTML tags
    if remove_html:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-z]+;", " ", text)   # HTML entities

    # 2. URLs
    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Twitter artifacts
    if remove_mentions:
        text = re.sub(r"@\w+", " ", text)
    if remove_hashtags:
        text = re.sub(r"#\w+", " ", text)

    # 4. Lowercase
    if lowercase:
        text = text.lower()

    # 5. Contractions expansion (basic)
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'ve": " have", "'ll": " will",
        "'d": " would", "'m": " am",
    }
    for pat, rep in contractions.items():
        text = text.replace(pat, rep)

    # 6. Special chars / punctuation (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 7. Extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 8. Tokenise for lemmatize / stopwords
    tokens = text.split()

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def batch_clean(
    texts: List[str],
    cfg: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Apply clean_text to a list of strings using project config."""
    if cfg is None:
        cfg = get_config()["preprocessing"]
    return [
        clean_text(
            t,
            remove_html=cfg.get("remove_html", True),
            remove_urls=cfg.get("remove_urls", True),
            remove_mentions=cfg.get("remove_mentions", True),
            remove_hashtags=cfg.get("remove_hashtags", False),
            lowercase=cfg.get("lowercase", True),
            remove_stopwords=cfg.get("remove_stopwords", False),
            lemmatize=cfg.get("lemmatize", True),
        )
        for t in texts
    ]


# ──────────────────────────────────────────────────────────
#  DataFrame Preprocessing
# ──────────────────────────────────────────────────────────
def preprocess_df(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Clean text column in-place and remove empty rows."""
    cfg = get_config()["preprocessing"]
    logger.info(f"Preprocessing {len(df)} rows …")
    df = df.copy()
    df["clean_text"] = batch_clean(df[text_col].tolist(), cfg)
    # Drop rows where cleaning produced empty string
    before = len(df)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} empty rows after cleaning.")
    return df


# ──────────────────────────────────────────────────────────
#  LSTM Vocabulary & Padding
# ──────────────────────────────────────────────────────────
class Vocabulary:
    """Simple word-to-index vocabulary for LSTM."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, max_vocab_size: int = 30000, min_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._freq: Dict[str, int] = {}
        self.built = False

    def build(self, texts: List[str]) -> None:
        """Build vocabulary from a list of pre-cleaned texts."""
        for text in texts:
            for word in text.split():
                self._freq[word] = self._freq.get(word, 0) + 1

        # Sort by frequency, keep top-N
        sorted_words = sorted(self._freq.items(), key=lambda x: -x[1])
        sorted_words = [
            (w, c) for w, c in sorted_words if c >= self.min_freq
        ][: self.max_vocab_size - 2]

        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        for word, _ in sorted_words:
            self.word2idx[word] = len(self.word2idx)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.built = True
        logger.info(f"Vocabulary size: {len(self.word2idx)}")

    def encode(self, text: str) -> List[int]:
        unk = self.word2idx[self.UNK_TOKEN]
        return [self.word2idx.get(w, unk) for w in text.split()]

    def __len__(self) -> int:
        return len(self.word2idx)


def pad_sequences(
    sequences: List[List[int]],
    max_len: int,
    pad_value: int = 0,
) -> np.ndarray:
    """Pad / truncate sequences to a fixed length."""
    result = np.full((len(sequences), max_len), pad_value, dtype=np.int64)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        result[i, :length] = seq[:length]
    return result


def texts_to_sequences(
    texts: List[str],
    vocab: Vocabulary,
    max_len: int = 200,
) -> np.ndarray:
    """Encode texts to padded integer sequences."""
    seqs = [vocab.encode(t) for t in texts]
    return pad_sequences(seqs, max_len=max_len)


# ──────────────────────────────────────────────────────────
#  Save / Load Processed Data
# ──────────────────────────────────────────────────────────
def save_processed(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset: str = "imdb",
) -> None:
    out = ensure_dir(ROOT / "data" / "processed" / dataset)
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out   / "val.csv",   index=False)
    test_df.to_csv(out  / "test.csv",  index=False)
    logger.info(f"Processed splits saved to {out}")


def load_processed(dataset: str = "imdb") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = ROOT / "data" / "processed" / dataset
    return (
        pd.read_csv(base / "train.csv"),
        pd.read_csv(base / "val.csv"),
        pd.read_csv(base / "test.csv"),
    )


# ──────────────────────────────────────────────────────────
#  Full Pipeline Entry-point
# ──────────────────────────────────────────────────────────
def run_preprocessing_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset: str = "imdb",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean all splits and persist to disk."""
    train_df = preprocess_df(train_df)
    val_df   = preprocess_df(val_df)
    test_df  = preprocess_df(test_df)
    save_processed(train_df, val_df, test_df, dataset=dataset)
    return train_df, val_df, test_df
