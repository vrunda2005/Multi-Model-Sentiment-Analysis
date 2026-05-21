
"""
Exploratory Data Analysis script.
Run: python -m src.data.eda
Generates word clouds, class distributions, text length plots → reports/figures/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud

from src.utils.helpers import get_logger, ensure_dir, get_project_root, get_config
from src.data.download import get_dataset
from src.data.preprocess import clean_text

logger  = get_logger("eda")
ROOT    = get_project_root()
FIG_DIR = ensure_dir(ROOT / "reports" / "figures")

plt.style.use("seaborn-v0_8-darkgrid")
BG    = "#1a1a2e"
TEXT  = "white"
CMAP  = ["#6C63FF", "#FF6B6B", "#4ECDC4", "#FFE66D"]


def _set_dark_style(ax, title=""):
    ax.set_facecolor("#16213e")
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor("rgba(255,255,255,0.1)")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")


# ─────────────────────────────────────────────────────────
#  1. Class Distribution
# ─────────────────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame, name: str = "train") -> None:
    counts = df["sentiment"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)

    # Bar
    bars = axes[0].bar(counts.index, counts.values, color=CMAP[:len(counts)], edgecolor="none")
    for bar, val in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20, f"{val:,}",
            ha="center", color=TEXT, fontsize=11, fontweight="bold"
        )
    _set_dark_style(axes[0], f"Class Distribution — {name} split")
    axes[0].set_ylabel("Count")

    # Pie
    axes[1].pie(
        counts.values,
        labels=[s.capitalize() for s in counts.index],
        autopct="%1.1f%%",
        colors=CMAP[:len(counts)],
        startangle=140,
        textprops={"color": TEXT, "fontsize": 12},
    )
    axes[1].set_title("Sentiment Proportions", fontsize=13, fontweight="bold", color=TEXT)
    axes[1].set_facecolor(BG)

    plt.tight_layout()
    path = FIG_DIR / f"class_distribution_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info(f"Saved → {path}")


# ─────────────────────────────────────────────────────────
#  2. Text Length Distribution
# ─────────────────────────────────────────────────────────
def plot_length_distribution(df: pd.DataFrame, name: str = "train") -> None:
    df = df.copy()
    df["word_count"] = df["text"].apply(lambda t: len(str(t).split()))
    df["char_count"] = df["text"].apply(len)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    sentiments = df["sentiment"].unique()

    for i, col in enumerate(["word_count", "char_count"]):
        for j, sent in enumerate(sentiments):
            subset = df[df["sentiment"] == sent][col]
            axes[i].hist(subset, bins=50, alpha=0.6,
                         color=CMAP[j % len(CMAP)], label=sent.capitalize(), edgecolor="none")
        _set_dark_style(axes[i], col.replace("_", " ").title())
        axes[i].set_xlabel("Count")
        axes[i].set_ylabel("Frequency")
        axes[i].legend(facecolor="#16213e", labelcolor=TEXT)

    fig.suptitle(f"Text Length Analysis — {name} split", color=TEXT, fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / f"text_length_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info(f"Saved → {path}")


# ─────────────────────────────────────────────────────────
#  3. Word Clouds per Sentiment
# ─────────────────────────────────────────────────────────
def plot_wordclouds(df: pd.DataFrame, name: str = "train") -> None:
    sentiments = sorted(df["sentiment"].unique())
    n = len(sentiments)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5), facecolor=BG)
    if n == 1:
        axes = [axes]

    wc_colors = [
        ["#6C63FF", "#a78bfa", "#c4b5fd"],
        ["#FF6B6B", "#f87171", "#fca5a5"],
        ["#4ECDC4", "#2dd4bf", "#99f6e4"],
    ]

    for ax, sent, palette in zip(axes, sentiments, wc_colors * 5):
        subset = df[df["sentiment"] == sent]["text"].astype(str)
        text   = " ".join(subset)
        # Clean for wordcloud
        text   = clean_text(text, lemmatize=False, remove_stopwords=True)

        wc = WordCloud(
            width=800, height=400,
            background_color="#16213e",
            colormap="cool",
            max_words=100,
            prefer_horizontal=0.8,
        ).generate(text)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{sent.capitalize()} Sentiment", color=TEXT,
                     fontsize=14, fontweight="bold")
        ax.set_facecolor(BG)

    fig.suptitle(f"Word Clouds — {name} split", color=TEXT, fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / f"wordclouds_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info(f"Saved → {path}")


# ─────────────────────────────────────────────────────────
#  4. Top N-grams
# ─────────────────────────────────────────────────────────
def plot_top_ngrams(df: pd.DataFrame, n: int = 1, top_k: int = 20, name: str = "train") -> None:
    sentiments = sorted(df["sentiment"].unique())
    ncols = len(sentiments)
    fig, axes = plt.subplots(1, ncols, figsize=(10 * ncols, 6), facecolor=BG)
    if ncols == 1:
        axes = [axes]

    for ax, sent, color in zip(axes, sentiments, CMAP):
        texts  = df[df["sentiment"] == sent]["text"].astype(str)
        tokens = " ".join(texts).lower().split()

        if n == 1:
            freqs = Counter(tokens).most_common(top_k)
        else:
            from nltk.util import ngrams as nltk_ngrams
            import nltk
            nltk.download("punkt_tab", quiet=True)
            all_ngrams = []
            for t in texts:
                toks = t.lower().split()
                all_ngrams.extend([" ".join(g) for g in nltk_ngrams(toks, n)])
            freqs = Counter(all_ngrams).most_common(top_k)

        words, counts = zip(*freqs) if freqs else ([], [])
        y_pos = range(len(words))
        ax.barh(list(y_pos), list(counts), color=color, edgecolor="none")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(list(words))
        ax.invert_yaxis()
        _set_dark_style(ax, f"Top {top_k} {'Bigrams' if n==2 else 'Words'} — {sent.capitalize()}")
        ax.set_xlabel("Frequency")

    gram_type = "bigrams" if n == 2 else "unigrams"
    path = FIG_DIR / f"top_{gram_type}_{name}.png"
    fig.suptitle(f"Top {top_k} {gram_type.title()} by Sentiment", color=TEXT,
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info(f"Saved → {path}")


# ─────────────────────────────────────────────────────────
#  Summary Stats
# ─────────────────────────────────────────────────────────
def print_summary(train_df, val_df, test_df) -> None:
    for split, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        df["word_count"] = df["text"].apply(lambda t: len(str(t).split()))
        logger.info(
            f"\n{'─'*40}\n{split}: {len(df)} samples\n"
            f"Avg words: {df['word_count'].mean():.1f} ± {df['word_count'].std():.1f}\n"
            f"Max words: {df['word_count'].max()}\n"
            f"Label distribution:\n{df['sentiment'].value_counts().to_string()}\n"
        )


# ─────────────────────────────────────────────────────────
#  Run EDA
# ─────────────────────────────────────────────────────────
def run_eda(dataset: str = "imdb", max_samples: int = 10000) -> None:
    logger.info(f"Running EDA for dataset: {dataset}")
    train_df, val_df, test_df = get_dataset(dataset=dataset, max_samples=max_samples)
    print_summary(train_df, val_df, test_df)

    for split_name, df in [("train", train_df), ("test", test_df)]:
        plot_class_distribution(df, split_name)
        plot_length_distribution(df, split_name)
        plot_wordclouds(df, split_name)
        plot_top_ngrams(df, n=1, top_k=20, name=split_name)
        plot_top_ngrams(df, n=2, top_k=15, name=split_name)

    logger.info(f"\n✓ EDA complete. Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    cfg = get_config()
    run_eda(
        dataset     = cfg["data"]["dataset"],
        max_samples = min(cfg["data"]["max_samples"] or 10000, 10000),
    )
