
"""
Evaluation framework: metrics, confusion matrix, error analysis, model comparison.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.utils.helpers import get_logger, ensure_dir, get_project_root

logger  = get_logger(__name__)
ROOT    = get_project_root()
FIG_DIR = ensure_dir(ROOT / "reports" / "figures")
MET_DIR = ensure_dir(ROOT / "reports" / "metrics")

plt.style.use("seaborn-v0_8-darkgrid")
PALETTE = ["#6C63FF", "#FF6B6B", "#4ECDC4", "#FFE66D"]


# ──────────────────────────────────────────────────────────
#  Core Metrics
# ──────────────────────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    latency_s: float = 0.0,
    n_samples: int   = 0,
) -> Dict[str, Any]:
    """
    Compute a comprehensive metrics dict for one model tier.
    """
    n_classes = len(np.unique(y_true))
    avg       = "binary" if n_classes == 2 else "weighted"

    metrics: Dict[str, Any] = {
        "accuracy":           round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_weighted":        round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_macro":           round(float(f1_score(y_true, y_pred, average="macro",    zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted":    round(float(recall_score(y_true, y_pred, average="weighted",    zero_division=0)), 4),
        "latency_ms_per_sample": round(latency_s / max(n_samples, 1) * 1000, 3),
        "n_samples":          n_samples,
        "classification_report": classification_report(y_true, y_pred, target_names=labels, zero_division=0),
    }

    # AUC-ROC (binary: use positive class prob; multiclass: OvR)
    if y_prob is not None:
        try:
            if n_classes == 2:
                metrics["roc_auc"] = round(
                    float(roc_auc_score(y_true, y_prob[:, 1])), 4
                )
            else:
                metrics["roc_auc"] = round(
                    float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")), 4
                )
        except Exception:
            metrics["roc_auc"] = None

    return metrics


def save_metrics(metrics: Dict[str, Any], name: str) -> Path:
    """Persist metrics JSON (excluding long text fields)."""
    out = MET_DIR / f"{name}_metrics.json"
    saveable = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(out, "w") as f:
        json.dump(saveable, f, indent=2)
    logger.info(f"Metrics saved → {out}")
    return out


# ──────────────────────────────────────────────────────────
#  Confusion Matrix
# ──────────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    name: str,
    normalize: bool = True,
) -> Path:
    cm  = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(f"{name} — Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    path = FIG_DIR / f"{name}_confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {path}")
    return path


# ──────────────────────────────────────────────────────────
#  Model Comparison Bar Chart
# ──────────────────────────────────────────────────────────
def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "f1_weighted",
) -> Path:
    """
    Bar chart comparing multiple models on a given metric.
    results = {"baseline": {...}, "lstm": {...}, "transformer": {...}}
    """
    names  = list(results.keys())
    values = [results[n].get(metric, 0) for n in names]
    colors = PALETTE[: len(names)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=1.5, zorder=3)
    ax.set_ylim(max(0, min(values) - 0.1), min(1.02, max(values) + 0.08))
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.yaxis.grid(True, zorder=0, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    plt.tight_layout()
    path = FIG_DIR / f"model_comparison_{metric}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Comparison chart saved → {path}")
    return path


# ──────────────────────────────────────────────────────────
#  Training Curves
# ──────────────────────────────────────────────────────────
def plot_training_curves(
    history: Dict[str, list],
    name: str,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (loss_key, acc_key), title in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc")],
        ["Loss", "Accuracy"],
    ):
        epochs = range(1, len(history[loss_key]) + 1)
        ax.plot(epochs, history[loss_key], "o-", color=PALETTE[0], label="Train")
        ax.plot(epochs, history[acc_key],  "s--", color=PALETTE[1], label="Validation")
        ax.set_title(f"{name} — {title}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()

    # Rewire: first subplot is loss, second is accuracy
    axes[0].set_ylabel("Loss")
    axes[0].lines[0].set_label("Train Loss")
    axes[0].lines[1].set_label("Val Loss")
    axes[1].set_ylabel("Accuracy")
    axes[1].lines[0].set_label("Train Acc")
    axes[1].lines[1].set_label("Val Acc")
    for ax in axes:
        ax.legend()

    plt.tight_layout()
    path = FIG_DIR / f"{name}_training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────
#  Error Analysis
# ──────────────────────────────────────────────────────────
def error_analysis(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    label_names: List[str],
    name: str,
    n_worst: int = 20,
) -> pd.DataFrame:
    """
    Build a DataFrame of misclassified samples sorted by prediction confidence.
    High-confidence wrong predictions are the most informative.
    """
    mask    = y_true != y_pred
    indices = np.where(mask)[0]

    errors = pd.DataFrame({
        "text":           [texts[i] for i in indices],
        "true_label":     [label_names[y_true[i]] for i in indices],
        "pred_label":     [label_names[y_pred[i]] for i in indices],
        "confidence":     [float(y_prob[i].max()) for i in indices],
    })

    errors = errors.sort_values("confidence", ascending=False).head(n_worst)

    out = MET_DIR / f"{name}_error_analysis.csv"
    errors.to_csv(out, index=False)
    logger.info(f"Error analysis saved → {out} ({len(errors)} samples)")
    return errors


# ──────────────────────────────────────────────────────────
#  Full Evaluation Pipeline
# ──────────────────────────────────────────────────────────
def full_evaluation(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    texts: List[str],
    label_names: List[str],
    latency_s: float = 0.0,
) -> Dict[str, Any]:
    """
    Run all evaluation steps for one model and persist artifacts.
    """
    n = len(y_true)
    logger.info(f"\n{'='*50}\nEvaluating: {name}\n{'='*50}")

    metrics = compute_metrics(y_true, y_pred, y_prob, label_names, latency_s, n)
    logger.info(f"\n{metrics['classification_report']}")
    save_metrics(metrics, name)

    plot_confusion_matrix(y_true, y_pred, label_names, name)

    if y_prob is not None:
        error_analysis(texts, y_true, y_pred, y_prob, label_names, name)

    return metrics
