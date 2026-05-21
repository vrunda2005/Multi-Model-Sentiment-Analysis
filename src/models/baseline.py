
"""
Tier 1: Classical ML Baseline
TF-IDF + Logistic Regression / SVM / Naive Bayes / Random Forest
"""

import time
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder

from src.utils.helpers import get_logger, get_config, ensure_dir, get_project_root

logger = get_logger(__name__)
ROOT   = get_project_root()


# ──────────────────────────────────────────────────────────
#  Classifier Factory
# ──────────────────────────────────────────────────────────
def _make_classifier(cfg: Dict[str, Any]):
    """Instantiate the configured classifier."""
    name = cfg["baseline"]["model"]

    if name == "logistic_regression":
        lr_cfg = cfg["baseline"]["logistic_regression"]
        return LogisticRegression(
            C=lr_cfg["C"],
            max_iter=lr_cfg["max_iter"],
            solver=lr_cfg["solver"],
            class_weight=lr_cfg["class_weight"],
        )
    elif name == "svm":
        svm_cfg = cfg["baseline"]["svm"]
        clf = LinearSVC(C=svm_cfg["C"], max_iter=2000, class_weight="balanced")
        # Wrap in Platt scaling to get probability estimates
        return CalibratedClassifierCV(clf, cv=3)
    elif name == "naive_bayes":
        return ComplementNB(alpha=0.1)
    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=20, n_jobs=-1,
            class_weight="balanced", random_state=42
        )
    else:
        raise ValueError(f"Unknown model: '{name}'")


# ──────────────────────────────────────────────────────────
#  Build sklearn Pipeline
# ──────────────────────────────────────────────────────────
def build_baseline_pipeline(cfg: Optional[Dict[str, Any]] = None) -> Pipeline:
    """
    Return an sklearn Pipeline:
        TfidfVectorizer → Classifier
    """
    if cfg is None:
        cfg = get_config()

    tfidf_cfg = cfg["baseline"]["tfidf"]

    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        sublinear_tf=tfidf_cfg["sublinear_tf"],
        min_df=tfidf_cfg["min_df"],
        max_df=tfidf_cfg["max_df"],
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-z][a-z]+\b",
    )

    classifier = _make_classifier(cfg)

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf",   classifier),
    ])

    logger.info(f"Baseline pipeline: TF-IDF → {cfg['baseline']['model']}")
    return pipeline


# ──────────────────────────────────────────────────────────
#  Train
# ──────────────────────────────────────────────────────────
def train_baseline(
    X_train: list,
    y_train: list,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Pipeline, float]:
    """
    Fit the baseline pipeline.
    Returns (fitted_pipeline, training_time_seconds)
    """
    if cfg is None:
        cfg = get_config()

    pipeline = build_baseline_pipeline(cfg)

    logger.info("Training baseline model …")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0

    logger.info(f"Training complete in {elapsed:.2f}s")
    return pipeline, elapsed


# ──────────────────────────────────────────────────────────
#  Predict
# ──────────────────────────────────────────────────────────
def predict_baseline(
    pipeline: Pipeline,
    texts: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (predictions, probabilities).
    probabilities shape: (n_samples, n_classes)
    """
    preds = pipeline.predict(texts)
    try:
        probs = pipeline.predict_proba(texts)
    except AttributeError:
        # SVM without calibration
        scores = pipeline.decision_function(texts)
        probs = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    return preds, probs


# ──────────────────────────────────────────────────────────
#  Persist
# ──────────────────────────────────────────────────────────
def save_baseline(pipeline: Pipeline, name: str = "baseline") -> Path:
    out_dir = ensure_dir(ROOT / "experiments" / "models" / name)
    path    = out_dir / "pipeline.joblib"
    joblib.dump(pipeline, path)
    logger.info(f"Saved baseline pipeline → {path}")
    return path


def load_baseline(name: str = "baseline") -> Pipeline:
    path = ROOT / "experiments" / "models" / name / "pipeline.joblib"
    logger.info(f"Loading baseline pipeline ← {path}")
    return joblib.load(path)
