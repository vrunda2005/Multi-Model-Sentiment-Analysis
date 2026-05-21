
"""
SHAP + LIME explainability for all three model tiers.
"""

from typing import List, Optional, Any, Dict
import numpy as np
import matplotlib.pyplot as plt

from src.utils.helpers import get_logger, ensure_dir, get_project_root

logger  = get_logger(__name__)
ROOT    = get_project_root()
FIG_DIR = ensure_dir(ROOT / "reports" / "figures")


# ──────────────────────────────────────────────────────────
#  Baseline: LIME + SHAP (via LinearExplainer)
# ──────────────────────────────────────────────────────────
def explain_baseline_lime(
    pipeline,
    texts: List[str],
    class_names: List[str],
    n_features: int = 10,
    n_samples: int  = 3,
) -> None:
    """Generate LIME explanations for baseline model."""
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        logger.warning("lime not installed. Run: pip install lime")
        return

    explainer = LimeTextExplainer(class_names=class_names)

    def predict_fn(texts_in):
        return pipeline.predict_proba(texts_in)

    for i, text in enumerate(texts[:n_samples]):
        exp = explainer.explain_instance(
            text,
            predict_fn,
            num_features=n_features,
            num_samples=500,
        )
        fig = exp.as_pyplot_figure()
        fig.suptitle(f"LIME Explanation — Sample {i+1}", fontsize=12, fontweight="bold")
        path = FIG_DIR / f"lime_baseline_sample_{i+1}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"LIME explanation saved → {path}")


def explain_baseline_shap(
    pipeline,
    X_background: List[str],
    X_test: List[str],
    n_samples: int = 100,
) -> None:
    """SHAP LinearExplainer for TF-IDF + Logistic Regression pipeline."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed. Run: pip install shap")
        return

    vectorizer  = pipeline.named_steps["tfidf"]
    classifier  = pipeline.named_steps["clf"]

    X_bg_vec    = vectorizer.transform(X_background[:n_samples])
    X_test_vec  = vectorizer.transform(X_test[:20])

    try:
        explainer  = shap.LinearExplainer(classifier, X_bg_vec, feature_perturbation="interventional")
        shap_vals  = explainer.shap_values(X_test_vec)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_vals,
            X_test_vec,
            feature_names=vectorizer.get_feature_names_out(),
            show=False,
            plot_type="bar",
        )
        path = FIG_DIR / "shap_baseline_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary saved → {path}")
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")


# ──────────────────────────────────────────────────────────
#  Transformer: SHAP (via partition explainer)
# ──────────────────────────────────────────────────────────
def explain_transformer_shap(
    model,
    tokenizer,
    texts: List[str],
    class_names: List[str],
    n_samples: int = 5,
) -> None:
    """SHAP Text explainer for transformer model."""
    try:
        import shap
        import torch
        import torch.nn.functional as F
    except ImportError:
        logger.warning("shap or torch not installed.")
        return

    device = next(model.parameters()).device

    def predict_fn(texts_in: List[str]) -> np.ndarray:
        enc = tokenizer(
            list(texts_in),
            padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        return F.softmax(logits, dim=-1).cpu().numpy()

    masker    = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_fn, masker, output_names=class_names)

    sample_texts = texts[:n_samples]
    shap_values  = explainer(sample_texts)

    for i in range(len(sample_texts)):
        shap.plots.text(shap_values[i], display=False)
        path = FIG_DIR / f"shap_transformer_sample_{i+1}.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP transformer plot → {path}")


# ──────────────────────────────────────────────────────────
#  LIME for Transformer
# ──────────────────────────────────────────────────────────
def explain_transformer_lime(
    model,
    tokenizer,
    texts: List[str],
    class_names: List[str],
    n_samples: int = 3,
) -> None:
    try:
        from lime.lime_text import LimeTextExplainer
        import torch
        import torch.nn.functional as F
    except ImportError:
        logger.warning("lime not installed.")
        return

    device = next(model.parameters()).device

    def predict_fn(texts_in: List[str]) -> np.ndarray:
        enc = tokenizer(
            list(texts_in),
            padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        return F.softmax(logits, dim=-1).cpu().numpy()

    explainer = LimeTextExplainer(class_names=class_names)

    for i, text in enumerate(texts[:n_samples]):
        exp  = explainer.explain_instance(text, predict_fn, num_features=12, num_samples=300)
        fig  = exp.as_pyplot_figure()
        fig.suptitle(f"LIME — Transformer Sample {i+1}", fontsize=12, fontweight="bold")
        path = FIG_DIR / f"lime_transformer_sample_{i+1}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"LIME transformer plot → {path}")
