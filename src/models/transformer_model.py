
"""
Tier 3: Fine-tune DistilBERT (or any HuggingFace model) for sentiment classification.
Uses HuggingFace Trainer API with mixed-precision support.
"""

import json
import time
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.preprocessing import LabelEncoder
# NOTE: `evaluate` is imported lazily inside compute_metrics()
# to avoid crashing when only baseline/lstm tiers are trained.

from src.utils.helpers import get_logger, get_config, ensure_dir, get_project_root, get_device

logger = get_logger(__name__)
ROOT   = get_project_root()


# ──────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────
class TransformerDataset(Dataset):
    """HuggingFace-compatible PyTorch dataset."""

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ──────────────────────────────────────────────────────────
#  Tokenization
# ──────────────────────────────────────────────────────────
def tokenize_texts(
    texts: List[str],
    tokenizer,
    max_length: int = 256,
) -> Dict[str, torch.Tensor]:
    """Batch tokenize a list of texts."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


# ──────────────────────────────────────────────────────────
#  Build Model & Tokenizer
# ──────────────────────────────────────────────────────────
def build_transformer_model(
    num_labels: int,
    cfg: Optional[Dict[str, Any]] = None,
):
    if cfg is None:
        cfg = get_config()

    model_name = cfg["transformer"]["model_name"]
    logger.info(f"Loading pretrained model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    logger.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    return tokenizer, model


# ──────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    """Lazy-import evaluate so it only loads during transformer training."""
    import evaluate as hf_evaluate
    accuracy_metric = hf_evaluate.load("accuracy")
    f1_metric       = hf_evaluate.load("f1")

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc   = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1    = f1_metric.compute(
        predictions=preds, references=labels, average="weighted"
    )["f1"]
    return {"accuracy": acc, "f1": f1}


# ──────────────────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────────────────
def train_transformer(
    train_texts: List[str],
    val_texts:   List[str],
    train_labels: List[int],
    val_labels:   List[int],
    num_labels: int,
    cfg: Optional[Dict[str, Any]] = None,
    save_dir: Optional[Path]      = None,
) -> Tuple[Any, Any]:
    """
    Fine-tune a transformer model.
    Returns (trained_model, tokenizer)
    """
    if cfg is None:
        cfg = get_config()

    tc = cfg["transformer"]

    if save_dir is None:
        save_dir = ensure_dir(ROOT / "experiments" / "models" / "transformer")

    tokenizer, model = build_transformer_model(num_labels, cfg)

    logger.info("Tokenizing datasets …")
    max_len = cfg["data"]["max_seq_length"]
    train_enc = tokenizer(
        train_texts, padding=True, truncation=True,
        max_length=max_len, return_tensors="pt"
    )
    val_enc = tokenizer(
        val_texts, padding=True, truncation=True,
        max_length=max_len, return_tensors="pt"
    )

    train_dataset = TransformerDataset(train_enc, train_labels)
    val_dataset   = TransformerDataset(val_enc,   val_labels)

    training_args = TrainingArguments(
        output_dir                  = str(save_dir / "checkpoints"),
        num_train_epochs            = tc["epochs"],
        per_device_train_batch_size = tc["batch_size"],
        per_device_eval_batch_size  = tc["batch_size"],
        learning_rate               = tc["learning_rate"],
        warmup_ratio                = tc["warmup_ratio"],
        weight_decay                = tc["weight_decay"],
        max_grad_norm               = tc["max_grad_norm"],
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        fp16                        = tc.get("fp16", False),
        logging_dir                 = str(save_dir / "logs"),
        logging_steps               = 50,
        report_to                   = "none",   # disable wandb/hub
        seed                        = 42,
        dataloader_num_workers      = 0,
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting transformer fine-tuning …")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.1f}s")

    # Save best model
    model.save_pretrained(save_dir / "best_model")
    tokenizer.save_pretrained(save_dir / "best_model")
    logger.info(f"Best model saved → {save_dir / 'best_model'}")

    return model, tokenizer


# ──────────────────────────────────────────────────────────
#  Inference
# ──────────────────────────────────────────────────────────
def predict_transformer(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 256,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch inference. Returns (predictions, probabilities).
    """
    import torch
    import torch.nn.functional as F

    device = get_device()
    model  = model.to(device)
    model.eval()

    all_preds, all_probs = [], []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
            preds  = probs.argmax(axis=-1)
            all_probs.append(probs)
            all_preds.append(preds)

    return np.concatenate(all_preds), np.vstack(all_probs)


# ──────────────────────────────────────────────────────────
#  Load Saved Model
# ──────────────────────────────────────────────────────────
def load_transformer(name: str = "transformer"):
    """Load saved fine-tuned transformer from disk."""
    base = ROOT / "experiments" / "models" / name / "best_model"
    logger.info(f"Loading transformer ← {base}")
    tokenizer = AutoTokenizer.from_pretrained(str(base))
    model     = AutoModelForSequenceClassification.from_pretrained(str(base))
    return model, tokenizer
