
"""
Tier 2: Bidirectional LSTM with optional GloVe embeddings.
Architecture: Embedding → BiLSTM → Attention → FC → Softmax
"""

import os
import math
import time
import json
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.helpers import get_logger, get_config, ensure_dir, get_project_root, get_device
from src.data.preprocess import Vocabulary, texts_to_sequences

logger = get_logger(__name__)
ROOT   = get_project_root()


# ──────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────
class SentimentDataset(Dataset):
    """PyTorch Dataset wrapping padded integer sequences."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(sequences, dtype=torch.long)
        self.y = torch.tensor(labels,    dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ──────────────────────────────────────────────────────────
#  Attention Layer
# ──────────────────────────────────────────────────────────
class AttentionLayer(nn.Module):
    """Additive (Bahdanau-style) self-attention over LSTM output."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden*2)
        scores = self.attn(lstm_out).squeeze(-1)          # (batch, seq_len)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1) # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)         # (batch, hidden*2)
        return context


# ──────────────────────────────────────────────────────────
#  BiLSTM Model
# ──────────────────────────────────────────────────────────
class BiLSTMSentiment(nn.Module):
    """
    Bidirectional LSTM with:
        - Optional pre-trained GloVe embeddings
        - Self-attention pooling
        - Dropout regularization
        - FC classification head
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int  = 2,
        dropout: float   = 0.3,
        pad_idx: int     = 0,
        pretrained_emb: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Embedding
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx
        )
        if pretrained_emb is not None:
            self.embedding.weight = nn.Parameter(pretrained_emb)
            logger.info("Loaded pre-trained GloVe embeddings.")

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention
        self.attention = AttentionLayer(hidden_dim)

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb  = self.dropout(self.embedding(x))        # (B, L, E)
        out, _ = self.lstm(emb)                        # (B, L, H*2)
        ctx  = self.attention(out)                     # (B, H*2)
        ctx  = self.dropout(ctx)
        return self.fc(ctx)                            # (B, num_classes)


# ──────────────────────────────────────────────────────────
#  GloVe Loader
# ──────────────────────────────────────────────────────────
def load_glove_embeddings(
    glove_path: str,
    vocab: Vocabulary,
    embedding_dim: int,
) -> torch.Tensor:
    """
    Load GloVe vectors and return an embedding matrix aligned to vocab.
    Initialises OOV words with random uniform vectors.
    """
    logger.info(f"Loading GloVe from {glove_path} …")
    glove: Dict[str, np.ndarray] = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            glove[word] = vec

    scale  = 1.0 / math.sqrt(embedding_dim)
    matrix = np.random.uniform(-scale, scale, (len(vocab), embedding_dim)).astype(np.float32)
    matrix[0] = 0.0  # PAD stays zero

    hits = 0
    for word, idx in vocab.word2idx.items():
        if word in glove:
            matrix[idx] = glove[word]
            hits += 1

    coverage = hits / max(len(vocab) - 2, 1) * 100
    logger.info(f"GloVe coverage: {hits}/{len(vocab)-2} ({coverage:.1f}%)")
    return torch.tensor(matrix)


# ──────────────────────────────────────────────────────────
#  Build Model
# ──────────────────────────────────────────────────────────
def build_lstm_model(
    vocab: Vocabulary,
    num_classes: int,
    cfg: Optional[Dict[str, Any]] = None,
) -> BiLSTMSentiment:
    if cfg is None:
        cfg = get_config()

    lc  = cfg["lstm"]
    emb = None

    if lc["use_pretrained_embeddings"]:
        glove_path = ROOT / "data" / "embeddings" / lc["glove_file"]
        if glove_path.exists():
            emb = load_glove_embeddings(str(glove_path), vocab, lc["embedding_dim"])
        else:
            logger.warning(
                f"GloVe file not found at {glove_path}. "
                "Training with random embeddings."
            )

    model = BiLSTMSentiment(
        vocab_size    = len(vocab),
        embedding_dim = lc["embedding_dim"],
        hidden_dim    = lc["hidden_dim"],
        num_classes   = num_classes,
        num_layers    = lc["num_layers"],
        dropout       = lc["dropout"],
        pretrained_emb= emb,
    )
    return model


# ──────────────────────────────────────────────────────────
#  Training Loop
# ──────────────────────────────────────────────────────────
def train_lstm(
    model: BiLSTMSentiment,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Optional[Dict[str, Any]] = None,
    save_dir: Optional[Path]      = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = get_config()

    lc     = cfg["lstm"]
    device = get_device()
    model  = model.to(device)

    optimizer = Adam(model.parameters(), lr=lc["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=1, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience_ctr = 0
    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }

    if save_dir is None:
        save_dir = ensure_dir(ROOT / "experiments" / "models" / "lstm")

    for epoch in range(1, lc["epochs"] + 1):
        # ── Train ──
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(1) == y_batch).sum().item()
            total      += len(y_batch)

        train_loss = total_loss / total
        train_acc  = correct / total

        # ── Validate ──
        val_loss, val_acc = _evaluate_lstm(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch {epoch:02d}/{lc['epochs']} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}"
        )

        # ── Checkpoint ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            logger.info(f"  ✔ New best val_acc={best_val_acc:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= lc["patience"]:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    # Save training history
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"LSTM training done. Best val_acc={best_val_acc:.4f}")
    return history


def _evaluate_lstm(
    model, loader: DataLoader, criterion, device: str
) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(1) == y_batch).sum().item()
            total      += len(y_batch)
    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────
#  Inference
# ──────────────────────────────────────────────────────────
def predict_lstm(
    model: BiLSTMSentiment,
    sequences: np.ndarray,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (predictions, probabilities)."""
    device  = get_device()
    model   = model.to(device)
    model.eval()
    all_preds, all_probs = [], []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.tensor(sequences[i: i + batch_size], dtype=torch.long).to(device)
            logits = model(batch)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
            preds  = probs.argmax(axis=-1)
            all_probs.append(probs)
            all_preds.append(preds)

    return np.concatenate(all_preds), np.vstack(all_probs)


# ──────────────────────────────────────────────────────────
#  Save / Load
# ──────────────────────────────────────────────────────────
def save_lstm_artifacts(
    model: BiLSTMSentiment,
    vocab: Vocabulary,
    label_encoder,
    name: str = "lstm",
) -> Path:
    out = ensure_dir(ROOT / "experiments" / "models" / name)
    torch.save(model.state_dict(), out / "best_model.pt")
    with open(out / "vocab.pkl",    "wb") as f:
        pickle.dump(vocab, f)
    with open(out / "label_enc.pkl","wb") as f:
        pickle.dump(label_encoder, f)
    logger.info(f"LSTM artifacts saved → {out}")
    return out


def load_lstm_artifacts(name: str = "lstm"):
    base = ROOT / "experiments" / "models" / name
    with open(base / "vocab.pkl",    "rb") as f:
        vocab = pickle.load(f)
    with open(base / "label_enc.pkl","rb") as f:
        label_enc = pickle.load(f)
    return vocab, label_enc, base / "best_model.pt"
