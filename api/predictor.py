
"""
Unified predictor used by both the API and the dashboard.
Loads whichever models are available and exposes a clean predict() interface.
"""

import pickle
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from src.utils.helpers import get_logger, get_project_root, get_config

logger = get_logger(__name__)
ROOT   = get_project_root()


# ──────────────────────────────────────────────────────────
#  Prediction Result Dataclass
# ──────────────────────────────────────────────────────────
class PredictionResult:
    __slots__ = ("label", "label_name", "confidence", "probabilities", "latency_ms", "model_used")

    def __init__(
        self,
        label: int,
        label_name: str,
        confidence: float,
        probabilities: Dict[str, float],
        latency_ms: float,
        model_used: str,
    ):
        self.label         = label
        self.label_name    = label_name
        self.confidence    = confidence
        self.probabilities = probabilities
        self.latency_ms    = latency_ms
        self.model_used    = model_used

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label":         self.label_name,
            "confidence":    round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "latency_ms":    round(self.latency_ms, 2),
            "model_used":    self.model_used,
        }


# ──────────────────────────────────────────────────────────
#  Predictor
# ──────────────────────────────────────────────────────────
class SentimentPredictor:
    """
    Lazy-loads the requested model tier and exposes predict() and predict_all().
    """

    def __init__(self):
        self._baseline   = None
        self._baseline_le = None
        self._lstm       = None
        self._lstm_vocab  = None
        self._lstm_le    = None
        self._transformer = None
        self._tokenizer  = None
        self._trans_le   = None
        self._cfg        = get_config()
        self._available: List[str] = []
        self._scan_available()

    # ── Discovery ────────────────────────────────────────
    def _scan_available(self) -> None:
        base = ROOT / "experiments" / "models"
        if (base / "baseline" / "pipeline.joblib").exists():
            self._available.append("baseline")
        if (base / "lstm" / "best_model.pt").exists():
            self._available.append("lstm")
        if (base / "transformer" / "best_model").exists():
            self._available.append("transformer")
        logger.info(f"Available models: {self._available}")

    @property
    def available_models(self) -> List[str]:
        return self._available

    # ── Lazy Loaders ─────────────────────────────────────
    def _load_baseline(self):
        if self._baseline is None:
            import joblib
            base = ROOT / "experiments" / "models" / "baseline"
            self._baseline    = joblib.load(base / "pipeline.joblib")
            with open(base / "label_enc.pkl", "rb") as f:
                self._baseline_le = pickle.load(f)
            logger.info("Baseline model loaded.")

    def _load_lstm(self):
        if self._lstm is None:
            import torch
            from src.models.lstm_model import BiLSTMSentiment, load_lstm_artifacts

            vocab, le, ckpt_path = load_lstm_artifacts()
            num_classes = len(le.classes_)

            model = BiLSTMSentiment(
                vocab_size    = len(vocab),
                embedding_dim = self._cfg["lstm"]["embedding_dim"],
                hidden_dim    = self._cfg["lstm"]["hidden_dim"],
                num_classes   = num_classes,
                num_layers    = self._cfg["lstm"]["num_layers"],
                dropout       = 0.0,
            )
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            model.eval()

            self._lstm      = model
            self._lstm_vocab = vocab
            self._lstm_le   = le
            logger.info("BiLSTM model loaded.")

    def _load_transformer(self):
        if self._transformer is None:
            from src.models.transformer_model import load_transformer
            base = ROOT / "experiments" / "models" / "transformer"

            self._transformer, self._tokenizer = load_transformer()
            self._transformer.eval()

            with open(base / "label_enc.pkl", "rb") as f:
                self._trans_le = pickle.load(f)
            logger.info("Transformer model loaded.")

    # ── Core Predict ─────────────────────────────────────
    def predict(
        self,
        text: str,
        model_name: Optional[str] = None,
    ) -> PredictionResult:
        if not model_name:
            model_name = self._cfg["api"]["default_model"]
        if model_name not in self._available:
            if self._available:
                model_name = self._available[-1]
            else:
                raise RuntimeError("No trained models found. Run train_all.py first.")

        t0 = time.time()

        if model_name == "baseline":
            self._load_baseline()
            probs = self._baseline.predict_proba([text])[0]
            le    = self._baseline_le

        elif model_name == "lstm":
            self._load_lstm()
            from src.data.preprocess import texts_to_sequences
            seq   = texts_to_sequences([text], self._lstm_vocab, self._cfg["data"]["max_len_lstm"])
            from src.models.lstm_model import predict_lstm
            _, probs_arr = predict_lstm(self._lstm, seq)
            probs = probs_arr[0]
            le    = self._lstm_le

        elif model_name == "transformer":
            self._load_transformer()
            from src.models.transformer_model import predict_transformer
            _, probs_arr = predict_transformer(
                self._transformer, self._tokenizer, [text],
                max_length=self._cfg["data"]["max_seq_length"],
            )
            probs = probs_arr[0]
            le    = self._trans_le

        else:
            raise ValueError(f"Unknown model: {model_name}")

        latency_ms = (time.time() - t0) * 1000
        pred_idx   = int(np.argmax(probs))
        label_name = str(le.classes_[pred_idx])

        return PredictionResult(
            label         = pred_idx,
            label_name    = label_name,
            confidence    = float(probs[pred_idx]),
            probabilities = {str(le.classes_[i]): float(p) for i, p in enumerate(probs)},
            latency_ms    = latency_ms,
            model_used    = model_name,
        )

    def predict_all(self, text: str) -> Dict[str, PredictionResult]:
        """Run prediction with every available model."""
        return {m: self.predict(text, m) for m in self._available}


# Singleton
_predictor: Optional[SentimentPredictor] = None


def get_predictor() -> SentimentPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SentimentPredictor()
    return _predictor
