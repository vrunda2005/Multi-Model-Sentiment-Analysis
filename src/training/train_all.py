
"""
Master training script — trains all three model tiers and saves a comparison report.
Run: python -m src.training.train_all
"""

import sys
import time
import json
import pickle
import argparse
import numpy as np
from pathlib import Path

# ── ensure project root is on sys.path ──
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torch.utils.data import DataLoader

from src.utils.helpers import get_config, get_logger, set_seed, ensure_dir, get_project_root, device_info
from src.data.download import get_dataset
from src.data.preprocess import (
    run_preprocessing_pipeline, load_processed,
    Vocabulary, texts_to_sequences
)
from src.models.baseline import train_baseline, predict_baseline, save_baseline
from src.models.lstm_model import (
    build_lstm_model, train_lstm, predict_lstm, save_lstm_artifacts,
    SentimentDataset
)
from src.models.transformer_model import train_transformer, predict_transformer
from src.evaluation.evaluate import full_evaluation, plot_model_comparison, save_metrics

logger = get_logger("train_all")
ROOT   = get_project_root()


def parse_args():
    parser = argparse.ArgumentParser(description="Train all sentiment analysis models")
    parser.add_argument("--tiers",   nargs="+",
                        choices=["baseline", "lstm", "transformer", "all"],
                        default=["all"],
                        help="Which model tiers to train")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset from config")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading data (use cached processed data)")
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = get_config()
    seed   = cfg["project"]["seed"]
    set_seed(seed)

    tiers  = args.tiers
    if "all" in tiers:
        tiers = ["baseline", "lstm", "transformer"]

    dataset = args.dataset or cfg["data"]["dataset"]
    logger.info(f"Device info: {device_info()}")
    logger.info(f"Dataset: {dataset} | Tiers: {tiers}")

    # ── 1. Data ────────────────────────────────────────────
    if args.skip_download:
        logger.info("Loading pre-processed data from disk …")
        train_df, val_df, test_df = load_processed(dataset)
    else:
        logger.info("Downloading and preprocessing data …")
        train_df, val_df, test_df = get_dataset(
            dataset    = dataset,
            max_samples= cfg["data"]["max_samples"],
        )
        train_df, val_df, test_df = run_preprocessing_pipeline(
            train_df, val_df, test_df, dataset=dataset
        )

    # Integer labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"].values)
    y_val   = le.transform(val_df["label"].values)
    y_test  = le.transform(test_df["label"].values)
    label_names  = list(le.classes_.astype(str))
    num_classes  = len(label_names)

    # Texts
    train_texts = train_df["clean_text"].tolist()
    val_texts   = val_df["clean_text"].tolist()
    test_texts  = test_df["clean_text"].tolist()

    results = {}  # name → metrics dict

    # ── 2. Baseline ────────────────────────────────────────
    if "baseline" in tiers:
        logger.info("\n" + "="*60 + "\n  TIER 1: TF-IDF + Logistic Regression\n" + "="*60)
        pipeline, t_train = train_baseline(train_texts, y_train, cfg)
        save_baseline(pipeline)

        # Save label encoder alongside pipeline
        enc_path = ROOT / "experiments" / "models" / "baseline" / "label_enc.pkl"
        with open(enc_path, "wb") as f:
            pickle.dump(le, f)

        t0 = time.time()
        preds, probs = predict_baseline(pipeline, test_texts)
        t_infer = time.time() - t0

        metrics = full_evaluation(
            name="baseline",
            y_true=y_test,
            y_pred=preds,
            y_prob=probs,
            texts=test_texts,
            label_names=label_names,
            latency_s=t_infer,
        )
        metrics["train_time_s"] = round(t_train, 2)
        results["baseline"] = metrics
        logger.info(f"Baseline → accuracy={metrics['accuracy']} | F1={metrics['f1_weighted']}")

    # ── 3. BiLSTM ──────────────────────────────────────────
    if "lstm" in tiers:
        logger.info("\n" + "="*60 + "\n  TIER 2: Bidirectional LSTM + Attention\n" + "="*60)

        vocab = Vocabulary(
            max_vocab_size=cfg["data"]["max_vocab_size"],
            min_freq=cfg["preprocessing"]["min_word_freq"],
        )
        vocab.build(train_texts)

        max_len = cfg["data"]["max_len_lstm"]
        X_train_seq = texts_to_sequences(train_texts, vocab, max_len)
        X_val_seq   = texts_to_sequences(val_texts,   vocab, max_len)
        X_test_seq  = texts_to_sequences(test_texts,  vocab, max_len)

        bs           = cfg["lstm"]["batch_size"]
        train_loader = DataLoader(SentimentDataset(X_train_seq, y_train), batch_size=bs, shuffle=True)
        val_loader   = DataLoader(SentimentDataset(X_val_seq,   y_val),   batch_size=bs)

        model   = build_lstm_model(vocab, num_classes, cfg)
        history = train_lstm(model, train_loader, val_loader, cfg)

        # Load best checkpoint
        import torch
        ckpt_path = ROOT / "experiments" / "models" / "lstm" / "best_model.pt"
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        save_lstm_artifacts(model, vocab, le)

        t0 = time.time()
        preds, probs = predict_lstm(model, X_test_seq)
        t_infer = time.time() - t0

        metrics = full_evaluation(
            name="lstm",
            y_true=y_test,
            y_pred=preds,
            y_prob=probs,
            texts=test_texts,
            label_names=label_names,
            latency_s=t_infer,
        )
        results["lstm"] = metrics
        logger.info(f"BiLSTM → accuracy={metrics['accuracy']} | F1={metrics['f1_weighted']}")

        # Training curves
        from src.evaluation.evaluate import plot_training_curves
        plot_training_curves(history, "BiLSTM")

    # ── 4. Transformer ─────────────────────────────────────
    if "transformer" in tiers:
        logger.info("\n" + "="*60 + "\n  TIER 3: DistilBERT Fine-tuning\n" + "="*60)

        t0 = time.time()
        model_t, tokenizer_t = train_transformer(
            train_texts=train_texts,
            val_texts=val_texts,
            train_labels=y_train.tolist(),
            val_labels=y_val.tolist(),
            num_labels=num_classes,
            cfg=cfg,
        )
        t_train = time.time() - t0

        # Save label encoder
        enc_dir = ensure_dir(ROOT / "experiments" / "models" / "transformer")
        with open(enc_dir / "label_enc.pkl", "wb") as f:
            pickle.dump(le, f)

        t0 = time.time()
        preds, probs = predict_transformer(model_t, tokenizer_t, test_texts)
        t_infer = time.time() - t0

        metrics = full_evaluation(
            name="transformer",
            y_true=y_test,
            y_pred=preds,
            y_prob=probs,
            texts=test_texts,
            label_names=label_names,
            latency_s=t_infer,
        )
        metrics["train_time_s"] = round(t_train, 2)
        results["transformer"] = metrics
        logger.info(f"Transformer → accuracy={metrics['accuracy']} | F1={metrics['f1_weighted']}")

    # ── 5. Comparison Report ───────────────────────────────
    if len(results) > 1:
        logger.info("\n" + "="*60 + "\n  FINAL COMPARISON\n" + "="*60)
        for name, m in results.items():
            logger.info(
                f"{name:12s} | acc={m['accuracy']:.4f} | "
                f"f1={m['f1_weighted']:.4f} | "
                f"latency={m['latency_ms_per_sample']:.2f}ms/sample"
            )

        # Save comparison
        comp_path = ROOT / "reports" / "metrics" / "comparison.json"
        saveable  = {
            n: {k: v for k, v in m.items() if k != "classification_report"}
            for n, m in results.items()
        }
        with open(comp_path, "w") as f:
            json.dump(saveable, f, indent=2)
        logger.info(f"Comparison saved → {comp_path}")

        plot_model_comparison(results, metric="f1_weighted")
        plot_model_comparison(results, metric="accuracy")
        plot_model_comparison(results, metric="latency_ms_per_sample")

    logger.info("All training complete! ✓")


if __name__ == "__main__":
    main()
