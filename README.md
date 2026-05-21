
# 🧠 Sentiment Analysis 
A production-grade, **learning-based sentiment analysis pipeline** covering the full ML lifecycle:
from classical baselines to transformer fine-tuning, deployed via REST API and an interactive dashboard.

---

## ✨ Features

| Feature | Description |
|---|---|
| **3-Tier Architecture** | TF-IDF baseline → BiLSTM + Attention → DistilBERT |
| **GloVe Embeddings** | Pre-trained word vectors for LSTM tier |
| **Explainability** | SHAP + LIME for all model tiers |
| **FastAPI Backend** | `/predict`, `/predict/batch`, `/predict/compare` endpoints |
| **Streamlit Dashboard** | Live predictions, model comparison, batch analysis, reports |
| **MLflow Ready** | Experiment tracking with metrics and artifacts |
| **Error Analysis** | Auto-generated CSVs of high-confidence wrong predictions |

---

## 🗂️ Project Structure

```
sentiment_analysis/
├── config/
│   └── config.yaml              ← All hyperparameters & paths
├── data/
│   ├── raw/                     ← Downloaded splits (train/val/test CSVs)
│   ├── processed/               ← Cleaned text splits
│   └── embeddings/              ← GloVe file goes here
├── src/
│   ├── data/
│   │   ├── download.py          ← Dataset downloaders (IMDb, Twitter)
│   │   ├── preprocess.py        ← Cleaning, Vocabulary, padding
│   │   └── eda.py               ← EDA plots & word clouds
│   ├── models/
│   │   ├── baseline.py          ← TF-IDF + sklearn classifiers
│   │   ├── lstm_model.py        ← BiLSTM + Attention (PyTorch)
│   │   └── transformer_model.py ← DistilBERT fine-tuning (HuggingFace)
│   ├── training/
│   │   └── train_all.py         ← Master training script
│   ├── evaluation/
│   │   └── evaluate.py          ← Metrics, confusion matrix, comparison plots
│   ├── explainability/
│   │   └── explain.py           ← SHAP + LIME explanations
│   └── utils/
│       └── helpers.py           ← Config, seed, logger, device
├── api/
│   ├── main.py                  ← FastAPI app
│   └── predictor.py             ← Unified inference interface
├── dashboard/
│   └── app.py                   ← Streamlit dashboard
├── experiments/
│   └── models/                  ← Saved model artifacts
├── reports/
│   ├── figures/                 ← Auto-generated plots
│   └── metrics/                 ← JSON metrics & comparison
├── Makefile
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Create a virtual environment

```bash
cd sentiment_analysis
python -m venv .venv
source .venv/bin/activate     # Linux/Mac
# .venv\Scripts\activate      # Windows
```

### 2. Install dependencies

```bash
make install
# or: pip install -r requirements.txt
make setup   # also downloads NLTK data
```

### 3. (Optional) Download GloVe embeddings

For better LSTM performance:
```bash
# Download glove.6B.zip from https://nlp.stanford.edu/data/glove.6B.zip
# Extract and place glove.6B.100d.txt inside:
mkdir -p sentiment_analysis/data/embeddings/
mv glove.6B.100d.txt sentiment_analysis/data/embeddings/
```

### 4. Run EDA

```bash
make eda
# → figures saved to reports/figures/
```

### 5. Train models

```bash
make train            # all 3 tiers
make train-baseline   # Tier 1 only (fastest, ~30s)
make train-lstm       # Tier 2 only (~5-10 min)
make train-transformer # Tier 3 only (~20-60 min, GPU recommended)
```

### 6. Start the API

```bash
make api
# → http://localhost:8000/docs (Swagger UI)
```

### 7. Launch the Dashboard

```bash
make dashboard
# → http://localhost:8501
```

---

## 🔌 API Usage

### Single prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This was an amazing movie!", "model": "transformer"}'
```

Response:
```json
{
  "label": "positive",
  "confidence": 0.9821,
  "probabilities": {"negative": 0.0179, "positive": 0.9821},
  "latency_ms": 42.3,
  "model_used": "transformer"
}
```

### Compare all models
```bash
curl -X POST http://localhost:8000/predict/compare \
  -H "Content-Type: application/json" \
  -d '{"text": "The product is okay but could be better."}'
```

### Batch prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great!", "Terrible.", "Just okay."], "model": "baseline"}'
```

---

## 📊 Model Performance (IMDb Dataset)

| Model | Accuracy | F1 (Weighted) | Latency |
|---|---|---|---|
| TF-IDF + LR | ~88% | ~0.88 | < 1ms/sample |
| BiLSTM + Attention | ~90% | ~0.90 | ~5ms/sample |
| DistilBERT | ~93% | ~0.93 | ~40ms/sample |

> *Results are approximate. Actual results depend on hardware, GloVe availability, and training duration.*

---

## 🧪 Key Concepts Demonstrated

- **Classical NLP**: TF-IDF vectorization, n-gram features, sklearn pipelines
- **Deep Learning**: BiLSTM, Attention mechanism, early stopping, learning rate scheduling
- **Transfer Learning**: Pre-trained DistilBERT fine-tuning for domain adaptation
- **Reproducibility**: YAML config, seed control, versioned artifacts
- **Explainability**: SHAP (feature importance), LIME (local explanations)
- **Production**: REST API, async batch inference, probability calibration

---

## 🛠️ Tech Stack

`Python 3.10+` · `PyTorch` · `HuggingFace Transformers` · `scikit-learn` · `FastAPI` · `Streamlit` · `SHAP` · `LIME` · `MLflow` · `Plotly`

---

*Built for the AI Associate role application — demonstrating end-to-end ML engineering.*
# Multi-Model-Sentiment-Analysis
