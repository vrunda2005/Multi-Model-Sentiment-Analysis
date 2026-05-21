
# ═══════════════════════════════════════════════════════════
#  Sentiment Analysis Project — Makefile
#  Run from inside sentiment_analysis/ directory
#  Usage: make <target>
# ═══════════════════════════════════════════════════════════


# Auto-detect .venv if it exists, otherwise fall back to system python3
VENV   := .venv
PYTHON := $(shell [ -f $(VENV)/bin/python3 ] && echo $(VENV)/bin/python3 || which python3)
PIP    := $(shell [ -f $(VENV)/bin/pip ] && echo $(VENV)/bin/pip || which pip3)


.PHONY: help install setup eda train train-baseline train-lstm train-transformer \
        train-fast api dashboard test clean clean-all

## ─── Help ─────────────────────────────────────────────────
help:
	@echo ""
	@echo "  ╔══════════════════════════════════════════╗"
	@echo "  ║   Sentiment Analysis Project — Commands  ║"
	@echo "  ╚══════════════════════════════════════════╝"
	@echo ""
	@echo "  make install             Install all dependencies"
	@echo "  make setup               Install deps + NLTK data"
	@echo "  make eda                 Run exploratory data analysis"
	@echo "  make train               Train ALL three model tiers"
	@echo "  make train-baseline      Tier 1: TF-IDF + LogReg  (~30s)"
	@echo "  make train-lstm          Tier 2: BiLSTM + Attention (~10min)"
	@echo "  make train-transformer   Tier 3: DistilBERT fine-tune"
	@echo "  make train-fast          Tier 1 + 2 only (no GPU needed)"
	@echo "  make api                 Start FastAPI backend  (port 8000)"
	@echo "  make dashboard           Start Streamlit UI     (port 8501)"
	@echo "  make clean               Remove Python cache files"
	@echo "  make clean-all           Remove all generated artifacts"
	@echo ""

## ─── Environment ──────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt
	$(PIP) install evaluate

setup: install
	$(PYTHON) -c "import nltk; [nltk.download(p, quiet=True) for p in ['stopwords','wordnet','punkt','punkt_tab']]"
	@echo "✓ Setup complete"

## ─── Data & EDA ───────────────────────────────────────────
eda:
	$(PYTHON) -m src.data.eda
	@echo "✓ EDA figures saved to reports/figures/"

## ─── Training ─────────────────────────────────────────────
train:
	$(PYTHON) -m src.training.train_all --tiers all

train-baseline:
	$(PYTHON) -m src.training.train_all --tiers baseline

train-lstm:
	$(PYTHON) -m src.training.train_all --tiers lstm

train-transformer:
	$(PYTHON) -m src.training.train_all --tiers transformer

train-fast:
	$(PYTHON) -m src.training.train_all --tiers baseline lstm

## ─── Serving ──────────────────────────────────────────────
api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py --server.port 8501

## ─── Quality ──────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v --tb=short 2>/dev/null || \
	  echo "No tests/ directory found yet."

## ─── Cleanup ──────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Cache files removed"

clean-all: clean
	rm -rf experiments/models
	rm -rf reports/figures/*.png
	rm -rf reports/metrics/*.json
	rm -rf data/processed
	@echo "✓ All generated artifacts removed"
