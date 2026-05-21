
"""
Streamlit Dashboard — Multi-Model Sentiment Analysis
Run: streamlit run dashboard/app.py
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import requests

# ──────────────────────────────────────────────────────────
#  Page Config
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "SentimentIQ Dashboard",
    page_icon  = "🧠",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ──────────────────────────────────────────────────────────
#  Custom CSS — Dark Premium Theme
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
}

/* Cards */
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.2);
}

/* Hero */
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6C63FF, #FF6B9D, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}
.hero-subtitle {
    color: rgba(255,255,255,0.55);
    text-align: center;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

/* Result badge */
.sentiment-positive {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    padding: 10px 24px;
    border-radius: 50px;
    font-weight: 600;
    font-size: 1.1rem;
    display: inline-block;
    color: white;
}
.sentiment-negative {
    background: linear-gradient(135deg, #c0392b, #e74c3c);
    padding: 10px 24px;
    border-radius: 50px;
    font-weight: 600;
    font-size: 1.1rem;
    display: inline-block;
    color: white;
}
.sentiment-neutral {
    background: linear-gradient(135deg, #f39c12, #f1c40f);
    padding: 10px 24px;
    border-radius: 50px;
    font-weight: 600;
    font-size: 1.1rem;
    display: inline-block;
    color: white;
}

/* Input area */
.stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(108,99,255,0.4) !important;
    border-radius: 12px !important;
    color: white !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: #6C63FF !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,0.3) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(108, 99, 255, 0.4) !important;
}

/* Section headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 600;
    color: white;
    border-left: 4px solid #6C63FF;
    padding-left: 12px;
    margin: 1.5rem 0 1rem 0;
}

/* Dividers */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
#  Config & Constants
# ──────────────────────────────────────────────────────────
API_URL   = "http://localhost:8000"
MODEL_INFO = {
    "baseline":    {"label": "TF-IDF + Logistic Regression", "icon": "⚡", "tier": "Tier 1"},
    "lstm":        {"label": "BiLSTM + Attention",            "icon": "🔁", "tier": "Tier 2"},
    "transformer": {"label": "DistilBERT Fine-tuned",         "icon": "🤖", "tier": "Tier 3"},
}
COLORS = {
    "positive": "#4ECDC4",
    "negative": "#FF6B6B",
    "neutral":  "#FFE66D",
    "baseline": "#6C63FF",
    "lstm":     "#FF6B9D",
    "transformer": "#4ECDC4",
}


# ──────────────────────────────────────────────────────────
#  API Helpers
# ──────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def fetch_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def predict_text(text: str, model: str) -> dict:
    try:
        r = requests.post(
            f"{API_URL}/predict",
            json={"text": text, "model": model},
            timeout=30,
        )
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def predict_compare(text: str) -> dict:
    try:
        r = requests.post(
            f"{API_URL}/predict/compare",
            json={"text": text},
            timeout=60,
        )
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def predict_batch(texts: list, model: str) -> dict:
    try:
        r = requests.post(
            f"{API_URL}/predict/batch",
            json={"texts": texts, "model": model},
            timeout=60,
        )
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────────────────
#  Visualization Helpers
# ──────────────────────────────────────────────────────────
def confidence_gauge(value: float, label: str) -> go.Figure:
    color = "#4ECDC4" if value >= 0.7 else "#FFE66D" if value >= 0.5 else "#FF6B6B"
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value * 100,
        title = {"text": f"Confidence — {label}", "font": {"color": "white", "size": 14}},
        number = {"suffix": "%", "font": {"color": "white", "size": 28}},
        gauge  = {
            "axis":      {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.4)"},
            "bar":       {"color": color},
            "bgcolor":   "rgba(255,255,255,0.05)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0,  50], "color": "rgba(255,107,107,0.15)"},
                {"range": [50, 70], "color": "rgba(255,230,109,0.15)"},
                {"range": [70, 100],"color": "rgba(78,205,196,0.15)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def prob_bar_chart(probabilities: dict, title: str) -> go.Figure:
    labels = list(probabilities.keys())
    values = [v * 100 for v in probabilities.values()]
    colors = [COLORS.get(l, "#6C63FF") for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="white", size=13),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=13)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 115], showgrid=False, showticklabels=False, color="white"),
        yaxis=dict(color="white", tickfont=dict(size=13)),
        height=max(150, len(labels) * 60),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def model_compare_chart(results: dict) -> go.Figure:
    models     = list(results.keys())
    sentiments = list(results[models[0]]["probabilities"].keys()) if models else []

    fig = go.Figure()
    for i, sentiment in enumerate(sentiments):
        vals   = [results[m]["probabilities"].get(sentiment, 0) * 100 for m in models]
        color  = COLORS.get(sentiment, PALETTE[i % 4] if (PALETTE := ["#6C63FF","#FF6B6B","#4ECDC4","#FFE66D"]) else "#fff")
        fig.add_trace(go.Bar(
            name=sentiment.capitalize(),
            x=[MODEL_INFO.get(m, {}).get("label", m) for m in models],
            y=vals,
            marker_color=color,
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            textfont=dict(color="white"),
        ))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        xaxis=dict(color="white", tickfont=dict(size=12)),
        yaxis=dict(color="white", title="Probability (%)", range=[0, 115]),
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
        height=320,
        margin=dict(l=10, r=10, t=20, b=20),
    )
    return fig


def latency_chart(results: dict) -> go.Figure:
    models  = list(results.keys())
    labels  = [MODEL_INFO.get(m, {}).get("label", m) for m in models]
    latency = [results[m].get("latency_ms", 0) for m in models]
    colors  = [COLORS.get(m, "#6C63FF") for m in models]

    fig = go.Figure(go.Bar(
        x=labels, y=latency,
        marker=dict(color=colors),
        text=[f"{v:.1f}ms" for v in latency],
        textposition="outside",
        textfont=dict(color="white"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        xaxis=dict(color="white"),
        yaxis=dict(color="white", title="Latency (ms)"),
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ──────────────────────────────────────────────────────────
#  Sidebar
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 SentimentIQ")
    st.markdown("---")

    health = fetch_health()
    if health:
        st.success(f"✅ API Online")
        available = health.get("available_models", [])
        st.markdown(f"**Device:** `{health.get('device','cpu').upper()}`")
    else:
        st.error("❌ API Offline\n\nStart it with:\n```\npython -m api.main\n```")
        available = []

    st.markdown("### Available Models")
    for m in ["baseline", "lstm", "transformer"]:
        info = MODEL_INFO[m]
        if m in available:
            st.markdown(f"- {info['icon']} **{info['tier']}** — {info['label']} ✓")
        else:
            st.markdown(f"- ~~{info['icon']} {info['tier']} — {info['label']}~~ _(not trained)_")

    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔍 Live Predict", "⚖️ Model Compare", "📦 Batch Analysis", "📊 Reports"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:rgba(255,255,255,0.35);'>Sentiment Analysis Project<br>AI Associate Portfolio</small>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────
#  HERO
# ──────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 SentimentIQ</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Multi-tier Sentiment Analysis — TF-IDF · BiLSTM · DistilBERT</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ══════════════════════════════════════════════════════════
#  PAGE 1 — Live Predict
# ══════════════════════════════════════════════════════════
if page == "🔍 Live Predict":
    st.markdown('<div class="section-header">Live Prediction</div>', unsafe_allow_html=True)

    col_input, col_config = st.columns([3, 1])

    with col_config:
        selected_model = st.selectbox(
            "Select Model",
            options=available if available else ["baseline"],
            format_func=lambda m: f"{MODEL_INFO[m]['icon']} {MODEL_INFO[m]['tier']}",
        )
        st.markdown(f"<small style='color:rgba(255,255,255,0.5);'>{MODEL_INFO.get(selected_model,{}).get('label','')}</small>", unsafe_allow_html=True)

    with col_input:
        text_input = st.text_area(
            "Enter text to analyse",
            placeholder="Type or paste any text here — movie review, tweet, product feedback…",
            height=140,
            label_visibility="collapsed",
        )

    col_btn, col_ex = st.columns([1, 3])
    with col_btn:
        analyse_btn = st.button("✨ Analyse Sentiment", use_container_width=True)

    with col_ex:
        example_texts = {
            "Positive 🎉": "This product is absolutely fantastic! Best purchase I've made all year.",
            "Negative 😤": "Terrible experience. Completely broken and customer service was useless.",
            "Mixed 🤔":    "The camera quality is amazing, but the battery life is disappointingly short.",
        }
        ex_choice = st.selectbox("Try an example", [""] + list(example_texts.keys()),
                                 label_visibility="collapsed")
        if ex_choice:
            text_input = example_texts[ex_choice]

    if analyse_btn and text_input.strip():
        if not available:
            st.warning("⚠️ No models available. Train them first with `python -m src.training.train_all`")
        else:
            with st.spinner("Analysing…"):
                result = predict_text(text_input, selected_model)

            if "error" in result:
                st.error(f"API error: {result['error']}")
            else:
                st.markdown("---")
                label    = result["label"]
                conf     = result["confidence"]
                probs    = result["probabilities"]
                lat      = result["latency_ms"]
                css_cls  = f"sentiment-{label}"

                # Result header
                col_badge, col_lat = st.columns([2, 1])
                with col_badge:
                    st.markdown(
                        f'<span class="{css_cls}">{label.upper()}</span>',
                        unsafe_allow_html=True
                    )
                with col_lat:
                    st.metric("Inference Time", f"{lat:.1f} ms")

                st.markdown("")
                col_gauge, col_bar = st.columns(2)
                with col_gauge:
                    st.plotly_chart(confidence_gauge(conf, label), use_container_width=True)
                with col_bar:
                    st.plotly_chart(
                        prob_bar_chart(probs, "Class Probabilities"),
                        use_container_width=True
                    )

                with st.expander("📝 Input Text Preview"):
                    st.write(text_input)


# ══════════════════════════════════════════════════════════
#  PAGE 2 — Model Compare
# ══════════════════════════════════════════════════════════
elif page == "⚖️ Model Compare":
    st.markdown('<div class="section-header">All-Model Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        "<small style='color:rgba(255,255,255,0.5);'>Run the same text through every trained model simultaneously.</small>",
        unsafe_allow_html=True,
    )

    compare_text = st.text_area(
        "Text to compare",
        placeholder="Enter text to compare across all models…",
        height=120,
        label_visibility="collapsed",
    )
    compare_btn = st.button("🔬 Compare All Models", use_container_width=False)

    if compare_btn and compare_text.strip():
        if not available:
            st.warning("No models available.")
        else:
            with st.spinner("Running all models…"):
                cmp = predict_compare(compare_text)

            if "error" in cmp:
                st.error(cmp["error"])
            else:
                results = cmp.get("results", {})
                st.markdown("---")

                # Summary cards
                cols = st.columns(len(results))
                for col, (mname, res) in zip(cols, results.items()):
                    info  = MODEL_INFO.get(mname, {})
                    label = res["label"]
                    conf  = res["confidence"]
                    lat   = res["latency_ms"]
                    css   = f"sentiment-{label}"
                    with col:
                        st.markdown(f"""
                        <div class="metric-card" style="text-align:center">
                            <div style="font-size:2rem">{info.get('icon','')}</div>
                            <div style="color:rgba(255,255,255,0.6);font-size:0.8rem">{info.get('tier','')}</div>
                            <div style="color:white;font-weight:600;margin:6px 0">{info.get('label','')}</div>
                            <span class="{css}">{label.upper()}</span>
                            <div style="color:rgba(255,255,255,0.5);margin-top:8px;font-size:0.85rem">
                                {conf*100:.1f}% conf · {lat:.1f}ms
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("")
                col_prob, col_lat = st.columns(2)
                with col_prob:
                    st.markdown("**Probability Comparison**")
                    st.plotly_chart(model_compare_chart(results), use_container_width=True)
                with col_lat:
                    st.markdown("**Inference Latency**")
                    st.plotly_chart(latency_chart(results), use_container_width=True)


# ══════════════════════════════════════════════════════════
#  PAGE 3 — Batch Analysis
# ══════════════════════════════════════════════════════════
elif page == "📦 Batch Analysis":
    st.markdown('<div class="section-header">Batch Text Analysis</div>', unsafe_allow_html=True)

    batch_model = st.selectbox(
        "Model",
        options=available if available else ["baseline"],
        format_func=lambda m: f"{MODEL_INFO[m]['icon']} {MODEL_INFO[m]['tier']} — {MODEL_INFO[m]['label']}",
    )

    batch_input = st.text_area(
        "Enter one text per line",
        placeholder="Line 1: Great product!\nLine 2: Terrible service.\nLine 3: Average experience.",
        height=200,
        label_visibility="collapsed",
    )

    uploaded = st.file_uploader("Or upload a CSV (column: 'text')", type=["csv"])

    batch_btn = st.button("🚀 Run Batch Analysis", use_container_width=False)

    if batch_btn:
        texts_to_analyse = []
        if uploaded is not None:
            df_up = pd.read_csv(uploaded)
            if "text" in df_up.columns:
                texts_to_analyse = df_up["text"].dropna().tolist()
            else:
                st.error("CSV must have a 'text' column.")
        elif batch_input.strip():
            texts_to_analyse = [t.strip() for t in batch_input.strip().splitlines() if t.strip()]

        if texts_to_analyse and available:
            with st.spinner(f"Analysing {len(texts_to_analyse)} texts…"):
                resp = predict_batch(texts_to_analyse, batch_model)

            if "error" in resp:
                st.error(resp["error"])
            else:
                preds = resp["predictions"]
                total_ms = resp.get("total_latency_ms", 0)

                df_result = pd.DataFrame([
                    {
                        "Text":       t[:80] + "…" if len(t) > 80 else t,
                        "Sentiment":  p.get("label", "—"),
                        "Confidence": f"{p.get('confidence',0)*100:.1f}%",
                        "Latency (ms)": f"{p.get('latency_ms',0):.1f}",
                    }
                    for t, p in zip(texts_to_analyse, preds)
                ])

                # Summary metrics
                c1, c2, c3, c4 = st.columns(4)
                label_counts = df_result["Sentiment"].value_counts()
                c1.metric("Total Texts",   len(preds))
                c2.metric("Positive",       label_counts.get("positive", 0))
                c3.metric("Negative",       label_counts.get("negative", 0))
                c4.metric("Total Time",    f"{total_ms:.1f}ms")

                # Pie chart
                pie_fig = px.pie(
                    values=label_counts.values,
                    names=[n.capitalize() for n in label_counts.index],
                    color=label_counts.index,
                    color_discrete_map={
                        "positive": "#4ECDC4", "negative": "#FF6B6B", "neutral": "#FFE66D"
                    },
                    hole=0.5,
                )
                pie_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor ="rgba(0,0,0,0)",
                    legend=dict(font=dict(color="white")),
                    height=280,
                    margin=dict(l=0,r=0,t=0,b=0),
                )
                st.plotly_chart(pie_fig, use_container_width=True)

                st.dataframe(df_result, use_container_width=True)

                # Download
                csv = df_result.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Results CSV",
                    csv,
                    "sentiment_results.csv",
                    "text/csv",
                )


# ══════════════════════════════════════════════════════════
#  PAGE 4 — Reports
# ══════════════════════════════════════════════════════════
elif page == "📊 Reports":
    st.markdown('<div class="section-header">Training Reports & Metrics</div>', unsafe_allow_html=True)

    report_root = Path(__file__).parent.parent / "reports"
    metrics_dir = report_root / "metrics"
    figures_dir = report_root / "figures"

    # ── Comparison JSON ──
    comp_path = metrics_dir / "comparison.json"
    if comp_path.exists():
        with open(comp_path) as f:
            comp = json.load(f)

        st.markdown("### 📈 Model Performance Comparison")
        rows = []
        for mname, m in comp.items():
            rows.append({
                "Model":       MODEL_INFO.get(mname, {}).get("label", mname),
                "Tier":        MODEL_INFO.get(mname, {}).get("tier", ""),
                "Accuracy":    f"{m.get('accuracy',0):.4f}",
                "F1 (Weighted)": f"{m.get('f1_weighted',0):.4f}",
                "ROC-AUC":     f"{m.get('roc_auc',0):.4f}" if m.get("roc_auc") else "—",
                "Latency (ms/sample)": f"{m.get('latency_ms_per_sample',0):.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No comparison report found. Train models first: `python -m src.training.train_all`")

    # ── Per-model metrics ──
    met_files = list(metrics_dir.glob("*_metrics.json")) if metrics_dir.exists() else []
    if met_files:
        st.markdown("### 🔎 Detailed Metrics")
        tabs = st.tabs([f.stem.replace("_metrics", "").capitalize() for f in met_files])
        for tab, met_file in zip(tabs, met_files):
            with tab:
                with open(met_file) as f:
                    m = json.load(f)
                cols = st.columns(4)
                cols[0].metric("Accuracy",    f"{m.get('accuracy',0):.4f}")
                cols[1].metric("F1 Weighted", f"{m.get('f1_weighted',0):.4f}")
                cols[2].metric("Precision",   f"{m.get('precision_weighted',0):.4f}")
                cols[3].metric("Recall",      f"{m.get('recall_weighted',0):.4f}")

    # ── Figures ──
    fig_files = sorted(figures_dir.glob("*.png")) if figures_dir.exists() else []
    if fig_files:
        st.markdown("### 🖼️ Saved Figures")
        # Group into rows of 2
        for i in range(0, len(fig_files), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(fig_files):
                    f = fig_files[i + j]
                    with col:
                        st.image(str(f), caption=f.stem.replace("_", " ").title(), use_container_width=True)
    else:
        st.info("No report figures yet. Run training to generate them.")
