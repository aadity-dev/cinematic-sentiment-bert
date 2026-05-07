"""
Streamlit Sentiment Analysis Web App
Loads the fine-tuned DistilBERT model and provides a beautiful UI for sentiment prediction.
"""

import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import pandas as pd
import json
import os
import time

# ─── Page Config ───
st.set_page_config(
    page_title="🎭 Sentiment Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .main { font-family: 'Inter', sans-serif; }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    
    .hero-subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.2rem;
        margin-top: 0;
    }
    
    .result-positive {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .result-negative {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
    }
    
    .confidence-bar {
        background: #1f2937;
        border-radius: 10px;
        padding: 3px;
        margin-top: 10px;
    }
    
    .confidence-fill {
        height: 24px;
        border-radius: 8px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        font-size: 1rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ───
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")

@st.cache_resource
def load_model():
    """Load the fine-tuned model and tokenizer."""
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.to(device)
    return model, tokenizer, device


def predict_sentiment(text, model, tokenizer, device):
    """Run inference on a single text."""
    start = time.time()
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred].item()
    
    elapsed = time.time() - start
    
    label_map = {0: "Negative", 1: "Positive"}
    return {
        "label": label_map[pred],
        "confidence": confidence,
        "time_ms": round(elapsed * 1000, 1),
        "positive_prob": probs[0][1].item(),
        "negative_prob": probs[0][0].item(),
    }


# ─── Main App ───
def main():
    # Check if model exists
    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        st.error("⚠️ Model not found! Please run `python train.py` first to train the model.")
        st.info("The model should be saved in the `model/` directory.")
        st.stop()
    
    model, tokenizer, device = load_model()
    
    # Hero Section
    st.markdown('<h1 class="hero-title">🎭 Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Powered by DistilBERT · Trained on 50K IMDB Reviews</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        analysis_mode = st.radio("Mode", ["Single Text", "Batch Analysis"])
        
        st.markdown("---")
        st.markdown("## 📊 Model Info")
        
        metrics_path = os.path.join(MODEL_DIR, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 'N/A')}%")
            st.metric("F1 Score", f"{metrics.get('test_f1', 'N/A')}%")
            st.metric("Training Time", f"{metrics.get('training_time_minutes', 'N/A')} min")
        
        st.markdown("---")
        st.markdown("## 💡 Example Inputs")
        examples = [
            "This movie was absolutely fantastic! The acting was superb.", # Clear Positive
            "Terrible film. Waste of time and money.", # Clear Negative
            "Oh brilliant, another superhero movie with the exact same plot. Just what we needed.", # Sarcasm
            "The cinematography was breathtaking, but the plot was weak and the acting fell flat.", # Mixed Sentiment
            "meh.", # Very short text
            "I can't believe how not bad this was!", # Double negative / complex phrasing
        ]
        for ex in examples:
            if st.button(ex[:50] + "...", key=ex):
                st.session_state["input_text"] = ex
    
    # Initialize session state
    if "history" not in st.session_state:
        st.session_state["history"] = []
    
    if analysis_mode == "Single Text":
        # ─── Single Analysis ───
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_input = st.text_area(
                "Enter your text:",
                value=st.session_state.get("input_text", ""),
                height=150,
                placeholder="Type or paste a movie review, tweet, or any text...",
            )
            
            analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
        
        if analyze_btn and text_input.strip():
            result = predict_sentiment(text_input, model, tokenizer, device)
            
            # Store in history
            st.session_state["history"].append({
                "text": text_input[:100] + ("..." if len(text_input) > 100 else ""),
                "sentiment": result["label"],
                "confidence": f"{result['confidence']*100:.1f}%",
            })
            
            with col2:
                # Result card
                css_class = "result-positive" if result["label"] == "Positive" else "result-negative"
                emoji = "😊" if result["label"] == "Positive" else "😞"
                st.markdown(
                    f'<div class="{css_class}">{emoji} {result["label"]}</div>',
                    unsafe_allow_html=True,
                )
                
                # Confidence bar
                color = "#10b981" if result["label"] == "Positive" else "#ef4444"
                conf_pct = result["confidence"] * 100
                st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {conf_pct}%; background: {color};">
                        {conf_pct:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("")
                
                # Probability breakdown
                c1, c2 = st.columns(2)
                c1.metric("👍 Positive", f"{result['positive_prob']*100:.1f}%")
                c2.metric("👎 Negative", f"{result['negative_prob']*100:.1f}%")
                
                st.caption(f"⚡ Inference: {result['time_ms']}ms")
        
        elif analyze_btn:
            st.warning("Please enter some text to analyze.")
    
    else:
        # ─── Batch Analysis ───
        st.markdown("### 📋 Batch Analysis")
        st.info("Enter multiple texts, one per line.")
        
        batch_input = st.text_area(
            "Texts (one per line):",
            height=200,
            placeholder="Line 1: First review...\nLine 2: Second review...\nLine 3: Third review...",
        )
        
        if st.button("🔍 Analyze All", type="primary", use_container_width=True):
            lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
            
            if lines:
                results = []
                progress = st.progress(0)
                
                for i, text in enumerate(lines):
                    result = predict_sentiment(text, model, tokenizer, device)
                    results.append({
                        "Text": text[:80] + ("..." if len(text) > 80 else ""),
                        "Sentiment": f"{'😊' if result['label'] == 'Positive' else '😞'} {result['label']}",
                        "Confidence": f"{result['confidence']*100:.1f}%",
                        "Pos %": f"{result['positive_prob']*100:.1f}",
                        "Neg %": f"{result['negative_prob']*100:.1f}",
                    })
                    progress.progress((i + 1) / len(lines))
                
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                # Summary stats
                pos_count = sum(1 for r in results if "Positive" in r["Sentiment"])
                neg_count = len(results) - pos_count
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total", len(results))
                col2.metric("😊 Positive", pos_count)
                col3.metric("😞 Negative", neg_count)
                
                # Download CSV
                csv_df = pd.DataFrame(results)
                csv = csv_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    "sentiment_results.csv",
                    "text/csv",
                    use_container_width=True,
                )
            else:
                st.warning("Please enter at least one text.")
    
    # ─── History ───
    if st.session_state["history"]:
        st.markdown("---")
        st.markdown("### 📜 Prediction History")
        history_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state["history"] = []
            st.rerun()


if __name__ == "__main__":
    main()
