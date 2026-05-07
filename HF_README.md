---
title: BERT Sentiment Analyzer
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.34.0
app_file: app/app.py
pinned: false
---
# 🎭 BERT Sentiment Analyzer

A deep learning-powered sentiment analysis web application that classifies text as **Positive** or **Negative** using a fine-tuned BERT transformer model.

## 🚀 Features

- **Fine-tuned BERT** on 50K IMDB movie reviews
- **Real-time inference** with confidence scores
- **Batch analysis** — analyze multiple texts at once
- **Interactive Streamlit UI** with color-coded results
- **Download results** as CSV

## 🏗️ Architecture

```
Input Text → BERT Tokenizer → BERT Model → Classification Head → Sentiment + Confidence
```

## 📁 Project Structure

```
Sentiment/
├── data/           # Dataset scripts & exploration
├── model/          # Saved model weights & tokenizer
├── app/            # Streamlit web application
├── train.py        # Model training script
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

- **Model**: BERT (bert-base-uncased) / DistilBERT
- **Framework**: PyTorch + HuggingFace Transformers
- **Dataset**: IMDB Movie Reviews (50K samples)
- **UI**: Streamlit
- **Deployment**: HuggingFace Spaces

## 📊 Results

| Metric    | Score |
|-----------|-------|
| Accuracy  | 91.22% |
| F1-Score  | 91.24% |
| Precision | 91.01% |
| Recall    | 91.47% |

*Model was fine-tuned for 3 epochs in ~110 minutes on an Apple Silicon MPS.*

## 📸 Screenshots

![Sentiment Analysis Web App](data/app_screenshot.png)

## 🏃 Quick Start

```bash
# Clone the repo
git clone <repo-url>
cd Sentiment

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```

## 📝 License

MIT


hf:-- 

https://huggingface.co/spaces/Adit-11/bert-sentiment-analyzer