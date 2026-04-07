# Healthcare Patient Feedback Analysis

> NLP pipeline for automated patient feedback analysis — Cambrian College AI Program, Winter 2026

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-orange)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

This project automatically analyzes patient drug reviews to extract actionable insights using three NLP tasks:

| Task | Type | Model | Key Metric |
|------|------|-------|-----------|
| Task 1 — Sentiment Classification | Supervised | LinearSVC | F1 Macro: 0.861 |
| Task 2 — Theme Classification | Supervised | LogReg Multinomial | Accuracy: 0.944 |
| Task 3 — Topic Modeling | Unsupervised | NMF | Reconstruction Error: 226.02 |

All three tasks share a single TF-IDF representation — no redundant vectorizer fitting.

---

## Dataset

**UCI Drug Reviews** — Kaggle (`jessicali9530/kuc-hackathon-winter-2018`)

- 66,657 real patient reviews
- 4 medical themes: Anxiety, Birth Control, Depression, Pain
- Sentiment derived from rating: 7-10 = Positive (1), 1-6 = Negative (0)
- License: CC BY 4.0

---

## Project Structure

```
healthcare-nlp/
├── data/
│   └── patient_feedback.csv        # UCI Drug Reviews dataset
├── notebooks/
│   └── proof_of_results.ipynb      # EDA + experiments + proof of results
├── src/
│   ├── preprocess.py               # Text cleaning, TF-IDF, lemmatization
│   ├── train.py                    # Task 1/2/3 training + MLflow tracking
│   ├── predict.py                  # Inference: sentiment + theme + NMF
│   └── llm_client.py               # OpenAI / Ollama fallback client
├── api/
│   └── main.py                     # FastAPI — all endpoints
├── static/
│   └── index.html                  # Dashboard frontend (Chart.js)
├── monitoring/
│   └── prometheus.yml              # Prometheus metrics config
├── config.py                       # API keys, paths, hyperparameters
├── docker-compose.yml              # api + mlflow + ollama + prometheus
├── Dockerfile                      # FastAPI image for Render
├── requirements.txt
├── render.yaml                     # Render deployment config
├── .env.example                    # Environment variables template
└── README.md
```

---

## NLP Pipeline

```
Raw Text
  └─► HTML unescape + lowercase + remove punctuation
  └─► Stopword removal + lemmatization (NLTK)
  └─► TF-IDF vectorization (fit on TRAIN only — no data leakage)
        ├─► Task 1: LinearSVC → Sentiment (0/1)
        ├─► Task 2: LogisticRegression → Theme (4 classes)
        └─► Task 3: NMF → 4 latent topics (unsupervised)
```

**Key preprocessing decisions:**
- `html.unescape()` required for drug review raw text (contains `&#039;`, `&amp;`, etc.)
- `min_df=2` removes hapax legomena
- `sublinear_tf=True` applies log normalization — better for long reviews
- Split before `vectorizer.fit()` prevents data leakage

---

## Setup

### Prerequisites

- Python 3.10+
- Git

### Installation

```bash
git clone https://github.com/your-username/healthcare-nlp.git
cd healthcare-nlp
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

### Environment variables

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

`.env.example`:
```
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest
MLFLOW_TRACKING_URI=mlruns
DATA_PATH=data/patient_feedback.csv
MODEL_PATH=models/sentiment_model.pkl
```

### Download dataset

```bash
kaggle datasets download -d jessicali9530/kuc-hackathon-winter-2018
unzip kuc-hackathon-winter-2018.zip -d data/
```

Then run the preprocessing notebook `notebooks/proof_of_results.ipynb` to generate `data/patient_feedback.csv`.

---

## Training

Runs all 3 tasks in a single command:

```bash
python -m src.train
```

Expected output:

```
[Preprocess] Loaded 66,657 rows | 26,765 unique tokens
[Preprocess] Train: 53,323 | Test: 13,331 | TF-IDF vocab: 10,000

TASK 1 — Sentiment Classification (SVM)
  Accuracy      : 0.8681
  F1 Macro      : 0.8608
  AUC           : 0.9327
  Recall (neg)  : 0.8209  ← primary metric

TASK 2 — Theme Classification (LogReg Multinomial)
  Accuracy      : 0.9444
  F1 Macro      : 0.9071

TASK 3 — NMF Topic Modeling (4 topics)
  Topic 1: period, month, birth, pill, control, birth control...
  Topic 2: anxiety, depression, feel, life, taking, day...
  Topic 3: pain, back, work, take, relief, hour...
  Topic 4: effect, side effect, side, experienced, nausea...
```

MLflow runs are stored in `mlruns/`. View them with:

```bash
mlflow ui
```

---

## Running the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` to access the dashboard.

### API Endpoints

| Method | Endpoint | Task | Description |
|--------|----------|------|-------------|
| GET | `/health` | — | API + model status |
| POST | `/analyze` | Task 1 | Predict sentiment (Positive/Negative) |
| POST | `/predict-theme` | Task 2 | Predict theme (4 classes) |
| GET | `/topics-nmf` | Task 3 | NMF latent topics + top words |
| GET | `/topics` | Dashboard | Sentiment distribution by theme |
| GET | `/samples` | Dashboard | Real feedback samples by theme |
| POST | `/insights` | LLM | AI-generated insight from real data |
| GET | `/metrics` | MLOps | Latest MLflow run metrics |

### Example requests

```bash
# Task 1 — Sentiment
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This medication completely changed my life for the better."}'

# Task 2 — Theme
curl -X POST http://localhost:8000/predict-theme \
  -H "Content-Type: application/json" \
  -d '{"text": "I have been struggling with anxiety attacks every morning."}'

# Task 3 — NMF Topics
curl http://localhost:8000/topics-nmf
```

---

## Docker

```bash
docker-compose up --build
```

Services started:
- `api` — FastAPI on port 8000
- `mlflow` — MLflow UI on port 5000
- `ollama` — Local LLM on port 11434
- `prometheus` — Metrics on port 9090

---

## Deployment on Render

The project is configured for Render via `render.yaml`.

```bash
git add .
git commit -m "feat: your message"
git push origin main
```

Render auto-deploys on every push to `main`.

Set the following environment variables in Render dashboard (Settings → Environment):
- `OPENAI_API_KEY`
- `DATA_PATH`
- `MODEL_PATH`

---

## Model Performance

### Task 1 — Sentiment Classification

```
              precision    recall  f1-score   support
    Negative       0.84      0.82      0.83      5187
    Positive       0.89      0.90      0.89      8144
    accuracy                           0.87     13331
   macro avg       0.86      0.86      0.86     13331
```

- **Primary metric: Recall (Negative) = 0.821** — undetected negative feedback is the highest risk in healthcare
- AUC-ROC: 0.933

### Task 2 — Theme Classification

```
               precision    recall  f1-score   support
      Anxiety       0.87      0.76      0.81      1622
Birth Control       0.99      1.00      0.99      7678
   Depression       0.85      0.91      0.88      2426
         Pain       0.95      0.94      0.95      1605
    accuracy                           0.94     13331
```

### Task 3 — NMF Topics

| Topic | Top Keywords |
|-------|-------------|
| Topic 1 | period, month, birth, pill, control, birth control, weight |
| Topic 2 | anxiety, depression, feel, life, taking, day, attack, medication |
| Topic 3 | pain, back, work, take, relief, hour, back pain, medicine |
| Topic 4 | effect, side effect, side, nausea, negative, bad side |

---

## Why These Metrics Are Realistic

The dataset has 107 unique tokens in the synthetic baseline vs **26,765 unique tokens** in the UCI Drug Reviews dataset. The previous perfect scores (F1 = 1.00) were caused by:

1. A synthetically generated dataset with only 107 vocabulary tokens
2. Data leakage: `vectorizer.fit()` was called on the full dataset before the train/test split

Both issues are fixed in the current version.

---

## Technologies

| Category | Tool |
|----------|------|
| Language | Python 3.10 |
| NLP | NLTK (stopwords, lemmatization) |
| ML | scikit-learn (TfidfVectorizer, LinearSVC, LogisticRegression, NMF) |
| Tracking | MLflow |
| API | FastAPI + Uvicorn |
| LLM | OpenAI GPT-4o-mini / Ollama llama3.2 |
| Monitoring | Prometheus |
| Frontend | Chart.js |
| Deployment | Docker + Render |

---


*Cambrian College — AI & Machine Learning Program — NLP Course — Winter 2026*
