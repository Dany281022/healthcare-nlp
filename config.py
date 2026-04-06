# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ─────────────────────────────────────────────
OPENAI_API_KEY: str  = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Ollama (local fallback) ─────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str    = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

# ── MLflow ──────────────────────────────────────────────
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME: str     = "healthcare-sentiment"

# ── Data & Model paths ──────────────────────────────────
DATA_PATH: str  = os.getenv("DATA_PATH",  "data/patient_feedback.csv")
MODEL_PATH: str = os.getenv("MODEL_PATH", "models/sentiment_model.pkl")

# Task 1 : sentiment classifier
SENTIMENT_MODEL_PATH: str = MODEL_PATH

# Task 2 : theme classifier
THEME_MODEL_PATH: str = MODEL_PATH.replace("sentiment_model", "theme_model")

# Task 3 : NMF topic model
NMF_MODEL_PATH: str = MODEL_PATH.replace("sentiment_model", "nmf_model")

# Shared TF-IDF vectorizer
TFIDF_PATH: str = MODEL_PATH.replace("sentiment_model.pkl", "tfidf_vectorizer.pkl")

# ── Dataset columns ─────────────────────────────────────
TEXT_COL:         str = "Feedback"
LABEL_COL:        str = "Sentiment"
THEME_COL:        str = "Theme"
SATISFACTION_COL: str = "Satisfaction"

# ── Themes ──────────────────────────────────────────────
# Top 4 conditions from UCI Drug Reviews dataset
THEMES: list[str] = ["Anxiety", "Birth Control", "Depression", "Pain"]

# ── NMF Topic Modeling ──────────────────────────────────
NMF_N_COMPONENTS: int = 4
NMF_MAX_ITER: int     = 200
NMF_TOP_WORDS: int    = 10
