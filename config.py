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
# Priorité : variable d'env → CSV drug reviews → XLSX synthétique (toujours dispo)
DATA_PATH: str  = os.getenv("DATA_PATH",  "data/patient_feedback.csv")
MODEL_PATH: str = os.getenv("MODEL_PATH", "models/sentiment_model.pkl")

# Task 1 : sentiment classifier (LogisticRegression / LinearSVC)
SENTIMENT_MODEL_PATH: str = MODEL_PATH

# Task 2 : theme classifier (LogisticRegression)
THEME_MODEL_PATH: str = MODEL_PATH.replace("sentiment_model", "theme_model")

# Task 3 : NMF topic model (unsupervised)
NMF_MODEL_PATH: str = MODEL_PATH.replace("sentiment_model", "nmf_model")

# Shared TF-IDF vectorizer — utilisé par les 3 tâches
TFIDF_PATH: str = MODEL_PATH.replace("sentiment_model.pkl", "tfidf_vectorizer.pkl")

# ── Dataset columns ─────────────────────────────────────
# Identiques pour les deux datasets (XLSX synthétique et CSV drug reviews)
TEXT_COL:         str = "Feedback"
LABEL_COL:        str = "Sentiment"
THEME_COL:        str = "Theme"
SATISFACTION_COL: str = "Satisfaction"

# ── Themes ──────────────────────────────────────────────
# Task 2 : labels de classification de thème
THEMES: list[str] = ["communication", "wait_time", "medication", "discharge"]

# ── NMF Topic Modeling ──────────────────────────────────
# Task 3 : hyperparamètres
NMF_N_COMPONENTS: int = 4    # 4 topics latents = 4 thèmes attendus
NMF_MAX_ITER: int     = 200
NMF_TOP_WORDS: int    = 10   # mots par topic dans les logs MLflow
