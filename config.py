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

# ── MLflow (local file mode — no server required) ───────
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME: str     = "healthcare-sentiment"

# ── Data & Model ────────────────────────────────────────
DATA_PATH: str  = os.getenv("DATA_PATH",  "data/patient_feedback_dataset.xlsx")
MODEL_PATH: str = os.getenv("MODEL_PATH", "models/sentiment_model.pkl")

# ── Dataset columns ─────────────────────────────────────
TEXT_COL:         str = "Feedback"
LABEL_COL:        str = "Sentiment"
THEME_COL:        str = "Theme"
SATISFACTION_COL: str = "Satisfaction"
READMISSION_COL:  str = "Readmission"

THEMES: list[str] = ["communication", "wait_time", "medication", "discharge"]