# src/predict.py
"""
Inference module: sentiment prediction + LLM-powered insight generation.
Loads trained model and vectorizer, runs predictions, calls LLM for summaries.
"""
import joblib
import pandas as pd
from config import MODEL_PATH, DATA_PATH, THEMES
from src.llm_client import call_llm


def load_model() -> tuple:
    """Load trained classifier and TF-IDF vectorizer from disk."""
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(MODEL_PATH.replace(".pkl", "_vectorizer.pkl"))
    print(f"[Predict] Model loaded from {MODEL_PATH}")
    return model, vectorizer


def predict_sentiment(text: str, model, vectorizer) -> dict:
    """
    Predict sentiment for a single feedback string.

    Args:
        text: Raw patient feedback text.
        model: Trained sklearn classifier.
        vectorizer: Fitted TF-IDF vectorizer.
    Returns:
        dict with label (Positive/Negative) and raw prediction.
    """
    from src.preprocess import clean_text
    cleaned = clean_text(text)
    X       = vectorizer.transform([cleaned])
    pred    = model.predict(X)[0]
    label   = "Positive" if pred == 1 else "Negative"
    return {"text": text, "prediction": int(pred), "label": label}


def get_topic_distribution(data_path: str = DATA_PATH) -> dict:
    """
    Return sentiment distribution grouped by Theme from the full dataset.
    Uses Theme column: communication, wait_time, medication, discharge.
    """
    df = pd.read_excel(data_path)
    result = {}
    for theme in THEMES:
        subset = df[df["Theme"] == theme]
        pos = int((subset["Sentiment"] == 1).sum())
        neg = int((subset["Sentiment"] == 0).sum())
        avg_sat = round(float(subset["Satisfaction"].mean()), 2)
        result[theme] = {
            "total":           len(subset),
            "positive":        pos,
            "negative":        neg,
            "avg_satisfaction": avg_sat,
        }
    return result


def generate_llm_insight(feedback_samples: list[str], theme: str = "general") -> str:
    """
    Use LLM to generate a human-readable insight from a list of feedback strings.

    Args:
        feedback_samples: List of raw patient feedback texts.
        theme: Healthcare theme context (e.g. 'communication').
    Returns:
        LLM-generated insight string.
    """
    joined = "\n- ".join(feedback_samples[:10])
    prompt = f"""You are a healthcare quality analyst.
Analyze the following patient feedback samples related to: {theme}.

Feedback:
- {joined}

Provide a concise insight (3-5 sentences) covering:
1. Main recurring issues or praise
2. Actionable recommendation for the healthcare team
3. Overall sentiment assessment

Be factual, professional, and concise."""

    return call_llm(prompt)


if __name__ == "__main__":
    model, vectorizer = load_model()
    sample = "The nurse was very attentive and explained everything clearly."
    result = predict_sentiment(sample, model, vectorizer)
    print(f"[Predict] {result}")