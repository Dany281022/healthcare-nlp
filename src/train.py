# api/main.py
"""
FastAPI application — Healthcare Patient Feedback Analysis.
Serves the dashboard frontend + REST endpoints.

Task 1 : POST /analyze          — sentiment prediction
Task 2 : POST /predict-theme    — theme prediction
Task 3 : GET  /topics-nmf       — NMF latent topics
         GET  /topics           — sentiment distribution by theme (dashboard)
         POST /insights         — LLM insight from real feedback samples
         GET  /samples          — real feedback samples by theme
         GET  /metrics          — MLflow latest run metrics
         GET  /health           — API + model status
"""
import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from config import (
    MODEL_PATH, DATA_PATH, THEMES,
    MLFLOW_TRACKING_URI, EXPERIMENT_NAME,
    TFIDF_PATH,
)
from src.preprocess import clean_text
from src.predict import (
    predict_sentiment,
    predict_theme,
    get_topics_nmf,
    get_topic_distribution,
    get_real_samples,
    generate_llm_insight,
)

app = FastAPI(
    title="Healthcare Feedback API",
    description="Task 1: Sentiment | Task 2: Theme | Task 3: NMF Topics | LLM Insights",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)


# ── Load models on startup ─────────────────────────────
sentiment_model = None
vectorizer      = None

@app.on_event("startup")
def load_models() -> None:
    """Load Task 1 sentiment model + shared TF-IDF vectorizer at startup."""
    global sentiment_model, vectorizer
    get_topic_distribution.cache_clear()
    try:
        sentiment_model = joblib.load(MODEL_PATH)
        try:
            vectorizer = joblib.load(TFIDF_PATH)
        except FileNotFoundError:
            vectorizer = joblib.load(MODEL_PATH.replace(".pkl", "_vectorizer.pkl"))
        print("[API] Sentiment model + vectorizer loaded.")
    except FileNotFoundError:
        print("[API] WARNING: Models not found. Run src/train.py first.")


# ── Schemas ────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    text: str

class InsightRequest(BaseModel):
    theme: str   = "general"
    samples: list[str] = []

class AnalyzeResponse(BaseModel):
    text:       str
    prediction: int
    label:      str

class ThemeResponse(BaseModel):
    text:       str
    theme:      str
    confidence: float

class InsightResponse(BaseModel):
    theme:   str
    insight: str

class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool


# ── Endpoints ──────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=sentiment_model is not None)


# Task 1
@app.post("/analyze", response_model=AnalyzeResponse, tags=["Task 1 — Sentiment"])
def analyze_feedback(request: FeedbackRequest) -> AnalyzeResponse:
    """Task 1 : predict sentiment (Positive / Negative)."""
    if sentiment_model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run src/train.py first.")
    try:
        result = predict_sentiment(request.text, sentiment_model, vectorizer)
        return AnalyzeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Task 2
@app.post("/predict-theme", response_model=ThemeResponse, tags=["Task 2 — Theme"])
def predict_theme_endpoint(request: FeedbackRequest) -> ThemeResponse:
    """Task 2 : predict healthcare theme (communication / wait_time / medication / discharge)."""
    if vectorizer is None:
        raise HTTPException(status_code=503, detail="Vectorizer not loaded. Run src/train.py first.")
    try:
        result = predict_theme(request.text, vectorizer)
        return ThemeResponse(**result)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Theme model not found. Run src/train.py first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Task 3
@app.get("/topics-nmf", tags=["Task 3 — NMF Topics"])
def get_nmf_topics() -> dict:
    """Task 3 : return NMF latent topics with top words per topic."""
    try:
        return get_topics_nmf()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="NMF model not found. Run src/train.py first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Dashboard
@app.get("/topics", tags=["Dashboard"])
def get_topics() -> dict:
    """Return sentiment distribution grouped by healthcare theme."""
    try:
        return get_topic_distribution(DATA_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/samples", tags=["Dashboard"])
def get_samples(theme: str = "communication", n: int = 6) -> dict:
    """Return n real feedback samples for a given theme."""
    if theme not in THEMES:
        raise HTTPException(status_code=400, detail=f"Unknown theme. Choose from: {THEMES}")
    try:
        return {"theme": theme, "samples": get_real_samples(theme, n)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/insights", response_model=InsightResponse, tags=["LLM"])
def generate_insight(request: InsightRequest) -> InsightResponse:
    """Generate an LLM-powered insight from real feedback samples."""
    try:
        samples = request.samples if request.samples else get_real_samples(request.theme)
        insight = generate_llm_insight(samples, theme=request.theme)
        return InsightResponse(theme=request.theme, insight=insight)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["MLOps"])
def get_model_metrics() -> dict:
    """Return latest MLflow run metrics for all 3 tasks."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        exp    = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            raise HTTPException(status_code=404, detail="No experiment found. Run src/train.py first.")
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=10,
        )
        if not runs:
            raise HTTPException(status_code=404, detail="No training runs found. Run src/train.py first.")

        result = {}
        for run in runs:
            task = run.data.tags.get("task", "unknown")
            if task not in result:
                result[task] = {
                    "run_id":   run.info.run_id,
                    "run_name": run.data.tags.get("mlflow.runName", ""),
                    "metrics":  run.data.metrics,
                    "params":   run.data.params,
                }
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dashboard ──────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_dashboard() -> FileResponse:
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)