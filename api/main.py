# api/main.py
"""
FastAPI application — Healthcare Patient Feedback Analysis.
Serves the dashboard frontend + REST endpoints.
"""
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from config import MODEL_PATH, DATA_PATH, THEMES

app = FastAPI(
    title="Healthcare Feedback API",
    description="NLP-powered patient feedback analysis with LLM insights",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

# ── Serve static files ─────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
def serve_dashboard() -> FileResponse:
    """Serve the main dashboard HTML page."""
    return FileResponse("static/index.html")

# ── Load model on startup ──────────────────────────────
model      = None
vectorizer = None

@app.on_event("startup")
def load_model() -> None:
    """Load trained model and vectorizer at startup."""
    global model, vectorizer
    try:
        model      = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(MODEL_PATH.replace(".pkl", "_vectorizer.pkl"))
        print("[API] Model loaded successfully.")
    except FileNotFoundError:
        print("[API] WARNING: Model not found. Run src/train.py first.")


# ── Request / Response schemas ─────────────────────────
class FeedbackRequest(BaseModel):
    text: str

class InsightRequest(BaseModel):
    theme: str = "general"
    samples: list[str]

class AnalyzeResponse(BaseModel):
    text:       str
    prediction: int
    label:      str

class InsightResponse(BaseModel):
    theme:   str
    insight: str

class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool


# ── Endpoints ──────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    """Health probe for Render and monitoring systems."""
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.post("/analyze", response_model=AnalyzeResponse, tags=["NLP"])
def analyze_feedback(request: FeedbackRequest) -> AnalyzeResponse:
    """Predict sentiment for a single patient feedback text."""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        from src.preprocess import clean_text
        cleaned = clean_text(request.text)
        X       = vectorizer.transform([cleaned])
        pred    = int(model.predict(X)[0])
        label   = "Positive" if pred == 1 else "Negative"
        return AnalyzeResponse(text=request.text, prediction=pred, label=label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/insights", response_model=InsightResponse, tags=["LLM"])
def generate_insight(request: InsightRequest) -> InsightResponse:
    """Generate an LLM-powered insight from feedback samples."""
    try:
        from src.predict import generate_llm_insight
        insight = generate_llm_insight(request.samples, theme=request.theme)
        return InsightResponse(theme=request.theme, insight=insight)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/topics", tags=["NLP"])
def get_topics() -> dict:
    """Return sentiment distribution grouped by healthcare theme."""
    try:
        from src.predict import get_topic_distribution
        return get_topic_distribution(DATA_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["MLOps"])
def get_model_metrics() -> dict:
    """Return latest model performance metrics from MLflow."""
    try:
        import mlflow
        from config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        exp    = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            return {"message": "No experiment found. Run training first."}
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            return {"message": "No runs found."}
        latest = runs[0]
        return {
            "run_id":  latest.info.run_id,
            "status":  latest.info.status,
            "metrics": latest.data.metrics,
            "params":  latest.data.params,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)