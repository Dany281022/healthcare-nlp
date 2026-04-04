# tests/test_api.py
"""Unit tests for the FastAPI endpoints."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np


def get_client():
    """Import app only when needed to avoid startup model loading."""
    from api.main import app
    return TestClient(app)


def test_health_endpoint():
    print("Test 1: GET /health...")
    client = get_client()
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    print("  ✅ /health OK")


def test_analyze_endpoint_no_model():
    print("Test 2: POST /analyze (no model loaded)...")
    client = get_client()
    response = client.post("/analyze", json={"text": "The nurse was very kind."})
    # 503 expected when model not loaded
    assert response.status_code in [200, 503]
    print(f"  ✅ /analyze returned {response.status_code} (expected without model)")


def test_analyze_endpoint_with_mock_model():
    print("Test 3: POST /analyze (mock model)...")
    import api.main as app_module

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_vectorizer = MagicMock()
    mock_vectorizer.transform.return_value = MagicMock()

    app_module.model      = mock_model
    app_module.vectorizer = mock_vectorizer

    client   = get_client()
    response = client.post("/analyze", json={"text": "Great care from the staff."})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["Positive", "Negative"]
    assert data["prediction"] in [0, 1]
    print(f"  ✅ /analyze returned label={data['label']}")


def test_topics_endpoint():
    print("Test 4: GET /topics...")
    with patch("src.predict.get_topic_distribution") as mock_dist:
        mock_dist.return_value = {
            "communication": {"total": 267, "positive": 150, "negative": 117, "avg_satisfaction": 3.2},
            "wait_time":     {"total": 256, "positive": 120, "negative": 136, "avg_satisfaction": 2.8},
        }
        client   = get_client()
        response = client.get("/topics")
        assert response.status_code == 200
        print("  ✅ /topics OK")


def test_insights_endpoint():
    print("Test 5: POST /insights (mock LLM)...")
    with patch("src.predict.call_llm", return_value="Patients appreciate attentive staff."):
        client   = get_client()
        response = client.post("/insights", json={
            "theme":   "communication",
            "samples": ["Doctor explained everything well.", "Nurse was very kind."],
        })
        assert response.status_code == 200
        data = response.json()
        assert "insight" in data
        print(f"  ✅ /insights OK → {data['insight'][:60]}...")


if __name__ == "__main__":
    tests = [
        test_health_endpoint,
        test_analyze_endpoint_no_model,
        test_analyze_endpoint_with_mock_model,
        test_topics_endpoint,
        test_insights_endpoint,
    ]
    passed = failed = 0
    print("=" * 45)
    print("  TESTS - FastAPI Endpoints")
    print("=" * 45)
    for t in tests:
        try:
            t(); passed += 1
        except AssertionError as e:
            print(f"  ❌ FAILED: {t.__name__} → {e}"); failed += 1
        except Exception as e:
            print(f"  💥 ERROR:  {t.__name__} → {e}"); failed += 1
    print("=" * 45)
    print(f"  Result: {passed} passed / {failed} failed")
    print("=" * 45)