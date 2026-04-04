# src/llm_client.py
"""
LLM client with automatic fallback: OpenAI → Ollama.
Adapter pattern: only this function is imported by other modules.
"""
import httpx
from openai import OpenAI, APIConnectionError, AuthenticationError
from config import OPENAI_API_KEY, OPENAI_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL


def _call_openai(prompt: str) -> str:
    """Call OpenAI GPT with the given prompt."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


def _call_ollama(prompt: str) -> str:
    """Call a local Ollama model via its REST API."""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    response = httpx.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"]


def call_llm(prompt: str) -> str:
    """
    Single entry point for all LLM calls.
    Tries OpenAI first, falls back to Ollama on failure.
    """
    if OPENAI_API_KEY:
        try:
            print("[LLM] Using OpenAI...")
            return _call_openai(prompt)
        except (APIConnectionError, AuthenticationError) as e:
            print(f"[LLM] OpenAI failed ({e}), falling back to Ollama...")
        except Exception as e:
            print(f"[LLM] Unexpected OpenAI error ({e}), falling back to Ollama...")

    try:
        print(f"[LLM] Using Ollama ({OLLAMA_MODEL})...")
        return _call_ollama(prompt)
    except Exception as e:
        raise RuntimeError(f"All LLM providers failed. Last error: {e}")