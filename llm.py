"""Groq wrapper. If there is no GROQ_API_KEY (or Groq fails) every function returns None
and the calling module falls back to a simple rule-based answer, so the demo never breaks.
The key can come from .env, the environment, or the app sidebar (set_key)."""
import os
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

def model():
    return os.getenv("GROQ_MODEL", MODELS[0])

def available():
    return bool(os.getenv("GROQ_API_KEY"))

def set_key(key, model_name=None):
    key = (key or "").strip()
    if key: os.environ["GROQ_API_KEY"] = key
    if model_name: os.environ["GROQ_MODEL"] = model_name

def clear_key():
    os.environ.pop("GROQ_API_KEY", None)

def chat(system, user, temperature=0.3, max_tokens=600, json_mode=False):
    if not available():
        return None
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=25, max_retries=1)
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        r = client.chat.completions.create(model=model(), temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
        return r.choices[0].message.content
    except Exception as e:
        print("LLM error:", e)
        return None

def test_connection():
    """Returns (ok, message). Used by the sidebar button."""
    if not available(): return False, "No key set."
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=20, max_retries=0)
        r = client.chat.completions.create(model=model(), max_tokens=8, messages=[{"role": "user", "content": "Say OK"}])
        return True, f"Connected. Model {model()} replied: {r.choices[0].message.content.strip()[:30]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"
