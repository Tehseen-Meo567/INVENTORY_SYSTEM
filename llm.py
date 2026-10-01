"""Groq wrapper. If there is no GROQ_API_KEY (or Groq fails) every function returns None
and the calling module falls back to a simple rule-based answer, so the demo never breaks."""
import os
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

def available():
    return bool(os.getenv("GROQ_API_KEY"))

def chat(system, user, temperature=0.3, max_tokens=600, json_mode=False):
    if not available():
        return None
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        r = client.chat.completions.create(model=MODEL, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
        return r.choices[0].message.content
    except Exception as e:
        print("LLM error:", e)
        return None
