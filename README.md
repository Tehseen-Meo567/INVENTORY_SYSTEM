# SupplySense v2 - run it
1. `pip install -r requirements.txt`   (Python 3.10-3.13; if pip complains, use a fresh virtual environment)
2. `streamlit run app.py`
3. Paste your Groq key in the sidebar (or copy `.env.example` to `.env`). No key = offline demo mode.

## What is new in v2
- **ChromaDB** document search (rag.py). Sidebar shows the engine. First run downloads a small embedding
  model (needs internet). If that fails it falls back automatically to ChromaDB with offline embeddings, then to TF-IDF.
- **CrewAI agents** (crew_run.py + the "CrewAI Agents" tab). Tick "Demo mode" to see the screen without a key.
- **Groq key box** in the sidebar, model chooser, Test button.
- **Doc Search tab** to see ChromaDB working and compare it with simple word matching.

## Commands
- `python generate_data.py`          rebuild sample data
- `python pipeline.py`               run the fast (non-CrewAI) pipeline in the terminal
- `python crew_run.py`               run the CrewAI crew in the terminal (needs GROQ_API_KEY)
- `python test_app.py`               automatic click-through test of the app
- `python test_crew_offline.py`      tests the CrewAI wiring with a scripted model (no key)

## Environment switches
- `USE_CHROMA=0` force TF-IDF | `USE_CHROMA=hash` force ChromaDB with offline embeddings

| File | Owner idea | What it does |
|---|---|---|
| stock_watcher.py, budget_guard.py | Member 2 | Stock risk, order quantity, money and space check |
| vendor_finder.py, vendor_checker.py, rag.py | Member 3 | Seller comparison, document search (ChromaDB), trust score |
| negotiator.py | Member 4 | Emails, reading replies, bargaining inside limits |
| crew_run.py, pipeline.py, llm.py | Member 1 | CrewAI crew, glue code, Groq helper |
| app.py | Member 5 | Streamlit dashboard |
