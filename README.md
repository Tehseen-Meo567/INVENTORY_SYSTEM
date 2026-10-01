# SupplySense - run in 3 commands
1. `pip install -r requirements.txt`
2. (optional, for real AI answers) copy `.env.example` to `.env` and paste your Groq key. Without it the app runs in offline demo mode.
3. `streamlit run app.py`

`python generate_data.py` rebuilds the sample data. `python pipeline.py` runs the whole chain in the terminal.

| File | Owner idea | What it does |
|---|---|---|
| stock_watcher.py | Member 2 | When will each product run out, how much to order |
| budget_guard.py | Member 2 | Money + storage check |
| vendor_finder.py | Member 3 | Compare sellers, rank them |
| rag.py, vendor_checker.py | Member 3 | Read vendor documents, trust score with proof |
| negotiator.py | Member 4 | Emails, understand replies, bargain inside limits |
| pipeline.py, llm.py | Member 1 | Connects everything, Groq helper, PO PDF |
| app.py | Member 5 | Streamlit dashboard |
