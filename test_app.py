from streamlit.testing.v1 import AppTest
import os
os.environ["USE_CHROMA"] = os.getenv("USE_CHROMA", "hash")
at = AppTest.from_file("app.py", default_timeout=120).run()
assert not at.exception, at.exception
print("1) loads OK | tabs:", len(at.tabs), "| metrics:", [(m.label, m.value) for m in at.metric][:5])
[b for b in at.button if "Find vendors" in b.label][0].click().run(); assert not at.exception, at.exception
print("2) case started:", at.session_state.pid)
[b for b in at.button if "Run AI negotiation" in b.label][0].click().run(); assert not at.exception, at.exception
case = at.session_state.cases[at.session_state.pid]; print("3) negotiation:", case["negotiation"]["status"], case["negotiation"]["final_price"])
[b for b in at.button if "Approve" in b.label][0].click().run(); assert not at.exception, at.exception
print("4) approved. ledger:", at.session_state.ledger)
# doc search tab
print("5) doc search metrics:", [(m.label, m.value) for m in at.metric if m.label in ("Search engine", "Text pieces stored", "Vendors")])
# crew demo
[b for b in at.button if "Run CrewAI crew" in b.label][0].click().run(); assert not at.exception, at.exception
print("6) crew demo run:", at.session_state.crew["final"][:70], "| trace items:", len(at.session_state.crew["trace"]))
# Groq key box (fake key; no network): must not crash, status flips
at.sidebar.text_input[0].set_value("gsk_fake").run(); assert not at.exception, at.exception
print("7) key applied -> env set:", bool(os.getenv("GROQ_API_KEY")))
at.chat_input[0].set_value("Which products will run out soon?").run(); assert not at.exception, at.exception
print("8) chat:", at.session_state.chat[-1]["content"][:60].replace("\n", " | "))
print("ALL OK")
