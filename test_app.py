from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=60).run()
assert not at.exception, at.exception
print("1) loads OK | tabs:", len(at.tabs), "| metrics:", [(m.label, m.value) for m in at.metric][:5])
# start case: click 'Find vendors' button
btn = [b for b in at.button if "Find vendors" in b.label][0]; btn.click().run(); assert not at.exception, at.exception
print("2) case started:", at.session_state.pid)
btn = [b for b in at.button if "Run AI negotiation" in b.label][0]; btn.click().run(); assert not at.exception, at.exception
case = at.session_state.cases[at.session_state.pid]; print("3) negotiation:", case["negotiation"]["status"], case["negotiation"]["final_price"])
btn = [b for b in at.button if "Approve" in b.label][0]; btn.click().run(); assert not at.exception, at.exception
print("4) approved. ledger:", at.session_state.ledger)
at.chat_input[0].set_value("Which products will run out soon?").run(); assert not at.exception, at.exception
print("5) chat:", at.session_state.chat[-1]["content"][:120].replace("\n"," | "))
print("ALL OK")
