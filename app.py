import datetime as dt
import pandas as pd, streamlit as st
import pipeline as pl, negotiator, llm

st.set_page_config(page_title="SupplySense", page_icon="📦", layout="wide")
COLORS = {"URGENT": "background-color:#fecaca", "WARNING": "background-color:#fde68a",
          "OK": "background-color:#bbf7d0", "OVERSTOCK": "background-color:#bfdbfe"}
def color_status(df, cols):
    s = df.style
    return (s.map if hasattr(s, "map") else s.applymap)(lambda v: COLORS.get(v, ""), subset=cols)

@st.cache_data
def get_data(): return pl.load_data()
@st.cache_resource
def get_rag(): return pl.get_retriever()

@st.cache_resource
def get_tfidf():
    from rag import Retriever
    return Retriever(pl.DATA / "vendor_docs", engine="tfidf")

data = get_data(); get_rag()
ss = st.session_state
ss.setdefault("cases", {}); ss.setdefault("ledger", []); ss.setdefault("chat", []); ss.setdefault("pid", None)

# ---------- sidebar ----------
with st.sidebar:
    st.title("📦 SupplySense")
    st.caption("AI supply chain assistant")
    as_of = st.date_input("Today's date (demo)", dt.date.fromisoformat(data["meta"]["as_of"]))
    budget = st.number_input("Budget for this order cycle (Rs.)", 0, 5_000_000, 650_000, 10_000)
    capacity = st.number_input("Warehouse capacity (units)", 500, 50_000, 6_000, 500)
    st.subheader("Owner's limits (guardrails)")
    target_pct = st.slider("Target price (% of list price)", 80, 100, 92)
    max_pct = st.slider("Never pay more than (% of list)", 90, 110, 100)
    rounds = st.slider("Max negotiation rounds", 1, 5, 3)
    st.divider()
    st.subheader("🧠 Groq AI brain")
    def _clear_key():
        llm.clear_key(); ss.applied_key = None; ss.key_input = ""; pl.reset_trust()
    key_in = st.text_input("Groq API key", type="password", key="key_input", placeholder="gsk_...  (or put it in .env)")
    model_in = st.selectbox("Model", llm.MODELS, index=llm.MODELS.index(llm.model()) if llm.model() in llm.MODELS else 0)
    if key_in and key_in != ss.get("applied_key"):
        llm.set_key(key_in); ss.applied_key = key_in; pl.reset_trust()
    if model_in != llm.model():
        llm.set_key(None, model_in); pl.reset_trust()
    b1, b2 = st.columns(2)
    if b1.button("Test Groq connection"):
        ok, msg = llm.test_connection(); (st.success if ok else st.error)(msg)
    b2.button("Clear key", on_click=_clear_key)
    st.write("Status:", "**Groq key set**" if llm.available() else "**Offline demo mode** (no key)")
    info = pl.get_retriever().info()
    st.write("📚 Document search:", f"**{info['engine']}**")

stock = pl.stock_table(data, as_of)
risky = stock[stock.status.isin(["URGENT", "WARNING"])]

tabs = st.tabs(["📊 Dashboard", "🚨 Stock Alerts", "🏭 Vendors", "🤝 Negotiation", "✅ Approval & Order", "📚 Doc Search", "🧠 CrewAI Agents", "💬 Ask Assistant"])

# ---------- dashboard ----------
with tabs[0]:
    st.header("Dashboard")
    saved = sum(x["saving"] for x in ss.ledger)
    c = st.columns(5)
    c[0].metric("Products tracked", len(stock)); c[1].metric("At risk of stock-out", len(risky))
    c[2].metric("Urgent", int((stock.status == "URGENT").sum())); c[3].metric("Orders approved", len(ss.ledger))
    c[4].metric("Money saved (Rs.)", f"{saved:,.0f}")
    st.subheader("How long will stock last vs. how long does delivery take?")
    chart = stock.set_index("name")[["days_left", "lead_days"]].rename(columns={"days_left": "Stock lasts (days)", "lead_days": "Delivery takes (days)"})
    st.bar_chart(chart)
    st.caption("If the first bar is shorter than the second, the product runs out before new stock can arrive.")
    st.subheader("Budget Guard: what we can afford this cycle")
    plan, summ = pl.plan_orders(data, stock, budget, capacity)
    if len(plan):
        st.dataframe(plan.rename(columns={"wanted_qty": "wanted", "approved_qty": "approved"}), hide_index=True, width="stretch")
        m = st.columns(3); m[0].metric("Budget", f"Rs. {summ['budget']:,.0f}"); m[1].metric("Planned spend", f"Rs. {summ['spent']:,.0f}"); m[2].metric("Left", f"Rs. {summ['left']:,.0f}")
    else: st.success("Nothing needs ordering.")

# ---------- stock alerts ----------
with tabs[1]:
    st.header("Stock Alerts")
    show = stock[["name", "stock", "daily_sales", "days_left", "lead_days", "status", "order_qty", "reason"]]
    st.dataframe(color_status(show, ["status"]), hide_index=True, width="stretch")
    if len(risky):
        pick = st.selectbox("Start a procurement case for:", risky.name.tolist())
        if st.button("Find vendors for this product ➜", type="primary"):
            ss.pid = risky[risky.name == pick].iloc[0].product_id; st.success("Case started. Open the **Vendors** tab.")

def current_row():
    if not ss.pid: return None
    r = stock[stock.product_id == ss.pid]; return r.iloc[0] if len(r) else None

# ---------- vendors ----------
with tabs[2]:
    st.header("Vendors")
    row = current_row()
    if row is None: st.info("Pick a product in the **Stock Alerts** tab first.")
    else:
        st.write(f"**{row['name']}** - need about **{row.order_qty}** units, stock lasts **{row.days_left:.0f}** days.")
        rv = pl.ranked_vendors(data, row); trust = pl.trust_for(data["vendors"])
        view = rv[["vendor_name", "city", "unit_price", "order_qty", "lead_days", "in_time", "trust", "score", "total_cost", "note"]]
        st.dataframe(view, hide_index=True, width="stretch")
        best = rv.iloc[0]; st.success(f"Recommended: **{best.vendor_name}** (score {best.score}, trust {best.trust}/100)")
        st.subheader("Why these trust scores? (proof from documents)")
        for _, v in rv.iterrows():
            t = trust[v.vendor_id]
            with st.expander(f"{v.vendor_name}: {t['score']}/100 - {t['summary']}  [{t['method']}]"):
                for fl in t["red_flags"]: st.error(fl)
                for ev in t["evidence"][:4]: st.caption(f"📄 {ev['source']}: {ev['text'][:200]}")

# ---------- negotiation ----------
with tabs[3]:
    st.header("Negotiation")
    row = current_row()
    if row is None: st.info("Pick a product in the **Stock Alerts** tab first.")
    else:
        rv = pl.ranked_vendors(data, row)
        name = st.selectbox("Negotiate with:", rv.vendor_name.tolist())
        v = rv[rv.vendor_name == name].iloc[0]; qty = int(v.order_qty)
        g = pl.default_guardrails(v.unit_price, max(v.lead_days + 3, int(row.days_left)), target_pct / 100, max_pct / 100, rounds)
        st.caption(f"List price Rs. {v.unit_price:,.0f} | target Rs. {g['target_price']:,} | max Rs. {g['max_price']:,} | max delivery {g['max_lead_days']} days | up to {g['max_rounds']} rounds")
        if st.button("Run AI negotiation (simulated vendor)", type="primary"):
            res = negotiator.run_negotiation(v, row["name"], qty, g)
            ss.cases[row.product_id] = dict(product_id=row.product_id, product=row["name"], qty=qty, vendor=v.to_dict(), negotiation=res,
                                            list_price=float(v.unit_price), date=as_of.isoformat(), approved=None)
        case = ss.cases.get(row.product_id)
        if case:
            for m in case["negotiation"]["transcript"]:
                with st.chat_message("user" if m["who"] == "vendor" else "assistant", avatar="🏭" if m["who"] == "vendor" else "🤖"):
                    st.write(f"**{'Vendor' if m['who']=='vendor' else 'Our AI'} - round {m['round']}**"); st.write(m["text"])
                    if m["who"] == "vendor": st.caption(f"Understood: {m['parsed']} → **{m['action']}**: {m['reason']}")
            r = case["negotiation"]
            if r["status"] == "accepted": st.success(f"Deal: Rs. {r['final_price']:,.0f} per piece (list Rs. {case['list_price']:,.0f}). Saving: Rs. {r['saving']:,.0f}")
            else: st.warning(f"No deal: {r.get('reason')}. Try another vendor.")
        with st.expander("Paste a real vendor reply and see what the AI does"):
            txt = st.text_area("Vendor reply", "We can do Rs. 1850 per piece for 180 units, delivery in 5 days. MOQ 100.")
            if st.button("Analyze reply"):
                p = negotiator.parse_reply(txt); a, nxt, why = negotiator.decide(p, g, 1, g["target_price"])
                st.json(p); st.write(f"Decision: **{a}**" + (f" - counter at Rs. {nxt}" if nxt else "") + f" ({why})")

# ---------- approval ----------
with tabs[4]:
    st.header("Approval & Order Paper")
    row = current_row(); case = ss.cases.get(row.product_id) if row is not None else None
    if not case or case["negotiation"]["status"] != "accepted": st.info("Finish a successful negotiation first.")
    else:
        n = case["negotiation"]; v = case["vendor"]
        st.subheader("Owner summary")
        a, b, c3 = st.columns(3); a.metric("Vendor", v["vendor_name"]); b.metric("Price per piece", f"Rs. {n['final_price']:,.0f}", f"-Rs. {case['list_price']-n['final_price']:,.0f}"); c3.metric("Total", f"Rs. {n['final_price']*case['qty']:,.0f}")
        st.write(f"Quantity **{case['qty']}**, delivery in **{n['lead_days']} days**, payment: {v['payment_terms']}, trust score **{pl.trust_for(data['vendors'])[v['vendor_id']]['score']}/100**, saving **Rs. {n['saving']:,.0f}**.")
        if case["approved"] is None:
            x, y = st.columns(2)
            if x.button("✅ Approve", type="primary"):
                case["approved"] = True; ss.ledger.append({"product": case["product"], "saving": n["saving"]}); st.rerun()
            if y.button("❌ Reject"): case["approved"] = False; st.rerun()
        elif case["approved"]:
            st.success("Approved. Order paper is ready.")
            st.download_button("⬇️ Download Purchase Order (PDF)", pl.make_po_pdf(case), file_name=f"PO_{case['product_id']}.pdf", mime="application/pdf")
        else: st.error("Rejected. Nothing was ordered.")

# ---------- document search (RAG) ----------
with tabs[5]:
    st.header("📚 Vendor Document Search (RAG)")
    rag = pl.get_retriever(); info = rag.info()
    m = st.columns(3); m[0].metric("Search engine", info["engine"]); m[1].metric("Text pieces stored", info["chunks"]); m[2].metric("Vendors", info["vendors"])
    st.caption(f"Stored in: {info['folder']}")
    if info["note"]: st.warning(info["note"] + " (the app fell back automatically)")
    st.write("Every paragraph in the vendor files (invoices, reviews, contracts) is stored as one piece. Ask a question and the search returns the closest pieces. This is what the Vendor Checker uses as proof.")
    names = data["vendors"].drop_duplicates("vendor_id").set_index("vendor_id").vendor_name.to_dict()
    q = st.text_input("Ask the documents", "Which vendor delivers late or sends damaged items?")
    c1, c2 = st.columns([3, 1])
    vsel = c1.selectbox("Search in", ["All vendors"] + [f"{k} - {v}" for k, v in names.items()])
    k = c2.slider("Results", 1, 10, 5)
    def show(hits):
        st.dataframe(pd.DataFrame([{"vendor": names[h["vendor_id"]], "file": h["source"], "match": h["score"], "text": h["text"]} for h in hits]), hide_index=True, width="stretch")
    if q:
        vid = None if vsel.startswith("All") else vsel.split(" - ")[0]
        st.subheader(f"Results from {info['engine']}"); show(rag.search(q, vid, k))
        if st.checkbox("Compare with simple word-matching (TF-IDF)"):
            st.subheader("Results from TF-IDF (no ChromaDB)"); show(get_tfidf().search(q, vid, k))
            st.caption("ChromaDB matches by meaning (with the default model), TF-IDF only by shared words. Try a question with different wording, like 'products arrived broken'.")

# ---------- CrewAI ----------
with tabs[6]:
    st.header("🧠 CrewAI Agents")
    st.write("Here the 5 agents are real **CrewAI agents**: each has a role and goal, chooses which tool to use, and hands its result to the next agent. The tools are the same modules used in the other tabs, so the owner's limits are still enforced by code.")
    demo = st.checkbox("Demo mode (scripted replies, no key needed). Shows the screen and proves the tools and handoffs work; it is NOT real AI.", value=not llm.available())
    if not demo and not llm.available(): st.warning("Paste a Groq key in the sidebar to run the real crew.")
    cc = st.columns(2)
    opts = ["Let the agents choose the most urgent"] + risky.name.tolist()
    pick = cc[0].selectbox("Product", opts)
    mode = cc[1].radio("Crew style", ["sequential", "hierarchical"], horizontal=True, help="Hierarchical adds a manager agent. Experimental, slower.")
    if st.button("▶ Run CrewAI crew", type="primary", disabled=not (demo or llm.available())):
        try:
            import crew_run
            fp = None if pick.startswith("Let") else risky[risky.name == pick].iloc[0].product_id
            with st.spinner("Agents are working... (real Groq runs can take 30-90 seconds)"):
                ss.crew = crew_run.run_crew(as_of.isoformat(), budget, capacity, target_pct / 100, max_pct / 100, rounds, fp, mode, "demo" if demo else None)
                ss.crew["demo"] = demo
        except Exception as e:
            ss.crew = None; st.error(f"Crew failed: {type(e).__name__}: {str(e)[:300]}. Tip: the Fast pipeline tabs still work, and Groq's free tier has rate limits.")
    res = ss.get("crew")
    if res:
        if res.get("demo"): st.info("This run used the scripted demo model, not a real AI.")
        st.subheader("Owner summary (final output of the crew)"); st.success(res["final"])
        case = res["case"]
        if case:
            n = case["negotiation"]
            st.write(f"**Deal found:** {case['vendor']['vendor_name']}, {n['status']}" + (f", Rs. {n['final_price']:,.0f} per piece, saving Rs. {n['saving']:,.0f}" if n["status"] == "accepted" else ""))
            if n["status"] == "accepted" and st.button("Send this deal to the Approval tab"):
                ss.cases[case["product_id"]] = case; ss.pid = case["product_id"]; st.success("Done. Open the Approval & Order tab.")
        st.subheader("What each agent did")
        for t in res["trace"]:
            if t["kind"] == "tool":
                st.markdown("🔧 " + t["text"])
                if t.get("thought"): st.caption("Thought: " + t["thought"])
            else:
                with st.expander(f"✅ {t.get('agent') or 'Agent'} finished"): st.write(t["text"])

# ---------- assistant ----------
with tabs[7]:
    st.header("Ask the Assistant")
    st.caption("Try: 'Which products will run out soon?' or 'Which vendor is not trustworthy?'")
    for m in ss.chat:
        with st.chat_message(m["role"]): st.write(m["content"])
    if q := st.chat_input("Ask about stock or vendors..."):
        ss.chat.append({"role": "user", "content": q}); ss.chat.append({"role": "assistant", "content": pl.ask_assistant(q, stock, data)}); st.rerun()
