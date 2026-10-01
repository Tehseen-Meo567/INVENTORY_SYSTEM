"""CrewAI version of SupplySense: 5 agents + tools that wrap the existing modules.

Run in terminal:   python crew_run.py            (needs GROQ_API_KEY in .env or environment)
Or use the 'CrewAI Agents' tab in the app.

Design rule: the AI agents THINK and CHOOSE which tool to call, but every money rule (budget, max price,
delivery limit) stays inside the tools as normal code, so an agent can never go past the owner's limits."""
import os, re
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true"); os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import pipeline as pl, negotiator, budget_guard, llm as llm_helper

CFG, STATE, TRACE = {}, {}, []

def _pid(x):
    m = re.search(r"P\d{2}", str(x).upper()); return m.group(0) if m else str(x).strip()
def _vid(x):
    m = re.search(r"V\d", str(x).upper()); return m.group(0) if m else str(x).strip()
def _row(pid):
    r = CFG["stock"][CFG["stock"].product_id == pid]; return r.iloc[0] if len(r) else None

# ------------------------------------------------------------------ TOOLS (wrap existing code)
@tool("check_stock")
def check_stock(scope: str) -> str:
    """Lists products by stock risk. Use scope 'risky' for urgent/warning products or 'all' for every product."""
    df = CFG["stock"] if "all" in str(scope).lower() else CFG["stock"][CFG["stock"].status.isin(["URGENT", "WARNING"])]
    if df.empty: return "No products are at risk."
    return "\n".join(f"{r.product_id} | {r['name']} | {r.status} | stock {r.stock} | sells {r.daily_sales}/day | stock lasts {r.days_left} days | "
                     f"delivery ~{r.lead_days} days | suggested order {r.order_qty} | {r.reason}" for _, r in df.iterrows())

@tool("find_vendors")
def find_vendors(product_id: str) -> str:
    """Returns the ranked list of sellers for a product id (like P01) with price, minimum order, delivery days, trust score and ranking score."""
    pid = _pid(product_id); row = _row(pid)
    if row is None: return f"Unknown product id {product_id}."
    r = row.copy()
    if r.order_qty == 0: r["order_qty"] = 50
    rv = pl.ranked_vendors(CFG["data"], r); STATE.setdefault("vendors", {})[pid] = rv
    return "\n".join(f"{v.vendor_id} | {v.vendor_name} | Rs.{v.unit_price} | order {int(v.order_qty)} units | delivery {v.lead_days} days | "
                     f"arrives in time: {v.in_time} | trust {v.trust}/100 | score {v.score} | {v.note}" for _, v in rv.iterrows())

@tool("search_vendor_documents")
def search_vendor_documents(vendor_id: str, question: str) -> str:
    """Searches one vendor's invoices, reviews and contract notes (ChromaDB) and returns the most relevant passages as proof."""
    hits = pl.get_retriever().search(str(question), _vid(vendor_id), k=5)
    return "\n".join(f"[{h['source']}] (match {h['score']}) {h['text']}" for h in hits) or "No documents found."

@tool("check_budget")
def check_budget(product_id: str, vendor_id: str) -> str:
    """Checks if buying the suggested quantity from this vendor fits the owner's budget and warehouse space."""
    pid, vid = _pid(product_id), _vid(vendor_id); row = _row(pid); rv = STATE.get("vendors", {}).get(pid)
    if row is None or rv is None or vid not in set(rv.vendor_id): return "Call find_vendors for this product first, using a valid vendor id."
    v = rv[rv.vendor_id == vid].iloc[0]
    plan, s = budget_guard.check_budget([dict(product_id=pid, name=row["name"], status=row.status, qty=int(v.order_qty), unit_price=float(v.unit_price))],
                                        CFG["budget"], CFG["capacity"], int(CFG["stock"].stock.sum()))
    r = plan.iloc[0]
    return f"Decision {r.decision}: approved {r.approved_qty} of {r.wanted_qty} units, cost Rs.{r.cost:,.0f}. Budget Rs.{s['budget']:,.0f}, left Rs.{s['left']:,.0f}. Note: {r.note}"

@tool("negotiate")
def negotiate(vendor_id: str, product_id: str) -> str:
    """Runs the multi-round price negotiation with a vendor inside the owner's limits and returns the transcript summary and the final deal."""
    pid, vid = _pid(product_id), _vid(vendor_id); row = _row(pid); rv = STATE.get("vendors", {}).get(pid)
    if row is None or rv is None or vid not in set(rv.vendor_id): return "Call find_vendors for this product first, using a valid vendor id."
    v = rv[rv.vendor_id == vid].iloc[0]; qty = int(v.order_qty)
    g = pl.default_guardrails(v.unit_price, max(v.lead_days + 3, int(row.days_left)), CFG["target_pct"], CFG["max_pct"], CFG["rounds"])
    res = negotiator.run_negotiation(v, row["name"], qty, g)
    STATE["case"] = dict(product_id=pid, product=row["name"], qty=qty, vendor=v.to_dict(), negotiation=res, list_price=float(v.unit_price), date=CFG["date"], approved=None)
    lines = [f"Owner limits: target Rs.{g['target_price']}, max Rs.{g['max_price']}, max delivery {g['max_lead_days']} days, {g['max_rounds']} rounds."]
    for m in res["transcript"]:
        if m["who"] == "vendor": lines.append(f"Round {m['round']}: vendor asked Rs.{m['parsed']['price']} -> our decision: {m['action']} ({m['reason']})")
    lines.append(f"RESULT: {res['status']}" + (f", final price Rs.{res['final_price']:,.0f}, saving Rs.{res['saving']:,.0f}, delivery {res.get('lead_days')} days." if res["status"] == "accepted" else f". {res.get('reason')}"))
    return "\n".join(lines)

# ------------------------------------------------------------------ TRACE (what each agent did)
def _step(step):
    try:
        tool_name = getattr(step, "tool", None)
        if tool_name: TRACE.append({"kind": "tool", "text": f"Used tool **{tool_name}** with input {getattr(step, 'tool_input', '')}", "thought": (getattr(step, "thought", "") or "").strip()[:400]})
    except Exception: pass
def _task_done(out):
    try: TRACE.append({"kind": "task", "agent": getattr(out, "agent", ""), "text": str(getattr(out, "raw", out))})
    except Exception: pass


# ------------------------------------------------------------------ SCRIPTED DEMO MODEL (no key needed)
# NOT real AI. It follows a fixed script in the same Thought/Action format a real model uses, so the app can show
# the CrewAI screen and prove the tools + handoffs work. A real Groq model decides these steps by itself.
from crewai import BaseLLM
class ScriptedDemoLLM(BaseLLM):
    def __init__(self):
        super().__init__(model="scripted-demo"); self.calls = {}
    def supports_function_calling(self): return False
    def call(self, messages, tools=None, callbacks=None, available_functions=None, from_task=None, from_agent=None, **kw):
        role = from_agent.role if from_agent else "x"; n = self.calls.get(role, 0); self.calls[role] = n + 1
        pid = CFG["stock"][CFG["stock"].status.isin(["URGENT", "WARNING"])].iloc[0].product_id if CFG.get("focus") is None else CFG["focus"]
        best = STATE.get("vendors", {}).get(pid)
        vid = best.iloc[0].vendor_id if best is not None else "V3"
        plan = {"Stock Watcher": ("check_stock", '{"scope": "risky"}', f"[Scripted demo] Most urgent product is {pid}. See the tool result above for days left and order quantity."),
                "Vendor Finder": ("find_vendors", f'{{"product_id": "{pid}"}}', "[Scripted demo] Shortlisted the top 3 sellers from the ranked table."),
                "Vendor Checker": ("search_vendor_documents", f'{{"vendor_id": "{vid}", "question": "late deliveries damaged goods reliability"}}', f"[Scripted demo] Evidence checked. Recommended seller: {vid}."),
                "Budget Guard": ("check_budget", f'{{"product_id": "{pid}", "vendor_id": "{vid}"}}', "[Scripted demo] Budget and storage checked (see tool result)."),
                "Negotiator": ("negotiate", f'{{"vendor_id": "{vid}", "product_id": "{pid}"}}', "[Scripted demo] Negotiation finished inside the owner's limits. Recommendation: APPROVE if the deal below looks right.")}[role]
        if n == 0: return f"Thought: I should use my tool.\nAction: {plan[0]}\nAction Input: {plan[1]}\n"
        return f"Thought: I now know the final answer\nFinal Answer: {plan[2]}"

# ------------------------------------------------------------------ CREW
def make_llm():
    return LLM(model=f"groq/{llm_helper.model()}", api_key=os.getenv("GROQ_API_KEY"), temperature=0.2, max_tokens=900)

def build_crew(llm_obj, focus_pid=None, mode="sequential"):
    A = lambda role, goal, back, tools: Agent(role=role, goal=goal, backstory=back, tools=tools, llm=llm_obj, verbose=False, allow_delegation=False, max_iter=6)
    watcher = A("Stock Watcher", "Find which products will run out before new stock can arrive and say how much to order",
                "You are an inventory analyst for a small online mobile accessories shop in Pakistan. You explain risks in simple words.", [check_stock])
    finder = A("Vendor Finder", "Shortlist the best sellers for the product that needs restocking",
               "You are a sourcing officer who compares price, delivery time and minimum order quantity.", [find_vendors])
    checker = A("Vendor Checker", "Judge which sellers are trustworthy using evidence from their invoices, reviews and contracts",
                "You are a careful supplier auditor. You never trust a seller without evidence and you always quote the documents.", [search_vendor_documents])
    guard = A("Budget Guard", "Make sure the order fits the owner's budget and warehouse space",
              "You are the shop's accountant. You protect cash flow.", [check_budget])
    nego = A("Negotiator", "Get the best possible price from the chosen seller without breaking the owner's limits",
             "You are a polite but firm purchasing manager. The owner's limits are never negotiable.", [negotiate])
    focus = f" Focus only on product {focus_pid}." if focus_pid else " Pick the single most urgent product."
    t1 = Task(description="Call check_stock with scope 'risky'." + focus + " Report the product id, name, days until stock-out, delivery time and the suggested order quantity, in plain words.",
              expected_output="Product id, name, risk reason and order quantity in 3-4 lines.", agent=watcher)
    t2 = Task(description="Using the product from the previous task, call find_vendors with its product id. Shortlist the top 3 sellers and give one line on each (price, delivery, trust).",
              expected_output="Top 3 sellers with vendor ids and one line each.", agent=finder, context=[t1])
    t3 = Task(description="For each shortlisted seller call search_vendor_documents with a question about late deliveries, damaged goods and reliability. Decide who is trustworthy and who is risky, quoting the evidence. Recommend ONE best seller by vendor id.",
              expected_output="Trust verdict for each seller with quoted evidence, and the single recommended vendor id.", agent=checker, context=[t1, t2])
    t4 = Task(description="Call check_budget with the product id and the recommended vendor id. Report if it fits the budget and storage and what to change if not.",
              expected_output="Budget and storage decision in 2-3 lines.", agent=guard, context=[t1, t2, t3])
    t5 = Task(description="Call negotiate with the recommended vendor id and the product id. If the result is a walk-away, call negotiate once more with the second-best seller from the shortlist. Then write a short summary for the owner.",
              expected_output="Owner summary: seller, final price, saving, delivery days, trust, and a clear recommendation to approve or reject.", agent=nego, context=[t1, t2, t3, t4])
    kw = dict(agents=[watcher, finder, checker, guard, nego], tasks=[t1, t2, t3, t4, t5], verbose=False, step_callback=_step, task_callback=_task_done, max_rpm=20)
    if mode == "hierarchical": kw.update(process=Process.hierarchical, manager_llm=llm_obj)   # experimental
    else: kw.update(process=Process.sequential)
    return Crew(**kw)

def run_crew(as_of, budget=650_000, capacity=6_000, target_pct=0.92, max_pct=1.0, rounds=3, product_id=None, mode="sequential", llm_obj=None):
    data = pl.load_data(); stock = pl.stock_table(data, as_of)
    CFG.clear(); CFG.update(focus=product_id, data=data, stock=stock, budget=budget, capacity=capacity, target_pct=target_pct, max_pct=max_pct, rounds=rounds, date=str(as_of))
    STATE.clear(); TRACE.clear()
    if llm_obj == "demo": llm_obj = ScriptedDemoLLM()
    if llm_obj is None:
        if not llm_helper.available(): raise RuntimeError("No GROQ_API_KEY set. Add it in .env or paste it in the app sidebar.")
        llm_obj = make_llm()
    out = build_crew(llm_obj, product_id, mode).kickoff()
    return {"final": str(getattr(out, "raw", out)), "trace": list(TRACE), "case": STATE.get("case")}

if __name__ == "__main__":
    import json
    data = pl.load_data(); r = run_crew(data["meta"]["as_of"])
    print("\n===== FINAL OWNER SUMMARY =====\n", r["final"])
    print("\n===== AGENT STEPS =====")
    for t in r["trace"]: print("-", t.get("agent", ""), t["text"][:200].replace("\n", " "))
