"""Connects all agents. Used by the Streamlit app, and runnable from the terminal:  python pipeline.py"""
import json, io
from pathlib import Path
import pandas as pd
import llm, stock_watcher, budget_guard, vendor_finder, vendor_checker, negotiator
from rag import Retriever

DATA = Path(__file__).parent / "data"
_retriever, _trust = None, {}

def load_data():
    d = {k: pd.read_csv(DATA / f"{k}.csv") for k in ["products", "sales_history", "events", "vendors"]}
    d["meta"] = json.load(open(DATA / "meta.json")); return d

def get_retriever():
    global _retriever
    if _retriever is None: _retriever = Retriever(DATA / "vendor_docs")
    return _retriever

def trust_for(vendors):
    names = vendors.drop_duplicates("vendor_id").set_index("vendor_id").vendor_name.to_dict()
    for vid, n in names.items():
        if vid not in _trust: _trust[vid] = vendor_checker.check_vendor(get_retriever(), vid, n)
    return _trust

def stock_table(data, as_of):
    return stock_watcher.analyze_stock(data["products"], data["sales_history"], data["events"], data["vendors"], as_of)

def ranked_vendors(data, row):
    v = vendor_finder.find_vendors(data["vendors"], row["product_id"], int(row["order_qty"]), row["days_left"])
    return vendor_finder.rank_vendors(v, trust_for(data["vendors"]))

def plan_orders(data, stock_df, budget, capacity):
    orders = []
    for _, r in stock_df[stock_df.order_qty > 0].iterrows():
        top = ranked_vendors(data, r).iloc[0]
        orders.append(dict(product_id=r.product_id, name=r["name"], status=r.status, qty=int(top.order_qty), unit_price=float(top.unit_price)))
    return budget_guard.check_budget(orders, budget, capacity, int(stock_df.stock.sum()))

def default_guardrails(list_price, max_lead, target_pct=0.92, max_pct=1.0, rounds=3):
    return dict(target_price=round(list_price * target_pct), max_price=round(list_price * max_pct), max_lead_days=int(max_lead), max_rounds=int(rounds))

def make_po_pdf(case):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    buf = io.BytesIO(); doc = SimpleDocTemplate(buf, pagesize=A4); st = getSampleStyleSheet()
    n = case["negotiation"]; v = case["vendor"]; qty = case["qty"]
    rows = [["Item", "Quantity", "Unit price (Rs.)", "Total (Rs.)"], [case["product"], qty, f"{n['final_price']:,.0f}", f"{n['final_price']*qty:,.0f}"]]
    t = Table(rows, colWidths=[200, 70, 100, 100]); t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#14213D")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("GRID", (0,0), (-1,-1), 0.5, colors.grey)]))
    doc.build([Paragraph("PURCHASE ORDER", st["Title"]), Paragraph(f"PO No: PO-{case['product_id']}-{case['date']}", st["Normal"]), Paragraph(f"Date: {case['date']}", st["Normal"]),
               Spacer(1, 12), Paragraph(f"<b>Vendor:</b> {v['vendor_name']} ({v['city']})", st["Normal"]),
               Paragraph(f"<b>Payment terms:</b> {v['payment_terms']}", st["Normal"]), Paragraph(f"<b>Expected delivery:</b> within {n['lead_days']} days", st["Normal"]),
               Spacer(1, 12), t, Spacer(1, 12), Paragraph(f"Saving vs list price: Rs. {n['saving']:,.0f}", st["Normal"]),
               Spacer(1, 20), Paragraph("Approved by owner via SupplySense.", st["Italic"])])
    return buf.getvalue()

def ask_assistant(question, stock_df, data):
    q = question.lower(); trust = trust_for(data["vendors"]); risky = stock_df[stock_df.status.isin(["URGENT", "WARNING"])]
    if llm.available():
        ctx = "STOCK:\n" + stock_df[["name", "stock", "daily_sales", "days_left", "lead_days", "status", "order_qty", "reason"]].to_string(index=False)
        ctx += "\nVENDOR TRUST:\n" + "\n".join(f"{k}: score {v['score']} - {v['summary']}" for k, v in trust.items())
        ans = llm.chat("You are SupplySense, an assistant for an online store owner. Answer briefly using ONLY this data.", f"{ctx}\n\nQuestion: {question}")
        if ans: return ans
    if any(w in q for w in ["run out", "stock", "low", "eid", "sale", "order"]):
        if risky.empty: return "No products are at risk right now."
        return "Products at risk:\n" + "\n".join(f"- {r['name']}: {r.status}, about {r.days_left:.0f} days left, suggested order {r.order_qty}" for _, r in risky.iterrows())
    if any(w in q for w in ["vendor", "seller", "supplier", "trust", "why"]):
        names = data["vendors"].drop_duplicates("vendor_id").set_index("vendor_id").vendor_name
        return "Vendor trust scores:\n" + "\n".join(f"- {names[k]}: {v['score']}/100 - {v['summary']}" for k, v in trust.items())
    return "Offline mode: I can answer about stock risk and vendor trust. Add a GROQ_API_KEY for full answers."

if __name__ == "__main__":
    import datetime as dt
    data = load_data(); as_of = data["meta"]["as_of"]
    df = stock_table(data, as_of); print("\n== STOCK WATCHER ==\n", df[["name","stock","daily_sales","days_left","lead_days","status","order_qty"]].to_string(index=False))
    plan, summ = plan_orders(data, df, 650000, 6000); print("\n== BUDGET GUARD ==\n", plan[["product","status","wanted_qty","approved_qty","cost","decision"]].to_string(index=False)); print(summ)
    r = df[df.order_qty > 0].iloc[0]; rv = ranked_vendors(data, r)
    print(f"\n== VENDORS for {r['name']} ==\n", rv[["vendor_name","unit_price","moq","lead_days","in_time","trust","score","note"]].to_string(index=False))
    top = rv.iloc[0]; g = default_guardrails(top.unit_price, max(top.lead_days + 3, r.days_left))
    res = negotiator.run_negotiation(top, r["name"], int(top.order_qty), g)
    print("\n== NEGOTIATION ==", g)
    for m in res["transcript"]: print(f"[{m['who']} r{m['round']}]", m["text"].replace("\n", " ")[:160], m.get("action", ""), m.get("reason", ""))
    print("RESULT:", {k: v for k, v in res.items() if k != "transcript"}, "| RAG mode:", get_retriever().mode, "| LLM:", llm.available())
