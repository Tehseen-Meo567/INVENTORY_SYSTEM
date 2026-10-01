"""AGENT 1 - Stock Watcher: predicts when each product runs out and how much to order."""
import math, numpy as np, pandas as pd

def _multiplier(date, events):
    m = 1.0
    for _, e in events.iterrows():
        if e.start_date <= date < e.start_date + pd.Timedelta(days=int(e.days)):
            m = max(m, float(e.multiplier))
    return m

def analyze_stock(products, sales, events, vendors, as_of, horizon=60, safety_days=2, cover_days=21, buffer=1.15):
    as_of = pd.Timestamp(as_of)
    ev = events.copy(); ev["start_date"] = pd.to_datetime(ev["start_date"])
    sales = sales.copy(); sales["date"] = pd.to_datetime(sales["date"])
    rows = []
    for _, p in products.iterrows():
        s = sales[(sales.product_id == p.product_id) & (sales.date < as_of)].sort_values("date")
        base = float(s.tail(28).units.mean())                       # normal daily sales (last 4 weeks)
        lead = int(math.ceil(vendors[vendors.product_id == p.product_id].lead_days.median()))
        demand = [base * _multiplier(as_of + pd.Timedelta(days=i), ev) for i in range(horizon)]
        cum, dus = 0.0, float("inf")                                  # dus = days until stock-out
        for i, d in enumerate(demand):
            if cum + d >= p.current_stock:
                dus = i + (p.current_stock - cum) / d; break
            cum += d
        need = sum(demand[:lead + cover_days]) * buffer - p.current_stock
        qty = int(math.ceil(max(need, 0) / 10) * 10)
        cover = p.current_stock / base
        upcoming = ev[(ev.start_date >= as_of) & (ev.start_date <= as_of + pd.Timedelta(days=lead + cover_days))].sort_values("start_date")
        event_txt = ""
        if len(upcoming):
            e = upcoming.iloc[0]; event_txt = f"{e['name']} in {(e.start_date - as_of).days} days (sales x{e.multiplier})"
        if dus <= lead + safety_days: status = "URGENT"
        elif dus <= lead + 10: status = "WARNING"
        elif cover > 90: status = "OVERSTOCK"
        else: status = "OK"
        if status in ("URGENT", "WARNING"):
            reason = f"Stock lasts about {dus:.0f} days but delivery takes about {lead} days."
            if event_txt: reason += f" Demand will jump: {event_txt}."
        elif status == "OVERSTOCK":
            reason = f"Stock covers about {cover:.0f} days of normal sales - slow mover, do not reorder."
        else:
            reason = "Enough stock for now."
        rows.append(dict(product_id=p.product_id, name=p["name"], category=p.category, stock=int(p.current_stock),
                         daily_sales=round(base, 1), days_left=round(min(dus, horizon), 1), lead_days=lead,
                         status=status, order_qty=qty if status in ("URGENT", "WARNING") else 0,
                         unit_cost=float(p.unit_cost), reason=reason))
    order = {"URGENT": 0, "WARNING": 1, "OK": 2, "OVERSTOCK": 3}
    df = pd.DataFrame(rows)
    return df.sort_values("status", key=lambda c: c.map(order)).reset_index(drop=True)
