"""AGENT 2 - Vendor Finder: lists sellers for a product and compares them."""
import pandas as pd

def find_vendors(vendors, product_id, qty, days_left):
    v = vendors[vendors.product_id == product_id].copy()
    v["order_qty"] = v.moq.where(v.moq > qty, qty)                 # seller's minimum may force a bigger order
    v["total_cost"] = v.order_qty * v.unit_price
    v["in_time"] = v.lead_days <= days_left                          # arrives before we run out?
    def note(r):
        n = []
        if r.order_qty > qty: n.append(f"minimum order forces +{int(r.order_qty - qty)} units")
        if not r.in_time: n.append("arrives after stock-out")
        return "; ".join(n) or "ok"
    v["note"] = v.apply(note, axis=1)
    return v.reset_index(drop=True)

def rank_vendors(v, trust):
    """Score = price 40% + trust 35% + speed 25%, minus a penalty if the vendor is too late."""
    v = v.copy()
    v["trust"] = v.vendor_id.map(lambda x: trust[x]["score"])
    price_s = v.unit_price.min() / v.unit_price * 100
    speed_s = v.lead_days.min() / v.lead_days * 100
    v["score"] = (0.40 * price_s + 0.35 * v.trust + 0.25 * speed_s - (~v.in_time) * 30).round(1)
    return v.sort_values("score", ascending=False).reset_index(drop=True)
