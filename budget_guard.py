"""AGENT 4 - Budget Guard: checks money and storage space before anything is ordered."""
import pandas as pd

def check_budget(orders, budget, capacity_units, current_units):
    """orders: list of dicts with product_id, name, status, qty, unit_price. Urgent items get money first."""
    rows, spent, units = [], 0.0, current_units
    for o in sorted(orders, key=lambda x: (0 if x["status"] == "URGENT" else 1, -x["qty"] * x["unit_price"])):
        qty, notes = o["qty"], []
        free = max(capacity_units - units, 0)
        if qty > free:
            qty = int(free); notes.append(f"reduced for storage space (only {int(free)} free)")
        affordable = int((budget - spent) // o["unit_price"])
        if qty > affordable:
            qty = max(affordable, 0); notes.append("reduced to fit budget")
        cost = qty * o["unit_price"]; spent += cost; units += qty
        decision = "FULL" if qty == o["qty"] else ("REDUCED" if qty > 0 else "NOT AFFORDABLE")
        rows.append(dict(product=o["name"], status=o["status"], wanted_qty=o["qty"], approved_qty=qty,
                         unit_price=o["unit_price"], cost=cost, decision=decision, note="; ".join(notes) or "within budget and space"))
    df = pd.DataFrame(rows)
    return df, {"budget": budget, "spent": spent, "left": budget - spent,
                "units_after": units, "capacity": capacity_units, "within_budget": spent <= budget}
