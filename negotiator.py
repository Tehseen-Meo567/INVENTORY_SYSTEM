"""AGENT 5 - Negotiator: writes emails, understands vendor replies, bargains inside the owner's limits."""
import re, json, llm

def draft_email(vendor_name, product, qty, offer, lead_days, round_no, last_price=None):
    last_price = int(last_price) if last_price else last_price
    if llm.available():
        goal = ("Open the negotiation politely" if round_no == 1 else f"Counter their last price of Rs. {last_price}")
        out = llm.chat("You write short, polite, professional purchase emails for a Pakistani online store. Max 90 words. No placeholders.",
                       f"{goal}. Supplier: {vendor_name}. Product: {product}. Quantity: {qty}. Our offer: Rs. {offer} per piece. "
                       f"We need delivery within {lead_days} days. Mention we can confirm today if agreed.", temperature=0.4)
        if out: return out.strip()
    if round_no == 1:
        return (f"Dear {vendor_name} team,\n\nWe would like to order {qty} units of {product}. "
                f"Could you supply them at Rs. {offer} per piece with delivery within {lead_days} days? "
                f"We can confirm the order today if the price works.\n\nRegards,\nPurchasing Team")
    return (f"Dear {vendor_name} team,\n\nThank you for your offer of Rs. {last_price}. For {qty} units and a quick confirmation, "
            f"could you meet Rs. {offer} per piece?\n\nRegards,\nPurchasing Team")

def parse_reply(text):
    """Pull price, quantity, delivery days and minimum order out of a free-text reply."""
    out = {"price": None, "qty": None, "lead_days": None, "moq": None}
    if llm.available():
        j = llm.chat('Extract from the supplier email. Reply JSON only: {"price": number|null (per unit, PKR), "qty": number|null, '
                     '"lead_days": number|null, "moq": number|null}', text, json_mode=True, temperature=0)
        try:
            d = json.loads(j); out.update({k: d.get(k) for k in out}); 
            if out["price"]: return out
        except Exception: pass
    t = text.replace(",", "")
    m = re.search(r"(?:rs\.?|pkr)\s*(\d+(?:\.\d+)?)", t, re.I) or re.search(r"(\d+(?:\.\d+)?)\s*(?:rs|pkr|per (?:piece|unit|pc))", t, re.I)
    if m: out["price"] = float(m.group(1))
    m = re.search(r"(\d+)\s*days?", t, re.I);                      out["lead_days"] = int(m.group(1)) if m else None
    m = re.search(r"(\d+)\s*(?:units|pieces|pcs)", t, re.I);       out["qty"] = int(m.group(1)) if m else None
    m = re.search(r"(?:moq|minimum(?: order)?)\D{0,15}(\d+)", t, re.I); out["moq"] = int(m.group(1)) if m else None
    return out

def decide(parsed, g, round_no, our_last_offer):
    """g = guardrails: target_price, max_price, max_lead_days, max_rounds. Returns (action, next_offer, reason)."""
    p, lead = parsed["price"], parsed["lead_days"]
    if p is None: return "ask_again", None, "Could not find a price in the reply."
    if lead and lead > g["max_lead_days"]: return "walk", None, f"Delivery {lead} days is above our limit of {g['max_lead_days']} days."
    if p > g["max_price"] and round_no >= g["max_rounds"]: return "walk", None, f"Rs. {p:.0f} is above our maximum of Rs. {g['max_price']:.0f}."
    if p <= g["target_price"]: return "accept", None, f"Rs. {p:.0f} meets our target price."
    if our_last_offer is not None and p <= our_last_offer: return "accept", None, "Vendor met our offer."
    if p <= g["max_price"] and round_no >= g["max_rounds"]: return "accept", None, f"Final round: Rs. {p:.0f} is inside our limit (max Rs. {g['max_price']:.0f})."
    base = our_last_offer if our_last_offer is not None else g["target_price"]
    return "counter", round((base + p) / 2), "Price still above target - counter-offer."

def simulate_vendor_reply(vendor, product, qty, our_offer, state):
    """Demo vendor: starts at list price, concedes step by step, never below its hidden floor."""
    L = float(vendor["unit_price"]); floor = L * float(vendor["flex_pct"])
    if our_offer >= floor: price = our_offer; txt = "Okay, we can agree to that price."
    else:
        ask = state.get("ask", L); price = max(round(floor), round(ask - (ask - our_offer) * 0.4)); txt = "We can improve the price a little."
    state["ask"] = price
    return (f"{txt} For {qty} units of {product} our price is Rs. {int(price)} per piece with delivery in "
            f"{int(vendor['lead_days'])} days. Minimum order is {int(vendor['moq'])} units.")

def run_negotiation(vendor, product, qty, guardrails, reply_fn=None):
    """Runs the full back-and-forth. reply_fn(our_offer, our_email, round)->vendor text; default = simulated vendor."""
    state, transcript = {}, []
    offer, last_price = guardrails["target_price"], None
    result = {"status": "walk", "final_price": None, "saving": 0, "transcript": transcript}
    for r in range(1, guardrails["max_rounds"] + 1):
        email = draft_email(vendor["vendor_name"], product, qty, round(offer), guardrails["max_lead_days"], r, last_price)
        transcript.append({"who": "us", "round": r, "text": email, "offer": round(offer)})
        reply = reply_fn(round(offer), email, r) if reply_fn else simulate_vendor_reply(vendor, product, qty, round(offer), state)
        parsed = parse_reply(reply)
        action, nxt, reason = decide(parsed, guardrails, r, round(offer))
        transcript.append({"who": "vendor", "round": r, "text": reply, "parsed": parsed, "action": action, "reason": reason})
        if action == "accept":
            result.update(status="accepted", final_price=parsed["price"], lead_days=parsed["lead_days"] or int(vendor["lead_days"]),
                          saving=(float(vendor["unit_price"]) - parsed["price"]) * qty, reason=reason); break
        if action in ("walk", "ask_again"):
            result.update(status="walk", reason=reason); break
        offer, last_price = nxt, parsed["price"]
    else:
        result["reason"] = "No agreement within the allowed rounds."
    return result
