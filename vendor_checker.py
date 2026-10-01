"""AGENT 3 - Vendor Checker (RAG): reads a vendor's documents and gives a trust score with proof."""
import json, re, llm

NEG = ["late", "delay", "damaged", "defect", "faulty", "refund", "poor", "unreliable", "difficult", "dented", "slow", "advance", "estimates only", "no late delivery penalty"]
POS = ["on time", "intact", "excellent", "reliable", "genuine", "responsive", "consistent", "five-star", "guaranteed", "penalty: 2%"]
QUERY = "late delivery delay damaged defective faulty refund complaint poor quality reliability"

def _heuristic(chunks):
    score, flags = 80, []
    for c in chunks:
        t = c["text"].lower()
        neg = sum(w in t for w in NEG); pos = sum(w in t for w in POS)
        if neg: flags.append(c["text"][:140])
        score += 3 * min(pos, 2) - 7 * min(neg, 2)
    return max(15, min(98, score)), flags

def check_vendor(retriever, vendor_id, vendor_name):
    chunks = retriever.search(QUERY, vendor_id, k=8)
    h_score, flags = _heuristic(chunks)
    result = {"vendor_id": vendor_id, "score": h_score, "method": "keyword rules",
              "summary": ("Red flags found in documents." if flags else "No problems found in documents."),
              "red_flags": flags[:3], "evidence": chunks}
    if llm.available():
        ctx = "\n".join(f"- ({c['source']}) {c['text']}" for c in chunks)
        out = llm.chat("You judge supplier reliability using ONLY the evidence given. Reply as JSON: "
                       '{"score": 0-100 integer, "summary": "one sentence", "red_flags": ["short strings"]}',
                       f"Supplier: {vendor_name}\nEvidence:\n{ctx}", json_mode=True, temperature=0.1)
        try:
            j = json.loads(out)
            result.update(score=int(j["score"]), summary=j["summary"], red_flags=j.get("red_flags", [])[:3], method="Groq LLM + RAG")
        except Exception:
            pass
    return result
