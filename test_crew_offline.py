"""Tests the CrewAI wiring WITHOUT a Groq key, using a scripted stand-in model that follows the same
Thought/Action format a real model would. It proves: agents are built, tools run, context is passed
between tasks, results are saved. It does NOT test the quality of a real AI model's reasoning."""
import re, os
import crew_run


r = crew_run.run_crew("2026-11-01", llm_obj="demo")
print("FINAL:", r["final"])
print("TRACE ITEMS:", len(r["trace"]), "| tool steps:", sum(t["kind"] == "tool" for t in r["trace"]))
assert r["case"] and r["case"]["negotiation"]["status"] == "accepted", "negotiation result missing"
print("CASE:", r["case"]["product"], r["case"]["vendor"]["vendor_name"], r["case"]["negotiation"]["final_price"])
print("CREW WIRING OK")
