"""No API key needed. Run from the project folder:  python tests/test_core.py"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import tools
from core import AgentSession, MockClient

ROOT = Path(__file__).resolve().parent.parent
GT = pd.read_csv(ROOT / "evals" / "ground_truth.csv").set_index("customer_id")
ok = True


def check(name, cond, detail=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (("  -> " + str(detail)) if not cond else ""))
    ok = ok and cond


# ---- tools
s = tools.new_state()
tools.run_tool("load_data", {"start_date": "2026-04-01", "end_date": "2026-09-30"}, s)
c = tools.run_tool("calculate_churn", {"period_start": "2026-07-01", "period_end": "2026-09-30"}, s)
check("Q3 churn 13.93", c["churn_rate_pct"] == 13.93 and c["active_at_start"] == 122 and c["customers_lost"] == 17, c)
v = tools.run_tool("verify_calculation", {"period_start": "2026-07-01", "period_end": "2026-09-30"}, s)
check("verify matches", v["matches_calculate_churn"] is True, v)
f = tools.run_tool("flag_at_risk", {"as_of_date": "2026-09-30"}, s)
check("24 at risk, all IDs returned", f["at_risk_count"] == 24 and len(f["at_risk_all_ids"]) == 24, f["at_risk_count"])
declining = set(GT[GT.planted_segment == "declining"].index)
check("23 of 24 flagged are planted declining", len(set(f["at_risk_all_ids"]) & declining) == 23)

# ---- per-session isolation (the bug a shared global would cause)
s2 = tools.new_state()
e = tools.run_tool("calculate_churn", {"period_start": "2026-07-01", "period_end": "2026-09-30"}, s2)
check("second session has no data loaded", "error" in e, e)

# ---- get_customer_history
h = tools.run_tool("get_customer_history", {"customer_id": "C0114"}, tools.new_state())
check("C0114 history: at_risk, 1 vs 6", h["status"] == "at_risk" and h["purchases_last_90d"] == 1
      and h["purchases_prior_90d"] == 6, h)
check("history has no names", "name" not in json.dumps(h).lower().replace("signup", ""))
check("C0001 healthy", tools.run_tool("get_customer_history", {"customer_id": "c0001"}, tools.new_state())["status"] == "healthy")
check("C0116 churned", tools.run_tool("get_customer_history", {"customer_id": "C0116"}, tools.new_state())["status"] == "churned")
check("C9999 not found", "not found" in tools.run_tool("get_customer_history", {"customer_id": "C9999"}, tools.new_state())["error"])
check("bad id rejected", "Invalid" in tools.run_tool("get_customer_history", {"customer_id": "../../etc"}, tools.new_state())["error"])
# status agrees with ground truth for every customer
bad = [cid for cid in GT.index
       if {"healthy": "healthy", "declining": "at_risk", "churned": "churned"}[GT.loc[cid, "planted_segment"]]
       != tools.run_tool("get_customer_history", {"customer_id": cid}, tools.new_state())["status"]]
# heuristic: 23 of 25 declining caught + 1 false positive -> exactly 3 disagreements are expected
check("history status matches planted segment (3 known disagreements)", len(bad) <= 3, bad)

# ---- size guard + report caps
check("size guard", "exceeds" in tools.run_tool("load_data", {"start_date": "2025-01-01", "end_date": "2026-09-30"}, tools.new_state())["error"])
check("report cap", "too long" in tools.run_tool("write_report", {"filename": "x", "content": "a" * 3001}, tools.new_state())["error"])
st = tools.new_state()
r = tools.run_tool("write_report", {"filename": "../../evil name", "content": "hi"}, st)
check("filename sanitized", r.get("saved") == ".._.._evil_name.md" and (Path(st["report_dir"]) / r["saved"]).exists(), r)
check("unknown tool", "Unknown" in tools.run_tool("rm_rf", {}, tools.new_state())["error"])
check("bad args handled", "error" in tools.run_tool("load_data", {"nope": 1}, tools.new_state()))


def types(events):
    return [e["type"] for e in events]


# ---- normal run
sess = AgentSession(MockClient("normal"))
ev = list(sess.ask("Q3 churn?"))
check("normal run finishes with final", ev[-1]["type"] == "final" and not sess.pending, types(ev))

# ---- gate: nothing runs until a human approves
sess = AgentSession(MockClient("report"))
ev = list(sess.ask("save a report"))
check("gate pauses before write_report", ev[-1]["type"] == "approval_needed" and sess.pending is not None, types(ev))
check("report NOT saved while pending", sess.state["last_report"] is None)
check("gate shows full arguments", set(ev[-1]["args"]) == {"filename", "content"} and "13.93" in ev[-1]["args"]["content"])
check("cannot ask while pending", list(sess.ask("another"))[0]["type"] == "error")
ev2 = list(sess.resolve(False))
check("reject -> not saved, run finishes", sess.state["last_report"] is None and ev2[-1]["type"] == "final", types(ev2))
sess = AgentSession(MockClient("report"))
list(sess.ask("save a report"))
ev2 = list(sess.resolve(True))
check("approve -> saved, run finishes", sess.state["last_report"]["filename"] == "q3_churn.md" and ev2[-1]["type"] == "final", types(ev2))
check("resolve with nothing pending", list(AgentSession(MockClient()).resolve(True))[0]["type"] == "error")

# ---- caps
sess = AgentSession(MockClient("runaway"), limits={"max_steps_per_turn": 4})
ev = list(sess.ask("x"))
check("step cap stops runaway", ev[-1]["type"] == "stop" and ev[-1]["reason"] == "max_steps" and sess.steps == 4, types(ev))
check("wrap-up notice sent", any(e["type"] == "notice" for e in ev))
sess = AgentSession(MockClient("burst"))
ev = list(sess.ask("x"))
skipped = [e for e in ev if e["type"] == "tool_call" and e.get("skipped")]
check("tool-call cap: 12 requested, 8 run, 4 skipped", sess.tool_calls == 8 and len(skipped) == 4, (sess.tool_calls, len(skipped)))
sess = AgentSession(MockClient("normal"), limits={"max_turns": 2})
list(sess.ask("a")); sess.client = MockClient("normal"); list(sess.ask("b"))
check("turn cap", list(sess.ask("c"))[0]["type"] == "error" and sess.turns == 2)
check("input length cap", list(AgentSession(MockClient()).ask("x" * 401))[0]["type"] == "error")
check("empty input rejected", list(AgentSession(MockClient()).ask("   "))[0]["type"] == "error")
sess = AgentSession(MockClient("normal"), limits={"max_session_tokens": 120})
ev = list(sess.ask("x"))
check("session token budget stops run", ev[-1]["type"] == "stop" and ev[-1]["reason"] == "token_budget", types(ev))


class Boom:
    class messages:
        @staticmethod
        def create(**kw):
            raise RuntimeError("secret-key-sk-ant-123 leaked?")


ev = list(AgentSession(Boom()).ask("x"))
check("API error handled and message does not leak details",
      ev[-1]["type"] == "error" and "sk-ant" not in ev[-1]["message"], ev[-1])

md = AgentSession(MockClient("report"))
list(md.ask("save")); list(md.resolve(True))
t = md.to_markdown()
check("trace markdown has gate + decision", "HUMAN GATE" in t and "HUMAN DECISION:** APPROVED" in t)

# ---- truncated tool call (answer cap hit while writing a long report) ----
def _truncation_check():
    from types import SimpleNamespace as NS
    import core as _c

    class Cut:
        def __init__(self):
            self.messages = self
        def create(self, **kw):
            blk = NS(type="tool_use", id="t1", name="write_report", input={"filename": "x"})
            return NS(content=[blk], stop_reason="max_tokens", usage=NS(input_tokens=10, output_tokens=1200))
    s = _c.AgentSession(Cut())
    ev = list(s.ask("Save a huge report"))
    finals = [e for e in ev if e["type"] == "final"]
    assert finals and "3,000" in finals[0]["text"], ev
    assert not s.pending and s.messages[-1]["role"] == "assistant" and isinstance(s.messages[-1]["content"], str)
    print("ok  truncated tool call gives a clear answer and no dangling tool_use")

_truncation_check()


print("\nALL PASSED" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
