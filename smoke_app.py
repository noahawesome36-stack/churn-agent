"""Smoke test for app.py WITHOUT Streamlit installed: runs the script against a stand-in
for the streamlit module and drives it through ask -> approval gate -> approve/reject.
This checks the control flow and the code paths; it is NOT a visual test. For that, run
`CHURN_MOCK=report streamlit run app.py` on your computer."""
import os, sys, runpy, types
from pathlib import Path
from unittest.mock import MagicMock
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["CHURN_MOCK"] = "report"

class Rerun(Exception): pass

class SS(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v

def make_st(script, ss, log):
    st = MagicMock(name="streamlit")
    st.session_state = ss
    st.secrets = {}
    st.cache_resource = lambda f: (lambda c={}: c.setdefault("v", f()))
    def rerun(): raise Rerun()
    st.rerun = rerun
    st.stop = lambda: (_ for _ in ()).throw(SystemExit)
    st.columns = lambda n: [MagicMock(button=lambda *a, _i=i, **k: script.get(("col", _i), False)) for i in range(n)]
    st.button = lambda label, **k: script.get(label, False)
    st.chat_input = lambda *a, **k: script.get("chat_input")
    for fn in ("markdown", "code", "warning", "caption", "error", "info", "title", "subheader", "divider", "download_button"):
        setattr(st, fn, (lambda name: lambda *a, **k: log.append((name, a)))(fn))
    return st

def run(script, ss):
    log = []
    sys.modules["streamlit"] = make_st(script, ss, log)
    for m in ("core",):
        sys.modules.pop(m, None)
    try:
        runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    except Rerun:
        pass
    return log

ss = SS()
ok = True
def check(n, c):
    global ok; print(("PASS " if c else "FAIL ") + n); ok = ok and c

run({"chat_input": "save a report"}, ss)
check("run 1: question processed, agent paused at gate", ss["sess"].pending is not None and ss["sess"].state["last_report"] is None)
log = run({}, ss)
txt = " ".join(str(a) for n, a in log)
check("run 2: gate screen shows filename and full content", "q3_churn" in txt and "13.93" in txt and "Human approval required" in txt)
check("run 2: report still not saved", ss["sess"].state["last_report"] is None)
run({("col", 1): True}, ss)   # Reject button
check("reject: nothing saved, gate cleared", ss["sess"].pending is None and ss["sess"].state["last_report"] is None)
log = run({}, ss)
check("final answer rendered after reject", any("Q3 churn is 13.93%" in str(a) for n, a in log))

ss2 = SS()
run({"chat_input": "save a report"}, ss2); run({("col", 0): True}, ss2)   # Approve
check("approve: report saved", ss2["sess"].state["last_report"]["filename"] == "q3_churn.md")
run({"Start over": True}, ss2); run({}, ss2)   # real Streamlit reruns after the reset
check("start over clears session", "sess" in ss2 and ss2["sess"].turns == 0)

ss3 = SS()
run({"Q3 churn + who is at risk": True}, ss3)
check("example button runs a question", ss3["sess"].turns == 1)
print("\nSMOKE OK" if ok else "\nSMOKE FAILED"); sys.exit(0 if ok else 1)
