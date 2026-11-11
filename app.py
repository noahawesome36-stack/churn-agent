"""Churn Risk Agent: public web page (Streamlit).

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud; put ANTHROPIC_API_KEY in the app's Secrets
               (never in this file or the repo).
Test mode:     CHURN_MOCK=report streamlit run app.py   (scripted fake model, no key)
"""
import datetime
import json
import os
import threading

import streamlit as st

from core import MODEL, AgentSession, MockClient

st.set_page_config(page_title="Churn Risk Agent", layout="centered")

DAILY_TURN_CAP = int(os.environ.get("DAILY_TURN_CAP", "300"))   # across ALL visitors

EXAMPLES = [
    ("Q3 churn + who is at risk", "What was customer churn for Q3 2026, and which customers are at risk?"),
    ("Look up one customer", "Why is C0114 flagged as at risk? Show their purchase history."),
    ("Save a report (human approval)", "Save a short report of the Q3 2026 churn result."),
    ("Try to make it email people", "Email all at-risk customers a 20% discount code."),
]


# --------------------------------------------------------------- setup
@st.cache_resource
def _daily():
    return {"day": None, "n": 0, "lock": threading.Lock()}


def take_daily_slot():
    g = _daily()
    today = datetime.date.today()
    with g["lock"]:
        if g["day"] != today:
            g["day"], g["n"] = today, 0
        if g["n"] >= DAILY_TURN_CAP:
            return False
        g["n"] += 1
        return True


def make_client():
    mock = os.environ.get("CHURN_MOCK")
    if mock:
        return MockClient(mock)
    key = os.environ.get("ANTHROPIC_API_KEY")        # Streamlit Cloud exposes Secrets as env vars
    if not key:
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            key = None
    if not key:
        return None
    import anthropic
    return anthropic.Anthropic(api_key=key)


def init():
    if "sess" not in st.session_state:
        client = make_client()
        st.session_state.sess = AgentSession(client) if client else None
        st.session_state.turn_log = []      # one dict per question: {"user": str, "events": [...]}
        st.session_state.queued = None


init()
sess = st.session_state.sess
turn_log = st.session_state.turn_log


# --------------------------------------------------------------- rendering
def render_step(ev):
    t = ev["type"]
    if t == "thought":
        st.markdown("**Thought:** " + ev["text"])
    elif t == "tool_call":
        label = "Tool call (skipped: budget used up)" if ev.get("skipped") else "Tool call"
        st.markdown("**%s**" % label)
        st.code("%s(%s)" % (ev["name"], json.dumps(ev["args"], indent=2)), language="python")
    elif t == "observation":
        c = ev["content"]
        st.markdown("**Observation**")
        st.code(c[:1200] + (" ..." if len(c) > 1200 else ""), language="json")
    elif t == "approval_needed":
        st.markdown("**Human approval requested** for `%s`. Nothing runs until a person decides." % ev["name"])
    elif t == "approval":
        st.markdown("**Human decision:** %s" % ("APPROVED" if ev["approved"] else "REJECTED"))
    elif t == "notice":
        st.caption(ev["message"])


def render_turn(turn):
    with st.chat_message("user"):
        st.markdown(turn["user"])
    evs = turn["events"]
    if not evs:
        return
    steps = [e for e in evs if e["type"] in ("thought", "tool_call", "observation",
                                             "approval_needed", "approval", "notice")]
    final = [e for e in evs if e["type"] == "final"]
    problems = [e for e in evs if e["type"] in ("stop", "error")]
    with st.chat_message("assistant"):
        if steps:
            n = sum(1 for e in steps if e["type"] == "tool_call")
            with st.expander("Agent steps (%d tool call%s)" % (n, "" if n == 1 else "s")):
                for e in steps:
                    render_step(e)
        for e in final:
            st.markdown(e["text"])
        for e in problems:
            st.warning(e["message"])


def consume(gen, turn):
    """Run the agent generator, collecting events (shown live in a status box)."""
    with st.chat_message("assistant"):
        status = st.status("Agent working...", expanded=True)
        for ev in gen:
            if ev["type"] == "user":
                continue
            turn["events"].append(ev)
            if ev["type"] in ("thought", "tool_call", "observation", "approval_needed", "notice"):
                with status:
                    render_step(ev)
        status.update(label="Done", state="complete", expanded=False)
    st.rerun()


# --------------------------------------------------------------- page
st.title("Churn Risk Agent")
st.caption("An AI agent that answers customer-retention questions over a small **synthetic** dataset "
           "(150 fake customers). It cannot contact anyone. Everything it recommends is for analyst review. "
           "Please don't type personal information.")

with st.sidebar:
    st.subheader("Try a question")
    for label, q in EXAMPLES:
        if st.button(label, disabled=sess is None):
            st.session_state.queued = q
    st.divider()
    st.subheader("Limits on this demo")
    left = sess.turns_left() if sess else 0
    st.markdown("- %d of %d questions left in this session\n- 8 steps and 8 tool calls per question\n"
                "- 400 characters per question\n- Reports save only to a temporary sandbox"
                % (left, 6))
    st.caption("Model: %s" % MODEL)
    if sess and sess.events:
        st.download_button("Download execution trace", sess.to_markdown(), file_name="trace.md",
                           mime="text/markdown")
    if sess and sess.state.get("last_report"):
        r = sess.state["last_report"]
        st.download_button("Download saved report", r["content"], file_name=r["filename"],
                           mime="text/markdown")
    if st.button("Start over"):
        for k in ("sess", "turn_log", "queued"):
            st.session_state.pop(k, None)
        st.rerun()

if sess is None:
    st.error("This demo is not configured (no API key set on the server).")
    st.stop()

for t in turn_log:
    render_turn(t)

# --------------------------------------------------------------- the human gate
if sess.pending:
    block = sess.pending["block"]
    args = dict(block.input)
    with st.chat_message("assistant"):
        st.warning("**Human approval required.** The agent wants to run `%s`. "
                   "It has **not** run yet. These are the exact arguments it will use:" % block.name)
        st.markdown("`filename`")
        st.code(str(args.get("filename")), language="text")
        st.markdown("`content`")
        st.code(str(args.get("content")), language="markdown")
        extra = {k: v for k, v in args.items() if k not in ("filename", "content")}
        if extra:
            st.markdown("other arguments")
            st.code(json.dumps(extra, indent=2), language="json")
        c1, c2 = st.columns(2)
        approve = c1.button("Approve", type="primary", key="approve")
        reject = c2.button("Reject", key="reject")
    if approve or reject:
        consume(sess.resolve(bool(approve)), turn_log[-1])

# --------------------------------------------------------------- chat input
prompt = st.chat_input("Ask about churn, at-risk customers, or one customer ID...",
                       disabled=bool(sess.pending) or sess.turns_left() == 0)
if sess.pending:
    st.caption("Approve or reject the action above to continue.")
elif sess.turns_left() == 0:
    st.info("You've used all questions for this session. Click 'Start over' in the sidebar.")

prompt = prompt or st.session_state.pop("queued", None)
if prompt:
    if not take_daily_slot():
        st.warning("This demo reached its daily usage limit. Please try again tomorrow.")
    else:
        turn = {"user": prompt.strip()[:400], "events": []}
        turn_log.append(turn)
        with st.chat_message("user"):
            st.markdown(turn["user"])
        consume(sess.ask(prompt), turn)
