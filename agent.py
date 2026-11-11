"""Command-line version of the Churn Risk Agent (same agent as the web app).

  python agent.py "What was churn for Q3 2026 and who is at risk?"
  python agent.py --mock report "..."     # no API key; tests the approval gate
  python agent.py --max-steps 2 "..."
"""
import argparse
from datetime import datetime
from pathlib import Path

from core import AgentSession, MockClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request", nargs="*", default=["What was customer churn for Q3 2026 and which customers are at risk?"])
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--mock", nargs="?", const="normal", help="scripted fake model: normal|report|runaway|burst")
    a = ap.parse_args()
    if a.mock:
        client = MockClient(a.mock)
    else:
        import anthropic  # reads ANTHROPIC_API_KEY from the environment
        client = anthropic.Anthropic()
    sess = AgentSession(client, limits={"max_steps_per_turn": a.max_steps})

    def show(events):
        for e in events:
            t = e["type"]
            if t == "thought":
                print("THOUGHT:", e["text"])
            elif t == "tool_call":
                print("TOOL CALL:", e["name"], e["args"])
            elif t == "observation":
                print("OBSERVATION:", e["content"][:600])
            elif t == "final":
                print("\nFINAL ANSWER:", e["text"])
            elif t in ("stop", "error", "notice"):
                print(t.upper() + ":", e["message"])
            elif t == "approval_needed":
                print("\n=== HUMAN APPROVAL NEEDED: %s ===" % e["name"])
                for k, v in e["args"].items():
                    print("%s:\n%s\n" % (k, v))

    show(sess.ask(" ".join(a.request)))
    while sess.pending:
        ok = input("Approve this action? [y/N] ").strip().lower() == "y"
        show(sess.resolve(ok))

    Path("traces").mkdir(exist_ok=True)
    path = Path("traces") / ("trace_%s.md" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    path.write_text(sess.to_markdown())
    print("\nTrace saved to", path)


if __name__ == "__main__":
    main()
