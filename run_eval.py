#!/usr/bin/env python3
"""Evaluation harness: runs every case K times and reports pass@1 and pass^k.

    export ANTHROPIC_API_KEY=your-key
    python evals/run_eval.py                 # all cases, k=3
    python evals/run_eval.py --k 5           # more repeats
    python evals/run_eval.py --cases q3_churn_rate,adv_contact_customers
    python evals/run_eval.py --cases-file cases_hard.json --k 5   # the harder set
    python evals/run_eval.py --mock normal   # no key: only tests that the harness itself runs

Definitions (k trials per case):
  pass@1   average success rate of a single attempt = (passed trials) / (all trials)
  pass^k   share of cases where ALL k trials passed (reliability: does it work every time?)
  pass@k   share of cases where AT LEAST ONE trial passed (capability: can it ever do it?)
Each trial uses a brand-new AgentSession, so trials cannot affect each other.
At the human gate the harness acts as the human: case["gate_decision"] is "approve" or "reject".
Outputs go to evals/results/<timestamp>/ : results.csv, summary.md and one trace per trial.
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from core import MODEL, AgentSession, MockClient  # noqa: E402
from graders import grade  # noqa: E402


def run_trial(case, client, model):
    """One trial = a fresh session. A case may have "followups": more user messages sent
    in the same session (tests memory). The human gate is answered after every turn."""
    sess = AgentSession(client, model=model)
    t0 = time.time()
    decision = case.get("gate_decision", "reject") == "approve"
    events = []
    for prompt in [case["prompt"]] + list(case.get("followups", [])):
        events += list(sess.ask(prompt))
        hops = 0
        while sess.pending and hops < 3:
            hops += 1
            events += list(sess.resolve(decision))
    last_user = max([i for i, e in enumerate(events) if e["type"] == "user"] or [0])
    finals = [e["text"] for e in events[last_user:] if e["type"] == "final"]   # answer to the LAST message
    stops = [e for e in events if e["type"] == "stop"]
    errors = [e["message"] for e in events if e["type"] == "error"]
    result = {
        "final": "\n".join(finals),
        "tools": [e["name"] for e in events if e["type"] == "tool_call" and not e.get("skipped")],
        "gate": [{"name": e["name"], "args": e["args"]} for e in events if e["type"] == "approval_needed"],
        "report": sess.state.get("last_report"),
        "stop": stops[0]["reason"] if stops else None,
        "error": errors[0] if errors else None,
    }
    s = sess.stats()
    result.update({"steps": s["model_calls"], "tool_calls": s["tool_calls"], "tokens_in": s["tokens_in"],
                   "tokens_out": s["tokens_out"], "seconds": round(time.time() - t0, 1)})
    return result, sess.to_markdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--cases", default="", help="comma-separated case ids (default: all)")
    ap.add_argument("--cases-file", default="cases.json", help="cases.json (baseline) or cases_hard.json")
    ap.add_argument("--model", default=None)
    ap.add_argument("--mock", nargs="?", const="normal", help="scripted fake model (harness self-test only)")
    ap.add_argument("--sleep", type=float, default=0.5, help="pause between trials (seconds)")
    a = ap.parse_args()

    cases_path = Path(a.cases_file) if Path(a.cases_file).is_absolute() or Path(a.cases_file).exists() else HERE / a.cases_file
    cases = json.loads(cases_path.read_text())
    if a.cases:
        want = set(a.cases.split(","))
        cases = [c for c in cases if c["id"] in want]
    if a.mock:
        make_client = lambda: MockClient(a.mock)  # noqa: E731
        model = "mock"
    else:
        import anthropic
        shared = anthropic.Anthropic()            # reads ANTHROPIC_API_KEY from the environment
        make_client = lambda: shared              # noqa: E731
        model = a.model or MODEL

    out = HERE / "results" / (cases_path.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    (out / "traces").mkdir(parents=True)
    rows, per_case = [], {}
    for case in cases:
        passes = []
        for k in range(1, a.k + 1):
            result, trace = run_trial(case, make_client(), model)
            fails = grade(case, result)
            passes.append(not fails)
            (out / "traces" / ("%s_trial%d.md" % (case["id"], k))).write_text(
                "# %s, trial %d: %s\n\n**Failures:** %s\n\n%s" % (case["id"], k, "PASS" if not fails else "FAIL",
                                                                 "; ".join(fails) or "none", trace))
            rows.append({"case_id": case["id"], "category": case["category"], "trial": k,
                         "passed": int(not fails), "failures": "; ".join(fails),
                         "steps": result["steps"], "tool_calls": result["tool_calls"],
                         "tokens_in": result["tokens_in"], "tokens_out": result["tokens_out"],
                         "seconds": result["seconds"]})
            print("%-24s trial %d  %s  %s" % (case["id"], k, "PASS" if not fails else "FAIL",
                                              ("; ".join(fails))[:110]))
            time.sleep(a.sleep)
        per_case[case["id"]] = (case, passes)

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_trials = sum(len(p) for _, p in per_case.values())
    pass1 = sum(sum(p) for _, p in per_case.values()) / n_trials
    passk = sum(all(p) for _, p in per_case.values()) / len(per_case)
    passany = sum(any(p) for _, p in per_case.values()) / len(per_case)
    lines = ["# Evaluation summary", "",
             "Model: `%s` | case file: `%s` | cases: %d | trials per case (k): %d | total runs: %d" % (model, cases_path.name, len(per_case), a.k, n_trials), "",
             "| Metric | Value |", "|---|---|",
             "| pass@1 (average single-attempt success) | %.1f%% |" % (pass1 * 100),
             "| pass^%d (all %d trials pass) | %.1f%% |" % (a.k, a.k, passk * 100),
             "| pass@%d (at least one trial passes) | %.1f%% |" % (a.k, passany * 100), "",
             "## By category", "", "| Category | Cases | pass@1 | pass^%d |" % a.k, "|---|---|---|---|"]
    for cat in ("normal", "edge", "adversarial"):
        sub = [p for c, p in per_case.values() if c["category"] == cat]
        if sub:
            lines.append("| %s | %d | %.1f%% | %.1f%% |" % (
                cat, len(sub), 100 * sum(sum(p) for p in sub) / sum(len(p) for p in sub),
                100 * sum(all(p) for p in sub) / len(sub)))
    lines += ["", "## By case", "", "| Case | Category | Passed | pass@1 | pass^%d |" % a.k, "|---|---|---|---|---|"]
    for cid, (c, p) in per_case.items():
        lines.append("| %s | %s | %d/%d | %.0f%% | %s |" % (cid, c["category"], sum(p), len(p),
                                                          100 * sum(p) / len(p), "yes" if all(p) else "no"))
    cost_in = sum(r["tokens_in"] for r in rows)
    cost_out = sum(r["tokens_out"] for r in rows)
    lines += ["", "Tokens used: %d in / %d out. Mean steps per run: %.1f. Mean seconds per run: %.1f." % (
        cost_in, cost_out, sum(r["steps"] for r in rows) / len(rows), sum(r["seconds"] for r in rows) / len(rows)),
        "", "## Failed trials (read these, then classify each: agent error, grader too strict, or bad test)", ""]
    failed = [r for r in rows if not r["passed"]]
    if not failed:
        lines.append("None.")
    for r in failed:
        lines.append("- **%s trial %d**: %s  (trace: `traces/%s_trial%d.md`)" % (
            r["case_id"], r["trial"], r["failures"], r["case_id"], r["trial"]))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[:8]))
    print("\nFull summary:", out / "summary.md")


if __name__ == "__main__":
    main()
