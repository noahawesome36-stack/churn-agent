# Churn Risk Agent (Milestone II)

A tool-using AI agent for customer-retention analysis over a small **synthetic** dataset.
Web page built with Streamlit. Never commit your API key.

## Files
| File | What it is |
|---|---|
| `app.py` | the web page (chat + visible human approval gate) |
| `core.py` | the agent loop, cost caps and approval gate |
| `tools.py` | 6 tools: load_data, calculate_churn, flag_at_risk, verify_calculation, **get_customer_history (new)**, write_report (gated) |
| `agent.py` | command-line version of the same agent |
| `customers.csv`, `purchases.csv` | synthetic data |
| `evals/` | evaluation harness: `cases.json` (15 cases), `graders.py`, `run_eval.py`, `build_cases.py`, `ground_truth.csv` |
| `tests/` | tests that need no API key |

## Run on your computer
```
pip install -r requirements.txt
CHURN_MOCK=report streamlit run app.py          # scripted fake model, no key: tests the page and the gate
export ANTHROPIC_API_KEY=your-key               # (PowerShell: $env:ANTHROPIC_API_KEY="your-key")
streamlit run app.py                            # real agent
```

## Tests (no key needed)
```
python tests/test_core.py        # tools, caps, approval gate
python tests/test_graders.py     # graders accept good answers, reject bad ones
python tests/smoke_app.py        # page control flow against a stand-in for streamlit
```

## Evaluation (needs the key; ~45 runs)
```
python evals/run_eval.py                     # 15 cases x 3 trials -> pass@1 and pass^3
python evals/run_eval.py --k 5               # more repeats
python evals/run_eval.py --cases q3_churn_rate,gate_reject
```
Results: `evals/results/<timestamp>/summary.md`, `results.csv`, and one trace per trial.

## Deploy (Streamlit Community Cloud)
1. Put these files in a public GitHub repo (no keys in it).
2. share.streamlit.io -> New app -> pick the repo, branch `main`, main file `app.py`.
3. Advanced settings -> Secrets: `ANTHROPIC_API_KEY = "your-key"`
4. Deploy, then open the link in a private window and try the example buttons.
Optional: `DAILY_TURN_CAP`, `CLAUDE_MODEL` can be set the same way.
