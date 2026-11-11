"""Tools for the Churn Risk Agent (v2).

Changes from Milestone 1:
  * State is PER SESSION (a dict passed to every tool), so two visitors on the
    same server never see each other's loaded data.
  * New tool: get_customer_history(customer_id)  (the "second real tool").
  * write_report writes only to a temporary sandbox folder and is size-capped.
    The agent loop requires a human to approve it (see core.py).

Every tool returns a small JSON-serializable dict. Errors are returned as
{"error": ...} instead of raised, so the agent can read them and recover.
Tools return customer IDs only, never names.
Python 3.9 compatible.
"""
import csv
import os
import re
import tempfile
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA_DIR = Path(os.environ.get("CHURN_DATA_DIR", HERE))

AS_OF = "2026-09-30"        # "today" in the synthetic dataset
CHURN_DAYS = 60             # churn = no purchase in this many days
MAX_RECORDS = 1500          # size guard on load_data (about 6 months of data)
SAMPLE = 10                 # max IDs echoed back in a result
MAX_REPORT_CHARS = 3000     # cap on what write_report may save
ID_RE = re.compile(r"^C\d{4}$")


def _d(s):
    return datetime.strptime(s, "%Y-%m-%d")


@lru_cache(maxsize=1)
def _all_purchases():
    return pd.read_csv(DATA_DIR / "purchases.csv", parse_dates=["purchase_date"])


@lru_cache(maxsize=1)
def _customers():
    return pd.read_csv(DATA_DIR / "customers.csv", parse_dates=["signup_date"])


def new_state():
    """Fresh per-session state. One of these per visitor / per eval trial."""
    return {"purchases": None, "window": None, "last_churn": None,
            "report_dir": None, "last_report": None}


def classify(days_since, recent_90, prior_90):
    """One rule used by both flag_at_risk and get_customer_history."""
    if days_since > CHURN_DAYS:
        return "churned"
    if prior_90 >= 3 and recent_90 <= prior_90 * 0.5:
        return "at_risk"
    return "healthy"


# ---------------------------------------------------------------- tool 1
def load_data(state, start_date, end_date):
    """Load purchases between two dates (inclusive) into this session, with a row cap."""
    s, e = _d(start_date), _d(end_date)
    if s > e:
        return {"error": "start_date is after end_date"}
    df = _all_purchases()
    sub = df[(df.purchase_date >= s) & (df.purchase_date <= e)].copy()
    if len(sub) > MAX_RECORDS:
        return {"error": f"{len(sub)} rows exceeds the {MAX_RECORDS}-row limit. "
                         "Narrow the date range. Roughly 6 months of data fits under the limit "
                         "(for example 2026-04-01 to 2026-09-30). If the user asked for a longer "
                         "period, analyze the most recent window that fits and say so in your answer."}
    state["purchases"] = sub
    state["window"] = (s, e)
    return {"rows_loaded": len(sub), "customers": int(sub.customer_id.nunique()),
            "window": [start_date, end_date]}


def _check_window(state, lo, hi):
    if state["purchases"] is None:
        return {"error": "No data loaded. Call load_data first."}
    ws, we = state["window"]
    if lo < ws or hi > we:
        return {"error": f"Loaded window {ws.date()}..{we.date()} does not cover "
                         f"{lo.date()}..{hi.date()}. Call load_data again with a wider range."}
    return None


# ---------------------------------------------------------------- tool 2
def calculate_churn(state, period_start, period_end):
    """Churn % = customers lost / customers active at start * 100.

    Active at start: bought in the 60 days before period_start.
    Lost: no purchase in the 60 days ending at period_end.
    """
    ps, pe = _d(period_start), _d(period_end)
    err = _check_window(state, ps - timedelta(days=CHURN_DAYS), pe)
    if err:
        return err
    df = state["purchases"]
    active = set(df[(df.purchase_date >= ps - timedelta(days=CHURN_DAYS))
                    & (df.purchase_date < ps)].customer_id)
    last = df[df.purchase_date <= pe].groupby("customer_id").purchase_date.max()
    lost = sorted(c for c in active if (pe - last[c]).days > CHURN_DAYS)
    rate = round(len(lost) / len(active) * 100, 2) if active else None
    result = {"period": [period_start, period_end], "active_at_start": len(active),
              "customers_lost": len(lost), "churn_rate_pct": rate,
              "lost_customer_ids_sample": lost[:SAMPLE]}
    state["last_churn"] = result
    return result


# ---------------------------------------------------------------- tool 3
def flag_at_risk(state, as_of_date):
    """Flag customers whose purchase frequency is dropping (not yet churned)."""
    t = _d(as_of_date)
    err = _check_window(state, t - timedelta(days=180), t)
    if err:
        return err
    df = state["purchases"]
    df = df[df.purchase_date <= t]
    last = df.groupby("customer_id").purchase_date.max()
    days_since = (t - last).dt.days
    churned = sorted(days_since[days_since > CHURN_DAYS].index)

    recent = df[df.purchase_date > t - timedelta(days=90)].groupby("customer_id").size()
    prior = df[(df.purchase_date > t - timedelta(days=180))
               & (df.purchase_date <= t - timedelta(days=90))].groupby("customer_id").size()
    flagged = []
    for c in days_since[days_since <= CHURN_DAYS].index:
        r, p = int(recent.get(c, 0)), int(prior.get(c, 0))
        if classify(int(days_since[c]), r, p) == "at_risk":
            flagged.append({"customer_id": c, "recent_90d": r, "prior_90d": p,
                            "days_since_last": int(days_since[c]),
                            "reason": f"{r} purchases in last 90 days vs {p} in the prior 90; "
                                      f"{int(days_since[c])} days since last purchase"})
    flagged.sort(key=lambda x: (x["recent_90d"] / x["prior_90d"], x["customer_id"]))
    return {"as_of": as_of_date, "at_risk_count": len(flagged),
            "at_risk_top": flagged[:10],
            "at_risk_all_ids": [x["customer_id"] for x in flagged],
            "churned_count": len(churned), "churned_ids_sample": churned[:SAMPLE],
            "note": "Flag for analyst review; do not contact customers."}


# ---------------------------------------------------------------- tool 4
def verify_calculation(state, period_start, period_end):
    """Independent recomputation of churn with plain csv/dict code (no pandas),
    compared against the most recent calculate_churn result."""
    prev = state["last_churn"]
    if prev is None or prev["period"] != [period_start, period_end]:
        return {"error": "Run calculate_churn for this exact period first."}
    ps, pe = _d(period_start), _d(period_end)
    last_by_cust, active = {}, set()
    with open(DATA_DIR / "purchases.csv", newline="") as f:
        for row in csv.DictReader(f):
            d = _d(row["purchase_date"])
            c = row["customer_id"]
            if d <= pe and (c not in last_by_cust or d > last_by_cust[c]):
                last_by_cust[c] = d
            if ps - timedelta(days=CHURN_DAYS) <= d < ps:
                active.add(c)
    lost = [c for c in active if (pe - last_by_cust[c]).days > CHURN_DAYS]
    rate = round(len(lost) / len(active) * 100, 2) if active else None
    return {"independent_active": len(active), "independent_lost": len(lost),
            "independent_churn_rate_pct": rate,
            "matches_calculate_churn": (len(active) == prev["active_at_start"]
                                        and len(lost) == prev["customers_lost"])}


# ---------------------------------------------------------------- tool 5 (NEW)
def get_customer_history(state, customer_id):
    """Purchase history and risk status for ONE customer. Does not need load_data."""
    cid = str(customer_id).strip().upper()
    if not ID_RE.match(cid):
        return {"error": "Invalid customer_id. IDs look like C0001 (letter C and four digits)."}
    cust = _customers()
    row = cust[cust.customer_id == cid]
    if row.empty:
        return {"error": f"Customer {cid} not found."}
    t = _d(AS_OF)
    df = _all_purchases()
    df = df[(df.customer_id == cid) & (df.purchase_date <= t)].sort_values("purchase_date")
    out = {"customer_id": cid, "signup_date": row.iloc[0].signup_date.date().isoformat(),
           "as_of": AS_OF, "total_purchases": int(len(df))}
    if df.empty:
        out["status"] = "no_purchases"
        return out
    dates = df.purchase_date
    gaps = dates.diff().dt.days.dropna()
    last = dates.max()
    days_since = int((t - last).days)
    recent = int(((dates > t - timedelta(days=90)) & (dates <= t)).sum())
    prior = int(((dates > t - timedelta(days=180)) & (dates <= t - timedelta(days=90))).sum())
    out.update({
        "first_purchase": dates.min().date().isoformat(),
        "last_purchase": last.date().isoformat(),
        "days_since_last_purchase": days_since,
        "total_spend": round(float(df.amount.sum()), 2),
        "avg_gap_days": round(float(gaps.mean()), 1) if len(gaps) else None,
        "purchases_last_90d": recent, "purchases_prior_90d": prior,
        "status": classify(days_since, recent, prior),
        "reason": f"{recent} purchases in last 90 days vs {prior} in the prior 90; "
                  f"{days_since} days since last purchase",
        "recent_purchases": [{"date": d.date().isoformat(), "amount": round(float(a), 2)}
                             for d, a in zip(dates.tail(12), df.amount.tail(12))],
    })
    return out


# ---------------------------------------------------------------- tool 6 (GATED)
def write_report(state, filename, content):
    """Save a markdown report into a temporary sandbox folder.
    The agent loop will not run this until a human approves the exact arguments."""
    content = str(content)
    if len(content) > MAX_REPORT_CHARS:
        return {"error": f"Report too long ({len(content)} chars). Limit is {MAX_REPORT_CHARS}."}
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(filename))[:60] or "report"
    if not name.endswith(".md"):
        name += ".md"
    if state["report_dir"] is None:
        state["report_dir"] = tempfile.mkdtemp(prefix="churn_reports_")
    (Path(state["report_dir"]) / name).write_text(content)
    state["last_report"] = {"filename": name, "content": content}
    return {"saved": name, "chars": len(content),
            "location": "temporary sandbox folder (deleted when the server restarts)"}


# ---------------------------------------------------------------- registry
FUNCS = {"load_data": load_data, "calculate_churn": calculate_churn,
         "flag_at_risk": flag_at_risk, "verify_calculation": verify_calculation,
         "get_customer_history": get_customer_history, "write_report": write_report}


def run_tool(name, args, state):
    if name not in FUNCS:
        return {"error": f"Unknown tool: {name}"}
    if not isinstance(args, dict):
        return {"error": "Tool arguments must be an object."}
    try:
        return FUNCS[name](state, **args)
    except Exception as exc:  # return errors to the agent instead of crashing
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def _schema(name, description, props, required):
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": props, "required": required}}


_date = {"type": "string", "description": "YYYY-MM-DD"}
TOOL_SCHEMAS = [
    _schema("load_data", "Load purchase records between two dates into memory (row-capped). "
            "Must be called before calculate_churn, flag_at_risk and verify_calculation. "
            "To compute churn for a period, load from at least 60 days before period start; "
            "to flag at-risk, load from 180 days before as_of.",
            {"start_date": _date, "end_date": _date}, ["start_date", "end_date"]),
    _schema("calculate_churn", "Customer churn % for a period = customers lost / customers active "
            "at start * 100. Churned = no purchase in 60 days.",
            {"period_start": _date, "period_end": _date}, ["period_start", "period_end"]),
    _schema("flag_at_risk", "List customers with dropping purchase frequency, with a reason for each "
            "(top 10 detailed, all IDs listed), and report already-churned customers separately.",
            {"as_of_date": _date}, ["as_of_date"]),
    _schema("verify_calculation", "Independently recompute churn for the same period and check it "
            "matches calculate_churn. Always run this before giving a final churn number.",
            {"period_start": _date, "period_end": _date}, ["period_start", "period_end"]),
    _schema("get_customer_history", "Look up ONE customer by ID (like C0114): signup date, purchase "
            "count, last purchase, spend, average gap between purchases, recent purchases, and risk "
            "status (healthy, at_risk or churned) as of 2026-09-30. Does not need load_data. "
            "Returns an error if the customer does not exist.",
            {"customer_id": {"type": "string", "description": "Customer ID such as C0114"}},
            ["customer_id"]),
    _schema("write_report", "Save a short markdown report (max 3000 characters) to a sandbox folder. "
            "REQUIRES HUMAN APPROVAL: a person sees these exact arguments and must approve before "
            "anything is saved. Only call it when the user asks to save a report.",
            {"filename": {"type": "string"}, "content": {"type": "string"}},
            ["filename", "content"]),
]
