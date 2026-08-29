"""
TIER 3 - THE TRACE STORE

Tier 1 and Tier 2 decide things. This file only REMEMBERS them.

Nothing here ever touches a live request. A trace is written after the answer
has already gone out, and every background job reads from this file rather
than from the running system. That separation is the whole point: an analysis
job that can slow down or break a user request is not a background job.

What a trace holds
------------------
One row per request: the question, the answer, every detector score, the
decision, the timings, the cost, and later the user's thumbs. That is enough
to answer every question the background jobs ask, without ever re-running the
pipeline.

Why local JSONL and not LangSmith directly
------------------------------------------
LangSmith is the production home for this - it does tracing, feedback and
dashboards properly. But a prototype that only works with an API key is a
prototype nobody can run. So the store is a plain file with the same shape as
a LangSmith run, and to_langsmith() shows the mapping.

One line per row, appended, never rewritten. That means a crash mid-write
loses one row instead of the file.
"""

import json
import os
import time
import uuid
from collections import defaultdict

TRACE_FILE = "traces.jsonl"


# ---------------------------------------------------------------- writing

def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def short_ref(run_id: str) -> str:
    """The reference id the user sees on a refusal. Same id, shorter."""
    return f"{run_id[:4]}-{run_id[4:6]}"


def record_trace(question, answer, t1_in, t1_out=None, t2=None, decision=None,
                 user="anon", thread=None, tokens=0, timings=None,
                 run_id=None, path=TRACE_FILE) -> dict:
    """
    Called AFTER the response. Flattens the scores so the jobs can read them
    without digging through nested dicts.
    """
    run_id = run_id or new_run_id()
    row = {
        "run_id": run_id,
        "ref": short_ref(run_id),
        "ts": time.time(),
        "user": user,
        "thread": thread or user,
        "question": question,
        "answer": answer,
        "tokens": tokens,

        # flattened so a job can do row["t1_in.injection"] style lookups
        "t1_in_score": t1_in.get("risk_score"),
        "t1_in_band": t1_in.get("band"),
        "detectors": t1_in.get("scores", {}),
        "strict_mode": t1_in.get("strict_mode", False),

        "t1_out_score": (t1_out or {}).get("risk_score"),
        "t2_score": (t2 or {}).get("risk_score"),
        "t2_scores": (t2 or {}).get("scores", {}),
        "t2_ran": (t2 or {}).get("ran", []),

        "action": (decision or {}).get("action", t1_in.get("action")),
        "trust_label": (decision or {}).get("trust_label"),
        "retry_count": (decision or {}).get("retry_count", 0),

        "timings": timings or {},
        "feedback": None,          # filled in later by record_feedback
        "sources_disagreed": (t2 or {}).get("sources_disagreed", False),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def record_feedback(run_id: str, thumbs: str, path=TRACE_FILE):
    """
    Thumbs arrive minutes after the answer, so they cannot be part of the
    original row. Appended as a separate event and stitched together on read -
    the same way LangSmith attaches feedback to a run id.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": run_id, "feedback": thumbs,
                            "ts": time.time(), "kind": "feedback"}) + "\n")


# ---------------------------------------------------------------- reading

def load_traces(path=TRACE_FILE):
    """Read every row, then stitch the feedback events onto their runs."""
    if not os.path.exists(path):
        return []
    rows, feedback = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue                      # a half-written line, skip it
            if d.get("kind") == "feedback":
                feedback[d["run_id"]] = d["feedback"]
            else:
                rows.append(d)
    for r in rows:
        if r["run_id"] in feedback:
            r["feedback"] = feedback[r["run_id"]]
    return rows


def clear_traces(path=TRACE_FILE):
    if os.path.exists(path):
        os.remove(path)


def summary(rows=None):
    rows = rows if rows is not None else load_traces()
    if not rows:
        return {"rows": 0}
    bands = defaultdict(int)
    actions = defaultdict(int)
    for r in rows:
        bands[r.get("t1_in_band", "?")] += 1
        actions[r.get("action", "?")] += 1
    thumbs = [r["feedback"] for r in rows if r.get("feedback")]
    return {
        "rows": len(rows),
        "users": len({r["user"] for r in rows}),
        "bands": dict(bands),
        "actions": dict(actions),
        "tokens": sum(r.get("tokens", 0) for r in rows),
        "thumbs_down": thumbs.count("down"),
        "thumbs_up": thumbs.count("up"),
        "tier2_ran": sum(1 for r in rows if r.get("t2_score") is not None),
    }


# ---------------------------------------------------------------- langsmith

def to_langsmith(row: dict) -> dict:
    """
    The mapping, so swapping the store for real LangSmith is a rename and not
    a redesign. Scores become feedback entries; everything else becomes run
    metadata attached to the same run id the user saw as a reference.
    """
    return {
        "id": row["run_id"],
        "name": "controlplane_request",
        "inputs": {"question": row["question"]},
        "outputs": {"answer": row["answer"], "trust_label": row["trust_label"]},
        "extra": {"metadata": {
            "user": row["user"], "thread": row["thread"],
            "band": row["t1_in_band"], "action": row["action"],
            "strict_mode": row["strict_mode"], "retry_count": row["retry_count"],
            **{f"detector.{k}": v for k, v in row["detectors"].items()},
            **{f"tier2.{k}": v for k, v in row["t2_scores"].items()},
        }},
        "feedback": ([{"key": "user_thumbs", "score": 1 if row["feedback"] == "up" else 0}]
                     if row["feedback"] else []),
    }


if __name__ == "__main__":
    clear_traces()
    fake_t1 = {"risk_score": 17.0, "band": "normal", "action": "allow",
               "scores": {"injection": 0.0, "unsafe": 0.0, "pii": 0.0,
                          "high_stakes": 0.45, "usage": 0.0},
               "strict_mode": True}
    r = record_trace("What is the notice period?", "Notice period is 60 days [S3].",
                     fake_t1, decision={"action": "allow",
                                        "trust_label": "verified against your documents"},
                     user="u1", thread="t1", tokens=120)
    record_feedback(r["run_id"], "up")
    print("run_id", r["run_id"], " reference shown to user:", r["ref"])
    print("summary:", summary())
    print("\nas a LangSmith run:")
    print(json.dumps(to_langsmith(load_traces()[0]), indent=2)[:520])
