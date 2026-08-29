"""
TIER 3 - THE ORCHESTRATOR AND THE LEARNING FLYWHEEL

Tier 3 runs on a schedule, not on a request. Nothing in this file is allowed
to make a user wait.

The flywheel, and the one rule that keeps it safe
-------------------------------------------------
The flywheel turns findings into changes: risk weights, thresholds, the source
trust list, the re-rank boosts, the cache.

It PROPOSES. It never applies anything on its own.

That single rule is the difference between a system that improves and a system
that quietly destroys itself. A flywheel fed on unverified data spins the wrong
way just as happily as the right way - and nobody notices, because the output
still looks like a clean number.

Three guards, and each one exists because of a specific failure:

    human-verified only   raw thumbs-down is noisy. People downvote correct
                          answers they did not like. Only decisions a reviewer
                          confirmed become training data.

    shadow mode first     a new threshold logs for a week and blocks nothing.
                          If false alarms climb, it is rolled back before any
                          user ever felt it.

    held-out validation   a threshold tuned on the data that suggested it will
                          always look excellent. It has to prove itself on data
                          it has never seen.
"""

import time

from tier3_trace import load_traces, summary
from tier3_jobs import (bias_pattern, cost_patterns, thumbs_clusters,
                        repeat_queries, conflict_rate)

SCHEDULE = {
    "bias_pattern": "weekly",
    "cost_patterns": "daily",
    "thumbs_clusters": "daily",
    "repeat_queries": "daily",
    "conflict_rate": "weekly",
}


def run_all(run_fn=None, rows=None) -> dict:
    """
    run_fn is only needed by the bias job, which has to re-run the pipeline on
    swapped variants. Every other job reads the traces and nothing else.
    """
    t0 = time.perf_counter()
    rows = rows if rows is not None else load_traces()
    out = {
        "cost_patterns": cost_patterns(rows),
        "thumbs_clusters": thumbs_clusters(rows),
        "repeat_queries": repeat_queries(rows),
        "conflict_rate": conflict_rate(rows),
    }
    if run_fn:
        out["bias_pattern"] = bias_pattern(run_fn, rows)
    return {
        "tier": 3,
        "traces": len(rows),
        "summary": summary(rows),
        "jobs": out,
        "time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


# ============================================================================
# THE FLYWHEEL
# ============================================================================

def propose_updates(report: dict) -> list:
    """
    Turn findings into concrete proposals. Every proposal carries the evidence
    that produced it, so a human can judge it without re-running anything.

    Nothing here is applied. The apply step is a person clicking approve.
    """
    jobs = report.get("jobs", {})
    props = []

    # ---- from the bias job
    b = jobs.get("bias_pattern")
    if b and b.get("pairs_that_differed"):
        props.append({
            "target": "detector thresholds",
            "change": "review the detectors that scored the swapped pairs differently",
            "why": f"{b['pairs_that_differed']} of {b['pairs_tested']} counterfactual "
                   f"pairs differed, worst gap {b['max_gap']}",
            "evidence": b["worst"][:2],
            "risk": "high - a bias fix that overcorrects creates a new bias",
            "requires": "human review, then shadow mode",
        })

    # ---- from the cost job
    c = jobs.get("cost_patterns", {})
    if c.get("top_user_share", 0) > 40:
        props.append({
            "target": "usage guard limits",
            "change": "lower per_day for outlier users",
            "why": f"one user is {c['top_user_share']}% of all tokens",
            "evidence": c.get("top_users", [])[:2],
            "risk": "low - throttling is reversible and visible",
            "requires": "shadow mode",
        })
    if c.get("retries", 0) > max(1, c.get("requests", 1)) * 0.1:
        props.append({
            "target": "retrieval",
            "change": "widen retrieval before the retry, not after",
            "why": f"{c['retries']} retries across {c['requests']} requests - "
                   f"the first attempt is under-supplied",
            "evidence": [], "risk": "low", "requires": "shadow mode",
        })

    # ---- from the thumbs job
    t = jobs.get("thumbs_clusters", {})
    for cl in t.get("clusters", [])[:2]:
        if cl["failure_rate"] >= 0.6 and cl["down"] >= 2:
            props.append({
                "target": "source trust list / documents",
                "change": f"audit the documents covering '{cl['topic']}'",
                "why": f"'{cl['topic']}' fails {int(cl['failure_rate']*100)}% "
                       f"of the time ({cl['down']} down, {cl['up']} up)",
                "evidence": [cl],
                "risk": "none - this is a document fix, not a model change",
                "requires": "content owner",
            })

    # ---- from the repeat job
    r = jobs.get("repeat_queries", {})
    for p in r.get("asked_repeatedly_overall", [])[:2]:
        props.append({
            "target": "cache",
            "change": f"pre-warm the cache for '{p['topic'][:32]}'",
            "why": f"asked {p['asked_by_many']} times across different threads",
            "evidence": [p],
            "risk": "low - only verified answers are ever cached",
            "requires": "automatic once an answer has a thumbs-up",
        })

    # ---- from the conflict job
    k = jobs.get("conflict_rate", {})
    if k.get("sources_disagreed"):
        props.append({
            "target": "source trust list",
            "change": "rank the internal policy above the older copy for "
                      f"{k['worst_topics'][0][0] if k.get('worst_topics') else 'the affected topic'}",
            "why": f"sources disagreed on {k['sources_disagreed']} of "
                   f"{k['answers_deep_checked']} deep-checked answers",
            "evidence": k.get("worst_topics", [])[:2],
            "risk": "medium - trust ranking changes every future answer",
            "requires": "human review, then shadow mode",
        })

    return props


def shadow_plan(proposal: dict) -> dict:
    """
    How a proposal reaches production. Nothing skips these steps.
    """
    return {
        "proposal": proposal["change"],
        "steps": [
            "1. a human reviews the evidence and approves or rejects",
            "2. the change goes live in SHADOW mode - it logs, it blocks nothing",
            "3. one week later, compare false alarms and misses against the "
            "week before",
            "4. worse on either? roll back. Better on both? enable it",
            "5. re-run the held-out tests, because a threshold tuned on live "
            "traffic can quietly break the cases the tests cover",
        ],
        "rollback": "keep the previous value in config, so rollback is a "
                    "one-line revert and not a rebuild",
    }


def print_report(report: dict, proposals: list):
    s = report["summary"]
    print("=" * 78)
    print(f"  TIER 3 REPORT   {report['traces']} traces   {report['time_ms']} ms")
    print("=" * 78)
    print(f"  bands {s.get('bands')}")
    print(f"  actions {s.get('actions')}")
    print(f"  tokens {s.get('tokens')}   thumbs up {s.get('thumbs_up')} "
          f"down {s.get('thumbs_down')}")
    print()
    for name, j in report["jobs"].items():
        print(f"  [{SCHEDULE.get(name,'?'):<6}] {name}")
        print(f"           {j.get('verdict', '')}")
    print()
    print(f"  FLYWHEEL PROPOSED {len(proposals)} CHANGES  (none applied)")
    print("  " + "-" * 74)
    for p in proposals:
        print(f"    target   {p['target']}")
        print(f"    change   {p['change']}")
        print(f"    why      {p['why']}")
        print(f"    risk     {p['risk']}")
        print()


if __name__ == "__main__":
    import tier1
    from tier3_trace import load_traces
    tier1.warmup()
    rows = load_traces()
    if not rows:
        print("No traces yet. Run simulate_traffic.py first.")
    else:
        rep = run_all(run_fn=lambda q: tier1.tier1(q, user="bias-test"), rows=rows)
        props = propose_updates(rep)
        print_report(rep, props)
        if props:
            print("  how the first one would reach production")
            for step in shadow_plan(props[0])["steps"]:
                print(f"    {step}")
