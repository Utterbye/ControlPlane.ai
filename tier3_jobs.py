"""
TIER 3 - THE FIVE BACKGROUND JOBS

Every job here answers a question that a single request cannot answer.

    bias_pattern      is the system treating similar people differently?
    cost_patterns     where is the money going, and is any of it waste?
    thumbs_clusters   which TOPIC is failing, not which answer?
    repeat_queries    who is asking the same thing over and over?
    conflict_rate     which topics do our sources disagree about?

The thing they have in common
-----------------------------
None of them can be answered by looking at one request. That is not a
limitation to apologise for - it is the definition of this tier. Tier 1 and
Tier 2 look at one request very fast. Tier 3 looks at thousands, slowly, and
finds the things that are only visible in aggregate.

The bias job is the clearest example. A single answer that says "you should
consider a lighter role" is not evidence of anything. The SAME answer given to
Rahul and not to Rekha, on the same question, is. You need the pair.
"""

import re
import time
from collections import Counter, defaultdict

from tier3_trace import load_traces

# ============================================================================
# JOB 1 - BIAS PATTERN
# ============================================================================
#
# The counterfactual test. Take a real question, change ONLY an attribute that
# should not matter, and see whether the system reacts differently.
#
# Anything that changes is a bias signal, because the only thing that changed
# was the name.

SWAPS = {
    "name": [("Rahul", "Rekha"), ("Arjun", "Ayesha"), ("Vikram", "Fatima"),
             ("Suresh", "Priya"), ("David", "Deepak")],
    "gender": [("he", "she"), ("his", "her"), ("him", "her"),
               ("male", "female"), ("husband", "wife")],
    "city": [("Mumbai", "Patna"), ("Bangalore", "Ranchi"),
             ("Delhi", "Guwahati")],
}


def make_counterfactuals(question: str):
    """
    Produce (variant_a, variant_b, attribute) for every swap that applies.
    Returns nothing if the question has no swappable attribute - most
    questions do not, and pretending otherwise would invent findings.
    """
    out = []
    for attr, pairs in SWAPS.items():
        for a, b in pairs:
            pattern = re.compile(rf"\b{re.escape(a)}\b", re.I)
            if pattern.search(question):
                out.append((question, pattern.sub(b, question), attr, a, b))
    return out


def bias_pattern(run_fn, rows=None, sample: int = 50) -> dict:
    """
    run_fn(text) -> dict with a comparable score. Usually tier1.tier1.

    Runs on REAL questions from the traces, not invented ones, so the test
    reflects what users actually ask.
    """
    t0 = time.perf_counter()
    rows = rows if rows is not None else load_traces()
    pairs, tested, diffs = [], 0, []

    for r in rows[:sample]:
        for q_a, q_b, attr, a, b in make_counterfactuals(r["question"]):
            ra, rb = run_fn(q_a), run_fn(q_b)
            tested += 1
            gap = abs(ra["risk_score"] - rb["risk_score"])
            if gap > 0 or ra["band"] != rb["band"]:
                # A gap with no culprit is not actionable. Naming the detector
                # that moved turns "there is bias somewhere" into a ticket.
                culprits = {
                    k: (ra["scores"].get(k, 0), rb["scores"].get(k, 0))
                    for k in set(ra["scores"]) | set(rb["scores"])
                    if ra["scores"].get(k, 0) != rb["scores"].get(k, 0)
                }
                diffs.append({
                    "attribute": attr, "swapped": f"{a} -> {b}",
                    "question": q_a[:60],
                    "score_a": ra["risk_score"], "score_b": rb["risk_score"],
                    "band_a": ra["band"], "band_b": rb["band"], "gap": round(gap, 1),
                    "detectors_that_moved": culprits,
                })
            pairs.append(gap)

    worst = sorted(diffs, key=lambda d: -d["gap"])[:5]
    blamed = Counter()
    for d in diffs:
        blamed.update(d["detectors_that_moved"].keys())
    return {
        "job": "bias_pattern",
        "detectors_blamed": blamed.most_common(),
        "pairs_tested": tested,
        "pairs_that_differed": len(diffs),
        "max_gap": round(max(pairs), 1) if pairs else 0.0,
        "mean_gap": round(sum(pairs) / len(pairs), 2) if pairs else 0.0,
        "worst": worst,
        "verdict": ("no difference found" if not diffs else
                    f"{len(diffs)} of {tested} pairs scored differently"
                    + (f", {blamed.most_common(1)[0][0]} moved most often"
                       if blamed else "")),
        "time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


# ============================================================================
# JOB 2 - COST PATTERNS
# ============================================================================
#
# Request count is not the money. Tokens are. One 40,000 character paste costs
# more than fifty ordinary questions, and a request counter cannot see that.

def cost_patterns(rows=None) -> dict:
    t0 = time.perf_counter()
    rows = rows if rows is not None else load_traces()
    if not rows:
        return {"job": "cost_patterns", "rows": 0}

    by_user = defaultdict(int)
    by_band = defaultdict(int)
    retries = tier2_runs = blocked_early = 0
    total = 0

    for r in rows:
        tk = r.get("tokens", 0)
        total += tk
        by_user[r["user"]] += tk
        by_band[r.get("t1_in_band", "?")] += tk
        retries += r.get("retry_count", 0)
        if r.get("t2_score") is not None:
            tier2_runs += 1
        if r.get("action") in ("block", "block_and_route_hr") and r.get("t1_out_score") is None:
            blocked_early += 1

    top = sorted(by_user.items(), key=lambda kv: -kv[1])[:3]
    share = round(top[0][1] / total * 100, 1) if top and total else 0.0

    return {
        "job": "cost_patterns",
        "requests": len(rows),
        "tokens_total": total,
        "tokens_per_request": round(total / len(rows), 1),
        "top_users": top,
        "top_user_share": share,
        "tokens_by_band": dict(by_band),
        "retries": retries,
        "tier2_rate": round(tier2_runs / len(rows) * 100, 1),
        "blocked_before_retrieval": blocked_early,
        # the saving that tiering actually bought, in requests that never
        # reached a model at all
        "saved_by_early_block": blocked_early,
        "verdict": (f"{share}% of all tokens came from one user; "
                    f"{blocked_early} requests were blocked before any model call"),
        "time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


# ============================================================================
# JOB 3 - THUMBS-DOWN CLUSTERS
# ============================================================================
#
# One thumbs-down is noise. Every reimbursement question getting a thumbs-down
# is a retrieval problem in one topic, and that is actionable.

STOP = set("""a an the is are was were be of to in on at for for with by from as that
this it its and or but if then so what which who how do does did can could i my me you
your we our they their""".split())


def _topic_words(q: str):
    return [w for w in re.findall(r"[a-z]+", q.lower()) if w not in STOP and len(w) > 3]


def thumbs_clusters(rows=None, min_size: int = 2) -> dict:
    t0 = time.perf_counter()
    rows = rows if rows is not None else load_traces()
    down = [r for r in rows if r.get("feedback") == "down"]
    up = [r for r in rows if r.get("feedback") == "up"]

    counts = Counter()
    for r in down:
        counts.update(set(_topic_words(r["question"])))

    # a word is only interesting if it is disproportionately in the DOWN set
    up_counts = Counter()
    for r in up:
        up_counts.update(set(_topic_words(r["question"])))

    clusters = []
    for word, n in counts.most_common(20):
        if n < min_size:
            continue
        u = up_counts.get(word, 0)
        rate = n / (n + u)
        clusters.append({"topic": word, "down": n, "up": u,
                         "failure_rate": round(rate, 2)})

    clusters.sort(key=lambda c: (-c["failure_rate"], -c["down"]))
    return {
        "job": "thumbs_clusters",
        "thumbs_down": len(down),
        "thumbs_up": len(up),
        "overall_down_rate": round(len(down) / max(1, len(down) + len(up)), 2),
        "clusters": clusters[:5],
        "verdict": (f"'{clusters[0]['topic']}' fails "
                    f"{int(clusters[0]['failure_rate']*100)}% of the time"
                    if clusters else "no topic stands out"),
        "time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


# ============================================================================
# JOB 4 - REPEAT QUERIES
# ============================================================================
#
# The usage guard catches repetition inside one session, live. This job finds
# it ACROSS sessions and across users - the same question asked by forty
# different people is not a frustrated user, it is a missing document.

def repeat_queries(rows=None) -> dict:
    t0 = time.perf_counter()
    rows = rows if rows is not None else load_traces()

    by_thread = defaultdict(list)
    normalised = Counter()
    for r in rows:
        key = " ".join(sorted(set(_topic_words(r["question"]))))
        if key:
            normalised[key] += 1
            by_thread[r["thread"]].append(key)

    frustrated = []
    for thread, keys in by_thread.items():
        c = Counter(keys)
        for key, n in c.items():
            if n >= 3:
                frustrated.append({"thread": thread, "topic": key[:48], "asked": n})

    popular = [{"topic": k[:48], "asked_by_many": n}
               for k, n in normalised.most_common(5) if n >= 3]

    return {
        "job": "repeat_queries",
        "threads": len(by_thread),
        "frustrated_threads": frustrated[:5],
        "asked_repeatedly_overall": popular,
        "verdict": ("one thread asked the same thing 3+ times - frustration"
                    if frustrated else
                    "no single thread is stuck; repeats are spread across users, "
                    "which points at a missing document rather than a bad answer"),
        "time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


# ============================================================================
# JOB 5 - CONFLICT RATE
# ============================================================================
#
# Which topics do our own sources disagree about? A high conflict rate on one
# topic means the documents themselves need fixing - no amount of better
# retrieval solves a policy that contradicts itself.

def conflict_rate(rows=None) -> dict:
    t0 = time.perf_counter()
    rows = rows if rows is not None else load_traces()
    checked = [r for r in rows if r.get("t2_score") is not None]
    conflicted = [r for r in checked if r.get("sources_disagreed")]

    by_topic = Counter()
    for r in conflicted:
        by_topic.update(set(_topic_words(r["question"])))

    return {
        "job": "conflict_rate",
        "answers_deep_checked": len(checked),
        "sources_disagreed": len(conflicted),
        "rate": round(len(conflicted) / max(1, len(checked)), 2),
        "worst_topics": by_topic.most_common(5),
        "verdict": ("sources contradict each other on "
                    f"{by_topic.most_common(1)[0][0]}" if by_topic
                    else "no source conflicts recorded"),
        "time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
