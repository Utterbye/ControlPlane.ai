"""
TIER 2 - THE DEEP CHECK, AND THE DECISION ENGINE

Tier 2 does not run on everything. It runs when Tier 1 says the answer is
risky, or when the high-stakes detector turned strict mode on. That is the
whole reason the system is fast: most answers never reach this file.

Three checks, and they are NOT equal
------------------------------------
    groundedness   is each claim actually in the evidence it cited?
    consistency    did the model give the same facts twice?
    output safety  is the wording unfair, unsafe, or overconfident?

Groundedness is the expensive and important one, so it always runs.
Consistency needs a SECOND generation, which costs a whole extra model call,
so it only runs when it is likely to earn that cost.

Parallel here, unlike Tier 1
----------------------------
In Tier 1 threads lost, because every detector finished in under a
millisecond and thread setup cost more than the work. Here the checks are
model calls taking hundreds of milliseconds, so running them together turns
three waits into one. Same flag, opposite answer, because the workload is
different. That is measurement, not preference.
"""

import time
from concurrent.futures import ThreadPoolExecutor

from tier2_groundedness import check_groundedness
from tier2_checks import check_consistency, check_output_safety

WEIGHTS = {"groundedness": 50, "self_consistency": 30, "output_safety": 20}


# ============================================================================
# TIER 2
# ============================================================================

def tier2(answer: str, evidence: dict, strict: bool = False,
          second_answer: str = None, llm_judge=None, parallel: bool = True) -> dict:
    """
    answer         the text about to be delivered
    evidence       {"S1": "...", "S2": "..."} - what the tags refer to
    strict         high stakes, so judge harder
    second_answer  a second generation from the same evidence. Without it the
                   consistency check is skipped rather than faked.
    llm_judge      optional real judge for the claims overlap cannot settle
    """
    t0 = time.perf_counter()

    jobs = {
        "groundedness": lambda: check_groundedness(answer, evidence, llm_judge=llm_judge),
        "output_safety": lambda: check_output_safety(answer, strict=strict),
    }
    if second_answer:
        jobs["self_consistency"] = lambda: check_consistency(answer, second_answer)

    if parallel and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            done = dict(zip(jobs, pool.map(lambda f: f(), jobs.values())))
    else:
        done = {k: f() for k, f in jobs.items()}

    # Weighted over the checks that actually ran, so skipping consistency does
    # not silently make every answer look 30 points safer.
    total_w = sum(WEIGHTS[k] for k in done)
    weighted = sum(done[k]["score"] * WEIGHTS[k] for k in done) / total_w * 100
    solo = max(done[k]["score"] * 100 for k in done)
    final = max(weighted, solo)

    return {
        "tier": 2,
        "risk_score": round(final, 1),
        "weighted": round(weighted, 1),
        "solo": round(solo, 1),
        "ran": sorted(done),
        "skipped": ["self_consistency"] if not second_answer else [],
        "scores": {k: v["score"] for k, v in done.items()},
        "detail": done,
        "time_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


# ============================================================================
# DECISION ENGINE
# ============================================================================
#
# Two inputs only: how bad is it, and how sure are we.
# One place decides, so the rules stay the same everywhere and any decision
# can still be explained months later in an audit.

def decide(tier1_result: dict, tier2_result: dict = None,
           retry_count: int = 0, strict: bool = False) -> dict:
    t0 = time.perf_counter()
    reasons = []

    t1_score = tier1_result["risk_score"]
    t2 = tier2_result or {}
    t2_score = t2.get("risk_score", 0.0)
    severity = max(t1_score, t2_score)

    # confidence = how certain the strongest signal was, not how bad it was
    confident = severity >= 80 or t1_result_confident(tier1_result)

    g = t2.get("detail", {}).get("groundedness", {})
    unsupported = g.get("unsupported", 0)

    # ---- hard rules first, in the order they must win
    if tier1_result["action"] in ("block", "block_and_route_hr", "care"):
        action = tier1_result["action"]
        reasons.append(f"Tier 1 asked for {action}")

    elif unsupported and retry_count == 0:
        action = "retry"
        reasons.append(f"{unsupported} claim(s) not supported by the cited evidence")

    elif unsupported and retry_count >= 1:
        action = "abstain"
        reasons.append("still unsupported after one retry - the loop brake")

    elif severity >= 80 and confident:
        action = "block"
        reasons.append(f"severity {severity} and the signal is certain")

    elif severity >= 40 and not confident:
        action = "escalate" if strict else "allow_with_note"
        reasons.append(f"severity {severity} but the signal is not certain")

    elif severity >= 40 and confident:
        # Strict mode refuses silent edits. A policy answer with one line
        # quietly deleted looks complete and is not, which is worse than
        # no answer at all.
        action = "escalate" if strict else "auto_fix"
        reasons.append("auto_fix is off in strict mode" if strict
                       else "small, and we are sure - fix it and deliver")

    else:
        action = "allow"
        reasons.append("nothing worth acting on")

    label = trust_label(tier1_result, t2, action)

    return {
        "action": action,
        "severity": round(severity, 1),
        "confident": confident,
        "strict": strict,
        "retry_count": retry_count,
        "trust_label": label,
        "reasons": reasons,
        "time_ms": round((time.perf_counter() - t0) * 1000, 3),
    }


def t1_result_confident(t1: dict) -> bool:
    """A detector allowed to act alone, scoring high, counts as certain."""
    return any(v >= 0.80 for k, v in t1.get("scores", {}).items()
               if k != "high_stakes")


def trust_label(t1: dict, t2: dict, action: str) -> str:
    """What the user actually sees next to the answer."""
    if action in ("block", "abstain", "block_and_route_hr"):
        return "could not verify"
    g = t2.get("detail", {}).get("groundedness", {})
    if g:
        if g.get("unsupported"):
            return "could not verify"
        if g.get("unsure"):
            return "partly verified"
        if g.get("claims"):
            return "verified against your documents"
    return "not checked in depth"


if __name__ == "__main__":
    evidence = {"S1": "Bereavement leave is up to 3 days, applied for in advance."}

    cases = [
        ("clean answer",
         "Bereavement leave is up to 3 days [S1]. Apply before taking it [S1].",
         "You get a maximum of 3 days, applied for beforehand.", False),
        ("invented number",
         "You can claim the bereavement fare within 90 days [S1].",
         "The bereavement fare can be claimed within 60 days.", True),
        ("overconfident, strict",
         "You will definitely get 3 days of bereavement leave [S1].",
         None, True),
    ]
    fake_t1 = {"risk_score": 20.0, "action": "allow", "scores": {"unsafe": 0.0}}

    for name, ans, second, strict in cases:
        r2 = tier2(ans, evidence, strict=strict, second_answer=second)
        d = decide(fake_t1, r2, retry_count=0, strict=strict)
        print(f"\n{name}")
        print(f"  tier2 {r2['risk_score']:>5}  ran={r2['ran']} "
              f"scores={r2['scores']}  {r2['time_ms']}ms")
        print(f"  decision: {d['action']}  label='{d['trust_label']}'")
        print(f"  because: {d['reasons'][0]}")
