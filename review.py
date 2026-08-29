"""
HUMAN REVIEW AND FEEDBACK

Two boxes on the chart that are easy to draw and easy to skip, and skipping
them quietly breaks the claim the whole system rests on.

Everything else in this project says "the flywheel learns from human
decisions". Without a review queue there ARE no human decisions - only raw
thumbs, which are not labels. A thumbs-down means "look at this", not "this
was wrong". People downvote correct answers they did not like.

So this file is where labelled data actually gets created:

    escalated answer  ->  a reviewer looks  ->  a verdict  ->  training data

Feedback is different, and the difference matters
-------------------------------------------------
    thumbs      cheap, plentiful, noisy.   A signal that something is worth
                looking at. Never a label.
    review      slow, rare, trustworthy.   A label.

The queue turns the first into the second, and only the second is allowed to
change anything.

Why the queue must stay small
-----------------------------
If everything escalates, the queue becomes the new bottleneck and nothing gets
reviewed at all. A queue nobody works is worse than no queue, because it looks
like oversight while providing none. So the decision engine only sends the
thin slice it is genuinely unsure about, and this file measures whether that
slice is staying thin.
"""

import json
import os
import time
from collections import Counter

QUEUE_FILE = "review_queue.jsonl"
SLA_HOURS = 24


# ============================================================================
# THE QUEUE
# ============================================================================

def escalate(run_id, question, answer, evidence, decision, tier2=None,
             path=QUEUE_FILE) -> dict:
    """
    Called when the decision engine says escalate. The reviewer needs
    everything needed to judge WITHOUT re-running anything - the question, the
    answer, the evidence it cited, and why the machine was unsure.
    """
    item = {
        "run_id": run_id,
        "queued_at": time.time(),
        "due_at": time.time() + SLA_HOURS * 3600,
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "machine_said": {
            "action": decision.get("action"),
            "severity": decision.get("severity"),
            "confident": decision.get("confident"),
            "reasons": decision.get("reasons", []),
            "trust_label": decision.get("trust_label"),
        },
        "tier2_scores": (tier2 or {}).get("scores", {}),
        "status": "waiting",
        "verdict": None,
        "reviewer": None,
        "note": None,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item) + "\n")
    return item


def review(run_id, verdict, reviewer, note="", path=QUEUE_FILE) -> dict:
    """
    A reviewer decides. Four verdicts, and each one means something different
    to the flywheel:

        correct        the machine was right to be unsure, the answer is fine
        wrong          the answer was wrong - this is the valuable one
        should_block   we under-reacted
        over_flagged   we over-reacted, this never needed a human

    "over_flagged" is the one teams forget to offer, and it is the only way
    the system ever learns to relax. Without it, every review round makes the
    checker stricter and nobody notices until users give up on it.
    """
    assert verdict in ("correct", "wrong", "should_block", "over_flagged")
    event = {"kind": "verdict", "run_id": run_id, "verdict": verdict,
             "reviewer": reviewer, "note": note, "at": time.time()}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    return event


def load_queue(path=QUEUE_FILE):
    """Items, with their verdicts stitched on."""
    if not os.path.exists(path):
        return []
    items, verdicts = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("kind") == "verdict":
                verdicts[d["run_id"]] = d
            else:
                items.append(d)
    for it in items:
        v = verdicts.get(it["run_id"])
        if v:
            it.update(status="reviewed", verdict=v["verdict"],
                      reviewer=v["reviewer"], note=v["note"],
                      reviewed_at=v["at"])
    return items


def queue_health(path=QUEUE_FILE) -> dict:
    """
    Is the queue staying thin, and is it being worked?

    An unworked queue is the failure mode to watch for. It looks like
    oversight on a diagram and provides none in reality.
    """
    items = load_queue(path)
    if not items:
        return {"queued": 0, "verdict": "queue is empty"}
    now = time.time()
    waiting = [i for i in items if i["status"] == "waiting"]
    overdue = [i for i in waiting if now > i["due_at"]]
    done = [i for i in items if i["status"] == "reviewed"]
    counts = Counter(i["verdict"] for i in done)

    over = counts.get("over_flagged", 0)
    over_rate = over / len(done) if done else 0.0

    return {
        "queued": len(items),
        "waiting": len(waiting),
        "overdue": len(overdue),
        "reviewed": len(done),
        "verdicts": dict(counts),
        "over_flag_rate": round(over_rate, 2),
        "verdict": (f"{len(overdue)} items past the {SLA_HOURS}h SLA - the queue "
                    f"is not being worked" if overdue else
                    f"{int(over_rate*100)}% of reviews say we over-flagged"
                    if done else "nothing reviewed yet"),
    }


# ============================================================================
# LABELLED DATA - the only output of this file that is allowed to train anything
# ============================================================================

def labelled_data(path=QUEUE_FILE):
    """
    Reviewed items only. This is what the flywheel is permitted to learn from.

    Note what is NOT here: thumbs. They are a signal for choosing what to
    review, never a label in their own right.
    """
    out = []
    for it in load_queue(path):
        if it["status"] != "reviewed":
            continue
        out.append({
            "run_id": it["run_id"],
            "text": it["question"],
            "answer": it["answer"],
            "machine_action": it["machine_said"]["action"],
            "human_verdict": it["verdict"],
            # what the flywheel should do with it
            "lesson": {
                "correct": "the threshold was right, leave it alone",
                "wrong": "we missed something - add this as a training example",
                "should_block": "we under-reacted - raise the score for this shape",
                "over_flagged": "we over-reacted - this belongs in the safe examples",
            }[it["verdict"]],
        })
    return out


def clear(path=QUEUE_FILE):
    if os.path.exists(path):
        os.remove(path)


# ============================================================================
# FEEDBACK - cheap, noisy, and deliberately kept separate
# ============================================================================

def thumbs_to_review_priority(rows):
    """
    Thumbs do not label anything. What they DO is tell you where to point the
    reviewer's limited time.

    An answer with a thumbs-down that the machine was confident about is the
    most interesting thing in the log: the machine was sure, and the user
    disagreed. That gap is where the checker is wrong in a way no test caught.
    """
    scored = []
    for r in rows:
        if r.get("feedback") != "down":
            continue
        confidence = r.get("t1_in_score", 0)
        # confident AND downvoted = the highest value review
        priority = confidence if r.get("action") in ("allow", "auto_fix") else confidence / 2
        scored.append({"run_id": r["run_id"], "question": r["question"][:60],
                       "machine_score": confidence, "action": r.get("action"),
                       "priority": round(priority, 1)})
    scored.sort(key=lambda s: -s["priority"])
    return scored


if __name__ == "__main__":
    clear()
    esc = escalate(
        run_id="abc123",
        question="Is the customer eligible for a bereavement fare?",
        answer="Yes, you can claim it within 90 days [S1].",
        evidence={"S1": "Bereavement fares must be booked in advance."},
        decision={"action": "escalate", "severity": 62.0, "confident": False,
                  "reasons": ["severity 62 but the signal is not certain"],
                  "trust_label": "partly verified"},
        tier2={"scores": {"groundedness": 0.6, "output_safety": 0.3}},
    )
    print("queued:", esc["run_id"], "due in", SLA_HOURS, "hours")
    print("health:", queue_health())

    review("abc123", "wrong", reviewer="hr-priya",
           note="the 90 day window does not exist in any policy")
    print("\nafter review:")
    print(" health:", queue_health())
    print(" labelled data the flywheel may use:")
    for d in labelled_data():
        print(f"   {d['human_verdict']:<13} {d['lesson']}")
