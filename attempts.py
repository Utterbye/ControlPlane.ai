"""
THE FRONT OF THE PIPELINE

Three things that happen before any retrieval, in this order:

    rewrite_query    turn a half-sentence into a standalone question
    attempt_state    is this the same question again? which attempt is it?
    cache_lookup     have we already answered this, in any wording?

Order matters, and this is the order
------------------------------------
Rewriting comes FIRST. "What about last month?" matches nothing in the cache
and retrieves nothing useful. Fix the question, then look it up. Doing it the
other way round means the cache and the retriever both work on a broken query
and both fail quietly.

The attempt check comes before the cache, because the cache must be SKIPPED
when someone is on their second or third try. Serving a stored answer to a
person who already rejected that answer is the single worst thing this
pipeline can do - it proves nobody is listening.

The two thresholds, and why they are different
----------------------------------------------
    cache hit needs 0.95   strict. Serving the wrong stored answer is costly.
    frustration needs 0.85 loose. People reword the same question constantly.

Same technique, different tolerance, because the cost of being wrong is
different in each direction.
"""

import re
import time
from datetime import date, timedelta

import numpy as np

import shared_encoder as enc

SPACE = "queries"   # own TF-IDF vocabulary, isolated from every other module

CACHE_SIM = 0.95
FRUSTRATION_SIM = 0.85
SESSION_WINDOW = 10          # how many recent questions to compare against

PUSHBACK = [
    "no that's not", "not what i asked", "that's wrong", "you misunderstood",
    "i already said", "again", "phir se", "read my question", "not helpful",
    "you didn't answer", "this is wrong", "try again",
]

SHORT_FORMS = {
    "wfh": "work from home", "pf": "provident fund", "ctc": "cost to company",
    "hr": "human resources", "od": "on duty", "lop": "loss of pay",
    "ncns": "no call no show", "pip": "performance improvement plan",
}


# ============================================================================
# 1. REWRITE
# ============================================================================

def rewrite_query(question: str, history=None, today: date = None) -> dict:
    """
    Four changes, each one because a later step breaks without it.
    """
    today = today or date(2026, 8, 26)
    history = history or []
    original, q = question, question
    changes, assumptions = [], []

    # a) short forms -> full words, so the search matches the document wording
    for short, full in SHORT_FORMS.items():
        pat = re.compile(rf"\b{short}\b", re.I)
        if pat.search(q):
            q = pat.sub(full, q)
            changes.append(f"{short} -> {full}")

    # b) relative time -> real dates. Not different WORDS - actual dates, so a
    #    search filter and a metadata comparison can both use them.
    rel = {
        "last month": (today.replace(day=1) - timedelta(days=1)).replace(day=1),
        "this month": today.replace(day=1),
        "last year": date(today.year - 1, 1, 1),
    }
    for phrase, start in rel.items():
        if re.search(rf"\b{phrase}\b", q, re.I):
            if phrase == "last month":
                end = today.replace(day=1) - timedelta(days=1)
            elif phrase == "this month":
                end = today
            else:
                end = date(today.year - 1, 12, 31)
            span = f"{start.isoformat()} to {end.isoformat()}"
            q = re.sub(rf"\b{phrase}\b", span, q, flags=re.I)
            changes.append(f"{phrase} -> {span}")
            # "last month" is ambiguous - calendar month or the past 30 days.
            # Pick one rule and SHOW the assumption rather than guess silently.
            assumptions.append(f"reading '{phrase}' as {span}")

    # c) pronouns -> the thing they refer to, taken from the last question
    if history and re.search(r"\b(it|that|this|those|they)\b", q, re.I):
        subject = _subject_of(history[-1])
        if subject:
            q = re.sub(r"\b(it|that|this)\b", subject, q, count=1, flags=re.I)
            changes.append(f"pronoun -> {subject}")

    # d) two questions in one -> split, so neither half gets ignored
    parts = [p.strip() for p in re.split(r"\band\b|\?", q) if len(p.strip()) > 12]
    split = parts if len(parts) > 1 else []
    if split:
        changes.append(f"split into {len(split)} questions")

    return {
        "original": original,
        "rewritten": q.strip(),
        "changes": changes,
        "assumptions": assumptions,
        "split": split,
        "changed": q.strip() != original.strip(),
    }


def _subject_of(prev: str):
    words = [w for w in re.findall(r"[a-z]+", prev.lower())
             if len(w) > 4 and w not in ("about", "which", "there", "would")]
    return words[-1] if words else None


# ============================================================================
# 2. ATTEMPT STATE
# ============================================================================

class AttemptTracker:
    """
    Counts how many times a person has asked the same thing in ONE thread.

    Per thread, not per user, and that distinction is the whole point. Three
    similar questions in one conversation is a frustrated person. Three across
    three conversations is a normal Tuesday.
    """

    def __init__(self):
        self.threads = {}     # thread -> [(question, vector)]
        self.counts = {}      # thread -> attempt number
        self.last_feedback = {}

    def note_feedback(self, thread, thumbs):
        self.last_feedback[thread] = thumbs

    def check(self, question: str, thread: str) -> dict:
        t0 = time.perf_counter()
        hist = self.threads.get(thread, [])
        pushed_back = any(p in question.lower() for p in PUSHBACK)
        thumbed_down = self.last_feedback.get(thread) == "down"

        best_sim, matched = 0.0, None
        if hist:
            v = enc.encode([question], space=SPACE)[0]
            sims = np.array([float(v @ h[1]) for h in hist[-SESSION_WINDOW:]])
            i = int(sims.argmax())
            best_sim = float(sims[i])
            matched = hist[-SESSION_WINDOW:][i][0]

        # A thumbs-down or an explicit complaint is proof on its own. No
        # similarity check needed - the person told us.
        if pushed_back or thumbed_down:
            attempt = self.counts.get(thread, 1) + 1
            why = "explicit push-back" if pushed_back else "thumbs-down on the last answer"
        elif best_sim >= FRUSTRATION_SIM:
            attempt = self.counts.get(thread, 1) + 1
            why = f"same meaning as an earlier question ({best_sim:.2f})"
        else:
            attempt = 1
            why = "new topic"

        self.counts[thread] = attempt
        return {
            "attempt": attempt,
            "why": why,
            "similarity": round(best_sim, 2),
            "matched_earlier": matched,
            "skip_cache": attempt >= 2,
            "strategy": STRATEGY[min(attempt, 3)],
            "time_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def record(self, question: str, thread: str):
        v = enc.encode([question], space=SPACE)[0]
        self.threads.setdefault(thread, []).append((question, v))

    def reset_on_success(self, thread):
        """A thumbs-up ends the thread of frustration. Start counting again."""
        self.counts[thread] = 1
        self.last_feedback[thread] = "up"


STRATEGY = {
    1: {"name": "normal flow", "cache": True, "widen": False,
        "model": "standard", "say": None},
    2: {"name": "change the strategy", "cache": False, "widen": True,
        "model": "stronger",
        "say": "Let me try this a different way."},
    3: {"name": "stop answering, start helping", "cache": False, "widen": True,
        "model": "stronger",
        "say": "I'm not getting this right. Here's exactly what I could and "
               "couldn't find, and I can put you in touch with someone."},
}


# ============================================================================
# 3. SEMANTIC CACHE
# ============================================================================

class SemanticCache:
    """
    Stores verified answers and serves them for questions that MEAN the same
    thing, not just questions spelled the same way.

    A cached answer is not only faster - it is an answer that already passed
    every check. That is why the storing conditions are strict.
    """

    def __init__(self):
        self.entries = []     # dicts with question, vector, answer, meta

    def store(self, question, answer, trust_label, doc_version,
              personal=False, thumbs=None, passed_checks=True):
        # Five conditions, and all of them must hold.
        if personal:
            return {"stored": False, "why": "personal answers are never cached - "
                                            "the answer differs per person"}
        if not passed_checks:
            return {"stored": False, "why": "the answer did not pass its checks"}
        if thumbs != "up":
            return {"stored": False, "why": "no thumbs-up yet"}
        self.entries.append({
            "question": question, "vector": enc.encode([question], space=SPACE)[0],
            "answer": answer, "trust_label": trust_label,
            "doc_version": doc_version, "stored_at": time.time(),
        })
        return {"stored": True, "why": "verified, liked, and not personal"}

    def lookup(self, question, doc_version, skip=False):
        t0 = time.perf_counter()
        if skip:
            return {"hit": False, "why": "cache skipped - the user is retrying",
                    "time_ms": 0.0}
        if not self.entries:
            return {"hit": False, "why": "cache is empty", "time_ms": 0.0}

        v = enc.encode([question], space=SPACE)[0]
        sims = np.array([float(v @ e["vector"]) for e in self.entries])
        i = int(sims.argmax())
        best = float(sims[i])
        entry = self.entries[i]

        if best < CACHE_SIM:
            return {"hit": False, "why": f"closest stored question is {best:.2f}, "
                                         f"below {CACHE_SIM}",
                    "time_ms": round((time.perf_counter() - t0) * 1000, 2)}
        if entry["doc_version"] != doc_version:
            # The documents changed underneath this answer. It may have been
            # right when it was stored and wrong now, and nothing about the
            # question tells you that.
            return {"hit": False, "why": "source documents changed since this "
                                         "answer was stored",
                    "time_ms": round((time.perf_counter() - t0) * 1000, 2)}
        return {"hit": True, "answer": entry["answer"],
                "trust_label": entry["trust_label"],
                "similarity": round(best, 3),
                "matched": entry["question"],
                "why": "verified answer to the same question",
                "time_ms": round((time.perf_counter() - t0) * 1000, 2)}


# ============================================================================

enc.register_corpus([
    "what is the notice period", "how much reimbursement can i claim",
    "what are the office timings", "what is the leave policy",
], space=SPACE)

if __name__ == "__main__":
    print("1. REWRITE")
    for q, hist in [("What is the WFH policy?", []),
                    ("How many claims did we get last month?", []),
                    ("What about it for probation?", ["What is the notice period?"])]:
        r = rewrite_query(q, hist)
        print(f"   {q}")
        print(f"   -> {r['rewritten']}")
        print(f"      changes {r['changes']}  assumptions {r['assumptions']}")

    print("\n2. ATTEMPT LADDER")
    tr = AttemptTracker()
    seq = [("What is the reimbursement limit?", None),
           ("How much can I claim as reimbursement?", None),
           ("no that's not what I asked", None)]
    for q, fb in seq:
        r = tr.check(q, "t1")
        tr.record(q, "t1")
        print(f"   attempt {r['attempt']}  sim {r['similarity']}  "
              f"skip_cache={r['skip_cache']}  {r['why']}")
        print(f"      strategy: {r['strategy']['name']}")

    print("\n3. CACHE")
    c = SemanticCache()
    print("  ", c.store("What is the notice period?", "60 days [S1].",
                        "verified", "v1", thumbs="up"))
    print("  ", c.store("What is my leave balance?", "You have 12.",
                        "verified", "v1", personal=True, thumbs="up"))
    print("  ", c.lookup("What is the notice period?", "v1"))
    print("  ", c.lookup("Tell me about the notice period", "v1"))
    print("  ", c.lookup("What is the notice period?", "v2"))
    print("  ", c.lookup("What is the notice period?", "v1", skip=True))
