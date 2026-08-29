"""
TIER 1 - DETECTOR 4 : HIGH-STAKES TOPIC

What it does:
    Decide whether a confidently wrong answer to THIS question would cause
    real damage. If yes, turn strict mode on.

The one detector that never blocks
----------------------------------
Detectors 1, 2 and 3 all look for something wrong with the query. This one
does not. The query is perfectly legitimate - that is the whole point.

    "Is the customer eligible for a bereavement fare?"

Zero injection. Zero unsafe intent. Zero PII. Every other detector waves it
through, and the Air Canada tribunal happened anyway. Nobody attacked the
chatbot. Someone asked an ordinary question and got a confident wrong answer.

So this detector's output is not "stop". It is "be careful":
    cite every claim, no silent edits, stronger model, force the deep check.

Its score is capped on purpose. It can push a query into the cautious band,
never into block. A salary question is high stakes AND completely legitimate,
and blocking it would be a false alarm.

How it decides
--------------
Same two-set trick as Detector 2, different purpose. Compare the query with
the high-stakes examples and with the low-stakes examples, and see which side
is nearer. Without the low-stakes set, every question matches something.

The encoder is shared with Detector 2, so the model is loaded once for both.
"""

import time

import numpy as np

import shared_encoder as enc

SPACE = "stakes"   # own TF-IDF vocabulary, isolated from every other module
from topics_stakes import STAKES_FLAT, LOW_STAKES, STRICT_MODE

# Cosine ranges differ per encoder, so the cut-offs do too.
THRESHOLDS = {
    "minilm": {"floor": 0.30, "strong": 0.50, "margin": 0.06},
    "tfidf":  {"floor": 0.14, "strong": 0.26, "margin": 0.03},
}

# Words that make a question high stakes on their own, whatever the topic is.
# These lift a borderline score - they never create one from nothing.
ESCALATORS = [
    "am i entitled", "am i eligible", "legally", "guarantee", "guaranteed",
    "promise", "commit", "assured", "refund", "compensation", "liable",
    "liability", "contract says", "as per policy", "confirm that",
    "can i tell the client", "can i tell the customer", "is it final",
]

# This detector can never block on its own.
MAX_SCORE = 0.65

_stakes_vecs = None
_low_vecs = None
_meta = None

# hand our examples to the shared encoder so the TF-IDF fallback can be fitted
enc.register_corpus([t for _, t in STAKES_FLAT] + LOW_STAKES, space=SPACE)


def load(verbose: bool = False):
    """Encode both example sets once, at start-up."""
    global _stakes_vecs, _low_vecs, _meta
    if _stakes_vecs is not None:
        return
    t0 = time.perf_counter()
    _meta = [c for c, _ in STAKES_FLAT]
    _stakes_vecs = enc.encode([t for _, t in STAKES_FLAT], space=SPACE)
    _low_vecs = enc.encode(LOW_STAKES, space=SPACE)
    if verbose:
        print(f"encoder={enc.name()}, {len(STAKES_FLAT)} high + {len(LOW_STAKES)} low "
              f"examples encoded in {(time.perf_counter()-t0)*1000:.0f} ms")


def detect_stakes(text: str) -> dict:
    load()
    t0 = time.perf_counter()
    cut = THRESHOLDS[enc.name()]

    q = enc.encode([text], space=SPACE)[0]
    hi = _stakes_vecs @ q
    lo = _low_vecs @ q

    h_best, l_best = float(hi.max()), float(lo.max())
    margin = h_best - l_best
    idx = int(hi.argmax())
    category = _meta[idx]
    nearest = STAKES_FLAT[idx][1]
    matched = []

    if h_best < cut["floor"]:
        score, category = 0.0, None
        matched.append("topic:nothing_matched")
    elif l_best >= h_best:
        score, category = 0.0, None
        matched.append(f"topic:low_stakes_is_nearer(low={l_best:.2f} vs high={h_best:.2f})")
    elif margin >= cut["margin"] and h_best >= cut["strong"]:
        score = 0.65
        matched.append(f"topic:{category}(sim={h_best:.2f}, margin={margin:.2f})")
    elif margin >= cut["margin"]:
        score = 0.45
        matched.append(f"topic:{category}(sim={h_best:.2f}, margin={margin:.2f})")
    else:
        score = 0.25
        matched.append(f"topic:weak({category}, margin={margin:.2f})")

    # phrases that carry stakes on their own - they lift, never start
    low = text.lower()
    hits = [e for e in ESCALATORS if e in low]
    if hits and score >= 0.20:
        score = min(MAX_SCORE, score + 0.20)
        matched.append(f"escalator:{hits[0]}")

    score = min(score, MAX_SCORE)
    strict = score >= 0.40

    return {
        "detector": "high_stakes",
        "score": round(score, 2),
        "category": category,
        "strict_mode": strict,
        "applies": STRICT_MODE if strict else [],
        "encoder": enc.name(),
        "signals": {
            "high_best": round(h_best, 3),
            "low_best": round(l_best, 3),
            "margin": round(margin, 3),
            "nearest_example": nearest if score >= 0.40 else None,
        },
        "matched": matched,
        "time_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


def band_of(score: float) -> str:
    """Note there is no block band here - MAX_SCORE stops short of 0.80."""
    if score >= 0.40:
        return "cautious"
    return "allow"


if __name__ == "__main__":
    load(verbose=True)
    demo = [
        "Is the customer eligible for a bereavement fare?",
        "How much reimbursement can I claim for travel?",
        "Am I entitled to a relieving letter?",
        "What does the non compete clause mean?",
        "Can I promise the client delivery by next week?",
        "What are the office timings?",
        "How do I book a meeting room?",
        "Translate this paragraph into Hindi",
    ]
    print(f"\n{'score':>6} {'strict':>8} {'category':>22} {'ms':>6}  query")
    print("-" * 104)
    for q in demo:
        r = detect_stakes(q)
        print(f"{r['score']:>6} {str(r['strict_mode']):>8} {str(r['category']):>22} "
              f"{r['time_ms']:>6}  {q}")
        print(f"{'':>46}-> {', '.join(r['matched'])}")
