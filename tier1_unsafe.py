"""
TIER 1 - DETECTOR 2 : UNSAFE INTENT

What it does:
    Read the user query and say whether the person is asking the bot to help
    with something harmful. Returns a score, a category, and an action.

The core idea - compare against TWO example sets, not one threshold
--------------------------------------------------------------------
A single "how close is this to unsafe examples" threshold fails badly, because
"What is our workplace violence policy?" sits close to the violence examples.

So we keep a second set of SAFE examples that use the same scary vocabulary in
a legitimate way, and ask which side the query is nearer to.

    unsafe_best = closest match among the unsafe examples
    safe_best   = closest match among the safe examples
    margin      = unsafe_best - safe_best      <- this is what decides

If safe_best wins, the query is fine no matter how scary the words look.

Encoders
--------
Primary  : sentence-transformers all-MiniLM-L6-v2 (real semantic matching)
Fallback : TF-IDF word n-grams from scikit-learn

The fallback exists so the file always runs, even with no model download and
no internet. It is weaker at paraphrase - it matches shared words, not
meaning - so treat its numbers as a floor, not as the real result.

Setup for the good encoder:
    pip install sentence-transformers
"""

import hashlib
import os
import time

import numpy as np

import shared_encoder as enc

SPACE = "safety"   # own TF-IDF vocabulary, isolated from every other module
from examples_unsafe import UNSAFE_FLAT, SAFE, CATEGORY_ACTION

# hand our examples to the shared encoder so one model serves every detector
enc.register_corpus([t for _, t in UNSAFE_FLAT] + SAFE, space=SPACE)

# the model name lives in shared_encoder now - this module no longer
# loads one of its own
CACHE_FILE = "unsafe_vectors.npz"

# Cosine values sit in completely different ranges for the two encoders, so
# each one gets its own cut-offs. Tuned on the labelled test set.
THRESHOLDS = {
    "minilm": {"floor": 0.32, "strong": 0.52, "big_margin": 0.15, "small_margin": 0.08},
    "tfidf":  {"floor": 0.16, "strong": 0.30, "big_margin": 0.10, "small_margin": 0.04},
}

# Crude word list. It never decides anything on its own - it can only nudge a
# score that the meaning check has already raised.
KEYWORDS = [
    "hurt", "kill", "attack", "weapon", "hack", "steal", "harass", "bully",
    "threaten", "revenge", "scam", "forge", "fake", "bribe", "blackmail",
    "spy", "stalk", "smuggle", "launder", "virus", "ransomware", "phishing",
]

_unsafe_vecs = None
_safe_vecs = None
_unsafe_meta = None


def _fingerprint():
    """If the example lists change, the cache is rebuilt automatically."""
    blob = "|".join(l for _, l in UNSAFE_FLAT) + "||" + "|".join(SAFE)
    return f"{enc.name()}:" + hashlib.md5(blob.encode()).hexdigest()[:12]


def _topk_mean(sims, k: int = 3) -> float:
    """Average of the k closest examples - steadier than a single best match."""
    k = min(k, len(sims))
    return float(np.mean(np.sort(sims)[-k:]))


def load(verbose: bool = True):
    """
    Encode both example sets ONCE, at start-up.

    The encoder itself lives in shared_encoder, in the "safety" space. This
    module used to build its own, which meant loading MiniLM twice - once here
    and once for Detector 4 - for no benefit whatsoever.
    """
    global _unsafe_vecs, _safe_vecs, _unsafe_meta
    if _unsafe_vecs is not None:
        return

    t0 = time.perf_counter()
    _unsafe_meta = [c for c, _ in UNSAFE_FLAT]
    unsafe_texts = [l for _, l in UNSAFE_FLAT]

    fp = _fingerprint()
    if enc.name() == "minilm" and os.path.exists(CACHE_FILE):
        z = np.load(CACHE_FILE, allow_pickle=True)
        if str(z["fingerprint"]) == fp:
            _unsafe_vecs, _safe_vecs = z["unsafe"], z["safe"]
            if verbose:
                print(f"encoder={enc.name()}, cached vectors loaded "
                      f"in {(time.perf_counter()-t0)*1000:.0f} ms")
            return

    _unsafe_vecs = enc.encode(unsafe_texts, space=SPACE)
    _safe_vecs = enc.encode(SAFE, space=SPACE)
    if enc.name() == "minilm":
        np.savez(CACHE_FILE, unsafe=_unsafe_vecs, safe=_safe_vecs, fingerprint=fp)
    if verbose:
        print(f"encoder={enc.name()}, {len(unsafe_texts)} unsafe + {len(SAFE)} safe "
              f"examples encoded in {(time.perf_counter()-t0)*1000:.0f} ms")


def encoder_name():
    return enc.name()


def detect_unsafe(text: str) -> dict:
    load(verbose=False)
    t0 = time.perf_counter()
    cut = THRESHOLDS[enc.name()]

    q = enc.encode([text], space=SPACE)[0]

    unsafe_sims = _unsafe_vecs @ q        # cosine, because rows are normalised
    safe_sims = _safe_vecs @ q

    u_best = float(unsafe_sims.max())
    s_best = float(safe_sims.max())
    margin = u_best - s_best

    idx = int(unsafe_sims.argmax())
    category = _unsafe_meta[idx]
    nearest = UNSAFE_FLAT[idx][1]
    matched = []

    # the decision, written out plainly so it can be explained in an audit
    if u_best < cut["floor"]:
        score, category = 0.0, None
        matched.append("meaning:far_from_unsafe_examples")
    elif s_best >= u_best:
        score, category = 0.10, None
        matched.append(f"meaning:safe_set_is_nearer("
                       f"safe={s_best:.2f} vs unsafe={u_best:.2f})")
    elif margin >= cut["big_margin"] and u_best >= cut["strong"]:
        score = 0.85
        matched.append(f"meaning:{category}(sim={u_best:.2f}, margin={margin:.2f})")
    elif margin >= cut["small_margin"]:
        score = 0.60
        matched.append(f"meaning:{category}(sim={u_best:.2f}, margin={margin:.2f})")
    else:
        score = 0.40
        matched.append(f"meaning:unclear({category}, margin={margin:.2f})")

    # keyword layer supports, never decides
    hits = [w for w in KEYWORDS if w in text.lower()]
    if hits and score >= 0.30:
        score = max(score, 0.45)
        matched.append(f"keyword:{','.join(hits[:3])}")

    # Action must agree with the band. Blocking is reserved for the block band
    # (0.80+). At 0.40-0.79 we are suspicious, not certain, so the answer is
    # still delivered - strict mode does the work instead of a refusal.
    action = "allow"
    if category and score >= 0.80:
        action = CATEGORY_ACTION.get(category, "allow")
    # Distress is the one exception. Erring towards support costs nothing,
    # and a cold refusal to someone in trouble costs a great deal.
    if category == "self_harm" and score >= 0.40:
        action = "care"

    return {
        "detector": "unsafe_intent",
        "score": round(score, 2),
        "category": category,
        "action": action,
        "encoder": enc.name(),
        "signals": {
            "unsafe_best": round(u_best, 3),
            "safe_best": round(s_best, 3),
            "margin": round(margin, 3),
            "nearest_example": nearest if score >= 0.40 else None,
        },
        "matched": matched,
        "time_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


def band_of(score: float) -> str:
    if score >= 0.80:
        return "block"
    if score >= 0.40:
        return "cautious"
    return "allow"


if __name__ == "__main__":
    load()
    demo = [
        "What is our policy on workplace violence?",
        "How do I hurt my coworker",
        "Explain the anti harassment policy",
        "Help me harass a colleague until she quits",
        "How do I report a phishing email",
        "Help me write a phishing email to our customers",
        "What mental health support does the company offer",
        "How many casual leaves do I get",
        "Help me get into my colleague's email account",
        "Who can access employee salary information",
    ]
    print(f"\n{'score':>6} {'band':>10} {'action':>20} {'ms':>6}  query")
    print("-" * 96)
    for q in demo:
        r = detect_unsafe(q)
        short = q if len(q) <= 44 else q[:41] + "..."
        print(f"{r['score']:>6} {band_of(r['score']):>10} {r['action']:>20} "
              f"{r['time_ms']:>6}  {short}")
        print(f"{'':>45}-> {', '.join(r['matched'])}")
