"""
TIER 2 - CHECK B : SELF-CONSISTENCY
TIER 2 - CHECK C : OUTPUT SAFETY AND FAIR WORDING

Both live here because both are small, and both answer a question that
groundedness cannot.

CHECK B - self-consistency
--------------------------
Groundedness asks "is this claim in the evidence?". It cannot tell you whether
the model was SURE. A model that is guessing will guess differently the second
time.

So: generate the answer twice from the same evidence, then compare the facts -
not the wording. Wording always differs. Numbers, dates and entities should
not. If version one says 3 days and version two says 5 days, the model was
inventing, and no amount of citation checking would have caught it.

The expensive part is the second generation, so this check only runs when it
is worth it: strict mode, or groundedness already looking shaky.

CHECK C - safety and fair wording
---------------------------------
Detector 2 in Tier 1 read the QUESTION. This reads the ANSWER, which is a
different problem: a polite, well-formed answer can still carry an unfair
generalisation about a group.

Note what this check does NOT do. Unfair WORDING is visible in one answer.
An unfair PATTERN - softer advice for one kind of name than another - is
invisible in one answer, because you need two answers to compare. That job
belongs to the background checks, not here.
"""

import re
import time

from tier2_groundedness import content_words, NUM

# ============================================================================
# CHECK B - SELF-CONSISTENCY
# ============================================================================

ENTITY = re.compile(r"\b[A-Z][a-z]{2,}\b")
DATEISH = re.compile(r"\b\d{1,2}\s*(?:days?|months?|weeks?|years?|hours?)\b", re.I)


def _facts(text: str) -> dict:
    """Pull out the parts that must NOT change between two generations."""
    return {
        "numbers": set(NUM.findall(text)),
        "durations": {d.lower().strip() for d in DATEISH.findall(text)},
        "entities": set(ENTITY.findall(text)),
    }


def check_consistency(answer_a: str, answer_b: str) -> dict:
    """
    answer_a  the answer we are about to deliver
    answer_b  a second generation from the SAME evidence

    Wording differences are ignored on purpose. Only facts count.
    """
    t0 = time.perf_counter()
    fa, fb = _facts(answer_a), _facts(answer_b)

    conflicts, checked = [], 0
    for key in ("numbers", "durations"):
        only_a, only_b = fa[key] - fb[key], fb[key] - fa[key]
        checked += len(fa[key] | fb[key])
        if only_a or only_b:
            conflicts.append({"kind": key,
                              "in_first_only": sorted(only_a),
                              "in_second_only": sorted(only_b)})

    # meaning drift, measured on content words rather than characters
    ca, cb = content_words(answer_a), content_words(answer_b)
    similarity = len(ca & cb) / len(ca | cb) if (ca | cb) else 1.0

    # Facts first. If there were facts to compare and they all matched, the
    # model was sure - stop there. Two paraphrases of the same answer share
    # only about half their words, so judging THAT as drift punishes the model
    # for writing naturally, which is exactly the false alarm we do not want.
    if conflicts:
        score = 0.85
        why = "the two generations disagree on a fact"
    elif checked >= 2:
        score = 0.0
        why = f"{checked} facts compared, all identical"
    elif similarity < 0.20:
        score = 0.55
        why = f"no facts to compare, and the answers barely overlap ({similarity:.2f})"
    else:
        score = 0.0
        why = f"no facts to compare, wording agrees ({similarity:.2f} overlap)"

    return {
        "check": "self_consistency",
        "score": round(score, 2),
        "similarity": round(similarity, 2),
        "facts_checked": checked,
        "conflicts": conflicts,
        "why": why,
        "time_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


# ============================================================================
# CHECK C - OUTPUT SAFETY AND FAIR WORDING
# ============================================================================

# Unfair generalisation is a SHAPE, not a word list: a group word, then a
# sweeping verb. "Women are usually less..." matches. "Women are eligible
# for..." does not.
GROUP = (r"(?:women|men|girls|boys|males|females|muslims|hindus|christians|sikhs|"
         r"dalits|brahmins|northerners|southerners|freshers|seniors|juniors|"
         r"immigrants|locals|older workers|younger workers|married women)")
SWEEP = (r"(?:are|is|tend to be|usually|always|never|generally|typically|"
         r"can't|cannot|should not|are not able|are less|are more|are better|are worse)")
UNFAIR = re.compile(rf"\b{GROUP}\b\s+(?:{SWEEP})\b", re.I)

HEDGE_MISSING = re.compile(
    r"\b(?:definitely|guaranteed|certainly|100%|without doubt|i promise|"
    r"you will definitely|absolutely will)\b", re.I)

UNSAFE_OUT = re.compile(
    r"\b(?:kill yourself|end your life|hurt (?:him|her|them)|"
    r"take revenge|make them suffer)\b", re.I)


def check_output_safety(answer: str, strict: bool = False) -> dict:
    t0 = time.perf_counter()
    flags, score = [], 0.0

    for m in UNFAIR.finditer(answer):
        score = max(score, 0.80)
        flags.append({"kind": "unfair_generalisation", "text": m.group()})

    for m in UNSAFE_OUT.finditer(answer):
        score = max(score, 0.90)
        flags.append({"kind": "unsafe_advice", "text": m.group()})

    # Overconfidence is not unsafe, but on a high-stakes answer it is exactly
    # how a chatbot creates a commitment the company has to honour.
    for m in HEDGE_MISSING.finditer(answer):
        lift = 0.55 if strict else 0.30
        score = max(score, lift)
        flags.append({"kind": "overclaim", "text": m.group()})

    return {
        "check": "output_safety",
        "score": round(score, 2),
        "flags": flags,
        "why": flags[0]["kind"] if flags else "nothing flagged",
        "time_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


if __name__ == "__main__":
    print("CHECK B - self consistency")
    pairs = [
        ("agree",
         "Bereavement leave is up to 3 days and must be applied for in advance.",
         "You get a maximum of 3 days bereavement leave, applied for beforehand."),
        ("disagree on a number",
         "Bereavement leave is up to 3 days.",
         "Bereavement leave is up to 5 days."),
        ("wandered off",
         "Bereavement leave is up to 3 days.",
         "Please contact the HR helpdesk for assistance with your query."),
    ]
    for name, a, b in pairs:
        r = check_consistency(a, b)
        print(f"  {r['score']:>5}  {name:<24} {r['why']}")
        for c in r["conflicts"]:
            print(f"         {c}")

    print("\nCHECK C - output safety and fair wording")
    outs = [
        ("clean", "You may apply for up to 3 days of bereavement leave.", False),
        ("unfair generalisation", "Women are usually less available for on-call duty.", False),
        ("overclaim, normal mode", "You will definitely get the refund.", False),
        ("overclaim, strict mode", "You will definitely get the refund.", True),
    ]
    for name, txt, strict in outs:
        r = check_output_safety(txt, strict)
        print(f"  {r['score']:>5}  {name:<26} {r['why']}")
