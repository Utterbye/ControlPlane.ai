"""
THE EVIDENCE PIPELINE

Three steps between "we have documents" and "the model may write an answer".

    retrieve_wide    fetch far more than you need
    rerank           cut it down to the few that actually answer the question
    check_conflict   ask whether those few agree with each other

Why fetch wide and then cut
---------------------------
Vector search is fuzzy. The piece that actually answers the question is often
at rank 11, not rank 3. Fetching only the top 5 means the right piece was
never a candidate, and no amount of clever prompting recovers it.

So fetch 25, then cut hard. The cutting is where the quality comes from.

Why re-ranking improves accuracy AND cost in the same step
----------------------------------------------------------
Plain search asks "does this text LOOK similar?". A re-ranker asks "does this
text ANSWER the question?". Similar and useful are not the same thing.

And five pieces instead of thirty-five is a much smaller prompt. Better
evidence and a cheaper call, from one operation.

Why the conflict check exists at all
------------------------------------
Documents contradict each other. An old policy PDF and a new one. A web page
from 2019 and the current rule. Quietly picking whichever ranked higher is
exactly how a chatbot becomes confidently wrong - and the user has no way to
know a second answer existed.

So: detect the disagreement, resolve it with stated rules, and when it cannot
be resolved, SAY SO rather than choose.
"""

import re
import time

import numpy as np

import shared_encoder as enc

SPACE = "documents"   # own TF-IDF vocabulary, isolated from every other module

# ---------------------------------------------------------------- config

FETCH_K = 25              # how many candidates to pull before cutting
KEEP_K = 5                # how many survive
# Below this, treat it as no evidence at all. Encoder-specific, because the
# two encoders produce cosines in completely different ranges - TF-IDF only
# counts shared words, so it scores everything lower. Measured on this corpus:
#   travel question   -> best 0.352   (a good match)
#   probation question-> best 0.194   (also a good match, just fewer shared words)
#   space travel      -> best 0.242   (a bad match)
# A single number cannot separate those under TF-IDF, which is precisely the
# weakness the real encoder removes.
FLOOR_BY_ENCODER = {"minilm": 0.30, "tfidf": 0.12}
DUP_THRESHOLD = 0.92      # near-identical chunks
# Two pieces must be this close to count as being ABOUT the same thing.
# Measured on this corpus, so the number is chosen rather than guessed:
#   p1 vs p2  same topic, different figure   0.514
#   p2 vs p3  same topic, different wording  0.405
#   p1 vs p5  notice period vs travel        0.013
#   p1 vs p7  notice period vs canteen       0.033
# The gap between "same topic" and "different topic" is enormous, so the line
# sits comfortably between them. TF-IDF needs a lower bar than MiniLM, because
# it only sees shared words - it scores a paraphrase far lower.
TOPIC_SIM_BY_ENCODER = {"minilm": 0.45, "tfidf": 0.20}


def _topic_sim():
    return TOPIC_SIM_BY_ENCODER.get(enc.name(), 0.45)


def _floor():
    return FLOOR_BY_ENCODER.get(enc.name(), 0.30)

# Where a piece came from changes how much it is worth. An internal policy
# beats a news article, and both beat a random blog.
TRUST_BOOST = {
    "internal_policy": 0.15,
    "internal_doc": 0.10,
    "official": 0.10,
    "news": 0.00,
    "blog": -0.10,
    "unknown": -0.05,
}
RECENCY_BOOST = 0.05      # published within the last year

NUM = re.compile(r"\b\d+(?:[.,]\d+)?\b")


# ---------------------------------------------------------------- retrieve

# Question words and filler. A word being absent from the corpus only means
# something if the word carries TOPIC. "long" and "much" are how people phrase
# a question, not what the question is about.
GENERIC = set("""what how when where why which who whom whose does doing done can
could would should shall will might must have has had been being about with
from into onto over under after before between during long much many more most
less least than then there here this that these those some any each every
other another same such just only very well make made makes making take taken
give given gets getting need needs needed want wants please tell show help
know knows allowed apply applying claim claims submit submitted receive
policy policies rule rules process procedure company employee employees office
work working days day year years month months time times going still even also
their there they them theirs your yours ours mine""".split())

MIN_TOPIC_LEN = 5     # "much" and "long" are 4; "space" and "probation" are not


def out_of_scope(question, corpus):
    """
    Does the question ask about something the corpus has never heard of?

    This exists because of a gap the other checks cannot close. Groundedness
    asks "is this claim in the evidence" - and an answer about travel
    reimbursement IS in the evidence, even when the question was about space
    travel. It is perfectly grounded and completely irrelevant.

    Similarity cannot separate those two either. Measured on this corpus:

        "how long is probation"        best 0.194   a real match
        "what is the policy on space
         travel"                       best 0.263   a false match

    The false match scores HIGHER than the true one, so no threshold - absolute
    or relative - sits between them. That is not a tuning problem, it is the
    fallback encoder being unable to tell them apart.

    But there is a lexical signal that works: "space" appears in no document
    at all. A distinctive word in the question that exists nowhere in the
    corpus means the question is about something we simply do not cover.
    """
    # Titles count. "Group Insurance 2026" covers insurance even though the
    # body says "health cover" - and a user will say insurance.
    vocab = set()
    for d in corpus:
        blob = d["text"] + " " + d.get("title", "")
        vocab |= {w for w in re.findall(r"[a-z]{4,}", blob.lower())}

    terms = [w for w in re.findall(r"[a-z]{%d,}" % MIN_TOPIC_LEN, question.lower())
             if w not in GENERIC]
    missing = [w for w in terms if w not in vocab and not any(
        w.startswith(v[:5]) or v.startswith(w[:5]) for v in vocab)]

    return {"out_of_scope": bool(missing),
            "unknown_terms": missing,
            "checked_terms": terms,
            "why": (f"'{missing[0]}' appears in no document" if missing
                    else "every distinctive term appears somewhere in the corpus")}


# A purely lexical scope check cannot know that "laptop" and "personal device"
# are the same idea. So an unknown term is a CAVEAT, not a refusal - unless the
# retrieval is also weak, in which case there really is nothing to say.
SCOPE_ABSTAIN_RELEVANCE = 0.30


def retrieve_wide(question, corpus, k=FETCH_K):
    """
    corpus: [{"id","text","source_type","year"}, ...]

    Deliberately generous. The point of this step is recall, not precision -
    precision is the next step's job.
    """
    t0 = time.perf_counter()
    if not corpus:
        return {"candidates": [], "time_ms": 0.0}

    q = enc.encode([question], space=SPACE)[0]
    mat = enc.encode([c["text"] for c in corpus], space=SPACE)
    sims = mat @ q

    order = np.argsort(sims)[::-1][:k]
    cands = []
    for i in order:
        c = dict(corpus[int(i)])
        c["retrieval_score"] = round(float(sims[int(i)]), 3)
        cands.append(c)
    return {"candidates": cands,
            "time_ms": round((time.perf_counter() - t0) * 1000, 2)}


# ---------------------------------------------------------------- rerank

def _dedupe(cands):
    """
    The same paragraph copied across four files should count once. Otherwise
    a duplicated document quietly outvotes a unique one.
    """
    if len(cands) < 2:
        return cands, 0
    vecs = enc.encode([c["text"] for c in cands], space=SPACE)
    keep, dropped = [], 0
    for i, c in enumerate(cands):
        if any(float(vecs[i] @ vecs[j]) > DUP_THRESHOLD for j in keep):
            dropped += 1
            continue
        keep.append(i)
    return [cands[i] for i in keep], dropped


def rerank(question, candidates, keep=KEEP_K, this_year=2026):
    """
    Three passes, cheap to expensive.

      a) drop near-duplicates
      b) score relevance properly
      c) apply source trust and recency

    Pass (b) should be a cross-encoder, which reads the question and the
    passage TOGETHER instead of comparing two separate vectors. That joint
    read is what makes it better than search. Without one installed we fall
    back to the embedding score, and the boosts still do real work.
    """
    t0 = time.perf_counter()
    if not candidates:
        return {"kept": [], "dropped_duplicates": 0, "reason": "nothing retrieved",
                "no_evidence": True, "time_ms": 0.0}

    cands, dupes = _dedupe(candidates)

    q = enc.encode([question], space=SPACE)[0]
    mat = enc.encode([c["text"] for c in cands], space=SPACE)
    base = mat @ q

    scored = []
    for i, c in enumerate(cands):
        s = float(base[i])
        s += TRUST_BOOST.get(c.get("source_type", "unknown"), -0.05)
        if c.get("year") and this_year - int(c["year"]) <= 1:
            s += RECENCY_BOOST
        d = dict(c)
        d["relevance"] = round(float(base[i]), 3)   # before any boost
        d["rerank_score"] = round(s, 3)
        scored.append(d)

    scored.sort(key=lambda d: -d["rerank_score"])
    top = scored[:keep]

    # The stop rule. If even the best surviving piece is weak, this is not a
    # thin answer waiting to be written - it is no answer.
    # Tested on RELEVANCE, not on the boosted score. A trust boost is meant to
    # order two relevant pieces, not to promote an irrelevant one.
    no_evidence = (not top) or max(d["relevance"] for d in top) < _floor()

    return {
        "kept": [] if no_evidence else top,
        "considered": len(candidates),
        "after_dedupe": len(cands),
        "dropped_duplicates": dupes,
        "best_score": top[0]["rerank_score"] if top else 0.0,
        "best_relevance": max((d["relevance"] for d in top), default=0.0),
        "no_evidence": no_evidence,
        "reason": (f"best relevance {max(d['relevance'] for d in top):.3f} is below "
                   f"the floor {_floor()}" if no_evidence and top
                   else "kept the strongest pieces"),
        "time_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


# ---------------------------------------------------------------- conflict

TRUST_RANK = {"internal_policy": 4, "internal_doc": 3,
              "official": 2, "news": 1, "blog": 0, "unknown": 0}


def check_conflict(pieces):
    """
    Do the surviving pieces disagree with each other?

    Numbers are the practical handle. If two pieces about the same topic state
    different numbers, that is a contradiction a user would notice - a limit,
    a duration, a deadline.

    Resolution order, applied in this order and no other:
      1 trust rank      internal policy beats a blog, always
      2 recency         for facts that change, newer wins
      3 independence    five copies of one article are one source, not five
      4 still tied      show both, and say they disagree
    """
    t0 = time.perf_counter()
    if len(pieces) < 2:
        return {"conflict": False, "reason": "only one piece of evidence",
                "winner": pieces[0] if pieces else None, "shown": pieces,
                "time_ms": 0.0}

    # Two pieces only contradict each other if they are ABOUT the same thing.
    # "notice period is 60 days" and "reimbursement is 15000 rupees" have
    # different numbers and no disagreement whatsoever.
    vecs = enc.encode([p["text"] for p in pieces], space=SPACE)
    groups = {}
    for i, p in enumerate(pieces):
        nums = frozenset(NUM.findall(p["text"]))
        if not nums:
            continue
        placed = False
        for key, (knums, members, kidx) in list(groups.items()):
            if float(vecs[i] @ vecs[kidx]) >= _topic_sim():
                members.append(p)
                knums.add(nums)
                placed = True
                break
        if not placed:
            groups[i] = ({nums}, [p], i)

    # a topic cluster is only a conflict if it holds two different number sets
    conflicted = [(numsets, members) for numsets, members, _ in groups.values()
                  if len(numsets) > 1]

    if not conflicted:
        return {"conflict": False, "reason": "no two pieces about the same topic disagree",
                "winner": pieces[0], "shown": pieces,
                "time_ms": round((time.perf_counter() - t0) * 1000, 2)}

    # More than one cluster can disagree internally. Only the one containing
    # the top-ranked piece matters, because that is the cluster the question
    # is actually about. Reporting a contradiction inside some unrelated
    # cluster would be technically true and completely useless.
    top_piece = pieces[0]
    conflicted.sort(key=lambda cm: 0 if top_piece in cm[1] else 1)
    numsets, members = conflicted[0]
    groups = {}
    for p in members:
        groups.setdefault(frozenset(NUM.findall(p["text"])), []).append(p)

    # 3 - collapse copies of the same source before counting support
    def independence(group):
        return len({p.get("source_id", p.get("id")) for p in group})

    ranked = []
    for nums, group in groups.items():
        best = max(group, key=lambda p: (TRUST_RANK.get(p.get("source_type", "unknown"), 0),
                                         int(p.get("year") or 0)))
        ranked.append({
            "numbers": sorted(nums),
            "members": group,
            "trust": TRUST_RANK.get(best.get("source_type", "unknown"), 0),
            "year": int(best.get("year") or 0),
            "independent_sources": independence(group),
            "piece": best,
        })

    ranked.sort(key=lambda g: (-g["trust"], -g["year"], -g["independent_sources"]))
    a, b = ranked[0], ranked[1]

    if a["trust"] > b["trust"]:
        why, resolved = f"trust rank {a['trust']} beats {b['trust']}", True
    elif a["year"] > b["year"]:
        why, resolved = f"{a['year']} is newer than {b['year']}", True
    elif a["independent_sources"] > b["independent_sources"]:
        why, resolved = "more independent sources", True
    else:
        why, resolved = "nothing separates them", False

    # Every piece in the losing group is a loser, not just the one that
    # represented it. Dropping only the representative leaves a blog repeating
    # the same wrong figure sitting in the evidence, which is the exact
    # outcome the conflict check exists to prevent.
    losing = [p for g in ranked[1:] for p in g["members"]] if resolved else []

    return {
        "conflict": True,
        "resolved": resolved,
        "reason": why,
        "winner": a["piece"] if resolved else None,
        "losing_pieces": losing,
        "shown": [a["piece"], b["piece"]],
        "groups": [{"numbers": g["numbers"], "trust": g["trust"],
                    "year": g["year"], "sources": g["independent_sources"]}
                   for g in ranked],
        "label": ("sources disagree" if not resolved else None),
        "time_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


def build_evidence(question, corpus):
    """The steps as one call, returning what the generator needs."""
    scope = out_of_scope(question, corpus)
    r1 = retrieve_wide(question, corpus)

    # Unknown topic word AND nothing retrieved strongly = genuinely out of scope.
    # Unknown word but a solid retrieval = probably a synonym, so answer it and
    # say plainly which term we found nothing about.
    best = max((c["retrieval_score"] for c in r1["candidates"]), default=0.0)
    if scope["out_of_scope"] and best < SCOPE_ABSTAIN_RELEVANCE:
        return {"evidence": {}, "no_evidence": True, "scope": scope,
                "retrieve": r1,
                "rerank": {"reason": scope["why"], "no_evidence": True,
                           "considered": len(r1["candidates"]), "after_dedupe": 0,
                           "dropped_duplicates": 0, "kept": []},
                "conflict": None}
    r2 = rerank(question, r1["candidates"])
    if r2["no_evidence"]:
        return {"evidence": {}, "no_evidence": True, "retrieve": r1,
                "rerank": r2, "conflict": None, "scope": scope}
    r3 = check_conflict(r2["kept"])

    # Start from what the re-ranker chose. The conflict check ADJUSTS that
    # list - it does not replace it. Replacing it threw away every other
    # relevant piece and answered the wrong question entirely.
    kept = list(r2["kept"])
    if r3["conflict"] and r3.get("resolved"):
        losers = {id(p) for p in r3.get("losing_pieces", [])}
        kept = [p for p in kept if id(p) not in losers]
    elif r3["conflict"]:
        # unresolved: both stay in, and the answer will carry the label
        for p in r3["shown"]:
            if p not in kept:
                kept.append(p)

    evidence = {f"S{i+1}": p["text"] for i, p in enumerate(kept)}
    return {"evidence": evidence, "no_evidence": False, "scope": scope,
            "caveat": (f"I don't have anything specifically about "
                       f"'{scope['unknown_terms'][0]}'. This is the closest "
                       f"I have." if scope["out_of_scope"] else None),
            "retrieve": r1, "rerank": r2, "conflict": r3,
            "sources_disagreed": bool(r3["conflict"] and not r3["resolved"])}


# ---------------------------------------------------------------- demo

import documents

CORPUS = documents.corpus()

enc.register_corpus([c["text"] for c in CORPUS], space=SPACE)

if __name__ == "__main__":
    for q in ["What is the notice period?",
              "How much travel reimbursement can I claim?",
              "What is the policy on space travel?"]:
        r = build_evidence(q, CORPUS)
        print(f"\nQ: {q}")
        print(f"   retrieved {len(r['retrieve']['candidates'])}  "
              f"-> after dedupe {r['rerank'].get('after_dedupe')} "
              f"(dropped {r['rerank'].get('dropped_duplicates')} duplicate) "
              f"-> kept {len(r['evidence'])}")
        if r["no_evidence"]:
            print(f"   NO EVIDENCE: {r['rerank']['reason']}")
            continue
        c = r["conflict"]
        print(f"   conflict={c['conflict']}  resolved={c.get('resolved')}  "
              f"because: {c['reason']}")
        for k, v in r["evidence"].items():
            print(f"   {k}: {v[:66]}")
