"""
TIER 2 - CHECK A : GROUNDEDNESS

The question this answers
    Every sentence in the answer claims something. Is that claim actually IN
    the evidence it cited, or did the model invent it?

This is the check that catches "confidently wrong" - the Air Canada failure.

The rule that makes it safe
---------------------------
The judge sees ONLY the claim and the evidence that claim cited. It is not
allowed to use its own knowledge. It answers one narrow question:

    "Is this claim stated in this text? yes / partly / no"

Checking is a much easier job than answering, so a judge doing only this is
far less likely to invent something than the model that wrote the answer.

Two judges, same interface
--------------------------
    llm_judge      a real model call. Accurate, ~200 ms per claim.
    overlap_judge  no model at all. Content-word overlap plus a hard numeric
                   check. Runs in microseconds, catches the obvious cases.

The overlap judge is not a toy. Most hallucinations invent a NUMBER - a limit,
a date, a percentage. A number in the claim that appears nowhere in the cited
evidence is a hallucination with no judgement required. That single rule
catches the Air Canada case, where the bot invented a refund window.

So the default is: overlap judge on every claim, LLM judge only on the claims
the overlap judge is unsure about. Tiering inside a tier.
"""

import re
import time

# words that carry no meaning, so overlap on them proves nothing
STOP = set("""a an the is are was were be been being of to in on at for with by from as
that this these those it its and or but if then than so such not no nor can could will
would shall should may might must do does did have has had you your we our they their
i me my he she his her them us what which who whom when where why how all any each""".split())

NUM = re.compile(r"\b\d+(?:[.,]\d+)?\b")
TAG = re.compile(r"\[([A-Za-z]?\d+)\]")


# ---------------------------------------------------------------- splitting

def split_claims(answer: str):
    """
    One sentence = one claim. Returns (text, [tags]) for each sentence.

    A sentence with NO tag is not "unverified" - it is a claim with no source
    at all, which is the strongest hallucination signal we have, and it costs
    nothing to detect.
    """
    parts = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        tags = TAG.findall(p)
        clean = TAG.sub("", p).strip(" .")

        # A fragment that is ONLY a tag belongs to the sentence before it.
        # Models write "...probation. [S3]" as often as "...probation [S3]."
        # and treating the first form as an untagged claim is a false alarm.
        if tags and not clean:
            if out:
                out[-1]["tags"].extend(tags)
            continue

        if len(clean) < 12:                # greetings, headings, list bullets
            continue
        out.append({"text": clean, "tags": tags})
    return out


def _stem(w: str) -> str:
    """
    Crude suffix stripping so "apply", "applied" and "applying" match.
    Without it, a correct paraphrase of the evidence scores zero overlap and
    a perfectly grounded claim gets flagged - which is a false alarm we can
    remove for four lines of code.
    """
    for suf in ("ingly", "edly", "ing", "ied", "ies", "ed", "es", "ly", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def content_words(text: str):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(w) for w in words if w not in STOP and len(w) > 2}


# ---------------------------------------------------------------- judges

def overlap_judge(claim: str, evidence: str) -> dict:
    """
    No model. Two signals:

      numbers  every number in the claim must appear in the evidence.
               An invented limit or date fails here immediately.
      overlap  share of the claim's content words that appear in the evidence.
    """
    ev_nums = set(NUM.findall(evidence))
    cl_nums = set(NUM.findall(claim))
    missing_nums = cl_nums - ev_nums

    cw = content_words(claim)
    ev = content_words(evidence)
    overlap = len(cw & ev) / len(cw) if cw else 0.0

    if missing_nums:
        return {"verdict": "not_supported", "confidence": 0.9,
                "why": f"number(s) {sorted(missing_nums)} appear nowhere in the cited evidence",
                "overlap": round(overlap, 2), "judge": "overlap"}
    if overlap >= 0.60:
        return {"verdict": "supported", "confidence": 0.7,
                "why": f"{int(overlap*100)}% of the claim's words are in the evidence",
                "overlap": round(overlap, 2), "judge": "overlap"}

    # Low overlap is suspicious, but a lexical judge must not be CONFIDENT
    # about it. A correct paraphrase can share almost no words. So this
    # becomes "unsure" and goes to the real judge, instead of being called
    # a hallucination by a word counter.
    return {"verdict": "unsure",
            "confidence": 0.4 if overlap < 0.25 else 0.3,
            "why": f"only {int(overlap*100)}% word overlap - too weak for a "
                   f"word counter to judge, needs the real judge",
            "overlap": round(overlap, 2), "judge": "overlap"}


JUDGE_PROMPT = """You are checking one claim against one piece of evidence.

RULES
- Use ONLY the evidence below. Your own knowledge is not allowed.
- If the evidence does not state it, the answer is no - even if you believe it.
- Answer with one word: supported, partly, or no.

EVIDENCE:
{evidence}

CLAIM:
{claim}

ANSWER:"""


def make_llm_judge(call_model):
    """
    call_model(prompt) -> str. Wire your own model in:

        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        judge = make_llm_judge(lambda p: llm.invoke(p).content)
    """
    def judge(claim: str, evidence: str) -> dict:
        raw = call_model(JUDGE_PROMPT.format(evidence=evidence[:3000], claim=claim)).lower()
        if "supported" in raw:
            v, c = "supported", 0.9
        elif "partly" in raw:
            v, c = "unsure", 0.5
        else:
            v, c = "not_supported", 0.9
        return {"verdict": v, "confidence": c, "why": raw.strip()[:120], "judge": "llm"}
    return judge


# ---------------------------------------------------------------- the check

def check_groundedness(answer: str, evidence: dict, llm_judge=None,
                       escalate_unsure: bool = True) -> dict:
    """
    answer    the generated text, with [S1] style tags on factual sentences
    evidence  {"S1": "text of source 1", "S2": ...}
    llm_judge optional. If given, only the claims the overlap judge is unsure
              about are sent to it - that is where the cost is worth paying.
    """
    t0 = time.perf_counter()
    claims = split_claims(answer)
    results, llm_calls = [], 0

    for c in claims:
        if not c["tags"]:
            results.append({**c, "verdict": "no_source", "confidence": 0.95,
                            "why": "this sentence cites nothing at all",
                            "judge": "rule"})
            continue

        cited = "\n".join(evidence.get(t, "") for t in c["tags"]).strip()
        if not cited:
            results.append({**c, "verdict": "bad_tag", "confidence": 0.95,
                            "why": f"tag {c['tags']} does not exist in the evidence",
                            "judge": "rule"})
            continue

        r = overlap_judge(c["text"], cited)
        if r["verdict"] == "unsure" and llm_judge and escalate_unsure:
            r = llm_judge(c["text"], cited)
            llm_calls += 1
        results.append({**c, **r})

    bad = [r for r in results if r["verdict"] in ("not_supported", "no_source", "bad_tag")]
    unsure = [r for r in results if r["verdict"] == "unsure"]

    if not claims:
        score = 0.0
    else:
        score = min(1.0, (len(bad) + 0.4 * len(unsure)) / len(claims))

    return {
        "check": "groundedness",
        "score": round(score, 2),
        "claims": len(claims),
        "supported": len(claims) - len(bad) - len(unsure),
        "unsure": len(unsure),
        "unsupported": len(bad),
        "llm_calls": llm_calls,
        "detail": results,
        "worst": sorted(bad, key=lambda r: -r["confidence"])[:3],
        "time_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


if __name__ == "__main__":
    evidence = {
        "S1": "Employees may apply for bereavement leave of up to 3 days. "
              "The application must be submitted before the leave is taken.",
        "S2": "Travel reimbursement is capped at 15000 rupees per trip.",
    }
    samples = [
        ("grounded",
         "You may apply for bereavement leave of up to 3 days [S1]. "
         "The application must be submitted before the leave [S1]."),
        ("invented number - the Air Canada shape",
         "You can apply for the bereavement fare within 90 days of travel [S1]."),
        ("no source at all",
         "The company always approves bereavement leave retroactively."),
        ("tag that does not exist",
         "Reimbursement is capped at 15000 rupees [S7]."),
        ("mixed",
         "Travel reimbursement is capped at 15000 rupees per trip [S2]. "
         "Managers can raise this to 25000 on request [S2]."),
    ]
    for name, ans in samples:
        r = check_groundedness(ans, evidence)
        print(f"\n{name}\n  score={r['score']}  claims={r['claims']} "
              f"supported={r['supported']} unsure={r['unsure']} bad={r['unsupported']}"
              f"  {r['time_ms']}ms")
        for d in r["detail"]:
            print(f"    [{d['verdict']:<14}] {d['text'][:56]}")
            print(f"    {'':<17} {d['why']}")
