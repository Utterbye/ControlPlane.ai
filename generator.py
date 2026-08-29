"""
THE GENERATOR

Writes the answer, and behaves differently depending on the band Tier 1 put
the question in.

    NORMAL     ordinary prompt, ordinary model, temperature 0.4
    STRICT     every claim must cite a source, no silent edits, temperature 0.1,
               stronger model, and the citations are VERIFIED after generation

Strict mode is not a note in a document. It is four concrete changes:

  1  a different prompt that demands a tag on every factual sentence
  2  temperature 0.1 instead of 0.4
  3  the stronger model instead of the cheap one
  4  a citation check after generation, with one reprompt if tags are missing

Why temperature matters here
----------------------------
Temperature controls how much the model varies its wording. High temperature
is pleasant for writing and dangerous for policy, because "creative" and
"makes things up" are the same behaviour seen from two angles. On a question
about someone's notice period you want boring and repeatable.

One catch worth knowing: the self-consistency check in Tier 2 needs SOME
variation to work. At temperature 0 both generations are identical and the
check tells you nothing. So the second generation deliberately runs warmer.

Pluggable model
---------------
call_model(prompt, temperature) -> str

The default is an extractive writer that needs no model at all: it picks the
sentences from the evidence that answer the question and tags them. That keeps
the whole pipeline runnable offline, and it can never hallucinate, which makes
it a poor test of the groundedness check - so there is a
demo_hallucinating_model() to prove the check actually fires.
"""

import random
import re
import time

MODES = {
    "normal": {"temperature": 0.4, "model": "standard",
               "require_citations": False, "allow_auto_fix": True},
    "strict": {"temperature": 0.1, "model": "stronger",
               "require_citations": True, "allow_auto_fix": False},
}

SECOND_PASS_TEMPERATURE = 0.6      # deliberately warmer, see the note above

NORMAL_PROMPT = """Answer the question using the evidence below.
Cite the source of each factual sentence with its tag, like [S1].
If the evidence does not answer the question, say so.

EVIDENCE
{evidence}

QUESTION
{question}

ANSWER"""

STRICT_PROMPT = """Answer the question using ONLY the evidence below.

RULES
- Every factual sentence MUST end with its source tag, like [S1].
- A sentence with no tag is not allowed. If you cannot cite it, do not say it.
- Do not add anything the evidence does not state, even if you believe it.
- If the evidence is incomplete, say exactly what is missing.
- Be plain and short. No hedging, no filler.

EVIDENCE
{evidence}

QUESTION
{question}

ANSWER"""

REPROMPT = """Your previous answer contained a sentence with no source tag:

  "{offending}"

Rewrite the whole answer. Every factual sentence must end with a tag like [S1].
If a sentence cannot be cited, delete it rather than keeping it untagged."""

TAG = re.compile(r"\[([A-Za-z]?\d+)\]")


# ---------------------------------------------------------------- offline model

def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 12]


def _overlap(a, b):
    wa = set(re.findall(r"[a-z]{4,}", a.lower()))
    wb = set(re.findall(r"[a-z]{4,}", b.lower()))
    return len(wa & wb) / len(wa) if wa else 0.0


def extractive_model(prompt: str, temperature: float = 0.4) -> str:
    """
    The offline default. Pulls the best-matching sentences out of the evidence
    and tags them. It cannot invent anything, which is exactly why it is safe
    to ship in a demo and useless for testing the hallucination checks.
    """
    ev_block = prompt.split("EVIDENCE")[1].split("QUESTION")[0]
    question = prompt.split("QUESTION")[1].split("ANSWER")[0].strip()

    pieces = {}
    for line in ev_block.strip().splitlines():
        m = re.match(r"\[([A-Za-z]?\d+)\]\s*(.+)", line.strip())
        if m:
            pieces[m.group(1)] = m.group(2)
    if not pieces:
        return "I could not find anything on that in the documents."

    scored = []
    for tag, text in pieces.items():
        for sent in _sentences(text) or [text]:
            scored.append((_overlap(question, sent), tag, sent))
    scored.sort(reverse=True)

    keep = 2 if temperature < 0.3 else 3      # colder answers stay shorter
    picked = [s for s in scored[:keep] if s[0] > 0.05]
    if not picked:
        return "I could not find anything on that in the documents."

    # a little wording variation at higher temperature, so the consistency
    # check has something real to compare
    out = []
    for score, tag, sent in picked:
        sent = sent.rstrip(".")
        if temperature > 0.5 and random.random() < 0.5:
            sent = re.sub(r"^Notice period is", "The notice period is", sent)
            sent = re.sub(r"^Travel reimbursement is", "Travel expenses are", sent)
        out.append(f"{sent} [{tag}].")
    return " ".join(out)


def demo_hallucinating_model(prompt: str, temperature: float = 0.4) -> str:
    """
    Deliberately invents a number, so you can watch the groundedness check
    catch it. This is the Air Canada shape.
    """
    base = extractive_model(prompt, temperature)
    return base + " You can also claim this retroactively within 90 days [S1]."


# ---------------------------------------------------------------- generate

def _format_evidence(evidence: dict) -> str:
    return "\n".join(f"[{k}] {v}" for k, v in evidence.items())


def _untagged_sentences(answer: str):
    out = []
    for s in re.split(r"(?<=[.!?])\s+|\n+", answer.strip()):
        s = s.strip()
        if len(s) < 12:
            continue
        if not TAG.search(s):
            out.append(s)
    return out


def generate(question: str, evidence: dict, strict: bool = False,
             call_model=None, second_pass: bool = False) -> dict:
    """
    Returns the answer plus everything the rest of the pipeline needs to know
    about HOW it was produced.
    """
    t0 = time.perf_counter()
    call_model = call_model or extractive_model
    mode = MODES["strict" if strict else "normal"]

    if not evidence:
        return {"answer": "I could not find anything on that in the documents.",
                "mode": "strict" if strict else "normal",
                "temperature": mode["temperature"], "model": mode["model"],
                "citations_ok": True, "untagged": [], "reprompted": False,
                "calls": 0, "time_ms": round((time.perf_counter() - t0) * 1000, 2)}

    temperature = SECOND_PASS_TEMPERATURE if second_pass else mode["temperature"]
    template = STRICT_PROMPT if strict else NORMAL_PROMPT
    prompt = template.format(evidence=_format_evidence(evidence), question=question)

    answer = call_model(prompt, temperature)
    calls, reprompted = 1, False

    # Strict mode VERIFIES the rule instead of trusting the prompt. A prompt is
    # a request; this is the check that the request was honoured.
    untagged = _untagged_sentences(answer)
    if strict and untagged:
        answer = call_model(prompt + "\n\n" + REPROMPT.format(offending=untagged[0][:90]),
                            temperature)
        calls += 1
        reprompted = True
        untagged = _untagged_sentences(answer)

    # Tags that point at evidence which does not exist are worse than no tag,
    # because they look verified. Strip them so the checks see the truth.
    valid = set(evidence)
    for tag in set(TAG.findall(answer)):
        if tag not in valid:
            answer = answer.replace(f"[{tag}]", "")

    return {
        "answer": answer.strip(),
        "mode": "strict" if strict else "normal",
        "temperature": temperature,
        "model": mode["model"],
        "require_citations": mode["require_citations"],
        "allow_auto_fix": mode["allow_auto_fix"],
        "citations_ok": not untagged,
        "untagged": untagged,
        "reprompted": reprompted,
        "calls": calls,
        "time_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


if __name__ == "__main__":
    ev = {"S1": "Notice period is 60 days for confirmed employees and 30 days "
                "during probation.",
          "S2": "Travel reimbursement is capped at 15000 rupees per trip."}

    print("NORMAL mode")
    r = generate("What is the notice period?", ev, strict=False)
    print(f"  temp {r['temperature']}  model {r['model']}  calls {r['calls']}")
    print(f"  {r['answer']}")

    print("\nSTRICT mode")
    r = generate("What is the notice period?", ev, strict=True)
    print(f"  temp {r['temperature']}  model {r['model']}  "
          f"citations_ok {r['citations_ok']}  reprompted {r['reprompted']}")
    print(f"  {r['answer']}")

    print("\nSTRICT mode against a model that skips tags")
    def sloppy(prompt, temperature=0.4):
        return "Notice period is 60 days. Managers can waive it entirely."
    r = generate("What is the notice period?", ev, strict=True, call_model=sloppy)
    print(f"  reprompted {r['reprompted']}  calls {r['calls']}  "
          f"citations_ok {r['citations_ok']}")
    print(f"  untagged: {r['untagged']}")

    print("\nA model that invents a number")
    r = generate("Can I claim it later?", ev, strict=True,
                 call_model=demo_hallucinating_model)
    print(f"  {r['answer']}")
    from tier2_groundedness import check_groundedness
    g = check_groundedness(r["answer"], ev)
    print(f"  groundedness {g['score']}  unsupported {g['unsupported']}")
    for w in g["worst"]:
        print(f"    {w['why']}")
