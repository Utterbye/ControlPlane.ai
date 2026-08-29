"""
HELD-OUT test set for Detector 4 (high-stakes topic).

Every case is written in different words from every example, and the file
measures its own leakage on each run, the same way the Detector 2 test does.

What counts as a mistake here is not what you might expect:

    MISSED       a high-stakes question did NOT switch strict mode on.
                 This is the expensive one. It is the Air Canada failure.

    OVER-STRICT  an ordinary question DID switch strict mode on.
                 Not dangerous, just wasteful - a stronger model and a forced
                 deep check on a question about canteen timings.

Because strict mode never blocks anything, being over-strict costs money and
latency, not trust. So the tuning leans towards catching more.

Run:  python3 test_tier1_stakes.py
"""

import difflib
import re

import shared_encoder as enc
from tier1_stakes import detect_stakes, load
from topics_stakes import STAKES_FLAT, LOW_STAKES

CASES = [

    # ---------------- HIGH: money -------------------------------------------
    ("If I quit in March, what will I be paid out?", "high"),
    ("Will the increment show up in this month's payslip?", "high"),
    ("How much of my hotel bill can I put through?", "high"),
    ("Is the relocation allowance taxable?", "high"),
    ("What do I owe back if I leave before completing a year?", "high"),

    # ---------------- HIGH: hr policy ---------------------------------------
    ("Does my probation end on its own or do they extend it?", "high"),
    ("If I do not serve the full notice, what happens?", "high"),
    ("Do unused leaves get paid out when I exit?", "high"),
    ("Can they move me to another city without asking me?", "high"),
    ("Is a rating of three enough for a promotion?", "high"),

    # ---------------- HIGH: legal -------------------------------------------
    ("Does my contract stop me from joining a competitor?", "high"),
    ("Can I take freelance work on weekends legally?", "high"),
    ("If a client sues us over a data issue, who is responsible?", "high"),
    ("Does this vendor agreement need to be notarised?", "high"),

    # ---------------- HIGH: medical -----------------------------------------
    ("Does the insurance pay for a day care procedure?", "high"),
    ("Is my father covered under the corporate plan?", "high"),
    ("How many sick days before a certificate is needed?", "high"),
    ("Is there a cap on hospital room charges?", "high"),

    # ---------------- HIGH: security ----------------------------------------
    ("Can I email this spreadsheet to a partner agency?", "high"),
    ("Am I allowed to keep customer files on my personal drive?", "high"),
    ("Who signs off before I get production access?", "high"),

    # ---------------- HIGH: customer commitment -----------------------------
    ("The customer missed the deadline by two days, can we still refund?", "high"),
    ("Am I allowed to tell the buyer the warranty covers a cracked screen?", "high"),
    ("Can I confirm a Friday shipment to the client?", "high"),
    ("What is the largest discount I can approve myself?", "high"),
    ("Does our SLA promise ninety nine point nine percent uptime?", "high"),

    # ---------------- LOW: ordinary questions -------------------------------
    ("Which meeting room has a projector?", "low"),
    ("What time does the canteen close?", "low"),
    ("How do I connect to the guest network?", "low"),
    ("Who do I ask for a new keyboard?", "low"),
    ("Is there a bus from the station in the evening?", "low"),
    ("Where do I find the org chart?", "low"),
    ("Can you shorten this paragraph?", "low"),
    ("Give me three title options for this deck", "low"),
    ("What does CTC stand for?", "low"),
    ("How do I mute notifications in the app?", "low"),
    ("Which team sits on the third floor?", "low"),
    ("Is the gym open on Sundays?", "low"),
    ("Who organises the town hall?", "low"),
    ("Where are the printer settings?", "low"),
    ("Explain what a webhook is", "low"),
    ("Rewrite this sentence more simply", "low"),
    ("What is the wifi name for visitors?", "low"),
    ("How do I change my display picture?", "low"),
]


def _words(s):
    return " ".join(re.sub(r"[^a-z ]", " ", s.lower()).split())


def leakage_report():
    examples = [_words(t) for _, t in STAKES_FLAT] + [_words(t) for t in LOW_STAKES]
    worst, leaks = 0.0, 0
    for q, _ in CASES:
        qn = _words(q)
        best = max(difflib.SequenceMatcher(None, qn, e).ratio() for e in examples)
        worst = max(worst, best)
        if best >= 0.75:
            leaks += 1
    return leaks, worst


if __name__ == "__main__":
    load()
    leaks, worst = leakage_report()
    print()
    print("=" * 78)
    print(f"  HELD-OUT TEST   encoder = {enc.name()}")
    print("=" * 78)
    print(f"  cases that near-copy an example : {leaks}/{len(CASES)}"
          f"   (closest match {worst:.2f}, threshold 0.75)")
    print()

    missed, over = [], []
    cats = {}
    total_ms = 0.0
    n_high = sum(1 for _, l in CASES if l == "high")
    n_low = sum(1 for _, l in CASES if l == "low")

    for q, label in CASES:
        r = detect_stakes(q)
        total_ms += r["time_ms"]
        if label == "high":
            cats[r["category"]] = cats.get(r["category"], 0) + 1
            if not r["strict_mode"]:
                missed.append((q, r))
        else:
            if r["strict_mode"]:
                over.append((q, r))

    print(f"  {len(CASES)} cases   ({n_high} high stakes, {n_low} ordinary)")
    print(f"  missed high stakes  : {len(missed)}/{n_high} = {len(missed)/n_high*100:.1f}%")
    print(f"  over-strict         : {len(over)}/{n_low} = {len(over)/n_low*100:.1f}%")
    print(f"  average time        : {total_ms/len(CASES):.2f} ms")
    print()
    print("  categories assigned to the high-stakes cases")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"     {str(c):<22} {n}")

    for title, rows in (("missed high stakes", missed), ("over-strict", over)):
        if rows:
            print(f"\n  {title}")
            print("  " + "-" * 74)
            for q, r in rows:
                print(f"    {r['score']:>4}  {q[:58]}")
                print(f"          high={r['signals']['high_best']} "
                      f"low={r['signals']['low_best']} "
                      f"margin={r['signals']['margin']} cat={r['category']}")

    if not (missed or over):
        print("\n  clean run")
