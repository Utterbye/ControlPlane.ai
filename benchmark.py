"""
BENCHMARK - test the detector on OTHER PEOPLE'S data

Why this file exists
--------------------
test_tier1_injection.py uses 114 cases that WE wrote. Those numbers look
great, but a judge can fairly ask: "did you just write test cases that your
own detector happens to pass?"

This file answers that. It downloads two public datasets that we did not
write and did not look at while building the phrase lists:

  ATTACKS  verazuo/jailbreak_llms - 666 real jailbreak prompts collected
           from Discord and Reddit communities (CCS 2024 research dataset)

  BENIGN   tatsu-lab/stanford_alpaca - 52,000 ordinary task instructions.
           None of these are attacks.

We measure two numbers:
    catch rate   - how many real attacks did we flag?
    false alarm  - how many ordinary instructions did we wrongly flag?

Note: the attack dataset contains offensive text. This script NEVER prints
prompt content - only counts.

Run:  python3 benchmark.py
"""

import json
import os
import random
import time
import urllib.request

import pandas as pd

from tier1_injection import detect_injection, band_of

ATTACK_URL = ("https://raw.githubusercontent.com/verazuo/jailbreak_llms/"
              "main/data/prompts/jailbreak_prompts_2023_05_07.csv")
BENIGN_URL = ("https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
              "main/alpaca_data.json")

ATTACK_FILE = "bench_attacks.csv"
BENIGN_FILE = "bench_benign.json"

BENIGN_SAMPLE = 2000       # how many ordinary instructions to test
SEED = 42                  # same sample every run, so numbers are comparable


def download(url: str, path: str):
    """Download once and keep it. No point pulling 22 MB on every run."""
    if os.path.exists(path):
        return
    print(f"downloading {os.path.basename(path)} ...")
    urllib.request.urlretrieve(url, path)


def load_attacks():
    download(ATTACK_URL, ATTACK_FILE)
    df = pd.read_csv(ATTACK_FILE)
    return [str(p) for p in df["prompt"].dropna().tolist()]


def load_benign():
    download(BENIGN_URL, BENIGN_FILE)
    data = json.load(open(BENIGN_FILE))
    # an alpaca row is {"instruction": ..., "input": ..., "output": ...}
    # we only care about what a user would actually type
    prompts = [d["instruction"] + (" " + d["input"] if d.get("input") else "")
               for d in data]
    random.seed(SEED)
    return random.sample(prompts, min(BENIGN_SAMPLE, len(prompts)))


def run(prompts, name):
    """Run the detector over a list of prompts and bucket the results."""
    buckets = {"allow": 0, "cautious": 0, "block": 0}
    total_ms = 0.0
    t0 = time.perf_counter()

    for p in prompts:
        r = detect_injection(p)          # no chat history in a benchmark
        buckets[band_of(r["score"])] += 1
        total_ms += r["time_ms"]

    wall = time.perf_counter() - t0
    print(f"  {name:<10} {len(prompts):>6} prompts   "
          f"{total_ms/len(prompts):.2f} ms each   {wall:.1f} s total")
    return buckets


if __name__ == "__main__":
    print("=" * 72)
    print("  BENCHMARK - public data the detector has never seen")
    print("=" * 72)

    attacks = load_attacks()
    benign = load_benign()
    print()

    a = run(attacks, "attacks")
    b = run(benign, "benign")

    n_a, n_b = sum(a.values()), sum(b.values())

    # a "catch" means we did not just wave it through
    caught_soft = a["cautious"] + a["block"]
    caught_hard = a["block"]

    # a "false alarm" is an ordinary instruction that got flagged
    fa_soft = b["cautious"] + b["block"]
    fa_hard = b["block"]

    print()
    print("  " + "-" * 68)
    print(f"  {'':<26}{'flagged (>=0.40)':>20}{'blocked (>=0.70)':>20}")
    print("  " + "-" * 68)
    print(f"  {'attacks caught':<26}"
          f"{caught_soft:>7} / {n_a:<10}{caught_hard:>7} / {n_a:<10}")
    print(f"  {'':<26}{caught_soft/n_a*100:>17.1f}%{caught_hard/n_a*100:>19.1f}%")
    print()
    print(f"  {'benign wrongly flagged':<26}"
          f"{fa_soft:>7} / {n_b:<10}{fa_hard:>7} / {n_b:<10}")
    print(f"  {'':<26}{fa_soft/n_b*100:>17.1f}%{fa_hard/n_b*100:>19.1f}%")
    print("  " + "-" * 68)
    print()
    print(f"  attack band split : allow {a['allow']}  "
          f"cautious {a['cautious']}  block {a['block']}")
    print(f"  benign band split : allow {b['allow']}  "
          f"cautious {b['cautious']}  block {b['block']}")
    print()
    print("  Reading this: the rule layers are cheap and explainable, so they")
    print("  carry the easy cases. Whatever they miss is exactly the gap that")
    print("  Layer C (the trained classifier) is meant to close.")
