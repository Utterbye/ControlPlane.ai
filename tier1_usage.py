"""
TIER 1 - DETECTOR 5 : USAGE GUARD

What it does:
    Protect against cost attacks and spam loops - somebody hammering the bot,
    pasting a novel into every message, or burning the daily budget.

The thing everyone gets wrong about this detector
-------------------------------------------------
It protects against the 50th request, NOT the first one.

On query number one the counters are all zero, and that is correct behaviour,
not a bug. There is no history yet. The only thing visible on a first request
is SIZE - a 40,000 character paste is a cost attack you can see immediately.

Everything else needs the session to build up:

    visible on request 1          needs history
    --------------------          -------------
    query length                  requests per minute
    pasted content size           requests today
    repeated character runs       spend today
    number of lines               retries this session
                                  how repetitive the queries are

So this detector reports cold_start=True while it is still blind, instead of
quietly returning a clean score. A clean score and "I cannot see anything yet"
are different statements, and the difference matters when you are reading logs.

Two different keys, because two different problems
--------------------------------------------------
    user_id    one person across ALL their chats
               -> money and rate. Opening ten tabs must not reset the budget.

    thread_id  one conversation
               -> loops and repetition. Asking the same thing three times in
                  ONE chat is a frustrated user. Asking three different
                  questions in three chats is a normal Tuesday.

Counting both in the same bucket breaks both jobs. So spend and rate are
counted per user, and retries and repetition are counted per thread.

Storage
-------
Counters live in memory here so the file runs with no setup. In production
they belong in Redis, which does the same job plus automatic expiry - you say
"keep this for 60 seconds" and it deletes itself. The Counters class is the
only thing that would change.

Tokens
------
tokens is passed in, not guessed. After a real chat call the model returns a
usage block with the exact input and output token count, and that is what you
record. The demo estimates len(text)/4 only because there is no model call.
"""

import time
from collections import defaultdict, deque

# ---------------------------------------------------------------- limits

LIMITS = {
    "chars_soft": 4000,        # long, but plausible
    "chars_hard": 20000,       # nobody types this
    "lines_hard": 400,
    "repeat_run": 200,         # same character in a row
    "per_minute": 20,
    "per_day": 500,
    "tokens_per_day": 200000,  # rough stand-in for spend
    "retries": 3,
    "repeat_ratio": 0.6,       # share of recent queries that are identical
}

WINDOW_MINUTE = 60
WINDOW_DAY = 86400


# ---------------------------------------------------------------- storage

class Counters:
    """
    In-memory stand-in for Redis. Every list is pruned by time, so old entries
    disappear the same way a Redis key with an expiry would.
    """

    def __init__(self):
        self.hits = defaultdict(deque)     # user -> timestamps
        self.tokens = defaultdict(deque)   # user -> (timestamp, tokens)
        self.recent = defaultdict(deque)   # user -> recent query texts
        self.retries = defaultdict(int)

    @staticmethod
    def _prune(dq, now, window, keyed=False):
        while dq and (now - (dq[0][0] if keyed else dq[0])) > window:
            dq.popleft()

    def record(self, user: str, text: str, tokens: int = 0, thread: str = None):
        """user gets the money counters, thread gets the loop counters."""
        now = time.time()
        thread = thread or user
        self.hits[user].append(now)                      # rate  -> per user
        self.tokens[user].append((now, tokens))          # spend -> per user
        self.recent[thread].append(text.strip().lower()[:200])   # loops -> per thread
        while len(self.recent[thread]) > 20:
            self.recent[thread].popleft()

    def snapshot(self, user: str, thread: str = None) -> dict:
        now = time.time()
        thread = thread or user
        self._prune(self.hits[user], now, WINDOW_DAY)
        self._prune(self.tokens[user], now, WINDOW_DAY, keyed=True)
        per_min = sum(1 for t in self.hits[user] if now - t <= WINDOW_MINUTE)
        recent = list(self.recent[thread])
        repeat_ratio = 0.0
        if len(recent) >= 4:
            repeat_ratio = 1 - (len(set(recent)) / len(recent))
        return {
            "per_minute": per_min,
            "per_day": len(self.hits[user]),
            "tokens_today": sum(tk for _, tk in self.tokens[user]),
            "retries": self.retries[thread],
            "repeat_ratio": round(repeat_ratio, 2),
            "history_len": len(self.hits[user]),
        }

    def reset(self, user: str = None):
        for d in (self.hits, self.tokens, self.recent, self.retries):
            d.clear() if user is None else d.pop(user, None)


COUNTERS = Counters()


# ---------------------------------------------------------------- helpers

def _longest_run(text: str) -> int:
    best = run = 1
    for i in range(1, len(text)):
        run = run + 1 if text[i] == text[i - 1] else 1
        best = max(best, run)
    return best if text else 0


def _ratio(value, limit):
    """0.0 at the limit, rising to 1.0 at twice the limit."""
    if value <= limit:
        return 0.0
    return min(1.0, (value - limit) / limit)


# ---------------------------------------------------------------- detector

def detect_usage(text: str, user: str = "anon", thread: str = None) -> dict:
    t0 = time.perf_counter()
    snap = COUNTERS.snapshot(user, thread)
    cold = snap["history_len"] == 0

    size_score, hist_score, matched = 0.0, 0.0, []

    # ---- visible on the very first request
    n_chars = len(text)
    if n_chars > LIMITS["chars_hard"]:
        size_score = max(size_score, 0.85)
        matched.append(f"size:chars={n_chars}")
    elif n_chars > LIMITS["chars_soft"]:
        size_score = max(size_score, 0.35 + 0.3 * _ratio(n_chars, LIMITS["chars_soft"]))
        matched.append(f"size:long_paste={n_chars}")

    n_lines = text.count("\n") + 1
    if n_lines > LIMITS["lines_hard"]:
        size_score = max(size_score, 0.60)
        matched.append(f"size:lines={n_lines}")

    run = _longest_run(text)
    if run > LIMITS["repeat_run"]:
        size_score = max(size_score, 0.55)
        matched.append(f"size:repeat_run={run}")

    # ---- needs history, silent until the session has one
    if not cold:
        for key, limit_key, label in (
            ("per_minute", "per_minute", "rate"),
            ("per_day", "per_day", "daily_requests"),
            ("tokens_today", "tokens_per_day", "spend"),
            ("retries", "retries", "retries"),
        ):
            r = _ratio(snap[key], LIMITS[limit_key])
            if r > 0:
                hist_score = max(hist_score, 0.45 + 0.5 * r)
                matched.append(f"{label}:{snap[key]}/{LIMITS[limit_key]}")

        if snap["repeat_ratio"] > LIMITS["repeat_ratio"]:
            hist_score = max(hist_score, 0.50)
            matched.append(f"loop:repeat_ratio={snap['repeat_ratio']}")

    score = max(size_score, hist_score)

    if score >= 0.80:
        action = "block"
    elif score >= 0.40:
        action = "throttle"
    else:
        action = "allow"

    return {
        "detector": "usage_guard",
        "score": round(score, 2),
        "action": action,
        "cold_start": cold,
        "visible_signals": ["size"] if cold else ["size", "history"],
        "keys": {"user": user, "thread": thread or user},
        "signals": {"chars": n_chars, "lines": n_lines, "longest_run": run, **snap},
        "matched": matched or (["cold_start:only size is visible"] if cold else []),
        "time_ms": round((time.perf_counter() - t0) * 1000, 3),
    }


def band_of(score: float) -> str:
    if score >= 0.80:
        return "block"
    if score >= 0.40:
        return "cautious"
    return "allow"


if __name__ == "__main__":
    COUNTERS.reset()
    print("\nrequest 1 - nothing in history yet")
    r = detect_usage("What is the leave policy?", "u1")
    print(f"   score={r['score']} cold_start={r['cold_start']} "
          f"visible={r['visible_signals']} matched={r['matched']}")

    print("\nrequest 1 but a huge paste - size IS visible immediately")
    COUNTERS.reset()
    r = detect_usage("x" * 25000, "u2")
    print(f"   score={r['score']} cold_start={r['cold_start']} matched={r['matched']}")

    print("\nsame user, 25 fast requests - now history can speak")
    COUNTERS.reset()
    for i in range(25):
        COUNTERS.record("u3", f"question number {i}", tokens=500)
    r = detect_usage("one more question", "u3")
    print(f"   score={r['score']} action={r['action']} matched={r['matched']}")

    print("\nsame user asking the same thing over and over")
    COUNTERS.reset()
    for _ in range(10):
        COUNTERS.record("u4", "what is my leave balance", tokens=200)
    r = detect_usage("what is my leave balance", "u4")
    print(f"   score={r['score']} action={r['action']} matched={r['matched']}")
