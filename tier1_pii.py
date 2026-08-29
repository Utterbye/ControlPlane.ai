"""
TIER 1 - DETECTOR 3 : SENSITIVE DATA (PII)

What it does:
    Find personal data in a piece of text, say how confident we are, and say
    what to do about it.

The two-mode idea
-----------------
The same detector runs twice, and the ACTION is opposite each time:

    mode="input"   the user typed their own Aadhaar into the chat.
                   They are allowed to. We MASK it before it reaches the model
                   or the logs, and we carry on answering.

    mode="output"  the bot is about to print someone's Aadhaar.
                   That is a leak. We BLOCK or redact.

Same finding, opposite response, decided by direction.

Why not just call Presidio
--------------------------
Presidio is excellent, but out of the box it is built around US and EU data.
Measured on this machine:

    email        -> EMAIL_ADDRESS 1.00   correct
    credit card  -> CREDIT_CARD   1.00   correct, Luhn validated
    phone        -> DATE_TIME     0.85   wrong label, and it outranks PHONE_NUMBER
    PAN          -> PERSON        0.85   wrong
    Aadhaar      -> DATE_TIME     0.85   wrong
    IFSC         -> nothing              missed
    bank account -> nothing              missed

So this file adds India-specific patterns with real validation on top, and
uses Presidio only for what Presidio is good at.

Validation is what keeps false alarms low
-----------------------------------------
A 12-digit number is not an Aadhaar. An Aadhaar passes the Verhoeff checksum.
A 16-digit number is not a card. A card passes Luhn. Checking the maths turns
a noisy regex into a precise one.

Presidio is optional. Without it the file still runs on regex plus validation.
    pip install presidio-analyzer
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
"""

import re
import time

# ============================================================================
# PART 1 - VALIDATORS
# A regex says "this looks like the right shape".
# A validator says "the maths actually works out".
# ============================================================================

# Verhoeff tables. Aadhaar uses this checksum, so a random 12-digit number
# has only about a 1 in 10 chance of passing.
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_ok(digits: str) -> bool:
    """Aadhaar checksum. Wrong length or a failed check means it is not one."""
    if len(digits) != 12 or not digits.isdigit():
        return False
    if digits[0] in "01":          # a real Aadhaar never starts with 0 or 1
        return False
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def luhn_ok(digits: str) -> bool:
    """Card checksum. Same idea, different algorithm."""
    if not digits.isdigit() or not (12 <= len(digits) <= 19):
        return False
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def pan_ok(text: str) -> bool:
    """
    PAN is AAAAA1234A. The 4th letter is the holder type, and only a fixed
    set of letters is legal there. That one rule removes most random matches.
    """
    if not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", text):
        return False
    return text[3] in "ABCFGHLJPTKE"


def gstin_ok(text: str) -> bool:
    """GSTIN is 2-digit state code + PAN + entity char + Z + check char."""
    if not re.fullmatch(r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]", text):
        return False
    return 1 <= int(text[:2]) <= 38          # valid Indian state codes


def always_ok(text: str) -> bool:
    return True


# ============================================================================
# PART 2 - PATTERNS
# (name, regex, validator, base_score, action_in, action_out)
#   action_in  - what to do when the USER typed it
#   action_out - what to do when the BOT is about to say it
# ============================================================================

PATTERNS = [
    ("AADHAAR",
     r"\b[2-9][0-9]{3}[\s-]?[0-9]{4}[\s-]?[0-9]{4}\b", verhoeff_ok, 0.95, "mask", "block"),

    ("PAN",
     r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", pan_ok, 0.90, "mask", "block"),

    ("GSTIN",
     r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z]\b", gstin_ok, 0.90, "mask", "block"),

    ("CREDIT_CARD",
     r"\b(?:[0-9][ -]?){12,19}\b", luhn_ok, 0.95, "mask", "block"),

    ("IFSC",
     r"\b[A-Z]{4}0[A-Z0-9]{6}\b", always_ok, 0.85, "mask", "block"),

    ("BANK_ACCOUNT",
     r"\b[0-9]{9,18}\b", always_ok, 0.40, "mask", "block"),   # needs context, see below

    ("UPI_ID",
     r"\b[\w.\-]{3,}@(?:okaxis|oksbi|okhdfcbank|okicici|ybl|paytm|upi|apl|ibl)\b",
     always_ok, 0.90, "mask", "block"),

    ("INDIAN_PHONE",
     r"(?:\+?91[\s-]?)?\b[6-9][0-9]{9}\b", always_ok, 0.70, "mask", "redact"),

    ("EMAIL",
     r"\b[\w.\-+]+@[\w\-]+\.[A-Za-z]{2,}\b", always_ok, 0.95, "allow", "redact"),

    ("VEHICLE_REG",
     r"\b[A-Z]{2}[\s-]?[0-9]{1,2}[\s-]?[A-Z]{1,3}[\s-]?[0-9]{4}\b", always_ok, 0.75, "mask", "redact"),

    ("PASSPORT_IN",
     r"\b[A-PR-WY][0-9]{7}\b", always_ok, 0.60, "mask", "block"),
]

# Words that, when they sit near a match, make it far more likely to be real.
# This is how BANK_ACCOUNT gets from a weak 0.40 to something usable, without
# flagging every long number in the text.
CONTEXT = {
    "AADHAAR": ["aadhaar", "aadhar", "uid", "uidai"],
    "PAN": ["pan", "permanent account"],
    "BANK_ACCOUNT": ["account", "a/c", "acct", "bank", "ifsc", "beneficiary"],
    "CREDIT_CARD": ["card", "credit", "debit", "cvv", "visa", "mastercard"],
    "IFSC": ["ifsc", "branch", "neft", "rtgs"],
    "INDIAN_PHONE": ["phone", "mobile", "call", "whatsapp", "contact", "number"],
    "GSTIN": ["gst", "gstin", "tax"],
    "PASSPORT_IN": ["passport"],
    "VEHICLE_REG": ["vehicle", "car", "registration", "number plate"],
}
# Words that mean the opposite: if one of these sits next to a match and no
# positive context word does, the number belongs to a system, not a person.
# This is more surgical than demanding positive context everywhere, which
# would miss a genuine leak that happens not to say "aadhaar" next to it.
NEGATIVE_CONTEXT = [
    "order", "invoice", "ticket", "build", "reference", "ref no", "po number",
    "room", "seat", "floor", "extension", "docket", "awb", "tracking",
    "batch", "serial", "sku", "transaction id", "txn", "job id", "case id",
]

CONTEXT_WINDOW = 40          # characters either side of the match
CONTEXT_BOOST = 0.35

# Company-owned values that must never be treated as personal data.
ALLOWLIST = {
    "support@company.com", "hr@company.com", "helpdesk@company.com",
    "1800123456", "18001234567",
}

_presidio = None
_presidio_tried = False


# ============================================================================
# PART 3 - PRESIDIO (optional, for what regex is bad at)
# ============================================================================

def _load_presidio():
    """
    Presidio is only used for PERSON names, international phones, IBAN and so
    on. It loads once. If it is missing, the detector still works on regex.
    """
    global _presidio, _presidio_tried
    if _presidio_tried:
        return _presidio
    _presidio_tried = True
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        cfg = {"nlp_engine_name": "spacy",
               "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}
        nlp = NlpEngineProvider(nlp_configuration=cfg).create_engine()
        _presidio = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
    except Exception:
        _presidio = None
    return _presidio


# entities we trust Presidio with. DATE_TIME is deliberately left out: it fired
# on Aadhaar and on phone numbers and outranked the correct label.
PRESIDIO_ENTITIES = {
    "PERSON": (0.60, "allow", "redact"),
    "IBAN_CODE": (0.90, "mask", "block"),
    "US_SSN": (0.90, "mask", "block"),
    "IP_ADDRESS": (0.60, "allow", "redact"),
}


# ============================================================================
# PART 4 - THE DETECTOR
# ============================================================================

def _around(text: str, start: int, end: int) -> str:
    left = max(0, start - CONTEXT_WINDOW)
    return text[left:end + CONTEXT_WINDOW].lower()


def _context_hit(text: str, start: int, end: int, kind: str) -> bool:
    words = CONTEXT.get(kind, [])
    if not words:
        return False
    return any(w in _around(text, start, end) for w in words)


def _negative_hit(text: str, start: int, end: int) -> bool:
    return any(w in _around(text, start, end) for w in NEGATIVE_CONTEXT)


def _overlaps(a, b):
    return not (a["end"] <= b["start"] or b["end"] <= a["start"])


def detect_pii(text: str, mode: str = "input", use_presidio: bool = True) -> dict:
    """
    text  - the query, or the answer
    mode  - "input" (user typed it) or "output" (bot is about to say it)
    """
    assert mode in ("input", "output")
    t0 = time.perf_counter()
    findings = []

    # ---- regex patterns with validation
    for kind, rx, validate, base, act_in, act_out in PATTERNS:
        for m in re.finditer(rx, text):
            raw = m.group()
            cleaned = re.sub(r"[\s-]", "", raw)

            if raw.strip().lower() in ALLOWLIST or cleaned.lower() in ALLOWLIST:
                continue
            if not validate(cleaned if kind != "EMAIL" else raw):
                continue

            score = base
            ctx = _context_hit(text, m.start(), m.end(), kind)
            if ctx:
                score = min(1.0, score + CONTEXT_BOOST)

            # a bare long number is only personal data if something says so
            if kind == "BANK_ACCOUNT" and not ctx:
                continue

            # a system id wearing the right shape. Positive context always wins,
            # so "aadhaar 2234..." still fires even inside an invoice sentence.
            if not ctx and _negative_hit(text, m.start(), m.end()):
                continue

            findings.append({
                "type": kind, "text": raw, "start": m.start(), "end": m.end(),
                "score": round(score, 2), "source": "regex",
                "context": ctx,
                "action": act_in if mode == "input" else act_out,
            })

    # ---- Presidio for the things regex cannot do
    if use_presidio:
        eng = _load_presidio()
        if eng is not None:
            try:
                for r in eng.analyze(text=text, language="en",
                                     entities=list(PRESIDIO_ENTITIES)):
                    base, act_in, act_out = PRESIDIO_ENTITIES[r.entity_type]
                    cand = {
                        "type": r.entity_type, "text": text[r.start:r.end],
                        "start": r.start, "end": r.end,
                        "score": round(min(base, r.score), 2),
                        "source": "presidio", "context": False,
                        "action": act_in if mode == "input" else act_out,
                    }
                    # our own validated match always wins over a Presidio guess
                    if not any(_overlaps(cand, f) for f in findings):
                        findings.append(cand)
            except Exception:
                pass

    findings.sort(key=lambda f: f["start"])

    # ---- one score and one action for the whole text
    if findings:
        score = max(f["score"] for f in findings)
        order = {"block": 3, "redact": 2, "mask": 1, "allow": 0}
        action = max((f["action"] for f in findings), key=lambda a: order[a])
    else:
        score, action = 0.0, "allow"

    return {
        "detector": "pii",
        "mode": mode,
        "score": round(score, 2),
        "action": action,
        "count": len(findings),
        "types": sorted({f["type"] for f in findings}),
        "findings": findings,
        "masked_text": mask(text, findings) if findings else text,
        "time_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


def mask(text: str, findings) -> str:
    """Replace each finding from the end backwards, so earlier offsets stay valid."""
    out = text
    for f in sorted(findings, key=lambda x: x["start"], reverse=True):
        if f["action"] == "allow":
            continue
        out = out[:f["start"]] + f"[{f['type']}]" + out[f["end"]:]
    return out


def band_of(score: float) -> str:
    if score >= 0.70:
        return "block"
    if score >= 0.40:
        return "cautious"
    return "allow"


if __name__ == "__main__":
    samples = [
        ("My Aadhaar is 2234 5678 9018 and PAN ABCPE1234F", "input"),
        ("My PAN is ABCDE1234F", "input"),   # D is not a legal holder type -> rejected
        ("Here is the employee record: Aadhaar 2234 5678 9018", "output"),
        ("Card 4111 1111 1111 1111, cvv 123", "input"),
        ("My account number is 50100234567890, IFSC HDFC0001234", "input"),
        ("Call me on 9876543210 or mail rahul@acme.com", "input"),
        ("Pay me at rahul@okaxis", "input"),
        ("The order id is 50100234567890 for the shipment", "input"),
        ("What is the leave policy?", "input"),
        ("Contact support@company.com or 1800123456", "output"),
    ]
    print(f"\n{'score':>6} {'band':>9} {'action':>8} {'ms':>7}  text")
    print("-" * 100)
    for txt, mode in samples:
        r = detect_pii(txt, mode)
        print(f"{r['score']:>6} {band_of(r['score']):>9} {r['action']:>8} "
              f"{r['time_ms']:>7}  [{mode}] {txt}")
        if r["findings"]:
            print(f"{'':>32}  {r['types']}")
            print(f"{'':>32}  masked -> {r['masked_text']}")
