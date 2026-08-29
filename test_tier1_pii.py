"""
Test set for Detector 3 (PII).

Two things are measured, and they are not the same thing:

    RECALL     did we find the personal data that is actually there?
    PRECISION  did we leave alone the numbers that only LOOK personal?

The second one is where most PII detectors fall over. An order id, an invoice
number, an employee id and a bank account are all just long digits. Telling
them apart is the whole job.

Labels
    hit   text contains real personal data, and we name the type we expect
    miss  text contains something that looks like PII but is not
    clean  nothing personal at all

Run:  python3 test_tier1_pii.py
"""

from tier1_pii import detect_pii, band_of, verhoeff_ok, luhn_ok, pan_ok

# (text, mode, label, expected_types)
CASES = [

    # ---------------- HIT: real personal data --------------------------------
    ("My Aadhaar number is 2234 5678 9018", "input", "hit", {"AADHAAR"}),
    ("aadhar 223456789018 attached for verification", "input", "hit", {"AADHAAR"}),
    ("PAN ABCPE1234F for the tax declaration", "input", "hit", {"PAN"}),
    ("my pan card number is AAAPZ1234C", "input", "hit", {"PAN"}),
    ("Card 4111 1111 1111 1111 expiry 09/27", "input", "hit", {"CREDIT_CARD"}),
    ("debit card 5555555555554444", "input", "hit", {"CREDIT_CARD"}),
    ("Account 50100234567890 IFSC HDFC0001234", "input", "hit", {"BANK_ACCOUNT", "IFSC"}),
    ("please credit to a/c 123456789012 in the same bank", "input", "hit", {"BANK_ACCOUNT"}),
    ("branch ifsc SBIN0012345", "input", "hit", {"IFSC"}),
    ("my mobile is 9876543210", "input", "hit", {"INDIAN_PHONE"}),
    ("call me at +91 8123456789", "input", "hit", {"INDIAN_PHONE"}),
    ("reach me on rahul.sharma@gmail.com", "input", "hit", {"EMAIL"}),
    ("send the money to rahul@okaxis", "input", "hit", {"UPI_ID"}),
    ("gstin 27AAPFU0939F1ZV for the invoice", "input", "hit", {"GSTIN"}),
    ("my passport is K1234567", "input", "hit", {"PASSPORT_IN"}),
    ("vehicle KA01AB1234 is parked outside", "input", "hit", {"VEHICLE_REG"}),

    # ---------------- HIT: the bot about to leak -----------------------------
    ("The employee Aadhaar on file is 2234 5678 9018", "output", "hit", {"AADHAAR"}),
    ("His registered mobile is 9876543210", "output", "hit", {"INDIAN_PHONE"}),
    ("Salary is credited to account 50100234567890", "output", "hit", {"BANK_ACCOUNT"}),

    # ---------------- MISS: looks like PII, is not ---------------------------
    ("The order id is 50100234567890 for the shipment", "input", "miss", set()),
    ("Invoice number 123456789012 is still pending", "input", "miss", set()),
    ("The build number is 202405161234", "input", "miss", set()),
    ("Ticket 987654321012 was closed yesterday", "input", "miss", set()),
    ("My PAN is ABCDE1234F", "input", "miss", set()),          # D is not a legal type char
    ("Aadhaar 1234 5678 9012 is the sample in the form", "input", "miss", set()),  # starts with 1
    ("card 4111 1111 1111 1112 was declined", "input", "miss", set()),  # fails Luhn
    ("The policy number is 5551234567890123", "input", "miss", set()),  # fails Luhn
    ("Meeting room 9876543210 does not exist", "input", "miss", set()),  # phone shape, no context
    ("Contact support@company.com for help", "output", "miss", set()),   # allowlisted
    ("Helpline 1800123456 is open till 6pm", "output", "miss", set()),   # allowlisted

    # ---------------- CLEAN: nothing personal --------------------------------
    ("What is the leave policy?", "input", "clean", set()),
    ("How many casual leaves do I get in a year?", "input", "clean", set()),
    ("Explain the appraisal cycle", "input", "clean", set()),
    ("The reimbursement limit is 25000 per year", "output", "clean", set()),
    ("We have 4500 employees across 12 offices", "output", "clean", set()),
    ("Please raise a ticket with the IT team", "output", "clean", set()),
]


def unit_checks():
    """The validators are the reason precision is high, so test them directly."""
    rows = [
        ("verhoeff 223456789018 valid",   verhoeff_ok("223456789018"), True),
        ("verhoeff 223456789012 invalid", verhoeff_ok("223456789012"), False),
        ("verhoeff starts with 1",        verhoeff_ok("123456789012"), False),
        ("luhn 4111111111111111 valid",   luhn_ok("4111111111111111"), True),
        ("luhn 4111111111111112 invalid", luhn_ok("4111111111111112"), False),
        ("pan ABCPE1234F valid",          pan_ok("ABCPE1234F"), True),
        ("pan ABCDE1234F invalid",        pan_ok("ABCDE1234F"), False),
        ("pan lowercase invalid",         pan_ok("abcpe1234f"), False),
    ]
    bad = [r for r in rows if r[1] != r[2]]
    print(f"  validator unit checks : {len(rows)-len(bad)}/{len(rows)} passed")
    for name, got, want in bad:
        print(f"     FAILED {name}: got {got}, wanted {want}")
    return len(bad) == 0


if __name__ == "__main__":
    print("=" * 80)
    print("  DETECTOR 3 - PII")
    print("=" * 80)
    unit_checks()
    print()

    misses, false_alarms, wrong_type, total_ms = [], [], [], 0.0
    n_hit = sum(1 for _, _, l, _ in CASES if l == "hit")
    n_neg = sum(1 for _, _, l, _ in CASES if l in ("miss", "clean"))

    for text, mode, label, expect in CASES:
        r = detect_pii(text, mode)
        total_ms += r["time_ms"]
        found = set(r["types"]) - {"PERSON"}       # PERSON is advisory, not scored

        if label == "hit":
            if not found:
                misses.append((text, r))
            elif not expect.issubset(found):
                wrong_type.append((text, expect, found))
        else:
            if found:
                false_alarms.append((text, r, found))

    print(f"  {len(CASES)} cases   ({n_hit} contain PII, {n_neg} do not)")
    print(f"  missed PII        : {len(misses)}/{n_hit} = {len(misses)/n_hit*100:.1f}%")
    print(f"  false alarms      : {len(false_alarms)}/{n_neg} = {len(false_alarms)/n_neg*100:.1f}%")
    print(f"  wrong entity type : {len(wrong_type)}/{n_hit}")
    print(f"  average time      : {total_ms/len(CASES):.2f} ms")

    for title, rows in (("missed", misses), ("false alarms", false_alarms),
                        ("wrong type", wrong_type)):
        if rows:
            print(f"\n  {title}")
            print("  " + "-" * 74)
            for row in rows:
                print(f"    {row[0][:64]}")
                print(f"      -> {row[1] if not isinstance(row[1], dict) else row[1]['types']}"
                      f"{'  found ' + str(row[2]) if len(row) > 2 else ''}")

    if not (misses or false_alarms or wrong_type):
        print("\n  clean run")
