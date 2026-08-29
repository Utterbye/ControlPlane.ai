"""
Example sentences for Detector 4 (high-stakes topic).

What "high stakes" means here
-----------------------------
NOT dangerous. NOT an attack. Just a question where a confidently wrong
answer causes real damage - money moves, a right is misstated, a commitment
is made on the company's behalf.

The Air Canada case is exactly this. Nobody attacked the chatbot. Someone
asked a completely ordinary question about a refund policy, got a confident
wrong answer, acted on it, and the airline was held liable. That query would
score zero on injection, zero on unsafe intent, and zero on PII. This is the
detector that would have caught it.

So the action is never "block". It is "raise the bar":
    every claim must cite a source
    no silent edits
    stronger model, low temperature
    the deep check runs even if the fast check looked clean

LOW_STAKES exists for the same reason the safe list exists in Detector 2.
Without something to compare against, every question looks like it belongs to
some category. "What are the office timings" must not trigger strict mode.
"""

STAKES = {

    # A wrong answer here moves money or misstates what someone is owed.
    "money": [
        "what is my salary revision this year",
        "how much reimbursement can I claim for travel",
        "what is the limit on the medical claim",
        "when will my bonus be credited",
        "how much notice pay do I get if I resign",
        "what is the per diem for an overseas trip",
        "can I claim the full amount for this invoice",
        "how much tax will be deducted from my salary",
        "what is my gratuity amount after five years",
        "is the joining bonus refundable if I leave early",
        "what is the reimbursement rate per kilometre",
        "how much of my provident fund can I withdraw",
    ],

    # A wrong answer misstates a right, an obligation, or a process outcome.
    "hr_policy": [
        "how many leaves am I entitled to this year",
        "what is my notice period as per my contract",
        "can the company terminate me without notice",
        "am I eligible for maternity leave",
        "what happens to my leave balance when I resign",
        "is my probation extended automatically",
        "can I be transferred without my consent",
        "what are the grounds for termination in the policy",
        "am I entitled to a relieving letter",
        "how does the appraisal rating affect my increment",
    ],

    # A wrong answer becomes a legal exposure.
    "legal": [
        "what does the non compete clause in my contract mean",
        "am I legally allowed to work a second job",
        "what are the company obligations under the labour law",
        "is this clause in the vendor contract enforceable",
        "what is our liability if the client data is leaked",
        "does this agreement need a stamp duty",
        "what notice does the law require for termination",
        "are we compliant with the data protection rules",
    ],

    # A wrong answer affects someone's health decision or insurance claim.
    "medical": [
        "what does my health insurance cover",
        "is this treatment covered under the company policy",
        "how many days of sick leave am I allowed",
        "can I claim for my parents medical bills",
        "what is the room rent limit in the insurance",
        "is a pre existing condition covered",
        "how do I add my spouse to the health cover",
    ],

    # A wrong answer creates a security or access hole.
    "security": [
        "how do I get admin access to the production system",
        "can I share this document with an external partner",
        "what is the process to reset a service account password",
        "am I allowed to store client data on my laptop",
        "who approves access to the customer database",
        "can I use a personal device for work email",
        "what is the retention period for customer records",
    ],

    # THE AIR CANADA CATEGORY.
    # The bot is about to state a promise the company must honour.
    "customer_commitment": [
        "can the customer get a refund after thirty days",
        "what is our cancellation policy for this booking",
        "does the warranty cover accidental damage",
        "can I promise the client delivery by next week",
        "what discount am I authorised to offer",
        "is the customer eligible for a bereavement fare",
        "what does our service level agreement guarantee",
        "can we waive the cancellation fee for this case",
        "what compensation is the customer entitled to",
        "how long does the customer have to claim a refund",
    ],
}

# Ordinary questions where a wrong answer costs nothing much.
# This is the comparison set - without it, everything looks high stakes.
LOW_STAKES = [
    "what are the office timings",
    "where is the cafeteria",
    "how do I book a meeting room",
    "is there a shuttle from the metro station",
    "what is the wifi password for guests",
    "who is the facilities manager",
    "how do I update my profile photo",
    "where do I find the holiday calendar",
    "what is the dress code for client meetings",
    "how do I install the vpn client",
    "which floor is the design team on",
    "what time does the gym open",
    "how do I raise an IT ticket",
    "is the parking free for employees",
    "what is the name of our travel desk vendor",
    "how do I join the internal book club",
    "where can I find the company logo files",
    "what is the agenda for the town hall",
    "how do I change my desk location",
    "is there a cricket team in the office",
    "summarise this document for me",
    "translate this paragraph into Hindi",
    "make this email sound more polite",
    "give me a title for this presentation",
    "explain what an API is",
    "what does this acronym stand for",
]

# What strict mode actually switches on. Same for every category, because the
# problem is the same - a confident wrong answer that someone acts on.
STRICT_MODE = [
    "every factual claim must carry a source tag",
    "no silent auto-fix - either show the note or abstain",
    "use the stronger model, low temperature",
    "run the deep check even if the fast check looked clean",
    "log the decision with a reference ID for audit",
]

STAKES_FLAT = [(cat, text) for cat, lines in STAKES.items() for text in lines]

if __name__ == "__main__":
    print(f"high-stakes examples: {len(STAKES_FLAT)} across {len(STAKES)} categories")
    for c, lines in STAKES.items():
        print(f"   {c:<22} {len(lines)}")
    print(f"low-stakes examples : {len(LOW_STAKES)}")
