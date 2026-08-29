"""
THE DOCUMENT STORE

Our own corpus, not borrowed from anywhere. Small on purpose - a demo needs
documents you can read in thirty seconds and still argue about.

Every document carries four things the pipeline actually uses:

    source_type   internal_policy > internal_doc > official > news > blog
                  This drives the trust boost in re-ranking and, more
                  importantly, it settles contradictions.
    year          for the recency boost, and for breaking ties when two
                  sources disagree.
    source_id     several chunks can come from ONE document. Counting them as
                  separate sources would let a single file outvote everything
                  else just by being long.
    version       bumped when the content changes. The cache checks this - an
                  answer that was right under v1 may be wrong under v2, and
                  nothing about the question tells you that.

Deliberate contradictions
-------------------------
The corpus contains real disagreements, because a demo where every document
agrees proves nothing:

    notice period    2026 policy says 60 days, the 2019 doc says 90
    reimbursement    2026 policy says 15000, a news article says 12000
    probation        two internal documents disagree on the duration

Each one is resolvable by a different rule, so you can watch the resolution
order actually do its job.
"""

DOC_VERSION = "v1"

DOCUMENTS = [
    # ---- notice period: a real contradiction, resolvable by trust rank
    {"id": "n1", "source_id": "hr_policy_2026", "source_type": "internal_policy",
     "year": 2026, "title": "HR Policy 2026 - Separation",
     "text": "Notice period is 60 days for confirmed employees and 30 days "
             "during probation. The notice period may be waived by mutual consent."},
    {"id": "n2", "source_id": "hr_handbook_2019", "source_type": "internal_doc",
     "year": 2019, "title": "Employee Handbook 2019",
     "text": "Notice period is 90 days for all confirmed employees."},
    {"id": "n3", "source_id": "careers_blog", "source_type": "blog",
     "year": 2020, "title": "Careers blog",
     "text": "At most Indian technology companies the notice period is 90 days."},

    # ---- reimbursement: resolvable by trust rank again, different sources
    {"id": "r1", "source_id": "travel_policy_2026", "source_type": "internal_policy",
     "year": 2026, "title": "Travel and Expense Policy 2026",
     "text": "Travel reimbursement is capped at 15000 rupees per trip. "
             "Claims must be submitted within 30 days of travel."},
    {"id": "r2", "source_id": "industry_news", "source_type": "news",
     "year": 2025, "title": "Industry salary and benefits report",
     "text": "Typical travel reimbursement in the sector is capped at 12000 rupees per trip."},

    # ---- probation: two internal docs, resolvable by RECENCY not trust
    {"id": "p1", "source_id": "onboarding_2026", "source_type": "internal_doc",
     "year": 2026, "title": "Onboarding Guide 2026",
     "text": "Probation lasts 3 months and may be extended once by 3 months."},
    {"id": "p2", "source_id": "onboarding_2022", "source_type": "internal_doc",
     "year": 2022, "title": "Onboarding Guide 2022",
     "text": "Probation lasts 6 months for all new joiners."},

    # ---- uncontested facts
    {"id": "l1", "source_id": "leave_policy_2026", "source_type": "internal_policy",
     "year": 2026, "title": "Leave Policy 2026",
     "text": "Employees receive 12 casual leaves and 15 earned leaves per year. "
             "Earned leave may be carried forward up to 45 days."},
    {"id": "l2", "source_id": "leave_policy_2026", "source_type": "internal_policy",
     "year": 2026, "title": "Leave Policy 2026",
     "text": "Bereavement leave of up to 3 days may be taken for an immediate "
             "family member. The application must be submitted before the leave."},
    {"id": "m1", "source_id": "insurance_2026", "source_type": "internal_policy",
     "year": 2026, "title": "Group Insurance 2026",
     "text": "Group health cover is 500000 rupees per family per year. "
             "Parents may be added at an additional premium."},
    {"id": "w1", "source_id": "wfh_policy_2026", "source_type": "internal_policy",
     "year": 2026, "title": "Work From Home Policy 2026",
     "text": "Employees may work from home up to 8 days per month with manager "
             "approval. Fully remote work requires HR approval."},
    {"id": "f1", "source_id": "facilities", "source_type": "internal_doc",
     "year": 2025, "title": "Facilities",
     "text": "The office is open from 8am to 8pm. The canteen serves lunch "
             "between 12pm and 3pm."},
    {"id": "s1", "source_id": "security_policy_2026", "source_type": "internal_policy",
     "year": 2026, "title": "Information Security Policy 2026",
     "text": "Customer data must not be copied to personal devices. Access to "
             "production systems requires approval from the data owner."},
]


def corpus():
    return DOCUMENTS


def by_id(doc_id):
    return next((d for d in DOCUMENTS if d["id"] == doc_id), None)


def bump_version(new_version):
    """
    Changing the documents must invalidate cached answers. This is the hook
    the cache checks - an answer stored under v1 is not served under v2.
    """
    global DOC_VERSION
    DOC_VERSION = new_version
    return DOC_VERSION


if __name__ == "__main__":
    from collections import Counter
    print(f"{len(DOCUMENTS)} documents, version {DOC_VERSION}")
    print("by source type:", dict(Counter(d["source_type"] for d in DOCUMENTS)))
    print("\ndeliberate contradictions:")
    print("  notice period   60 (policy 2026) vs 90 (handbook 2019, blog)")
    print("  reimbursement   15000 (policy 2026) vs 12000 (news 2025)")
    print("  probation       3 months (2026 doc) vs 6 months (2022 doc)")
