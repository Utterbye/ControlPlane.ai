ControlPlane Checker

A multi-layer AI safety, reliability, and governance system for enterprise chatbots.

ControlPlane Checker acts as a control layer around an LLM. Instead of directly sending every user query to the model, it evaluates the request, retrieves supporting evidence, verifies the generated answer, makes a final decision, and continuously analyses historical behaviour.

🚀 Overview

The system is divided into three tiers, with each tier serving a different purpose:

Tier 1 — Fast Risk Detection: Quickly identifies potentially unsafe or sensitive requests.
Tier 2 — Deep Verification: Performs detailed checks on generated answers.
Tier 3 — Background Learning: Analyses historical traces and proposes system improvements.

The overall objective is to provide safe, reliable, explainable, and auditable AI responses.

🛡️ Tier 1 — Fast Risk Detection

Tier 1 is the first line of defence.

It performs five major checks:

Injection / Jailbreak Detection
Unsafe Intent Detection
PII Detection
High-Stakes Topic Detection
Usage / Cost Abuse Detection

Each detector produces a risk score. These scores are combined into an overall risk score, risk band, action, and strict-mode decision.

The unsafe-intent detector compares a query against both unsafe and safe examples, reducing false positives when legitimate queries contain dangerous-looking vocabulary.

High-stakes queries, such as financial, HR, legal, medical, security, and customer-commitment questions, can activate stricter verification rather than simply being blocked.

🔎 Query Processing

Before retrieval and generation, the system prepares the query through three stages:

1. Query Rewriting

Converts incomplete questions, short forms, and relative dates into clearer queries.

Example:

"What is the WFH policy?"
            ↓
"What is the work from home policy?"
2. Attempt Detection

Detects whether the user is repeating or correcting a previous request.

Examples:

"That's not what I asked."
"Try again."
"Again."

This prevents the system from returning a cached answer that the user has already rejected.

3. Cache Lookup

Checks whether a sufficiently similar verified answer already exists, reducing unnecessary model calls.

📚 Evidence Retrieval

If there is no suitable cached answer, the system retrieves relevant documents and ranks the available evidence.

The purpose is to ensure that the LLM generates its response using relevant supporting information rather than relying only on its internal knowledge.

The selected evidence is then passed to the model for answer generation.

🔬 Tier 2 — Deep Verification

Tier 2 is the deeper verification layer.

It checks the generated response using:

Groundedness

Checks whether the claims made in the answer are actually supported by the retrieved evidence.

Self-Consistency

The system can generate the answer twice using the same evidence and compare important facts such as numbers, dates, durations, and entities.

If two generations disagree on an important fact, the answer may be unreliable.

Output Safety & Fair Wording

Checks the generated answer, rather than the original question.

It looks for:

Unsafe advice
Unfair generalisations
Overconfident claims
Problematic wording

⚖️ Decision Engine

The Decision Engine combines the results from the different checks and decides what should happen to the response.

Possible outcomes include:

ALLOW
AUTO-FIX
RETRY
ABSTAIN
ESCALATE
BLOCK

For example, if the generated answer contains unsupported claims, the system can retry the generation. If the claims remain unsupported after the retry, the system can abstain instead of returning an unreliable answer.

The decision also includes a trust label and explanation, making the result easier to audit.

📝 Trace Store

After a request is processed, the system stores a trace containing information such as:

User question
Generated answer
Tier 1 scores
Tier 2 scores
Decision
Timing information
Token usage
Trust label
User feedback

The trace store allows the background system to analyse historical behaviour without interfering with live requests.

🧠 Tier 3 — Background Learning

Tier 3 does not run directly on a live request.

Instead, it analyses historical traces to identify patterns that cannot be detected from a single request.

It analyses:

Analysis	Purpose
Bias Patterns	Detect whether similar users are treated differently
Cost Patterns	Identify inefficient token/resource usage
Feedback Clusters	Identify topics receiving repeated negative feedback
Repeated Queries	Find questions being asked repeatedly
Conflict Rate	Detect disagreement between sources

For example, the bias analysis creates counterfactual versions of real questions by changing attributes such as names while keeping the rest of the question unchanged. A different risk result can indicate a possible bias signal.

👨‍💻 Human-Controlled Improvement

One of the most important principles of the system is:

Tier 3 proposes changes — it never applies them automatically.

Tier 3 can propose changes to:

Detector thresholds
Retrieval behaviour
Source trust lists
Caching
Usage limits

However, proposed changes require human review, shadow-mode testing, and validation before affecting users.

This creates a controlled learning loop:

Traces → Analysis → Finding → Proposal → Human Review → Shadow Testing → Validation → Approved Change

🖥️ Streamlit Application

The project includes a Streamlit-based web interface called ControlPlane Checker.

Run the application with:

streamlit run app.py

The application provides four main tabs:

💬 Ask

Allows the user to enter a question and run it through the complete pipeline.

👤 Human Review

Allows reviewers to examine flagged cases and provide verified feedback.

🧠 Tier 3

Displays the background analysis and improvement proposals generated by Tier 3.

📊 Traces

Displays the stored request records used by the background analysis.

When the user clicks Run the pipeline, Streamlit sends the question, user ID, thread ID, and run ID into the backend graph.

⭐ Key Features
🛡️ Multi-layer AI safety
🔐 Jailbreak & prompt-injection detection
🚨 Unsafe-intent detection
🔏 PII protection
⚠️ High-stakes query detection
📚 Evidence-based generation
🔬 Groundedness verification
🔄 Self-consistency checking
⚖️ Fairness & output-safety checks
⚙️ Centralized decision engine
📝 Complete request tracing
👨‍💻 Human review workflow
🧠 Background bias and reliability analysis
🔁 Human-controlled improvement loop
🖥️ Interactive Streamlit interface
🎯 Core Design Philosophy

The project is built around five principles:

1. Safety First
Potentially unsafe requests and outputs are detected before reaching the user.

2. Evidence-Based Answers
Generated responses are checked against supporting evidence.

3. Fail Safely
When the system cannot verify an answer, it can retry, abstain, or escalate instead of confidently returning unreliable information.

4. Full Auditability
Requests, scores, decisions, timings, and feedback are recorded.

5. Human-Controlled Learning
The system can identify improvements, but humans remain responsible for approving changes.

🏗️ Project Structure
ControlPlane Checker/
│
├── app.py                  # Streamlit application
├── graph.py                # Main pipeline orchestration
│
├── tier1.py                # Tier 1 orchestration
├── tier1_unsafe.py         # Unsafe-intent detection
├── tier1_*                 # Other Tier 1 detectors
│
├── tier2.py                # Tier 2 orchestration
├── tier2_checks.py         # Consistency & output-safety checks
├── tier2_groundedness.py   # Evidence verification
│
├── tier3.py                # Tier 3 orchestrator
├── tier3_jobs.py           # Background analysis jobs
├── tier3_trace.py          # Trace storage & processing
│
├── attempts.py             # Rewrite, attempt & cache logic
├── documents.py            # Document/evidence handling
├── review.py               # Human review workflow
├── examples_unsafe.py      # Safety examples
│
├── traces.jsonl            # Request traces
└── docs/
    └── architecture.png    # System architecture
🔄 Complete System

ControlPlane Checker is not simply a chatbot. It is a control and governance layer around an LLM that combines real-time safety detection, retrieval, answer verification, decision-making, tracing, human review, and background learning.

The key idea is:

Detect → Retrieve → Generate → Verify → Decide → Trace → Learn → Improve

This architecture allows the system to become safer and more reliable over time without allowing automated changes to silently affect users., human review, and background learning into one architecture.
