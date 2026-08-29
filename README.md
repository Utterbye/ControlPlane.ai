ControlPlane Checker
Overview

ControlPlane Checker is a multi-layer AI safety, reliability, and governance system designed for enterprise chatbot applications. The main objective is to prevent unsafe or unreliable responses from reaching users while also making the complete AI pipeline auditable, explainable, and continuously improvable.

Instead of directly sending every user query to an LLM, the system processes the request through three different tiers. Each tier has a specific responsibility: Tier 1 provides fast risk detection, Tier 2 performs deeper verification, and Tier 3 analyses historical behaviour and proposes improvements.

System Architecture
                    USER QUERY
                        │
                        ▼
                ┌─────────────────┐
                │     TIER 1      │
                │ Fast Risk Check │
                └────────┬────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
           High Risk             Normal
              │                     │
              ▼                     ▼
          Block/Care        Query Processing
                                    │
                                    ▼
                         Rewrite → Attempt → Cache
                                    │
                                    ▼
                           Evidence Retrieval
                                    │
                                    ▼
                              Re-ranking
                                    │
                                    ▼
                              LLM Answer
                                    │
                                    ▼
                         Tier 1 Output Check
                                    │
                                    ▼
                ┌──────────────────────────┐
                │         TIER 2           │
                │   Deep Verification     │
                └────────────┬─────────────┘
                             │
                             ▼
                      Decision Engine
                             │
                   ┌─────────┼─────────┐
                   ▼         ▼         ▼
                 Allow     Retry     Abstain
                             │
                             ▼
                        Final Answer
                             │
                             ▼
                      Trace + Feedback
                             │
                             ▼
                ┌─────────────────────────┐
                │         TIER 3          │
                │ Background Analysis    │
                └─────────────────────────┘
                             │
                             ▼
                     Improvement Proposals
                             │
                             ▼
                       Human Approval
Tier 1 — Fast Risk Detection

Tier 1 is the first line of defence. Its purpose is to analyse a request quickly before expensive processing takes place.

It contains five major detectors:

Injection / Jailbreak Detection
Unsafe Intent Detection
PII Detection
High-Stakes Detection
Usage / Cost Detection

Each detector produces a score. These scores are combined into an overall risk score, risk band, and recommended action.

For unsafe-intent detection, the system does not simply search for dangerous words. It compares the query against both unsafe and safe examples, which helps prevent legitimate questions containing words such as attack, violence, or security from being incorrectly classified.

The high-stakes detector handles questions where an incorrect answer could have real consequences, such as financial, HR, legal, medical, security, or customer-commitment questions. Instead of automatically blocking these questions, the system raises the verification standard.

Query Processing

After the initial safety check, the query goes through several preparation stages.

1. Query Rewriting

The system converts incomplete or ambiguous questions into clearer standalone queries. It can expand short forms and resolve relative dates.

For example:

"What is the WFH policy?"
            ↓
"What is the work from home policy?"

This happens before cache lookup and retrieval because both depend on having a meaningful query.

2. Attempt Detection

The system checks whether the user is repeating or pushing back on a previous question.

For example:

"That's not what I asked."
"Try again."
"Again."

This prevents the system from simply returning a cached answer that the user has already rejected.

3. Cache Lookup

The system checks whether a sufficiently similar verified answer already exists. A strict similarity threshold is used because returning the wrong cached answer can be costly.

Evidence Retrieval and Generation

If there is no suitable cached answer, the system retrieves relevant documents and ranks the available evidence.

The purpose is to make the LLM answer using company-approved evidence rather than relying entirely on its internal knowledge.

The retrieved evidence is then passed to the model for answer generation.

After generation, the answer is checked again before it reaches the user.

Tier 2 — Deep Verification

Tier 2 is the second line of defence. It performs more expensive checks when additional verification is necessary.

The main checks are:

Groundedness

Groundedness asks:

Is the answer actually supported by the retrieved evidence?

If claims are unsupported, the system can retry the generation rather than immediately returning a potentially hallucinated answer.

Self-Consistency

The system can generate the answer twice using the same evidence and compare the important facts.

For example:

Answer 1: "The notice period is 30 days."
Answer 2: "The notice period is 60 days."

The difference indicates that the model may not be reliable about that fact. The check focuses on facts such as numbers and durations rather than requiring identical wording.

Output Safety and Fair Wording

Tier 1 examines the question, whereas this check examines the generated answer.

It looks for things such as:

unsafe advice
unfair generalisations
excessive certainty
problematic wording

This is important because a harmless-looking question can still result in an unsafe or unfair answer.

Decision Engine

After Tier 1 and Tier 2 results are available, the Decision Engine determines what should happen.

Possible actions include:

ALLOW
AUTO-FIX
RETRY
ABSTAIN
ESCALATE
BLOCK

For example, if the answer contains unsupported claims, the system can retry once. If the claims are still unsupported after the retry, the system can abstain instead of repeatedly generating unreliable answers.

The decision engine also produces a trust label and records the reason behind the decision, making the result easier to audit.

Trace Store

Every completed request is recorded in a trace store.

A trace can contain:

User question
Generated answer
Tier 1 scores
Tier 2 scores
Detector information
Decision
Timings
Token usage
Thread/user information
Trust label
User feedback

The important design principle is that the trace is written after the response, so background analysis does not interfere with live requests.

The prototype uses a local JSONL trace store, while the structure is designed so it can map to a production tracing system such as LangSmith.

Tier 3 — Background Learning and Monitoring

Tier 3 does not run directly on a user's request. It works on historical traces and looks for patterns that cannot be detected from a single query.

It analyses:

Bias patterns — whether similar users receive different treatment
Cost patterns — where token usage and resources are being consumed
Feedback clusters — which topics repeatedly receive negative feedback
Repeated queries — questions that users repeatedly ask
Conflict rate — topics where sources disagree

These jobs are specifically designed to analyse many requests together rather than one request at a time.

For example, the bias job can create counterfactual versions of a question by changing an attribute such as a name while keeping everything else the same. If the risk score changes significantly, it becomes a possible bias signal.

Human-Controlled Improvement

One of the most important design principles is:

Tier 3 proposes changes; it never applies them automatically.

The system may suggest changes to:

detector thresholds
source-trust lists
retrieval behaviour
caching
usage limits

But proposed changes must go through human review, shadow mode, and validation before they affect users.

This creates a controlled improvement loop:

User Requests
      ↓
Traces
      ↓
Tier 3 Analysis
      ↓
Findings
      ↓
Improvement Proposal
      ↓
Human Review
      ↓
Shadow Testing
      ↓
Validation
      ↓
Approved Change
Streamlit Application

The project is presented through a Streamlit-based web application called ControlPlane Checker. The application can be started using:

streamlit run app.py

The Streamlit interface provides four main tabs:

1. Ask

This is the main interface where a user enters a question and runs it through the complete pipeline. The application sends the question, user ID, thread ID, and run ID into the graph.

2. Human Review

This interface allows reviewers to examine flagged cases and provide verified feedback. This feedback becomes useful for improving the system.

3. Tier 3

This displays the background analysis performed by Tier 3 and shows the improvement proposals generated from the findings.

4. Traces

This displays the stored request records that are later used by Tier 3 for analysis.

The application therefore acts as the visual front-end of the complete control-plane pipeline, allowing the user to see how a query moves through safety checks, retrieval, generation, verification, decision-making, tracing, and background improvement.

Key Design Principles

The project is built around five major principles:

Safety first — risky requests and unsafe outputs are detected before reaching the user.
Evidence-based answers — generated answers are checked against retrieved evidence.
Fail safely — unsupported or unreliable answers can be retried or rejected.
Full auditability — requests, decisions, scores, and feedback are recorded.
Human-controlled improvement — Tier 3 can recommend changes, but humans remain responsible for approving them.

Overall, ControlPlane Checker acts as a control layer around an enterprise LLM, combining real-time safety checks, evidence verification, decision-making, tracing, human review, and background learning into one architecture.
