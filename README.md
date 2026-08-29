
Deployed- https://controlplaneai-z79ax6heyhlsihhanz2xns.streamlit.app/
# ControlPlane Checker

ControlPlane Checker is a multi-layer AI safety, reliability, and governance system for enterprise chatbot applications. Instead of sending every user query directly to an LLM, the system evaluates the request, prepares and retrieves supporting evidence, verifies the generated response, makes a final decision, and analyses historical behaviour for continuous improvement.

## Overview

The system is organized into three tiers:

- **Tier 1 — Fast Risk Detection:** Performs rapid checks for injection/jailbreak attempts, unsafe intent, PII, high-stakes topics, and usage/cost abuse.
- **Tier 2 — Deep Verification:** Performs deeper checks on generated responses, including groundedness, self-consistency, and output safety/fair wording.
- **Tier 3 — Background Analysis:** Analyses stored traces for bias, cost patterns, repeated queries, feedback patterns, and source conflicts, and proposes improvements under human control.

The overall goal is to make AI responses **safer, more reliable, explainable, and auditable**.

## System Architecture

![ControlPlane Architecture](docs/architecture.png)

The request-processing pipeline follows this general sequence:

**User Query → Tier 1 → Query Processing → Evidence Retrieval → LLM Generation → Output Checks → Tier 2 → Decision Engine → Final Answer → Trace & Feedback → Tier 3**

## Tier 1 — Fast Risk Detection

Tier 1 is the first line of defence. It evaluates requests before expensive processing and produces an overall risk score, risk band, recommended action, and strict-mode decision.

The main detectors are:

1. **Injection / Jailbreak Detection**
2. **Unsafe Intent Detection**
3. **PII Detection**
4. **High-Stakes Topic Detection**
5. **Usage / Cost Abuse Detection**

The unsafe-intent detector compares a query against both unsafe and safe examples. This helps distinguish genuinely unsafe requests from legitimate questions that happen to contain risky vocabulary.

High-stakes detection covers areas such as financial, HR, legal, medical, security, and customer-commitment questions. These queries can trigger stricter verification because an incorrect answer may have real-world consequences.

## Query Processing

After the initial safety checks, the request passes through several preparation stages.

### Query Rewriting

The system converts incomplete questions, short forms, and relative dates into clearer standalone queries.

Example:

```text
"What is the WFH policy?"
        ↓
"What is the work from home policy?"
```

Rewriting occurs before retrieval and cache lookup so that later stages work with a more complete query.

### Attempt Detection

The system checks whether the user is repeating or correcting a previous request.

Examples:

```text
"That's not what I asked."
"Try again."
"Again."
```

This prevents the system from simply returning a cached answer that the user has already rejected.

### Cache Lookup

The system checks whether a sufficiently similar verified answer already exists. A strict similarity threshold is used because serving an incorrect cached answer can be costly.

## Evidence Retrieval

If there is no suitable cached answer, the system retrieves relevant documents and ranks the available evidence.

The purpose is to provide the LLM with relevant supporting information so that responses are based on available evidence rather than relying entirely on model knowledge.

The selected evidence is passed to the model during answer generation.

## Tier 2 — Deep Verification

Tier 2 is the deeper verification layer. It is used when additional checking is required, particularly for risky or high-stakes responses.

### Groundedness

Groundedness checks whether the claims made in the generated answer are supported by the retrieved evidence.

If claims are unsupported, the system can retry the generation instead of immediately returning a potentially unreliable answer.

### Self-Consistency

The system can generate the answer twice using the same evidence and compare important facts.

The check focuses on facts such as:

- Numbers
- Durations
- Dates
- Entities

If two generations disagree on an important fact, the response may be unreliable.

### Output Safety and Fair Wording

Tier 1 evaluates the user's question, while this check evaluates the generated answer.

It checks for issues such as:

- Unsafe advice
- Unfair generalisations
- Excessive certainty
- Problematic wording

## Decision Engine

The Decision Engine combines the results from Tier 1 and Tier 2 and determines the appropriate action.

Possible outcomes include:

- **ALLOW**
- **AUTO-FIX**
- **RETRY**
- **ABSTAIN**
- **ESCALATE**
- **BLOCK**

For example, if the generated response contains unsupported claims, the system can retry the generation. If the claims remain unsupported after the retry, the system can abstain rather than repeatedly generating an unreliable answer.

The decision also includes a trust label and reasons, making the result easier to understand and audit.

## Trace Store

After a request is processed, the system stores a trace containing information such as:

- User question
- Generated answer
- Tier 1 scores
- Tier 2 scores
- Detector results
- Decision
- Timing information
- Token usage
- Trust label
- User feedback

The trace store allows Tier 3 to analyse historical behaviour without interfering with live requests.

The prototype uses a local JSONL trace store. Its structure is designed so that the stored information can be mapped to a production tracing system.

## Tier 3 — Background Learning and Monitoring

Tier 3 does not run directly on a live request. It works on historical traces and identifies patterns that cannot be reliably detected from a single request.

The background jobs analyse:

| Analysis | Purpose |
|---|---|
| **Bias Patterns** | Detect whether similar users or counterfactual requests are treated differently |
| **Cost Patterns** | Identify token usage and inefficient resource consumption |
| **Feedback Clusters** | Identify topics that repeatedly receive negative feedback |
| **Repeated Queries** | Find questions that are asked repeatedly |
| **Conflict Rate** | Detect topics where available sources disagree |

The bias analysis can create counterfactual versions of real questions by changing attributes such as names while keeping the rest of the question unchanged. Differences in risk scores can then be investigated as possible bias signals.

## Human-Controlled Improvement

A central design principle of Tier 3 is:

> **Tier 3 proposes changes; it never applies them automatically.**

The system can propose changes to areas such as:

- Detector thresholds
- Source-trust lists
- Retrieval behaviour
- Caching
- Usage limits

Proposed changes require human review and controlled validation before they can affect users. This includes safeguards such as shadow-mode testing and held-out validation.

The improvement process is:

**Traces → Analysis → Finding → Proposal → Human Review → Shadow Testing → Validation → Approved Change**

## Streamlit Application

The project includes a Streamlit web interface called **ControlPlane Checker**.

Run the application with:

```bash
streamlit run app.py
```

The application provides four main tabs:

### Ask

Allows a user to enter a question and run it through the complete pipeline.

### Human Review

Allows reviewers to examine flagged cases and provide verified feedback.

### Tier 3

Displays background analysis and improvement proposals generated from historical traces.

### Traces

Displays the stored request records used by the background analysis.

When the user clicks **Run the pipeline**, Streamlit passes the question, user ID, thread ID, and run ID into the backend graph.

## Project Structure

```text
ControlPlane Checker/
│
├── app.py                  # Streamlit application
├── graph.py                # Main pipeline orchestration
│
├── tier1.py                # Tier 1 orchestration
├── tier1_*.py              # Tier 1 detector modules
│
├── tier2.py                # Tier 2 orchestration
├── tier2_checks.py         # Consistency and output-safety checks
├── tier2_groundedness.py   # Evidence verification
│
├── tier3.py                # Tier 3 orchestrator
├── tier3_jobs.py           # Background analysis jobs
├── tier3_trace.py          # Trace storage and processing
│
├── attempts.py             # Query rewriting, attempts, and cache
├── documents.py            # Document and evidence handling
├── review.py               # Human review workflow
├── examples_unsafe.py      # Safety examples
│
├── traces.jsonl            # Request traces
└── docs/
    └── architecture.png    # System architecture image
```

## Key Features

- **Multi-layer AI safety**
- **Jailbreak and prompt-injection detection**
- **Unsafe-intent detection**
- **PII protection**
- **High-stakes query detection**
- **Evidence-based generation**
- **Groundedness verification**
- **Self-consistency checking**
- **Output safety and fair-wording checks**
- **Centralized decision engine**
- **Retry and abstention mechanisms**
- **Request tracing and feedback**
- **Human review workflow**
- **Background bias and reliability analysis**
- **Human-controlled improvement loop**
- **Interactive Streamlit interface**

## Design Principles

### Safety First

Potentially unsafe requests and generated outputs are identified before they can cause harm.

### Evidence-Based Responses

Generated answers are checked against retrieved evidence.

### Fail Safely

When the system cannot verify an answer, it can retry, abstain, or escalate instead of confidently returning an unreliable response.

### Full Auditability

Requests, detector scores, decisions, timings, and feedback are recorded for later analysis.

### Human-Controlled Learning

Tier 3 can identify potential improvements, but changes are not silently applied to the live system.

## Complete System

The core concept of ControlPlane Checker is:

**Detect → Retrieve → Generate → Verify → Decide → Trace → Analyse → Improve**

The result is a control layer around an enterprise LLM that combines real-time safety detection, evidence verification, decision-making, tracing, human review, and controlled background improvement.
