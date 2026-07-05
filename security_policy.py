"""Security policy and enforcement for RAG answer generation.

Section 6 defense #4 (enforcement layer). This is the piece that was missing:
the earlier defenses could *detect* risk, but nothing stopped a risky or
untrusted chunk from being used to generate the answer. This module turns
detection into a decision.

It provides three things:

1. ``classify_trust`` - decide whether a source is a trusted authority or an
   untrusted upload. Uploaded / user-provided documents default to *untrusted*.
2. ``is_sensitive_question`` - recognize business-sensitive topics (refunds,
   legal, finance, security, HR, compliance, contracts) that must only be
   answered from trusted sources.
3. ``apply_security_policy`` - the pure enforcement function. Given retrieved
   results and the question, it returns the chunks that are *allowed* into the
   prompt and the chunks that were *quarantined*, with reasons. When nothing is
   allowed, the pipeline returns a safe fallback instead of calling the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from config import settings
from injection_detector import assess_injection

TRUSTED = "trusted"
UNTRUSTED = "untrusted"

# Business-sensitive topics. A question touching any of these may only be
# answered from a trusted source; an untrusted document cannot be treated as
# authoritative policy.
_SENSITIVE_TERMS = [
    "refund", "refunds", "chargeback",
    "policy", "policies", "terms", "warranty", "guarantee",
    "contract", "contracts", "agreement", "sla",
    "legal", "law", "liability", "compliance", "regulation", "regulatory",
    "finance", "financial", "billing", "invoice", "payment", "pricing", "price", "discount",
    "security", "credential", "password", "secret",
    "hr", "salary", "compensation", "payroll", "benefits",
    "medical", "health", "insurance",
]
_SENSITIVE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in _SENSITIVE_TERMS) + r")\b", re.I
)

SAFE_FALLBACK = (
    "I found documents related to your question, but I can't answer from them safely. "
    "The relevant content was either flagged as containing suspicious instructions "
    "(possible prompt injection) or came from an untrusted source that shouldn't be "
    "treated as authoritative for this kind of question. I won't repeat unverified "
    "claims as if they were policy. Please provide the information from an approved, "
    "trusted document, or verify it with the responsible team before relying on it."
)


def classify_trust(source: str | Path) -> str:
    """Return ``TRUSTED`` or ``UNTRUSTED`` for a document source path.

    A source is trusted only if it lives under one of the configured
    *trusted authority* directories. Everything else - notably user uploads -
    defaults to untrusted, so a poisoned upload is never treated as official
    policy just because it was successfully ingested.
    """
    try:
        resolved = Path(source).resolve()
    except OSError:
        return UNTRUSTED

    for root in settings.trusted_authority_dirs:
        try:
            root_resolved = Path(root).resolve()
        except OSError:
            continue
        try:
            resolved.relative_to(root_resolved)
            return TRUSTED
        except ValueError:
            continue
    return UNTRUSTED


def is_sensitive_question(question: str) -> bool:
    """True if the question is about a business-sensitive topic."""
    return bool(_SENSITIVE_RE.search(question or ""))


@dataclass
class ChunkDecision:
    """The security decision made about one retrieved chunk."""

    result: object  # SearchResult; kept loose to avoid an import cycle
    allowed: bool
    reason: str
    trust: str
    injection_score: int


@dataclass
class PolicyOutcome:
    allowed: list  # list[SearchResult] safe to use for generation
    decisions: list[ChunkDecision]
    sensitive: bool

    @property
    def quarantined(self) -> list[ChunkDecision]:
        return [d for d in self.decisions if not d.allowed]

    @property
    def has_context(self) -> bool:
        return bool(self.allowed)


def apply_security_policy(results: list, question: str) -> PolicyOutcome:
    """Decide which retrieved chunks may be used to answer ``question``.

    A chunk is quarantined (excluded from the model's context) when:

    * it scores at/above the injection threshold (prompt injection), or
    * the question is business-sensitive and the chunk's source is untrusted.

    Everything else is allowed. This runs *before* generation, so quarantined
    content can never influence the answer.
    """
    sensitive = is_sensitive_question(question)
    threshold = settings.injection_block_threshold

    decisions: list[ChunkDecision] = []
    allowed: list = []

    for result in results:
        record = result.record
        trust = classify_trust(record.source)
        assessment = assess_injection(record.text, threshold=threshold)

        if assessment.is_injection:
            reason = "Quarantined: prompt injection detected (" + "; ".join(
                assessment.reasons[:3]
            ) + ")"
            decisions.append(
                ChunkDecision(result, False, reason, trust, assessment.score)
            )
            continue

        if sensitive and trust == UNTRUSTED:
            reason = (
                "Quarantined: sensitive question answered from an untrusted "
                "source; a trusted/approved document is required."
            )
            decisions.append(
                ChunkDecision(result, False, reason, trust, assessment.score)
            )
            continue

        decisions.append(
            ChunkDecision(result, True, "Allowed", trust, assessment.score)
        )
        allowed.append(result)

    return PolicyOutcome(allowed=allowed, decisions=decisions, sensitive=sensitive)
