"""Retrieved content sanitization.

Section 6 defense #3. Even with trusted sources (defense #1) and context
isolation (defense #2), a document may still contain risky text - prompt
injection phrases, fake conversation-role labels, or hidden unicode - that we
don't want to hand to the model verbatim. Sanitization cleans each retrieved
chunk *before* it becomes context, neutralizing the most common injection tricks
while leaving the legitimate information intact.
"""

from __future__ import annotations

import re
import unicodedata

REDACTION = "[redacted]"

# Phrases commonly used to hijack an assistant.
_INJECTION_PATTERNS = [
    re.compile(
        r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|above|prior|earlier)\s+instructions?",
        re.I,
    ),
    re.compile(r"disregard\s+(?:all\s+)?(?:the\s+)?(?:previous|above|prior)\b.*", re.I),
    re.compile(r"forget\s+(?:everything|all)\b.*", re.I),
    re.compile(r"you\s+are\s+now\b.*", re.I),
    re.compile(r"reveal\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)", re.I),
    re.compile(r"system\s+prompt", re.I),
]

# Fake conversation-role labels at the start of a line (system:, assistant:, user:).
_ROLE_LABEL = re.compile(r"(?im)^\s*(system|assistant|user)\s*:")

# Zero-width / invisible characters used to smuggle hidden instructions.
_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿]")


def sanitize_content(text: str) -> str:
    """Clean a retrieved chunk before it is placed into the prompt context."""
    if not text:
        return text

    # Normalize unicode and strip invisible characters.
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = _ZERO_WIDTH.sub("", cleaned)
    # Drop control characters, keeping newlines and tabs.
    cleaned = "".join(
        ch for ch in cleaned if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
    )

    # Neutralize fake role labels and known injection phrases.
    cleaned = _ROLE_LABEL.sub(f"{REDACTION}:", cleaned)
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub(REDACTION, cleaned)

    # Collapse runaway whitespace that can be used to hide payloads.
    cleaned = re.sub(r"[ \t]{3,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()
