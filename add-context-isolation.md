feature: add-context-isolation
goal: In this lesson you'll add context isolation so retrieved document content is kept structurally separate from the assistant's instructions.
subtitle: Separate instructions from retrieved content so documents can't give orders
conclusion: And that's context isolation - the second Section 6 defense. Retrieved chunks are now wrapped in explicit untrusted-content markers, and the model is told to treat anything inside them as data, not commands. Next we'll sanitize the retrieved content itself to neutralize risky text.
sequence: context_isolation.py -> llm_service.py
target_learner: Intermediate

## new: context_isolation.py

A brand-new module that defines the boundary markers, the isolation guidance added to the system prompt, and `isolate_context()`, which wraps retrieved content and neutralizes any forged boundary markers hidden inside a document.

## modified: llm_service.py

Import the isolation helpers, append the isolation guidance to the system prompt, and wrap the retrieved context with `isolate_context` before it goes into the user message.

```python
from context_isolation import ISOLATION_GUIDANCE, isolate_context
```

```python
                    "Keep the answer clear, beginner-friendly, and concise. "
                    + ISOLATION_GUIDANCE
```

```python
                    "Retrieved context:\n"
                    f"{isolate_context(context)}\n\n"
```
