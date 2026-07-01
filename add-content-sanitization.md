feature: add-content-sanitization
goal: In this lesson you'll add a sanitization layer that cleans risky text out of retrieved chunks before they reach the model.
subtitle: Clean or neutralize risky retrieved text before it becomes context
conclusion: And that's content sanitization - the third and final Section 6 defense. Retrieved chunks are now scrubbed of prompt-injection phrases, fake role labels, and hidden unicode before they enter the prompt. Combined with trusted source validation and context isolation, the assistant no longer blindly trusts the documents it retrieves.
sequence: content_sanitizer.py -> rag_pipeline.py
target_learner: Intermediate

## new: content_sanitizer.py

A brand-new module whose `sanitize_content()` normalizes unicode, strips invisible characters, and redacts known injection phrases and fake conversation-role labels, while leaving the legitimate information intact.

## modified: rag_pipeline.py

Import the sanitizer and apply it to each retrieved chunk inside `build_context`, so cleaned text is what gets assembled into the model's context.

```python
from content_sanitizer import sanitize_content
```

```python
                    f"Similarity: {result.score:.4f}",
                    "Text:",
                    sanitize_content(record.text),
```
