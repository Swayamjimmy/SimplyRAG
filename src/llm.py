```python
# src/citations.py

import re

from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel


# ============================================================
# Citation Data Models
# ============================================================

class Citation(BaseModel):
    """
    Links a claim to its source passage
    with verification status.
    """

    source_doc: str
    page_number: int
    passage: str
    verified: bool = False


class CitationResponse(BaseModel):
    """
    Complete response containing:
        - generated answer
        - extracted citations
    """

    answer: str
    citations: list[Citation]


# ============================================================
# Citation Extraction
# ============================================================

def extract_citations(
    text: str
) -> list[tuple[str, int]]:
    """
    Parse inline [N] markers from LLM output.

    Returns:
        list of:
            (claim_sentence, citation_index)
    """

    # Gemini/LangChain may return response.content
    # as a list of content blocks rather than a plain string.
    if not isinstance(text, str):
        if isinstance(text, list):
            text = "\n".join(
                item.get("text", "")
                if isinstance(item, dict)
                else str(item)
                for item in text
            )
        else:
            text = str(text)

    results = []

    # Split output into sentences.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    for sentence in sentences:
        # Find all citation markers such as [1], [2], [15].
        markers = re.findall(
            r"\[(\d+)\]",
            sentence
        )

        for marker in markers:
            # Remove citation markers to obtain the clean claim.
            claim = re.sub(
                r"\[\d+\]",
                "",
                sentence
            ).strip()

            results.append(
                (
                    claim,
                    int(marker)
                )
            )

    return results


# ============================================================
# LLM Content Normalization
# ============================================================

def _extract_response_text(content) -> str:
    """
    Convert different LangChain/Gemini response.content
    formats into a plain string.

    Gemini/LangChain can return content as:
        - str
        - list[str]
        - list[dict]
        - other provider-specific formats
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                # Common Gemini/LangChain content-block format.
                if "text" in item:
                    parts.append(str(item["text"]))

                # Handle nested text structures if present.
                elif "content" in item:
                    parts.append(str(item["content"]))

            else:
                parts.append(str(item))

        return "".join(parts)

    return str(content)


# ============================================================
# Citation Verification
# ============================================================

def verify_citation(
    claim: str,
    passage: str,
    llm: BaseChatModel
) -> bool:
    """
    Verify whether a source passage supports a claim
    using the Gemini chat model.

    Gemini is instructed to return ONLY:
        yes
        or
        no
    """

    verification_prompt = f"""
Determine whether the following source passage supports the claim.

Claim:

{claim}

Source passage:

{passage}

Question:

Does the source passage provide evidence that supports
the claim?

Answer with ONLY one word:

yes
or
no
"""

    response = llm.invoke(
        verification_prompt.strip()
    )

    # Gemini/LangChain may return response.content
    # as a string OR a list of content blocks.
    answer = _extract_response_text(
        response.content
    ).strip().lower()

    # Normalize possible punctuation/formatting.
    answer = re.sub(
        r"[^a-z]",
        "",
        answer
    )

    return answer == "yes"


# ============================================================
# Citation Accuracy
# ============================================================

def compute_citation_accuracy(
    citations: list[Citation]
) -> float:
    """
    Return the percentage of citations
    verified as grounded in source text.
    """

    if not citations:
        return 0.0

    verified_count = sum(
        1
        for citation in citations
        if citation.verified
    )

    return (
        verified_count
        / len(citations)
    )
```
