import re
import os
from pydantic import BaseModel
from groq import Groq


# Data model for a single citation
class Citation(BaseModel):
    """Links a claim to its source passage with verification status."""
    source_doc: str
    page_number: int
    passage: str
    verified: bool = False


# Structured response containing the answer and all citations
class CitationResponse(BaseModel):
    """Complete response with answer text and verified citation list."""
    answer: str
    citations: list[Citation]

def extract_citations(text: str) -> list[tuple[str, int]]:
    """Parse inline [N] markers from LLM output.

    Returns list of (claim_sentence, citation_index) tuples.
    """
    results = []
    # Split text into sentences at period/question/exclamation boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        # Find all [N] markers in this sentence
        markers = re.findall(r'\[(\d+)\]', sentence)
        for marker in markers:
            # Remove citation markers to get the clean claim text
            claim = re.sub(r'\[\d+\]', '', sentence).strip()
            results.append((claim, int(marker)))

    return results

def verify_citation(claim: str, passage: str, llm: Groq) -> bool:
    """Check if the cited passage supports the claim using Groq LLM."""
    # Build an entailment prompt asking if the source backs the claim
    verification_prompt = (
        "Determine if the following source passage supports the claim.\n\n"
        f"Claim: {claim}\n\n"
        f"Source passage: {passage}\n\n"
        "Does the source passage provide evidence that supports this claim?\n"
        "Answer with ONLY 'yes' or 'no'."
    )

    response = llm.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": verification_prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content.strip().lower()
    return answer == "yes"


def compute_citation_accuracy(citations: list[Citation]) -> float:
    """Return percentage of citations verified as grounded in source text."""
    if not citations:
        return 0.0
    verified_count = sum(1 for c in citations if c.verified)
    return verified_count / len(citations)