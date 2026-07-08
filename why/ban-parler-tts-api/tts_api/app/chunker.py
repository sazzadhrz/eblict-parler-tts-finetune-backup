import re
from typing import List


# Ordered split patterns: tried from coarsest to finest before word fallback.
# Lookahead keeps the delimiter attached to the NEXT chunk (conjunctions stay with their clause).
_PATTERNS = [
    r"(?<=[,;:])\s+",                                                              # clause boundaries
    r"\s+(?=\b(?:and|but|or|because|although|however|so|yet|while|when|if)\b)",   # conjunctions
]


def chunk_text(text: str, tokenizer, limit: int = 50) -> List[str]:
    """
    Split text into ordered chunks each within `limit` tokens.
    Prefers sentence → clause → conjunction boundaries; falls back to greedy
    word-based splitting. Never overlaps, never repeats content.
    """
    text = text.strip()
    if not text:
        return []

    # Level 1: sentence boundaries (Bengali danda included)
    sentences = [s.strip() for s in re.split(r"(?<=[।.!?])\s+", text) if s.strip()]
    if not sentences:
        sentences = [text]

    out: List[str] = []
    for sentence in sentences:
        _chunk(sentence, tokenizer, limit, out)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _token_count(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _chunk(text: str, tokenizer, limit: int, out: List[str]) -> None:
    """
    Split `text` into chunks ≤ limit tokens using pattern hierarchy, then
    greedy word split. Appends results to `out` in order.
    """
    if _token_count(text, tokenizer) <= limit:
        out.append(text)
        return

    # Try each pattern level in order
    for pattern in _PATTERNS:
        parts = [p.strip() for p in re.split(pattern, text, flags=re.IGNORECASE) if p.strip()]
        if len(parts) > 1:
            _accumulate(parts, tokenizer, limit, out)
            return

    # Fallback: greedy forward word split (no overlap)
    _word_split(text, tokenizer, limit, out)


def _accumulate(parts: List[str], tokenizer, limit: int, out: List[str]) -> None:
    """
    Greedily merge parts into chunks ≤ limit.
    If a single part still exceeds limit, recurse into it for finer splitting.
    """
    accumulated = ""
    for part in parts:
        candidate = (accumulated + " " + part).strip() if accumulated else part
        if _token_count(candidate, tokenizer) <= limit:
            accumulated = candidate
        else:
            if accumulated:
                out.append(accumulated)           # accumulated is known to fit
            if _token_count(part, tokenizer) <= limit:
                accumulated = part                # start fresh accumulation
            else:
                _chunk(part, tokenizer, limit, out)  # recurse for finer split
                accumulated = ""
    if accumulated:
        out.append(accumulated)


def _word_split(text: str, tokenizer, limit: int, out: List[str]) -> None:
    """
    Greedy forward word split. Takes as many words as fit within limit,
    then advances strictly past them — no overlap, guaranteed progress.
    """
    words = text.split()
    if not words:
        return

    start = 0
    while start < len(words):
        # Extend greedily to the right while within limit
        end = start + 1
        while end <= len(words):
            if _token_count(" ".join(words[start:end]), tokenizer) > limit:
                break
            end += 1
        end -= 1  # last valid end

        if end <= start:
            end = start + 1  # take at least one word even if it exceeds limit

        out.append(" ".join(words[start:end]))
        start = end  # strict forward progress — no overlap
