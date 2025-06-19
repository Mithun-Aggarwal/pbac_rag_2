# utils.py

import re

def clean_text(text: str) -> str:
    """
    Strips extra whitespace, newlines, and makes text LLM-friendly.
    
    Args:
        text (str): Raw chunk text

    Returns:
        str: Cleaned and trimmed version
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def shorten_text(text: str, max_chars: int = 300) -> str:
    """
    Truncates a string to a max character length with ellipsis.

    Args:
        text (str): Full text
        max_chars (int): Character limit

    Returns:
        str: Truncated summary
    """
    return text if len(text) <= max_chars else text[:max_chars] + "..."

def format_scores(chunks: list, top_k: int = 3) -> str:
    """
    Prints formatted similarity scores for chunks.
    """
    lines = []
    for i, chunk in enumerate(chunks[:top_k]):
        lines.append(f"[{i+1}] sim: {chunk['score']:.4f} — {shorten_text(chunk['text'])}")
    return "\n".join(lines)
