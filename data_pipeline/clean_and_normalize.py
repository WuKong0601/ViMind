import unicodedata
import re
from typing import Optional


def normalize_vietnamese_text(text: str) -> str:
    """Normalize Vietnamese text to Unicode NFC and clean excessive whitespaces and junk artifacts."""
    if not text or not isinstance(text, str):
        return ""
    # Normalize Unicode characters (compose diacritics)
    text = unicodedata.normalize("NFC", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove common crawled promotional domains
    text = re.sub(r"(?i)(montoan\.com\.vn|loigiaihay\.com|vietjack\.com|tuyensinh247\.com)", "", text)
    # Remove HTML tags if present
    text = re.sub(r"<[^>]+>", " ", text)
    # Replace non-breaking spaces and special spaces with standard space
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    # Normalize multiple whitespaces (keep single spaces)
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize multiple newlines (at most 2 consecutive newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_textbook_qa(question: str, answer: str) -> str:
    """Format textbook question and answer into coherent educational reading passage."""
    q_clean = normalize_vietnamese_text(question)
    a_clean = normalize_vietnamese_text(answer)
    if not q_clean or not a_clean:
        return ""
    # Filter out empty or trivial questions
    if len(q_clean.split()) < 4 and len(a_clean.split()) < 10:
        return ""
    return f"Bài học / Bài tập: {q_clean}\n\nNội dung / Lời giải chi tiết:\n{a_clean}"


def is_valid_text(text: str, min_words: int = 15, max_repeat_ratio: float = 0.5) -> bool:
    """Check if the text has at least a minimum number of words and isn't spam/repetitive."""
    if not text:
        return False
    words = text.split()
    if len(words) < min_words:
        return False

    # Check for excessive repetition (e.g. loops of identical words)
    if len(words) > 40:
        unique_words = set(w.lower() for w in words)
        ratio = len(unique_words) / len(words)
        if ratio < (1.0 - max_repeat_ratio):
            return False

    return True
