import unicodedata
import re

def normalize_vietnamese_text(text: str) -> str:
    """Normalize Vietnamese text to Unicode NFC and remove excessive whitespace."""
    if not text or not isinstance(text, str):
        return ""
    # Normalize Unicode characters (compose diacritics)
    text = unicodedata.normalize("NFC", text)
    # Remove multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_valid_text(text: str, min_words: int = 10) -> bool:
    """Check if the text has at least a minimum number of words."""
    if not text:
        return False
    words = text.split()
    return len(words) >= min_words
