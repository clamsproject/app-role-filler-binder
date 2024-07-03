"""
Utility functions for cleaning OCR data.
"""
from typing import List
import re


def has_alnum(string: str) -> bool:
    """Returns True if the input string contains any alphanumeric characters."""
    return any(char.isalnum() for char in string)


def has_alpha(string: str) -> bool:
    """Returns True if the input string contains any alpha characters."""
    return any(char.isalpha() for char in string)


def contains_year(string: str) -> bool:
    """Returns True if the string contains a valid year from 1000-2999."""
    return re.search(r"^[12][0-9]{3}$", string) is not None


def segment_string(string: str):
    """Inserts whitespace in a string to segment improperly merged words"""
    string = re.sub(r"(?<=[A-Za-z][,.])(?=[A-Za-z])", " ", string)
    return re.sub(r"(?<=[a-z]{3})(?=[A-Z])", " ", string)


def clean_ocr(text_document: str) -> List[str]:
    """Cleans ocr text document"""
    allowable_chars = {r'&'}
    cleaned = []
    for line in text_document.split('\n'):
        line = line.strip()
        if not has_alnum(line):
            continue
        else:
            line = segment_string(line)
            line = [
                w for w in line.split() if (len(w) > 1 and has_alpha(w)) or w in allowable_chars or contains_year(w)
            ]
        if line:
            cleaned.extend(line)
    return cleaned
