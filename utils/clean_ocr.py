"""
Utility functions for cleaning OCR data.
"""

import pandas as pd


def has_alnum(string: str) -> bool:
    """Returns True if any character in the string is an alphanumeric."""
    return any(char.isalnum() for char in string)


def has_alpha(string: str) -> bool:
    """Returns True if any character in the string contains alpha characters."""
    return any(char.isalpha() for char in string)


def clean_ocr(text_document: str) -> list[str]:
    """Cleans ocr text document"""
    allowable_chars = {r'&'}
    cleaned = []
    for line in text_document.split('\n'):
        if not has_alpha(line):
            continue
        else:
            line = [s for s in line.split() if (len(s) > 1 and has_alpha(s)) or s in allowable_chars]
        cleaned.extend(line)
    return cleaned


if __name__ == '__main__':
    df = pd.read_csv('doctr-preds.csv', usecols=['textdocument'])
    df['cleaned'] = df['textdocument'].map(clean_ocr)
    df.to_csv('cleaned_ocr_chyrons.csv', index=True)
