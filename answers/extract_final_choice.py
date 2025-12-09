# answers/extract_final_choice.py
"""
Unified answer extraction for multiple-choice questions.

This module provides a single source of truth for extracting MCQ answers
from model-generated text, with strong regex patterns and fallbacks.
"""

from typing import Optional, Tuple, List, Set
import re

CHOICE_SET_DEFAULT = {"A", "B", "C", "D"}

# Order matters; first match wins.
# Patterns are ordered from most explicit to most general.
_PATTERNS: List[re.Pattern] = [
    # Most explicit: "Final Answer: X" or "Final answer: X"
    re.compile(r'(?i)\bfinal\s*answer\s*[:\-]\s*\(?\s*([A-D])\s*\)?\b'),
    # Common: "Answer: X"
    re.compile(r'(?i)\banswer\s*[:\-]\s*\(?\s*([A-D])\s*\)?\b'),
    # Explicit phrasing: "The correct/final choice is X"
    re.compile(r'(?i)\b(correct|final)\s*(choice|option)\s*(is|:)\s*\(?\s*([A-D])\s*\)?\b'),
    # Parenthesized: "(X)"
    re.compile(r'\(([A-D])\)'),
    # Bare letter with boundary: "X." or "X)" or "X" at end
    re.compile(r'\b([A-D])\b(?=[\.\)]|$)'),
    # Option format: "Option X"
    re.compile(r'(?i)\boption\s*([A-D])\b'),
    # Choice format: "Choice X"
    re.compile(r'(?i)\bchoice\s*([A-D])\b'),
]


def extract_choice_regex(text: str, options: Set[str] = CHOICE_SET_DEFAULT) -> Tuple[bool, Optional[str], Tuple[int, int]]:
    """
    Extract MCQ choice using regex patterns.
    
    Searches for answer patterns in priority order. For patterns with multiple
    capturing groups, selects the rightmost group that matches a valid option.
    
    Args:
        text: Generated text to search
        options: Set of valid answer choices (default: A,B,C,D)
    
    Returns:
        Tuple of (found, letter, char_span):
        - found: True if a choice was extracted
        - letter: The extracted letter (uppercase) or None
        - char_span: (start, end) character positions in text
    
    Example:
        >>> extract_choice_regex("Final Answer: C")
        (True, 'C', (14, 15))
    """
    for pat in _PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        
        # Prefer rightmost capturing group that is a single letter
        for g in reversed(m.groups()):
            if g and len(g) == 1 and g.upper() in options:
                s, e = m.span()
                # Refine span to only the letter within the match
                idx = text[s:e].upper().find(g.upper())
                if idx >= 0:
                    return True, g.upper(), (s + idx, s + idx + 1)
                return True, g.upper(), (s, e)
    
    return False, None, (len(text), len(text))


def extract_choice_with_fallback(
    text: str,
    token_texts_tail: Optional[List[str]] = None,
    options: Set[str] = CHOICE_SET_DEFAULT
) -> Tuple[bool, Optional[str], Tuple[int, int]]:
    """
    Extract MCQ choice with multiple fallback strategies.
    
    Extraction strategy:
    1. Try regex patterns on full text (most reliable)
    2. If not found and token_texts_tail provided, search in last ~32 tokens
    3. Return (False, None, ...) if no answer found
    
    Args:
        text: Complete generated text
        token_texts_tail: Optional list of decoded token strings from last ~32 tokens
        options: Set of valid answer choices
    
    Returns:
        Tuple of (found, letter, char_span):
        - found: True if a choice was extracted
        - letter: The extracted letter (uppercase) or None
        - char_span: (start, end) character positions in original text
    
    Example:
        >>> text = "Let me think... The answer is B."
        >>> extract_choice_with_fallback(text)
        (True, 'B', (31, 32))
    """
    # 1) Try regex on full text
    found, letter, span = extract_choice_regex(text, options)
    if found:
        return True, letter, span
    
    # 2) Windowed fallback over last ~32 tokens if provided
    if token_texts_tail:
        tail = "".join(token_texts_tail)
        m = re.search(r'\b([A-D])\b', tail)
        if m and m.group(1).upper() in options:
            # Approximate char span near end
            # This will be re-mapped to token positions later
            tail_start = len(text) - len(tail)
            return True, m.group(1).upper(), (tail_start + m.start(1), tail_start + m.end(1))
    
    # 3) Final fallback: None
    return False, None, (len(text), len(text))
