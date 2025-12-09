"""
Unified answer parsing for MCQ, Numeric, and Labelset datasets.

This module provides consistent parsing across Step 1 (generation) and Step 2 (analysis).
Supports multiple format variations and dynamic alphabets to handle model output diversity.

Enhanced with:
- Typo tolerance (MQC, MCQQ, etc.)
- Expanded synonym mappings
- Robust block detection with fallback
- Token-level answer phrase locator for Phase-B interventions
- Unified answer extraction via answers.extract_final_choice module
"""

import re
from typing import Optional, Tuple, List, Dict, Any

# Import unified answer extractor
from answers.extract_final_choice import extract_choice_with_fallback, CHOICE_SET_DEFAULT


# ==================== Token-Level Answer Phrase Locator ====================
# These functions enable robust Phase-B interventions across LLaMA/Mistral/Qwen tokenizers

def _encode_phrase_ids(tokenizer, phrase: str) -> List[int]:
    """
    Encode phrase with BPE-quirk handling for leading spaces/newlines.
    
    Tries multiple variants (raw, with leading space, with leading newline) to find
    the tokenization that produces valid IDs. This handles differences in how
    LLaMA, Mistral, and Qwen tokenizers handle leading whitespace.
    
    Args:
        tokenizer: HuggingFace tokenizer
        phrase: Text phrase to encode (e.g., "Final answer:")
    
    Returns:
        List of token IDs for the phrase, or empty list if encoding fails
    
    Example:
        >>> ids = _encode_phrase_ids(tokenizer, "Final answer:")
        >>> # Might return [9550, 4320, 25] for one tokenizer
    """
    trials = [phrase]
    if not phrase.startswith(" "):
        trials.append(" " + phrase)
    if not phrase.startswith("\n"):
        trials.append("\n" + phrase)
        trials.append("\n " + phrase)
    
    # Also try stripped if phrase has trailing punctuation
    stripped = phrase.rstrip()
    if stripped != phrase:
        trials.append(stripped)
        if not stripped.startswith(" "):
            trials.append(" " + stripped)
    
    for cand in trials:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if ids:
            return ids
    
    return []


def _find_subseq(haystack_ids: List[int], needle_ids: List[int]) -> int:
    """
    Find first occurrence of needle subsequence in haystack token list.
    
    Args:
        haystack_ids: List of token IDs to search in
        needle_ids: List of token IDs to search for
    
    Returns:
        Index of first occurrence, or -1 if not found
    
    Example:
        >>> haystack = [10, 20, 30, 40, 50]
        >>> needle = [30, 40]
        >>> _find_subseq(haystack, needle)
        2
    """
    if not needle_ids or len(needle_ids) > len(haystack_ids):
        return -1
    
    first = needle_ids[0]
    i = 0
    while True:
        try:
            i = haystack_ids.index(first, i)
        except ValueError:
            return -1
        
        if haystack_ids[i:i+len(needle_ids)] == needle_ids:
            return i
        i += 1


# Answer phrase aliases to try when explicit phrase not found
ANSWER_ALIASES = [
    "<answer>", "Final answer:", "Final Answer:", "Therefore, the answer is",
    "Answer:", "final answer:", "So, the answer is"
]


def locate_answer_span(
    full_ids: List[int],
    prompt_len: int,
    tokenizer,
    answer_phrase: Optional[str] = None
) -> Tuple[int, int, bool]:
    """
    Locate answer phrase span using token-level subsequence search.
    
    Searches only within generated segment (full_ids[prompt_len:]) for robustness.
    Returns contract: if not found, returns (end, end, False) for empty span at end.
    
    Args:
        full_ids: Full token ID sequence [prompt + generated]
        prompt_len: Length of prompt portion
        tokenizer: HuggingFace tokenizer for encoding phrases
        answer_phrase: Explicit answer phrase to search for (e.g., "MCQ ANSWER:")
    
    Returns:
        Tuple of (answer_start, answer_end, found_bool):
        - answer_start: Index in full_ids where answer phrase starts
        - answer_end: Index in full_ids where answer phrase ends (exclusive)
        - found_bool: True if phrase was found, False otherwise
    
    Example:
        >>> # Suppose full_ids = [prompt tokens] + [cot tokens] + [answer phrase] + [answer]
        >>> start, end, found = locate_answer_span(full_ids, prompt_len, tokenizer, "MCQ ANSWER:")
        >>> # Returns (start_idx, end_idx, True) if found
    """
    gen_ids = full_ids[prompt_len:]
    
    # 1) Try explicit phrase
    cand = _encode_phrase_ids(tokenizer, answer_phrase) if answer_phrase else []
    idx = _find_subseq(gen_ids, cand) if cand else -1
    
    # 2) Fallback to aliases
    if idx == -1:
        for alias in ANSWER_ALIASES:
            cand = _encode_phrase_ids(tokenizer, alias)
            if cand:
                idx = _find_subseq(gen_ids, cand)
                if idx != -1:
                    break
    
    if idx == -1:
        # Not found → contract: empty span at end
        return len(full_ids), len(full_ids), False
    
    start = prompt_len + idx
    end = start + len(cand)
    return start, end, True


def _pick_text(raw: str) -> str:
    """
    Extract text, preferring answer block but falling back gracefully.
    
    Handles:
    - Complete block: <answer>...</answer>
    - Missing closing tag: <answer>...
    - Missing opening tag: ...</answer>
    - No tags: full text
    
    Args:
        raw: Raw model output
    
    Returns:
        Extracted text for parsing
    """
    # Try strict match first (both tags present)
    m = re.search(r'(?is)<\s*answer\s*>(.*?)</\s*answer\s*>', raw)
    if m:
        return m.group(1)
    
    # Try with only opening tag (take everything after)
    m = re.search(r'(?is)<\s*answer\s*>(.*?)$', raw)
    if m:
        return m.group(1)
    
    # Try with only closing tag (take everything before)
    m = re.search(r'(?is)^(.*?)</\s*answer\s*>', raw)
    if m:
        return m.group(1)
    
    # No tags, return full text
    return raw


def extract_mcq_answer(text: str, allowed_letters: Optional[List[str]] = None) -> Tuple[Optional[str], str]:
    """
    Extract a multiple-choice answer letter from model output with dynamic alphabet support.
    
    Now delegates to the unified extractor in answers.extract_final_choice for consistency.
    
    Args:
        text: Model-generated text (raw or cleaned)
        allowed_letters: List of valid letters (e.g., ["A","B","C","D","E"]). 
                        If None, defaults to ["A","B","C","D"]
    
    Returns:
        Tuple of (letter or None, status string)
        Status is 'valid' if match found, 'no_match' otherwise
    
    Example:
        >>> extract_mcq_answer("MCQ ANSWER: B)\\n", ["A","B","C","D"])
        ('B', 'valid')
        >>> extract_mcq_answer("Final Answer: C", ["A","B","C","D"])
        ('C', 'valid')
    """
    if allowed_letters is None:
        allowed_letters = ["A", "B", "C", "D"]
    
    # Convert to set for unified extractor
    options = set(allowed_letters)
    
    # Use unified extractor
    found, letter, _ = extract_choice_with_fallback(text, token_texts_tail=None, options=options)
    
    if found and letter:
        return letter, "valid"
    else:
        return None, "no_match"


def extract_numeric_answer(text: str) -> Tuple[Optional[str], str]:
    """
    Extract a numeric final answer from model output.
    
    Accepts (case-insensitive, multiline):
      - 'NUM ANSWER: -1234.56' or 'NUMBER ANS: -1234.56' (with typo tolerance)
      - '<answer> -1234.56' (legacy format)
      - 'Final answer: -1,234.56' or 'Final answer: $1,234.56'
      - 'Final answer: 2.3e-4'
      - '\\boxed{1234.56}'
      - Bare number line
    
    Extracts only within <answer>...</answer> window if present.
    
    Args:
        text: Model-generated text (raw or cleaned)
    
    Returns:
        Tuple of (normalized_number_string or None, status string)
        Status is 'valid' if match found, 'no_match' otherwise
    
    Notes:
      - Strips $, commas, and whitespace
      - Exponent forms like 1e-3 are preserved as lowercase 'e'
      - Supports \\boxed{...} LaTeX notation
      - Supports signs, decimals, thousands separators, and scientific notation
    
    Example:
        >>> extract_numeric_answer("NUM ANSWER: 1,234.56")
        ('1234.56', 'valid')
        >>> extract_numeric_answer("NUMBER ANS: $1,234.56")  # typo tolerance
        ('1234.56', 'valid')
        >>> extract_numeric_answer("\\\\boxed{2.3e-4}")
        ('2.3e-4', 'valid')
    """
    # Extract content with fallback logic
    search_text = _pick_text(text)
    take_first = "<answer>" in text.lower()
    
    # Number pattern: sign? optional $, digits with optional thousands, optional decimal, optional exponent
    num_core = r'\$?\s*[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?|\$?\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?'
    
    def normalize_number(s: str) -> str:
        """Strip $, commas, whitespace and normalize exponent."""
        s = s.replace('$', '').replace(',', '').replace(' ', '')
        s = re.sub(r'[E]', 'e', s)  # Normalize exponent
        return s.strip()
    
    # 0) Check for \boxed{...} format first (common in math problems)
    pat_boxed = re.compile(r'\\boxed\{([^\}]+)\}', re.IGNORECASE)
    matches = pat_boxed.findall(search_text)
    if matches:
        # Try to extract number from boxed content
        for match in (matches if take_first else reversed(matches)):
            # Match could be "1234" or "$1,234.56" etc
            num_match = re.search(num_core, match)
            if num_match:
                return normalize_number(num_match.group(0)), "valid"
    
    # 1) Prefer new structured format with typo tolerance
    # Accepts: NUM, NUMBER, NUMER, etc. + optional "ANSWER" or just "ANS"
    pat_num_answer = re.compile(rf'(?i)\bNUM(?:BER)?\s*ANS(?:WER)?\s*:\s*({num_core})', re.IGNORECASE)
    matches = pat_num_answer.findall(search_text)
    if matches:
        s = matches[0 if take_first else -1]
        return normalize_number(s), "valid"
    
    # 2) Legacy '<answer>' format
    pat_answer_tag = re.compile(rf'<answer>\s*({num_core})', re.IGNORECASE)
    matches = pat_answer_tag.findall(search_text)
    if matches:
        s = matches[0 if take_first else -1]
        return normalize_number(s), "valid"
    
    # 3) Fallback to explicit 'Final answer:' line
    pat_final = re.compile(rf'Final\s*answer\s*:\s*({num_core})', re.IGNORECASE)
    matches = pat_final.findall(search_text)
    if matches:
        s = matches[0 if take_first else -1]
        return normalize_number(s), "valid"
    
    # 4) Fallback: bare number line
    pat_bare = re.compile(rf'^\s*({num_core})\s*$', re.MULTILINE | re.IGNORECASE)
    matches = pat_bare.findall(search_text)
    if matches:
        s = matches[0 if take_first else -1]
        return normalize_number(s), "valid"
    
    return None, "no_match"


def extract_labelset_answer(text: str, labels: List[str]) -> Tuple[Optional[str], str]:
    """
    Extract a labelset answer from model output with synonym normalization.
    
    Accepts (case-insensitive, multiline):
      - 'CLS ANSWER: supported' or 'CLASS ANS: supported' (with typo tolerance)
      - '<answer> supported' (legacy format)
      - 'Final answer: supported'
      - Bare label on its own line
    
    Matches against provided labels using case-insensitive comparison.
    Normalizes common synonyms (e.g., proved→entailed, unknown→neither).
    
    Args:
        text: Model-generated text (raw or cleaned)
        labels: List of valid label strings (e.g., ["entailed", "disproved", "both", "neither"])
    
    Returns:
        Tuple of (matched_label or None, status string)
        Status is 'valid' if match found, 'no_match' otherwise
        Returns the label from the provided list (preserves original casing)
    
    Example:
        >>> labels = ["entailed", "disproved", "both", "neither"]
        >>> extract_labelset_answer("CLS ANSWER: proved\\n", labels)
        ('entailed', 'valid')
        >>> extract_labelset_answer("CLASS ANS: unknown\\n", labels)  # typo tolerance
        ('neither', 'valid')
    """
    if not labels:
        return None, "no_labels_provided"
    
    # Expanded synonym mappings for common variations
    SYNONYMS = {
        # Entailed variants
        'proved': 'entailed',
        'proven': 'entailed',
        'supported': 'entailed',
        'true': 'entailed',
        'yes': 'entailed',
        'correct': 'entailed',
        'valid': 'entailed',
        'confirmed': 'entailed',
        'verified': 'entailed',
        
        # Disproved variants
        'disproved': 'disproved',
        'refuted': 'disproved',
        'false': 'disproved',
        'no': 'disproved',
        'incorrect': 'disproved',
        'invalid': 'disproved',
        'contradicted': 'disproved',
        
        # Neither variants
        'unknown': 'neither',
        'unclear': 'neither',
        'inconclusive': 'neither',
        'uncertain': 'neither',
        'ambiguous': 'neither',
        'indeterminate': 'neither',
        
        # German (seen in some outputs)
        'wahr': 'entailed',
        'falsch': 'disproved',
    }
    
    def normalize_candidate(candidate: str) -> str:
        """Normalize a candidate answer using synonyms."""
        candidate_lower = candidate.lower().strip()
        # Check synonyms first
        if candidate_lower in SYNONYMS:
            return SYNONYMS[candidate_lower]
        return candidate_lower
    
    def match_label(candidate: str) -> Optional[str]:
        """Try to match candidate against labels with normalization."""
        candidate_norm = normalize_candidate(candidate)
        
        # Try exact case-insensitive match
        for label in labels:
            if candidate_norm == label.lower():
                return label
        
        # Try prefix match as fallback
        for label in labels:
            if candidate_norm.startswith(label.lower()):
                return label
        
        # Try original candidate without normalization
        for label in labels:
            if candidate.lower() == label.lower():
                return label
        
        for label in labels:
            if candidate.lower().startswith(label.lower()):
                return label
        
        return None
    
    # Extract content with fallback logic
    search_text = _pick_text(text)
    take_first = "<answer>" in text.lower()
    
    # 1) Prefer new structured format with typo tolerance
    # Accepts: CLS, CLASS, CLASSIFY, etc. + optional "ANSWER" or just "ANS"
    pat_cls_answer = re.compile(r'(?i)\b(?:CLS|CLASS(?:IFY)?)\s*ANS(?:WER)?\s*:\s*(.+?)(?:\n|$)', re.IGNORECASE)
    matches = pat_cls_answer.findall(search_text)
    if matches:
        candidate = matches[0 if take_first else -1].strip()
        matched = match_label(candidate)
        if matched:
            return matched, "valid"
    
    # 2) Legacy '<answer>' format
    pat_answer_tag = re.compile(r'<answer>\s*(.+?)\s*(?:</answer>|\n|$)', re.IGNORECASE)
    matches = pat_answer_tag.findall(search_text)
    if matches:
        candidate = matches[0 if take_first else -1].strip()
        matched = match_label(candidate)
        if matched:
            return matched, "valid"
    
    # 3) Fallback to 'Final answer:' format
    pat_final = re.compile(r'Final\s*answer\s*:\s*(.+?)(?:\n|$)', re.IGNORECASE)
    matches = pat_final.findall(search_text)
    if matches:
        candidate = matches[0 if take_first else -1].strip()
        matched = match_label(candidate)
        if matched:
            return matched, "valid"
    
    # 4) Fallback: search for any label appearing as standalone word/phrase
    lines = search_text.split('\n')
    for line in (lines if take_first else reversed(lines)):
        line_stripped = line.strip()
        if line_stripped:
            matched = match_label(line_stripped)
            if matched:
                return matched, "valid"
    
    return None, "no_match"


def extract_reason(text: str) -> Optional[str]:
    """
    Extract the reason/explanation from structured output.
    
    Looks for "REASON: ..." pattern and returns everything after it
    as the explanation text (can be multi-line).
    
    Args:
        text: Model-generated text
    
    Returns:
        Reason text if found, None otherwise
    
    Example:
        >>> text = "MCQ ANSWER: B\\nREASON: The answer is B because..."
        >>> extract_reason(text)
        ' The answer is B because...'
    """
    pattern = re.compile(r'(?im)^\s*REASON\s*:\s*(.*)$', re.MULTILINE)
    match = pattern.search(text)
    
    if not match:
        return None
    
    # Get everything after "REASON:" including rest of document
    start_pos = match.end()
    return text[start_pos:].strip() if start_pos < len(text) else ""


def get_answer_context(text: str, max_lines: int = 3) -> str:
    """
    Extract context around answer markers for debugging.
    
    Returns the last few lines that contain or precede answer markers
    like '<answer>', 'Final answer:', 'MCQ ANSWER:', etc.
    
    Args:
        text: Model-generated text
        max_lines: Maximum number of lines to return
    
    Returns:
        Context string (last N lines around answer area)
    
    Example:
        >>> text = "reasoning...\\n</cot>\\nMCQ ANSWER: B\\n"
        >>> get_answer_context(text, 2)
        '</cot>\\nMCQ ANSWER: B'
    """
    lines = text.split('\n')
    
    # Find lines with answer markers
    marker_indices = []
    for i, line in enumerate(lines):
        if re.search(r'(?i)<answer>|final\s*answer\s*:|mcq\s+answer\s*:|num\s+answer\s*:|cls\s+answer\s*:', line):
            marker_indices.append(i)
        elif re.search(r'(?i)^\s*[ABCDEFGHIJKLMNOPQRSTUVWXYZ](?:\s*[\)\.])?', line):
            marker_indices.append(i)
    
    if not marker_indices:
        # No markers found, return last N lines
        start_idx = max(0, len(lines) - max_lines)
        return '\n'.join(lines[start_idx:])
    
    # Get context around the last marker
    last_marker = marker_indices[-1]
    start_idx = max(0, last_marker - max_lines + 1)
    end_idx = min(len(lines), last_marker + 2)
    
    return '\n'.join(lines[start_idx:end_idx])


# ==================== Robust Answer Phrase Detection for Phase-B ====================

def find_answer_span(text: str) -> Dict[str, Any]:
    """
    Find answer phrase span in generated text using unified extractor.
    
    Now delegates to the unified extractor in answers.extract_final_choice for consistency.
    
    Args:
        text: Complete generated text (prompt + CoT + answer)
    
    Returns:
        Dictionary with keys:
            - found (bool): True if answer phrase detected
            - char_span (tuple): (start_char, end_char) inclusive start, exclusive end
            - label (str|None): Normalized answer ('A','B','C','D' or short answer)
    
    Example:
        >>> text = "Let me think... Final Answer: C"
        >>> result = find_answer_span(text)
        >>> result
        {'found': True, 'char_span': (25, 26), 'label': 'C'}
    """
    # Use unified extractor
    found, letter, char_span = extract_choice_with_fallback(text, token_texts_tail=None)
    
    return {
        'found': found,
        'char_span': char_span,
        'label': letter
    }


def extract_prediction(text: str) -> Optional[str]:
    """
    Extract final prediction from generated text using find_answer_span.
    
    Never returns 'nan' - returns None if no answer detected.
    
    Args:
        text: Complete generated text
    
    Returns:
        Normalized prediction string ('A'/'B'/'C'/'D' or stripped answer),
        or None if not detectable
    
    Example:
        >>> extract_prediction("Reasoning... Final Answer: C")
        'C'
        >>> extract_prediction("No clear answer here")
        None
    """
    result = find_answer_span(text)
    
    if not result['found'] or result['label'] is None:
        return None
    
    # Return normalized label
    label = result['label']
    
    # Additional normalization: uppercase, strip trailing punctuation
    label = label.upper().rstrip('.!?,;:')
    
    return label if label else None
