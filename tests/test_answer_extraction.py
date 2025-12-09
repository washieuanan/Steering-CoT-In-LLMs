"""
Tests for unified answer extraction.

This module tests the answer.extract_final_choice module to ensure
robust extraction across various formats.
"""

import pytest
from answers.extract_final_choice import (
    extract_choice_regex,
    extract_choice_with_fallback,
    CHOICE_SET_DEFAULT,
)


# Test cases covering various answer formats
BASIC_CASES = [
    ("Final Answer: C", "C"),
    ("final answer - (B)", "B"),
    ("Answer: D", "D"),
    ("The correct option is A.", "A"),
    ("Therefore, the final choice is C.", "C"),
    ("I pick (B).", "B"),
    ("Option D.", "D"),
    ("Choice A.", "A"),
    ("The answer is (C).", "C"),
    ("C", "C"),  # Bare letter at end
]

EDGE_CASES = [
    ("Let me think... Final Answer: B", "B"),
    ("After analysis, the answer is A.", "A"),
    ("So (D) is correct.", "D"),
    ("Therefore: C", "C"),
    ("My final choice: (A)", "A"),
]

# Cases that should NOT match (invalid)
NEGATIVE_CASES = [
    "No clear answer here",
    "The options are A, B, C, and D",
    "E is the answer",  # E not in default set
    "",
]


class TestExtractChoiceRegex:
    """Test the regex-based extraction."""
    
    def test_basic_patterns(self):
        """Test that basic patterns are recognized."""
        hits = 0
        for text, expected in BASIC_CASES:
            found, letter, span = extract_choice_regex(text, CHOICE_SET_DEFAULT)
            if found and letter == expected:
                hits += 1
            else:
                print(f"MISS: '{text}' -> got {letter}, expected {expected}")
        
        # Allow 1 miss out of 10
        assert hits >= 9, f"Only {hits}/10 basic cases matched"
    
    def test_edge_cases(self):
        """Test edge cases and variations."""
        hits = 0
        for text, expected in EDGE_CASES:
            found, letter, span = extract_choice_regex(text, CHOICE_SET_DEFAULT)
            if found and letter == expected:
                hits += 1
            else:
                print(f"MISS: '{text}' -> got {letter}, expected {expected}")
        
        # Be lenient with edge cases
        assert hits >= 3, f"Only {hits}/5 edge cases matched"
    
    def test_negative_cases(self):
        """Test that invalid inputs don't match."""
        for text in NEGATIVE_CASES:
            found, letter, span = extract_choice_regex(text, CHOICE_SET_DEFAULT)
            assert not found or letter is None, \
                f"False positive: '{text}' matched as {letter}"
    
    def test_span_accuracy(self):
        """Test that character spans are accurate."""
        text = "blah blah. Final Answer: C\n"
        found, letter, (s, e) = extract_choice_regex(text, CHOICE_SET_DEFAULT)
        
        assert found
        assert letter == "C"
        # The span should point to just the letter
        assert text[s:e] == "C"
    
    def test_custom_options(self):
        """Test with custom option set."""
        text = "The answer is F"
        options = {"A", "B", "C", "D", "E", "F"}
        
        found, letter, span = extract_choice_regex(text, options)
        assert found
        assert letter == "F"


class TestExtractChoiceWithFallback:
    """Test the full extraction with fallback strategies."""
    
    def test_regex_primary(self):
        """Test that regex is tried first."""
        text = "Let me think... Final Answer: C"
        found, letter, span = extract_choice_with_fallback(text, token_texts_tail=None)
        
        assert found
        assert letter == "C"
    
    def test_token_tail_fallback(self):
        """Test fallback to token tail when regex fails."""
        # Text with no explicit answer phrase, but letter in tail
        text = "Complex reasoning without explicit format"
        # Simulate last few tokens containing "B"
        token_texts = [" complex", " without", " B"]
        
        found, letter, span = extract_choice_with_fallback(
            text, token_texts_tail=token_texts
        )
        
        assert found
        assert letter == "B"
    
    def test_no_match_returns_false(self):
        """Test that no match returns (False, None, ...)."""
        text = "No answer here at all"
        found, letter, span = extract_choice_with_fallback(text, token_texts_tail=None)
        
        assert not found
        assert letter is None
        # Span should point to end
        assert span == (len(text), len(text))
    
    def test_case_insensitive(self):
        """Test that extraction is case-insensitive."""
        cases = [
            "final answer: c",
            "FINAL ANSWER: C",
            "Final Answer: c",
        ]
        
        for text in cases:
            found, letter, span = extract_choice_with_fallback(text)
            assert found
            assert letter == "C"  # Always uppercase
    
    def test_span_boundary(self):
        """Test span is within text boundaries."""
        text = "Answer: D"
        found, letter, (s, e) = extract_choice_with_fallback(text)
        
        assert found
        assert 0 <= s < len(text)
        assert s < e <= len(text)


class TestIntegration:
    """Integration tests simulating real usage."""
    
    def test_mcq_format(self):
        """Test typical MCQ output format."""
        output = """
        Let me analyze this question step by step.
        
        Looking at the choices:
        A) This seems incorrect because...
        B) This could be right, but...
        C) This is the best answer because...
        D) This doesn't fit.
        
        Final Answer: C
        """
        
        found, letter, span = extract_choice_with_fallback(output)
        assert found
        assert letter == "C"
    
    def test_cot_with_answer(self):
        """Test CoT followed by answer."""
        output = """
        <cot>
        First, I'll consider...
        Then, analyzing further...
        </cot>
        
        The correct option is B.
        """
        
        found, letter, span = extract_choice_with_fallback(output)
        assert found
        assert letter == "B"
    
    def test_minimal_format(self):
        """Test minimal acceptable format."""
        outputs = [
            "(A)",
            "Answer: B",
            "C",
        ]
        expected = ["A", "B", "C"]
        
        for text, exp in zip(outputs, expected):
            found, letter, span = extract_choice_with_fallback(text)
            assert found, f"Failed to extract from '{text}'"
            assert letter == exp


def test_recovery_rate_overall():
    """Test overall recovery rate across all cases."""
    all_cases = BASIC_CASES + EDGE_CASES
    hits = 0
    
    for text, expected in all_cases:
        found, letter, span = extract_choice_with_fallback(text)
        if found and letter == expected:
            hits += 1
    
    total = len(all_cases)
    rate = hits / total
    
    print(f"\nOverall recovery rate: {hits}/{total} = {rate:.1%}")
    
    # Require at least 80% recovery rate
    assert rate >= 0.80, f"Recovery rate {rate:.1%} below threshold"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
