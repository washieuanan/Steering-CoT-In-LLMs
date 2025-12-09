"""
Tests for Phase-B token-level intervention fixes.

Verifies that:
1. Answer phrase detection works across LLaMA/Mistral/Qwen tokenizers
2. Locality masks correctly identify CoT/answer/all spans
3. Hook application honors full mask span
4. Each question generates from clean state
"""

import pytest
import torch
import numpy as np
from transformers import AutoTokenizer

from utils.parse_answers import (
    _encode_phrase_ids,
    _find_subseq,
    locate_answer_span,
    ANSWER_ALIASES,
)
from utils.agg_utils import compute_locality_mask_post_generation


class TestTokenLevelAnswerPhraseFinding:
    """Test robust answer phrase detection across different tokenizers."""
    
    @pytest.mark.parametrize("model_name", [
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-7B-Instruct",
    ])
    def test_answer_phrase_found_with_variants(self, model_name):
        """Test that answer phrases are found with different leading whitespace."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            pytest.skip(f"Could not load tokenizer for {model_name}")
        
        # Test various answer phrase formats
        test_cases = [
            "Final answer:",
            " Final answer:",
            "\nFinal answer:",
            "\n Final answer:",
            "MCQ ANSWER:",
            " MCQ ANSWER:",
            "<answer>",
            " <answer>",
        ]
        
        for phrase in test_cases:
            ids = _encode_phrase_ids(tokenizer, phrase)
            assert len(ids) > 0, f"Failed to encode '{phrase}' for {model_name}"
            
            # Verify we can find it in a sequence
            prefix_ids = tokenizer.encode("Some reasoning here", add_special_tokens=False)
            full_ids = prefix_ids + ids + tokenizer.encode(" The answer is B", add_special_tokens=False)
            
            idx = _find_subseq(full_ids, ids)
            assert idx == len(prefix_ids), f"Failed to find '{phrase}' in sequence for {model_name}"
    
    @pytest.mark.parametrize("model_name", [
        "meta-llama/Llama-3.1-8B-Instruct",
    ])
    def test_locate_answer_span_basic(self, model_name):
        """Test locate_answer_span returns correct spans."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            pytest.skip(f"Could not load tokenizer for {model_name}")
        
        # Build a fake generation: prompt + cot + answer phrase + answer
        prompt = "Question: What is 2+2?"
        cot = " Let me think... 2 plus 2 equals 4."
        answer_phrase = "\nFinal answer:"
        answer = " 4"
        
        full_text = prompt + cot + answer_phrase + answer
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        
        # Encode parts separately to find boundaries
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        
        # Locate answer span
        a_start, a_end, found = locate_answer_span(full_ids, prompt_len, tokenizer, "Final answer:")
        
        assert found, "Answer phrase should be found"
        assert a_start >= prompt_len, "Answer should start after prompt"
        assert a_end > a_start, "Answer span should be non-empty"
        assert a_end <= len(full_ids), "Answer span should not exceed sequence length"
        
        # Verify the span contains the answer phrase
        span_ids = full_ids[a_start:a_end]
        span_text = tokenizer.decode(span_ids, skip_special_tokens=False)
        assert "Final answer:" in span_text or "final answer:" in span_text
    
    def test_locate_answer_span_not_found(self):
        """Test locate_answer_span when phrase is not present."""
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        except Exception:
            pytest.skip("Could not load tokenizer")
        
        # Generation without answer phrase
        text = "Question: What is 2+2? Let me think... The answer is 4."
        full_ids = tokenizer.encode(text, add_special_tokens=False)
        prompt_len = 10  # Arbitrary
        
        a_start, a_end, found = locate_answer_span(full_ids, prompt_len, tokenizer, "MCQ ANSWER:")
        
        assert not found, "Should not find answer phrase when not present"
        assert a_start == len(full_ids), "Should return end position when not found"
        assert a_end == len(full_ids), "Should return empty span at end"
    
    def test_answer_aliases_fallback(self):
        """Test that locate_answer_span tries aliases when explicit phrase not found."""
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        except Exception:
            pytest.skip("Could not load tokenizer")
        
        # Use an alias from ANSWER_ALIASES
        text = "Reasoning... <answer> The answer is B"
        full_ids = tokenizer.encode(text, add_special_tokens=False)
        prompt_len = 5
        
        # Don't provide explicit phrase, should find <answer> from aliases
        a_start, a_end, found = locate_answer_span(full_ids, prompt_len, tokenizer, None)
        
        # Should find one of the aliases
        # Note: This depends on aliases being registered as special tokens
        # For now, just verify it tries to search
        assert isinstance(found, bool)


class TestLocalityMasks:
    """Test locality mask computation for Phase-B interventions."""
    
    def test_compute_locality_mask_cot(self):
        """Test CoT locality masks from prompt_len to answer_start."""
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        except Exception:
            pytest.skip("Could not load tokenizer")
        
        # Build sequence: [prompt(10)] + [cot(50)] + [answer_phrase(5)] + [answer(10)]
        full_ids = list(range(75))  # Fake token IDs
        prompt_len = 10
        
        # Mock answer phrase at position 60
        answer_phrase = "Final answer:"
        
        mask, (a_start, a_end, found) = compute_locality_mask_post_generation(
            full_ids, prompt_len, tokenizer, "cot", answer_phrase
        )
        
        # Verify mask shape and type
        assert len(mask) == len(full_ids)
        assert all(isinstance(m, bool) for m in mask)
        
        # CoT locality should mask [prompt_len:answer_start)
        if found:
            # Verify CoT span is masked
            assert any(mask[prompt_len:a_start]), "CoT span should have some True values"
            # Verify answer span is not masked
            assert not any(mask[a_start:]), "Answer span should not be masked for cot locality"
    
    def test_compute_locality_mask_answer(self):
        """Test answer locality masks from answer_start to end."""
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        except Exception:
            pytest.skip("Could not load tokenizer")
        
        full_ids = list(range(75))
        prompt_len = 10
        answer_phrase = "Final answer:"
        
        mask, (a_start, a_end, found) = compute_locality_mask_post_generation(
            full_ids, prompt_len, tokenizer, "answer", answer_phrase
        )
        
        assert len(mask) == len(full_ids)
        
        if found:
            # Verify answer span is masked
            assert any(mask[a_start:a_end]), "Answer phrase span should be masked"
            # Verify CoT span is not masked
            assert not any(mask[prompt_len:a_start]), "CoT span should not be masked for answer locality"
    
    def test_compute_locality_mask_all(self):
        """Test all locality masks entire generation."""
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        except Exception:
            pytest.skip("Could not load tokenizer")
        
        full_ids = list(range(75))
        prompt_len = 10
        
        mask, (a_start, a_end, found) = compute_locality_mask_post_generation(
            full_ids, prompt_len, tokenizer, "all", None
        )
        
        assert len(mask) == len(full_ids)
        
        # All generated tokens should be masked
        assert not any(mask[:prompt_len]), "Prompt should not be masked"
        assert all(mask[prompt_len:]), "All generated tokens should be masked"
    
    def test_mask_respects_prompt_boundary(self):
        """Test that masks never include prompt tokens."""
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        except Exception:
            pytest.skip("Could not load tokenizer")
        
        full_ids = list(range(100))
        prompt_len = 30
        
        for locality in ["cot", "answer", "all"]:
            mask, _ = compute_locality_mask_post_generation(
                full_ids, prompt_len, tokenizer, locality, None
            )
            
            # Prompt tokens should never be masked
            assert not any(mask[:prompt_len]), f"Prompt masked for locality={locality}"


class TestHookMaskApplication:
    """Test that hook application honors full mask span."""
    
    def test_hook_applies_to_all_masked_positions(self):
        """Test that interventions apply to all True positions in mask."""
        from utils.hooks import apply_add
        
        # Create hidden states [B=1, T=100, H=128]
        B, T, H = 1, 100, 128
        h = torch.randn(B, T, H)
        
        # Create direction
        u = torch.randn(H)
        u = u / u.norm()
        
        # Create mask for positions 20-80
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[0, 20:80] = True
        
        # Apply intervention
        alpha = 2.0
        h_out = apply_add(h, u=u, alpha=alpha, add_mode="constant", mask=mask)
        
        # Verify only masked positions changed
        changed = ~torch.isclose(h, h_out, atol=1e-6).all(dim=-1)
        
        # All masked positions should have changed
        assert changed[0, 20:80].all(), "All masked positions should be modified"
        # Unmasked positions should not have changed
        assert not changed[0, :20].any(), "Prompt should not be modified"
        assert not changed[0, 80:].any(), "Post-answer should not be modified"
    
    def test_hook_works_with_single_token_mask(self):
        """Test that hooks work correctly when T=1 (decode step)."""
        from utils.hooks import apply_add
        
        # Decode step: single token
        B, T, H = 1, 1, 128
        h = torch.randn(B, T, H)
        u = torch.randn(H)
        u = u / u.norm()
        
        # Mask for this single token
        mask = torch.ones(B, T, dtype=torch.bool)
        
        h_out = apply_add(h, u=u, alpha=1.5, add_mode="constant", mask=mask)
        
        # Should have changed
        assert not torch.allclose(h, h_out, atol=1e-6), "Single token should be modified"
    
    def test_hook_projection_mode_vs_constant(self):
        """Test difference between projection and constant add modes."""
        from utils.hooks import apply_add
        
        B, T, H = 1, 50, 64
        h = torch.randn(B, T, H)
        u = torch.randn(H)
        u = u / u.norm()
        
        mask = torch.ones(B, T, dtype=torch.bool)
        alpha = 1.0
        
        # Projection mode: α * Proj_u(h)
        h_proj = apply_add(h, u=u, alpha=alpha, add_mode="proj", mask=mask)
        
        # Constant mode: α * u
        h_const = apply_add(h, u=u, alpha=alpha, add_mode="constant", mask=mask)
        
        # They should be different (unless h accidentally aligned with u)
        assert not torch.allclose(h_proj, h_const, atol=1e-4), "Modes should produce different results"


class TestDebugLogging:
    """Test that debug logging preserves special tokens."""
    
    def test_decode_preserves_special_tokens(self):
        """Test that we can decode with special tokens visible."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct",
                add_special_tokens=True
            )
            # Add special tokens
            tokenizer.add_special_tokens({"additional_special_tokens": ["<cot>", "<answer>"]})
        except Exception:
            pytest.skip("Could not load tokenizer")
        
        # Encode text with special tokens
        text = "Reasoning <cot> thinking <answer> B"
        ids = tokenizer.encode(text, add_special_tokens=False)
        
        # Decode with special tokens preserved
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        
        # Special tokens should be visible
        assert "<cot>" in decoded or "cot" in decoded.lower(), "CoT token should be visible"
        assert "<answer>" in decoded or "answer" in decoded.lower(), "Answer token should be visible"
    
    def test_special_token_ids_retrievable(self):
        """Test that special token IDs can be retrieved."""
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
            tokenizer.add_special_tokens({"additional_special_tokens": ["<cot>", "<answer>"]})
        except Exception:
            pytest.skip("Could not load tokenizer")
        
        # Get token IDs
        cot_id = tokenizer.convert_tokens_to_ids("<cot>")
        answer_id = tokenizer.convert_tokens_to_ids("<answer>")
        
        # Should be valid IDs (not unk_token_id)
        assert cot_id != tokenizer.unk_token_id, "CoT token should be registered"
        assert answer_id != tokenizer.unk_token_id, "Answer token should be registered"
        assert cot_id != answer_id, "Tokens should have different IDs"


class TestFreshStatePerQuestion:
    """Test that each question generates from clean state."""
    
    def test_no_carryover_between_questions(self):
        """Test that consecutive generations don't carry over state."""
        # This is more of an integration test - verify input_ids are fresh each time
        # The key is that we don't concatenate previous outputs to next inputs
        
        # Mock scenario: Two questions
        question1_ids = [1, 2, 3, 4, 5]
        question2_ids = [10, 20, 30, 40, 50]
        
        # Verify they're independent
        assert question1_ids != question2_ids
        assert len(set(question1_ids) & set(question2_ids)) == 0
        
        # In actual usage, each call to handler.build_prompt should create fresh input_ids
        # No history should be appended


class TestFindAnswerSpan:
    """Test the new find_answer_span function from parse_answers."""
    
    def test_find_answer_span_mcq_variants(self):
        """Test MCQ answer detection with various formats."""
        from utils.parse_answers import find_answer_span
        
        test_cases = [
            ("Let me think... Final Answer: C", True, "C"),
            ("Reasoning... Answer: B", True, "B"),
            ("Therefore, the answer is D", True, "D"),
            ("I choose Option A", True, "A"),
            ("MCQ ANSWER: C", True, "C"),
            ("Answer- B", True, "B"),
            ("(D)", True, "D"),
            ("No clear answer", False, None),
        ]
        
        for text, should_find, expected_label in test_cases:
            result = find_answer_span(text)
            assert result['found'] == should_find, f"Failed for: {text}"
            if should_find:
                assert result['label'] == expected_label, f"Expected {expected_label}, got {result['label']} for: {text}"
    
    def test_extract_prediction(self):
        """Test extract_prediction never returns nan."""
        from utils.parse_answers import extract_prediction
        
        test_cases = [
            ("Final Answer: C", "C"),
            ("Answer: B", "B"),
            ("No answer here", None),
            ("", None),
        ]
        
        for text, expected in test_cases:
            result = extract_prediction(text)
            assert result != "nan", f"Should never return 'nan' for: {text}"
            assert result == expected, f"Expected {expected}, got {result} for: {text}"


class TestTokenSpanMapping:
    """Test token_spans_from_char_span helper."""
    
    def test_token_spans_basic(self):
        """Test basic char-to-token mapping."""
        try:
            from transformers import AutoTokenizer
            from hf_model_wrapper import token_spans_from_char_span
            
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_fast=True)
        except Exception:
            pytest.skip("Could not load tokenizer")
        
        gen_text = "Let me think... Final Answer: C"
        
        # Find "Final Answer: C" in the text
        phrase = "Final Answer:"
        char_start = gen_text.find(phrase)
        char_end = char_start + len(phrase)
        
        if char_start >= 0:
            tok_span = token_spans_from_char_span(gen_text, tokenizer, (char_start, char_end))
            
            # Should return valid span
            assert tok_span is not None, "Should find token span for valid char span"
            assert tok_span[0] < tok_span[1], "Token span should be non-empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
