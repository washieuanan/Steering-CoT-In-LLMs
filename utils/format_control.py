"""
Finite-state decoding control for structured MCQ, Numeric, and Labelset outputs.

This module provides logits processors and stopping criteria to enforce:
1. CoT reasoning (minimum tokens)
2. Structured answer line: "\nMCQ ANSWER: X" or "\nNUM ANSWER: <number>" or "\nCLS ANSWER: <label>"
3. Structured reason line: "\nREASON: ..."

Supports:
- Dynamic alphabets for MCQ (A-Z)
- Trie-based matching for labelset (multi-token labels)
- EOS/EOT blocking during answer generation
- Leading newlines to prevent mid-line insertions
"""

from transformers import LogitsProcessor, StoppingCriteria
import torch
from typing import Set, List, Optional, Dict

# Constants for answer delimiters - single source of truth
COT_SENTINEL = "<cot>"
ANSWER_SENTINEL_TEXT = "MCQ ANSWER:"  # literal phrase before final choice
ANSWER_CLOSE = "</answer>"


def get_answer_sentinel_text() -> str:
    """Get the canonical answer sentinel text."""
    return ANSWER_SENTINEL_TEXT


def get_cot_sentinel_text() -> str:
    """Get the canonical CoT sentinel text."""
    return COT_SENTINEL


def get_blocked_special_ids(tokenizer) -> Set[int]:
    """
    Get set of special token IDs that should be blocked during answer generation.
    
    Args:
        tokenizer: HuggingFace tokenizer
    
    Returns:
        Set of token IDs to block (EOS, EOT, etc.)
    
    Example:
        >>> blocked = get_blocked_special_ids(tokenizer)
        >>> # Use in controller to prevent early termination
    """
    blocked = set()
    
    # Block EOS token
    if tokenizer.eos_token_id is not None:
        blocked.add(tokenizer.eos_token_id)
    
    # Check for common end-of-turn tokens
    for token_str in ["<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"]:
        tok_id = tokenizer.convert_tokens_to_ids(token_str)
        if tok_id != tokenizer.unk_token_id:
            blocked.add(tok_id)
    
    return blocked


def collect_single_token_variants(tok, s_list: List[str]) -> Set[int]:
    """
    Return a set of token ids such that tok.decode([id]) is any of:
    s, ' ' + s, or '\n' + s for each s in s_list, provided they are single-token.
    
    Args:
        tok: Tokenizer instance
        s_list: List of strings to check for single-token variants
    
    Returns:
        Set of token IDs that represent these strings as single tokens
    """
    out = set()
    for s in s_list:
        for prefix in ["", " ", "\n"]:
            ids = tok.encode(prefix + s, add_special_tokens=False)
            if len(ids) == 1:
                out.add(ids[0])
    return out


def phrase_ids(tok, s: str) -> List[int]:
    """
    Get token IDs for a phrase without special tokens.
    
    Args:
        tok: Tokenizer instance
        s: Phrase to tokenize
    
    Returns:
        List of token IDs
    """
    return tok.encode(s, add_special_tokens=False)


def tokenize_phrase_variants(tokenizer, phrase: str) -> List[List[int]]:
    """
    Return multiple tokenization variants for phrase to handle BPE quirks.
    
    Handles cases where leading newline/space may merge with following tokens
    differently in Mistral/LLaMA tokenizers.
    
    Args:
        tokenizer: HuggingFace tokenizer
        phrase: Base phrase (e.g., "MCQ ANSWER:")
    
    Returns:
        List of token ID sequences, with canonical newline version first
    
    Example:
        >>> variants = tokenize_phrase_variants(tokenizer, "MCQ ANSWER:")
        >>> # Returns [[10, 44, 23, 85], [44, 23, 85], ...]
    """
    variants_txt = [
        "\n" + phrase,      # canonical with leading newline
        phrase,             # no leading newline
        "\n " + phrase,     # newline + space
        " " + phrase,       # leading space
    ]
    
    seen = set()
    variants = []
    
    for t in variants_txt:
        ids = tokenizer.encode(t, add_special_tokens=False)
        key = tuple(ids)
        if ids and key not in seen:
            variants.append(ids)
            seen.add(key)
    
    return variants


def get_newline_token_ids(tokenizer) -> Set[int]:
    """
    Get set of token IDs that represent newlines.
    
    Common newline tokens across different tokenizers:
    - 10: '\n' in many tokenizers
    - 13: '\r' or '\n' in some
    - 271: '\n' in Mistral/LLaMA
    
    Args:
        tokenizer: HuggingFace tokenizer
    
    Returns:
        Set of newline token IDs
    """
    newline_ids = set()
    
    # Try encoding newline directly
    ids = tokenizer.encode("\n", add_special_tokens=False)
    if ids:
        newline_ids.update(ids)
    
    # Add common newline token IDs
    common_newline_ids = {10, 13, 271}
    for token_id in common_newline_ids:
        try:
            decoded = tokenizer.decode([token_id])
            if '\n' in decoded or '\r' in decoded:
                newline_ids.add(token_id)
        except:
            pass
    
    return newline_ids if newline_ids else {10}  # Fallback to \n


class TrieNode:
    """Simple trie node for matching token sequences."""
    
    def __init__(self):
        self.children: Dict[int, 'TrieNode'] = {}
        self.is_end = False
        self.label_index = -1  # Index in original labels list


class LabelTrie:
    """
    Trie structure for matching multi-token label sequences.
    
    Supports incremental matching during generation to handle labels
    that span multiple tokens (e.g., "dis proved" might be 2 tokens).
    """
    
    def __init__(self, label_token_sequences: List[List[int]]):
        """
        Initialize trie from label token sequences.
        
        Args:
            label_token_sequences: List of token ID lists, one per label
        """
        self.root = TrieNode()
        
        for idx, token_seq in enumerate(label_token_sequences):
            node = self.root
            for token_id in token_seq:
                if token_id not in node.children:
                    node.children[token_id] = TrieNode()
                node = node.children[token_id]
            node.is_end = True
            node.label_index = idx
    
    def get_valid_next_tokens(self, current_path: List[int]) -> Set[int]:
        """
        Get set of valid next token IDs given current path.
        
        Args:
            current_path: List of token IDs generated so far
        
        Returns:
            Set of token IDs that can validly extend the path
        """
        node = self.root
        for token_id in current_path:
            if token_id not in node.children:
                return set()  # Invalid path
            node = node.children[token_id]
        
        return set(node.children.keys())
    
    def is_complete_label(self, token_path: List[int]) -> bool:
        """
        Check if token path represents a complete label.
        
        Args:
            token_path: List of token IDs
        
        Returns:
            True if this is a complete label
        """
        node = self.root
        for token_id in token_path:
            if token_id not in node.children:
                return False
            node = node.children[token_id]
        return node.is_end


class UnifiedFormatController(LogitsProcessor):
    """
    Robust FSM-based logits processor to enforce CoT + structured answer format.
    
    This processor ensures the model generates:
    1. Free-form reasoning (>= min_cot_tokens)
    2. Exactly: "\n<PHRASE>" where PHRASE is "MCQ ANSWER:" / "NUM ANSWER:" / "CLS ANSWER:"
    3. The answer (letter / number / label)
    4. A newline
    5. Optionally "REASON:" + explanation
    
    States:
    - cot: Free CoT generation (block phrase starts until min_cot_tokens)
    - need_ans_phrase: Force phrase token-by-token with variant support
    - need_answer: Allow only answer tokens (letter / numeric / label-trie)
    - need_newline: Force a single newline token
    - need_reason_phrase: Force "\nREASON:" token-by-token
    - reason: Free explanation text
    - done: Complete, allow EOS
    
    Args:
        tok: Tokenizer instance
        mode: 'mcq', 'numeric', or 'labelset'
        min_cot_tokens: Minimum reasoning tokens before allowing answer
        phrase_text: Text of answer phrase (e.g., "MCQ ANSWER:")
        allowed_letters: For MCQ - list of valid letters
        label_token_sequences: For labelset - list of tokenized labels
        blocked_special_ids: Set of token IDs to block (EOS, EOT, etc.)
        require_reason: Whether to require REASON: phrase (default True)
        max_answer_tokens: Max tokens for numeric answers (default 32)
    """
    
    def __init__(
        self,
        tok,
        mode: str,
        min_cot_tokens: int = 24,
        phrase_text: Optional[str] = None,
        allowed_letters: Optional[List[str]] = None,
        label_token_sequences: Optional[List[List[int]]] = None,
        blocked_special_ids: Optional[Set[int]] = None,
        require_reason: bool = True,
        max_answer_tokens: int = 32,
        max_reason_tokens: int = 40,
    ):
        assert mode in ("mcq", "numeric", "labelset"), f"mode must be 'mcq', 'numeric', or 'labelset', got {mode}"
        self.tok = tok
        self.mode = mode
        self.min_cot_tokens = int(min_cot_tokens)
        self.require_reason = require_reason
        self.max_answer_tokens = max_answer_tokens
        self.max_reason_tokens = max_reason_tokens
        
        # Determine phrase text if not provided
        if phrase_text is None:
            if mode == "mcq":
                phrase_text = "MCQ ANSWER:"
            elif mode == "numeric":
                phrase_text = "NUM ANSWER:"
            else:
                phrase_text = "CLS ANSWER:"
        
        # Generate phrase variants to handle tokenizer quirks
        self.phrase_ids_list = tokenize_phrase_variants(tok, phrase_text)
        self.reason_phrase_ids_list = tokenize_phrase_variants(tok, "REASON:")
        
        # Tokenize </answer> closing tag variants for detection
        self.answer_close_ids_list = tokenize_phrase_variants(tok, "</answer>")
        
        # Collect ALL first tokens from ALL variants for blocking
        self.phrase_first_tokens = set()
        for variant in self.phrase_ids_list:
            if variant:
                self.phrase_first_tokens.add(variant[0])
        
        # Collect closing tag first tokens for detection in reason phase
        self.answer_close_first_tokens = set()
        for variant in self.answer_close_ids_list:
            if variant:
                self.answer_close_first_tokens.add(variant[0])
        
        # Get newline token IDs
        self.newline_ids = get_newline_token_ids(tok)
        
        # Get space token ID for safety valve
        space_ids = tok.encode(" ", add_special_tokens=False)
        self.space_id = space_ids[0] if space_ids else None
        
        # Mode-specific setup
        if mode == "mcq":
            if allowed_letters is None:
                allowed_letters = ["A", "B", "C", "D"]
            self.letters = collect_single_token_variants(tok, allowed_letters)
        elif mode == "labelset":
            if label_token_sequences is None:
                raise ValueError("labelset mode requires label_token_sequences")
            self.label_trie = LabelTrie(label_token_sequences)
        else:  # numeric
            # For numeric, allow digits, signs, decimal, comma, and scientific notation
            self.num_chars = collect_single_token_variants(
                tok, list("0123456789+-.,eE%$")
            )
        
        self.blocked_special_ids = blocked_special_ids if blocked_special_ids is not None else set()
        
        # Per-batch state tracking
        self.state = {}  # batch_idx -> dict
    
    def get_current_phase(self, batch_idx: int) -> str:
        """Get current phase for a batch index (for external access)."""
        if batch_idx not in self.state:
            return "cot"
        return self.state[batch_idx].get("phase", "cot")
    
    def _get_state(self, batch_idx: int) -> dict:
        """Get or initialize state for a batch index."""
        if batch_idx not in self.state:
            self.state[batch_idx] = {
                "phase": "cot",
                "generated_cot": 0,
                "phrase_pos": 0,
                "active_variant": None,
                "answer_tokens": 0,
                "answer_path": [],  # For labelset trie matching
                "mismatch_count": 0,  # Track consecutive mismatches for variant switching
            }
        return self.state[batch_idx]
    
    def _apply_allowlist(self, scores: torch.Tensor, allowed_ids: Set[int], safety_ids: Optional[Set[int]] = None) -> torch.Tensor:
        """
        Apply allowlist to scores, with safety valve.
        
        Args:
            scores: Score tensor for single batch item
            allowed_ids: Set of allowed token IDs
            safety_ids: Optional fallback IDs if allowlist is empty
        
        Returns:
            Modified scores
        """
        if not allowed_ids and safety_ids:
            # Safety valve: empty allowlist, use safety fallback
            allowed_ids = safety_ids
            import logging
            logging.warning(f"[FormatCtrl] Empty allowlist, using safety fallback: {safety_ids}")
        
        if not allowed_ids:
            # Last resort: allow everything (shouldn't happen)
            import logging
            logging.error(f"[FormatCtrl] No allowed tokens available! Allowing all.")
            return scores
        
        # Create mask that blocks everything except allowed tokens
        mask = torch.full_like(scores, float('-inf'))
        for token_id in allowed_ids:
            if token_id < len(mask):
                mask[token_id] = 0.0
        
        return scores + mask
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Process logits to enforce format constraints with robust state machine.
        
        Args:
            input_ids: Generated tokens so far [batch_size, seq_len]
            scores: Raw logits [batch_size, vocab_size]
        
        Returns:
            Modified logits with format constraints applied
        """
        batch_size = input_ids.shape[0]
        
        for b in range(batch_size):
            st = self._get_state(b)
            last_token = input_ids[b, -1].item()
            
            # Block special tokens unless we're done
            if st["phase"] not in ("reason", "done"):
                for blocked_id in self.blocked_special_ids:
                    if blocked_id < len(scores[b]):
                        scores[b, blocked_id] = float('-inf')
            
            # ========== STATE: CoT generation ==========
            if st["phase"] == "cot":
                st["generated_cot"] += 1
                
                # Block ALL phrase start tokens until minimum is reached
                if st["generated_cot"] < self.min_cot_tokens:
                    for token_id in self.phrase_first_tokens:
                        if token_id < len(scores[b]):
                            scores[b, token_id] = float('-inf')
                else:
                    # Check if we just generated a phrase start token
                    if last_token in self.phrase_first_tokens:
                        # Find which variant started
                        for var_idx, variant in enumerate(self.phrase_ids_list):
                            if variant and variant[0] == last_token:
                                st["active_variant"] = var_idx
                                st["phrase_pos"] = 1
                                st["phase"] = "need_ans_phrase"
                                break
            
            # ========== STATE: Force answer phrase token-by-token ==========
            elif st["phase"] == "need_ans_phrase":
                if st["active_variant"] is None:
                    st["active_variant"] = 0
                    st["phrase_pos"] = 0
                
                target_phrase = self.phrase_ids_list[st["active_variant"]]
                
                if st["phrase_pos"] < len(target_phrase):
                    # Force next token in phrase
                    next_token = target_phrase[st["phrase_pos"]]
                    safety = self.newline_ids | ({self.space_id} if self.space_id else set())
                    scores[b] = self._apply_allowlist(scores[b], {next_token}, safety)
                    st["phrase_pos"] += 1
                else:
                    # Phrase complete, transition to answer
                    st["phase"] = "need_answer"
                    st["answer_tokens"] = 0
                    st["answer_path"] = []
            
            # ========== STATE: Generate answer ==========
            elif st["phase"] == "need_answer":
                st["answer_tokens"] += 1
                
                if self.mode == "mcq":
                    # MCQ: allow only valid letters
                    if last_token in self.letters:
                        # Just generated letter, move to newline
                        st["phase"] = "need_newline"
                    else:
                        # Force letter
                        safety = self.newline_ids
                        scores[b] = self._apply_allowlist(scores[b], self.letters, safety)
                
                elif self.mode == "numeric":
                    # Numeric: allow digits + newline
                    if last_token in self.newline_ids:
                        # End of number
                        if self.require_reason:
                            st["phase"] = "need_reason_phrase"
                            st["phrase_pos"] = 0
                            st["active_variant"] = None
                        else:
                            st["phase"] = "done"
                    elif st["answer_tokens"] >= self.max_answer_tokens:
                        # Max length reached, force newline
                        scores[b] = self._apply_allowlist(scores[b], self.newline_ids, set())
                    else:
                        # Allow numeric chars + newline
                        allowed = self.num_chars | self.newline_ids
                        scores[b] = self._apply_allowlist(scores[b], allowed, self.newline_ids)
                
                else:  # labelset
                    # Trie-based label matching
                    if len(st["answer_path"]) == 0:
                        # Start of label
                        st["answer_path"].append(last_token)
                        valid_next = self.label_trie.get_valid_next_tokens(st["answer_path"])
                        
                        if self.label_trie.is_complete_label(st["answer_path"]):
                            # Single-token label complete
                            st["phase"] = "need_newline"
                        elif valid_next:
                            # Continue with trie
                            safety = self.newline_ids
                            scores[b] = self._apply_allowlist(scores[b], valid_next, safety)
                        else:
                            # Invalid start, force newline and bail
                            st["phase"] = "need_newline"
                    else:
                        # Continuing label
                        st["answer_path"].append(last_token)
                        
                        if self.label_trie.is_complete_label(st["answer_path"]):
                            # Label complete
                            st["phase"] = "need_newline"
                        else:
                            valid_next = self.label_trie.get_valid_next_tokens(st["answer_path"])
                            if valid_next:
                                safety = self.newline_ids
                                scores[b] = self._apply_allowlist(scores[b], valid_next, safety)
                            else:
                                # Dead end, force newline
                                st["phase"] = "need_newline"
            
            # ========== STATE: Force newline ==========
            elif st["phase"] == "need_newline":
                # Force a newline token
                scores[b] = self._apply_allowlist(scores[b], self.newline_ids, set())
                
                # After newline, decide next phase
                if last_token in self.newline_ids:
                    if self.require_reason:
                        st["phase"] = "need_reason_phrase"
                        st["phrase_pos"] = 0
                        st["active_variant"] = None
                    else:
                        st["phase"] = "done"
            
            # ========== STATE: Force REASON: phrase ==========
            elif st["phase"] == "need_reason_phrase":
                if st["active_variant"] is None:
                    # Check if we just started with a reason phrase token
                    for var_idx, variant in enumerate(self.reason_phrase_ids_list):
                        if variant and variant[0] == last_token:
                            st["active_variant"] = var_idx
                            st["phrase_pos"] = 1
                            break
                    
                    if st["active_variant"] is None:
                        st["active_variant"] = 0
                        st["phrase_pos"] = 0
                
                target_phrase = self.reason_phrase_ids_list[st["active_variant"]]
                
                if st["phrase_pos"] < len(target_phrase):
                    # Force next token
                    next_token = target_phrase[st["phrase_pos"]]
                    safety = self.newline_ids | ({self.space_id} if self.space_id else set())
                    scores[b] = self._apply_allowlist(scores[b], {next_token}, safety)
                    st["phrase_pos"] += 1
                else:
                    # REASON: phrase complete
                    st["phase"] = "reason"
            
            # ========== STATE: Free reason explanation ==========
            elif st["phase"] == "reason":
                # Track reason tokens
                st["reason_tokens"] = st.get("reason_tokens", 0) + 1
                
                # After max_reason_tokens, force closing tag
                if st["reason_tokens"] >= self.max_reason_tokens:
                    # Force transition to closing answer tag
                    st["phase"] = "closing_answer"
                    st["close_variant"] = 0  # Use first variant
                    st["close_pos"] = 0
                    # Force first token of closing tag
                    target_phrase = self.answer_close_ids_list[0]
                    if target_phrase:
                        scores[b] = self._apply_allowlist(scores[b], {target_phrase[0]}, self.newline_ids)
                        st["close_pos"] = 1
                # Check if model naturally starts closing tag
                elif last_token in self.answer_close_first_tokens:
                    # Find which variant started
                    for var_idx, variant in enumerate(self.answer_close_ids_list):
                        if variant and variant[0] == last_token:
                            # If single-token closing tag, transition to done
                            if len(variant) == 1:
                                st["phase"] = "done"
                            else:
                                # Multi-token closing tag, need to track it
                                st["phase"] = "closing_answer"
                                st["close_variant"] = var_idx
                                st["close_pos"] = 1
                            break
                # Allow natural generation otherwise
            
            # ========== STATE: Closing answer tag ==========
            elif st["phase"] == "closing_answer":
                # Force completion of </answer> closing tag
                target_phrase = self.answer_close_ids_list[st.get("close_variant", 0)]
                close_pos = st.get("close_pos", 1)
                
                if close_pos < len(target_phrase):
                    # Force next token in closing tag
                    next_token = target_phrase[close_pos]
                    safety = self.newline_ids | ({self.space_id} if self.space_id else set())
                    scores[b] = self._apply_allowlist(scores[b], {next_token}, safety)
                    st["close_pos"] = close_pos + 1
                else:
                    # Closing tag complete
                    st["phase"] = "done"
            
            # ========== STATE: Done ==========
            elif st["phase"] == "done":
                # Allow EOS/EOT
                pass
        
        return scores


class StopWhenStructuredTailComplete(StoppingCriteria):
    """
    Stop generation when structured output is complete.
    
    Stops only when controller reaches 'done' state or after minimum reason tokens
    in 'reason' state. Never stops during phrase forcing or answer generation.
    
    Example:
        >>> tokenizer = wrapper.tokenizer
        >>> controller = UnifiedFormatController(...)
        >>> stopping_criteria = StoppingCriteriaList([
        ...     StopWhenStructuredTailComplete(tokenizer, controller)
        ... ])
        >>> outputs = model.generate(..., stopping_criteria=stopping_criteria)
    """
    
    def __init__(self, tokenizer, controller: Optional[UnifiedFormatController] = None, min_reason_tokens: int = 5):
        """
        Args:
            tokenizer: HuggingFace tokenizer for decoding tokens
            controller: Optional controller to check phase state
            min_reason_tokens: Minimum tokens to generate in reason phase before allowing stop
        """
        self.tok = tokenizer
        self.controller = controller
        self.min_reason_tokens = min_reason_tokens
        self.buffer = ""
        self.reason_token_count = {}  # Track reason tokens per batch
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs
    ) -> bool:
        """
        Check if generation should stop based on controller state.
        
        Args:
            input_ids: Generated token IDs so far [batch_size, seq_len]
            scores: Model output scores (unused)
            **kwargs: Additional arguments (unused)
        
        Returns:
            True if generation should stop, False otherwise
        """
        if self.controller is None:
            # Fallback to text-based detection
            self.buffer = self.tok.decode(input_ids[0], skip_special_tokens=False)
            has_answer = ("MCQ ANSWER:" in self.buffer) or ("NUM ANSWER:" in self.buffer) or ("CLS ANSWER:" in self.buffer)
            has_reason = ("REASON:" in self.buffer)
            return has_answer and has_reason
        
        # Use controller state
        phase = self.controller.get_current_phase(0)  # Assume batch_idx 0
        
        # Never stop during structural phases
        if phase in ("cot", "need_ans_phrase", "need_answer", "need_newline", "need_reason_phrase"):
            return False
        
        # Stop immediately if done
        if phase == "done":
            return True
        
        # In reason phase, wait for minimum tokens
        if phase == "reason":
            batch_idx = 0
            if batch_idx not in self.reason_token_count:
                self.reason_token_count[batch_idx] = 0
            self.reason_token_count[batch_idx] += 1
            
            # Allow stopping after minimum reason tokens
            return self.reason_token_count[batch_idx] >= self.min_reason_tokens
        
        return False
