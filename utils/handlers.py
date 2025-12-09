"""
Dataset-specific handlers for Phase-A reasoning experiments.

Each handler encapsulates dataset-specific logic for:
- Prompt building with proper format (newline-anchored answer phrases)
- Structured-decoding configuration (mode, constraints)
- Answer parsing and evaluation

Handlers pass per-example constraints to the unified FSM controller,
keeping the controller dataset-agnostic.
"""

from typing import Dict, Any, List, Optional, Tuple, Protocol, Literal
from transformers import PreTrainedTokenizerBase
import torch

from utils.parse_answers import extract_mcq_answer, extract_numeric_answer, extract_labelset_answer
from utils.format_control import get_blocked_special_ids


class PromptFactory:
    """
    Factory for building dataset-specific prompts with strict format enforcement.
    
    Each method produces a prompt that:
    - For 'cot' mode: Ends with "<cot>\n" to start generation with CoT scaffolding
    - For 'direct' mode: Just asks the question without CoT instructions
    - For 'phase_b' mode: Answer format but NO explicit CoT instruction
    - Enforces exact output format: <cot>...</cot><answer>...\n</answer> (cot mode)
    - Does not mention dataset names in content
    - Only specifies allowed options and format rules
    """
    
    @staticmethod
    def arc_prompt(question: str, choices: dict, allowed_letters: list[str]) -> str:
        """Build ARC prompt with MCQ format (CoT scaffolded)."""
        letters = ", ".join(allowed_letters)
        choices_block = "\n".join([f"{k}) {choices[k]}" for k in allowed_letters])
        return f"""You are answering a multiple-choice question. First, think privately in <cot>…</cot>. Then give the final answer in <answer>…</answer> EXACTLY as specified.

Rules:
- Begin with a literal line "<cot>" and end that section with "</cot>".
- Immediately after, start the "<answer>" block.
- Inside <answer>, output **exactly two lines**:
  MCQ ANSWER: <LETTER>
  REASON: <one short sentence>
- Allowed letters: {letters}
- Do not print anything after </answer>.
- Do not include code fences or role markers.

Question:
{question}

Choices:
{choices_block}

Begin.
<cot>
"""
    
    @staticmethod
    def arc_prompt_direct(question: str, choices: dict, allowed_letters: list[str]) -> str:
        """Build ARC prompt WITHOUT CoT scaffolding - just the question.
        
        This is used in Phase B experiments to test whether injecting the
        reasoning subspace can induce reasoning behavior even without explicit
        CoT instructions.
        """
        choices_block = "\n".join([f"{k}) {choices[k]}" for k in allowed_letters])
        return f"""Question:
{question}

{choices_block}

Answer:"""
    
    @staticmethod
    def arc_prompt_phase_b(question: str, choices: dict, allowed_letters: list[str]) -> str:
        """Build ARC prompt for Phase B: answer format but NO CoT instruction.
        
        This tests whether injecting the reasoning subspace induces reasoning
        behavior without explicitly asking the model to reason step-by-step.
        The model is given the answer format but not told to think.
        """
        letters = ", ".join(allowed_letters)
        choices_block = "\n".join([f"{k}) {choices[k]}" for k in allowed_letters])
        return f"""You are answering a multiple-choice question.

Question:
{question}

Choices:
{choices_block}

Respond with your final answer in this exact format:
<answer>
MCQ ANSWER: <LETTER>
REASON: <brief explanation>
</answer>

Allowed letters: {letters}
"""
    
    @staticmethod
    def mmlu_pro_prompt(question: str, choices: dict, allowed_letters: list[str]) -> str:
        """Build MMLU-Pro prompt with MCQ format (CoT scaffolded)."""
        letters = ", ".join(allowed_letters)
        choices_block = "\n".join([f"{k}) {choices[k]}" for k in allowed_letters])
        return f"""You are answering a multiple-choice question. First, think privately in <cot>…</cot>. Then give the final answer in <answer>…</answer> EXACTLY as specified.

Rules:
- Begin with a literal line "<cot>" and end that section with "</cot>".
- Immediately after, start the "<answer>" block.
- Inside <answer>, output **exactly two lines**:
  MCQ ANSWER: <LETTER>
  REASON: <one short sentence>
- Allowed letters: {letters}
- Do not print anything after </answer>.
- Do not include code fences or role markers.

Question:
{question}

Choices:
{choices_block}

Begin.
<cot>
"""
    
    @staticmethod
    def mmlu_pro_prompt_direct(question: str, choices: dict, allowed_letters: list[str]) -> str:
        """Build MMLU-Pro prompt WITHOUT CoT scaffolding - just the question.
        
        This is used in Phase B experiments to test whether injecting the
        reasoning subspace can induce reasoning behavior even without explicit
        CoT instructions.
        """
        choices_block = "\n".join([f"{k}) {choices[k]}" for k in allowed_letters])
        return f"""Question:
{question}

{choices_block}

Answer:"""
    
    @staticmethod
    def mmlu_pro_prompt_phase_b(question: str, choices: dict, allowed_letters: list[str]) -> str:
        """Build MMLU-Pro prompt for Phase B: answer format but NO CoT instruction.
        
        This tests whether injecting the reasoning subspace induces reasoning
        behavior without explicitly asking the model to reason step-by-step.
        The model is given the answer format but not told to think.
        """
        letters = ", ".join(allowed_letters)
        choices_block = "\n".join([f"{k}) {choices[k]}" for k in allowed_letters])
        return f"""You are answering a multiple-choice question.

Question:
{question}

Choices:
{choices_block}

Respond with your final answer in this exact format:
<answer>
MCQ ANSWER: <LETTER>
REASON: <brief explanation>
</answer>

Allowed letters: {letters}
"""
    
    @staticmethod
    def gsm8k_prompt(problem: str) -> str:
        """Build GSM8K prompt with numeric format (CoT scaffolded)."""
        return f"""You are solving an arithmetic word problem. First, think privately in <cot>…</cot>. Then give the final answer in <answer>…</answer> EXACTLY as specified.

Rules:
- Begin with a literal line "<cot>" and end that section with "</cot>".
- Immediately after, start the "<answer>" block.
- Inside <answer>, output **exactly two lines**:
  NUM ANSWER: <number>   (digits only; optional leading '-' for negatives; no commas, spaces, or currency)
  REASON: <one short sentence>
- Do not print anything after </answer>.
- Do not include code fences or role markers.

Problem:
{problem}

Begin.
<cot>
"""
    
    @staticmethod
    def gsm8k_prompt_direct(problem: str) -> str:
        """Build GSM8K prompt WITHOUT CoT scaffolding - just the problem.
        
        This is used in Phase B experiments to test whether injecting the
        reasoning subspace can induce reasoning behavior even without explicit
        CoT instructions.
        """
        return f"""Problem:
{problem}

Answer:"""
    
    @staticmethod
    def gsm8k_prompt_phase_b(problem: str) -> str:
        """Build GSM8K prompt for Phase B: answer format but NO CoT instruction.
        
        This tests whether injecting the reasoning subspace induces reasoning
        behavior without explicitly asking the model to reason step-by-step.
        The model is given the answer format but not told to think.
        """
        return f"""You are solving an arithmetic word problem.

Problem:
{problem}

Respond with your final answer in this exact format:
<answer>
NUM ANSWER: <number>
REASON: <brief explanation>
</answer>

The number should be digits only (optional '-' for negatives; no commas, spaces, or currency).
"""


def build_answer_block(task_type: str, allowed_letters=None):
    """
    Build dataset-agnostic but task-specific answer block prefix.
    
    Constructs a fixed answer prefix that the model will continue from.
    The model is not asked to "follow a format"; instead we prewrite 
    the first line(s) and the model fills in after the colon.
    
    Args:
        task_type: One of "mcq", "numeric", or "labelset"
        allowed_letters: For MCQ - list of allowed letters (unused but kept for API consistency)
    
    Returns:
        Tuple of (answer_prefix, reason_header, answer_suffix)
        - answer_prefix: String to prepend to generation (includes opening tag and first line)
        - reason_header: String for reason line header
        - answer_suffix: Closing tag
    
    Example:
        >>> answer_prefix, reason_header, answer_suffix = build_answer_block("mcq")
        >>> # answer_prefix = "<answer>\nMCQ ANSWER: "
        >>> # reason_header = "\nREASON: "
        >>> # answer_suffix = "\n</answer>"
    """
    if task_type == "mcq":
        first_line = "MCQ ANSWER: "
    elif task_type == "numeric":
        first_line = "NUM ANSWER: "
    elif task_type == "labelset":
        first_line = "CLS ANSWER: "
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    answer_prefix = "<answer>\n" + first_line  # model continues here
    reason_header = "\nREASON: "
    answer_suffix = "\n</answer>"
    
    return answer_prefix, reason_header, answer_suffix


class Example(dict):
    """Typed alias for dataset examples (already dict-like in loaders)."""
    pass


class BaseHandler(Protocol):
    """
    Protocol defining the interface for dataset-specific handlers.
    
    Each handler owns:
    - task_type detection
    - prompt building with structured format
    - controller configuration (mode, constraints)
    - answer parsing and comparison
    """
    
    name: str  # e.g., "arc", "gsm8k", "mmlu_pro"
    
    def task_type(self, ex: Example) -> str:
        """Return 'mcq' | 'numeric' | 'labelset'."""
        ...
    
    def build_prompt(
        self, tokenizer: PreTrainedTokenizerBase, ex: Example
    ) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, Any]]:
        """
        Build prompt and return inputs + controller config.
        
        Returns:
            Tuple of (input_ids, attention_mask, cot_start_idx, controller_kwargs)
            
        controller_kwargs feeds your structured controller:
          - mode: 'mcq'|'numeric'|'labelset'
          - allowed_letters: List[str]          (mcq)
          - label_token_sequences: List[List[int]] (labelset)
          - blocked_special_ids: Set[int]
          - min_cot_tokens: int
        
        The method must append <cot> as the last pre-gen token (after chat template)
        and compute cot_start_idx accordingly.
        """
        ...
    
    def parse_pred(self, raw_decoded: str, ex: Example) -> Tuple[Optional[str], str]:
        """
        Parse prediction from raw decoded output.
        
        Returns:
            Tuple of (prediction, status) where:
            - prediction: normalized answer ('A' for mcq, '1234.5' for numeric, 'supported' for labelset)
            - status: 'valid' if parsed successfully, error code otherwise
        """
        ...
    
    def gold_target(self, ex: Example) -> str:
        """
        Return normalized gold target.
        
        Returns:
            Normalized gold answer string
        """
        ...
    
    def compare(self, pred: str, gold: str) -> bool:
        """
        Compare prediction to gold answer.
        
        Returns:
            True if correct, False otherwise
        """
        ...


class ARCHandler:
    """Handler for AI2 ARC (MCQ with dynamic alphabet)."""
    
    name = "arc"
    
    def task_type(self, ex: Example) -> str:
        return "mcq"
    
    def build_prompt(
        self, tokenizer: PreTrainedTokenizerBase, ex: Example, prompt_mode: str = "cot"
    ) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, Any]]:
        """Build ARC prompt with MCQ format.
        
        Args:
            tokenizer: HuggingFace tokenizer
            ex: Dataset example
            prompt_mode: "cot" for CoT-scaffolded prompt, "direct" for bare question,
                        "phase_b" for answer format without CoT instruction
        
        Returns:
            Tuple of (input_ids, attention_mask, cot_start_idx, controller_kwargs)
        """
        from hf_model_wrapper import build_input_with_cot, build_input_direct
        
        # Get choices and build dynamic alphabet
        choices_list = getattr(ex, "choices", [])
        if not choices_list:
            choices_list = ["A", "B", "C", "D"]  # Fallback
        
        allowed_letters = [chr(65 + i) for i in range(len(choices_list))]
        
        # Build choices dict for prompt
        choices_dict = {letter: choice for letter, choice in zip(allowed_letters, choices_list)}
        
        # Get question
        question = getattr(ex, "question", "")
        
        if prompt_mode == "direct":
            # Build direct prompt WITHOUT CoT scaffolding
            prompt = PromptFactory.arc_prompt_direct(question, choices_dict, allowed_letters)
            input_ids, attention_mask = build_input_direct(tokenizer, prompt, torch.device('cpu'))
            cot_start_idx = -1  # No CoT token in direct mode
            
            # Build controller kwargs for free-form generation
            controller_kwargs = {
                'mode': 'direct',  # Special mode for free-form generation
                'allowed_letters': allowed_letters,
                'blocked_special_ids': get_blocked_special_ids(tokenizer),
            }
        elif prompt_mode == "phase_b":
            # Build Phase B prompt: answer format but NO CoT instruction
            prompt = PromptFactory.arc_prompt_phase_b(question, choices_dict, allowed_letters)
            input_ids, attention_mask = build_input_direct(tokenizer, prompt, torch.device('cpu'))
            cot_start_idx = -1  # No CoT token in phase_b mode
            
            # Build controller kwargs for phase_b generation
            # Uses MCQ mode for answer extraction but no CoT enforcement
            controller_kwargs = {
                'mode': 'phase_b',  # Phase B mode: answer format, no CoT
                'allowed_letters': allowed_letters,
                'blocked_special_ids': get_blocked_special_ids(tokenizer),
                'require_reason': True,
            }
        else:
            # Build CoT-scaffolded prompt (default)
            prompt = PromptFactory.arc_prompt(question, choices_dict, allowed_letters)
            input_ids, attention_mask, cot_start_idx = build_input_with_cot(
                tokenizer, prompt, torch.device('cpu')
            )
            
            # Build controller kwargs for structured generation
            controller_kwargs = {
                'mode': 'mcq',
                'phrase_text': 'MCQ ANSWER:',
                'allowed_letters': allowed_letters,
                'blocked_special_ids': get_blocked_special_ids(tokenizer),
                'min_cot_tokens': 24,
                'require_reason': True,
            }
        
        return input_ids, attention_mask, cot_start_idx, controller_kwargs
    
    def parse_pred(self, raw_decoded: str, ex: Example) -> Tuple[Optional[str], str]:
        """Parse MCQ answer from output."""
        # Get allowed letters from example
        choices = getattr(ex, "choices", [])
        allowed_letters = [chr(65 + i) for i in range(len(choices))] if choices else None
        
        return extract_mcq_answer(raw_decoded, allowed_letters)
    
    def gold_target(self, ex: Example) -> str:
        """Return gold letter."""
        gold = getattr(ex, "correct_answer", "")
        return str(gold).strip().upper()
    
    def compare(self, pred: str, gold: str) -> bool:
        """Case-insensitive letter comparison."""
        return pred.upper() == gold.upper()


class GSM8KHandler:
    """Handler for GSM8K (numeric answers)."""
    
    name = "gsm8k"
    
    def task_type(self, ex: Example) -> str:
        return "numeric"
    
    def build_prompt(
        self, tokenizer: PreTrainedTokenizerBase, ex: Example, prompt_mode: str = "cot"
    ) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, Any]]:
        """Build GSM8K prompt with numeric format.
        
        Args:
            tokenizer: HuggingFace tokenizer
            ex: Dataset example
            prompt_mode: "cot" for CoT-scaffolded prompt, "direct" for bare question,
                        "phase_b" for answer format without CoT instruction
        
        Returns:
            Tuple of (input_ids, attention_mask, cot_start_idx, controller_kwargs)
        """
        from hf_model_wrapper import build_input_with_cot, build_input_direct
        
        # Get question
        question = getattr(ex, "question", "")
        
        if prompt_mode == "direct":
            # Build direct prompt WITHOUT CoT scaffolding
            prompt = PromptFactory.gsm8k_prompt_direct(question)
            input_ids, attention_mask = build_input_direct(tokenizer, prompt, torch.device('cpu'))
            cot_start_idx = -1  # No CoT token in direct mode
            
            # Build controller kwargs for free-form generation
            controller_kwargs = {
                'mode': 'direct',  # Special mode for free-form generation
                'blocked_special_ids': get_blocked_special_ids(tokenizer),
            }
        elif prompt_mode == "phase_b":
            # Build Phase B prompt: answer format but NO CoT instruction
            prompt = PromptFactory.gsm8k_prompt_phase_b(question)
            input_ids, attention_mask = build_input_direct(tokenizer, prompt, torch.device('cpu'))
            cot_start_idx = -1  # No CoT token in phase_b mode
            
            # Build controller kwargs for phase_b generation
            controller_kwargs = {
                'mode': 'phase_b',  # Phase B mode: answer format, no CoT
                'blocked_special_ids': get_blocked_special_ids(tokenizer),
                'require_reason': True,
                'max_answer_tokens': 32,
            }
        else:
            # Build CoT-scaffolded prompt (default)
            prompt = PromptFactory.gsm8k_prompt(question)
            input_ids, attention_mask, cot_start_idx = build_input_with_cot(
                tokenizer, prompt, torch.device('cpu')
            )
            
            # Build controller kwargs for structured generation
            controller_kwargs = {
                'mode': 'numeric',
                'phrase_text': 'NUM ANSWER:',
                'blocked_special_ids': get_blocked_special_ids(tokenizer),
                'min_cot_tokens': 24,
                'require_reason': True,
                'max_answer_tokens': 32,
            }
        
        return input_ids, attention_mask, cot_start_idx, controller_kwargs
    
    def parse_pred(self, raw_decoded: str, ex: Example) -> Tuple[Optional[str], str]:
        """Parse numeric answer from output."""
        return extract_numeric_answer(raw_decoded)
    
    def gold_target(self, ex: Example) -> str:
        """Return normalized gold number."""
        gold = getattr(ex, "correct_answer", "")
        return str(gold).strip()
    
    def compare(self, pred: str, gold: str, tolerance: float = 1e-6) -> bool:
        """Numeric comparison with tolerance."""
        try:
            pred_num = float(pred)
            gold_num = float(gold)
            return abs(pred_num - gold_num) < tolerance
        except ValueError:
            # Fallback to string comparison
            return pred == gold


class MMLUProHandler:
    """Handler for MMLU-Pro (MCQ with variable K)."""
    
    name = "mmlu_pro"
    
    def task_type(self, ex: Example) -> str:
        return "mcq"
    
    def build_prompt(
        self, tokenizer: PreTrainedTokenizerBase, ex: Example, prompt_mode: str = "cot"
    ) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, Any]]:
        """Build MMLU-Pro prompt with MCQ format.
        
        Args:
            tokenizer: HuggingFace tokenizer
            ex: Dataset example
            prompt_mode: "cot" for CoT-scaffolded prompt, "direct" for bare question,
                        "phase_b" for answer format without CoT instruction
        
        Returns:
            Tuple of (input_ids, attention_mask, cot_start_idx, controller_kwargs)
        """
        from hf_model_wrapper import build_input_with_cot, build_input_direct
        
        # Get choices and build dynamic alphabet
        choices_list = getattr(ex, "choices", [])
        if not choices_list:
            choices_list = ["A", "B", "C", "D"]  # Fallback
        
        allowed_letters = [chr(65 + i) for i in range(len(choices_list))]
        
        # Build choices dict for prompt
        choices_dict = {letter: choice for letter, choice in zip(allowed_letters, choices_list)}
        
        # Get question
        question = getattr(ex, "question", "")
        
        if prompt_mode == "direct":
            # Build direct prompt WITHOUT CoT scaffolding
            prompt = PromptFactory.mmlu_pro_prompt_direct(question, choices_dict, allowed_letters)
            input_ids, attention_mask = build_input_direct(tokenizer, prompt, torch.device('cpu'))
            cot_start_idx = -1  # No CoT token in direct mode
            
            # Build controller kwargs for free-form generation
            controller_kwargs = {
                'mode': 'direct',  # Special mode for free-form generation
                'allowed_letters': allowed_letters,
                'blocked_special_ids': get_blocked_special_ids(tokenizer),
            }
        elif prompt_mode == "phase_b":
            # Build Phase B prompt: answer format but NO CoT instruction
            prompt = PromptFactory.mmlu_pro_prompt_phase_b(question, choices_dict, allowed_letters)
            input_ids, attention_mask = build_input_direct(tokenizer, prompt, torch.device('cpu'))
            cot_start_idx = -1  # No CoT token in phase_b mode
            
            # Build controller kwargs for phase_b generation
            controller_kwargs = {
                'mode': 'phase_b',  # Phase B mode: answer format, no CoT
                'allowed_letters': allowed_letters,
                'blocked_special_ids': get_blocked_special_ids(tokenizer),
                'require_reason': True,
            }
        else:
            # Build CoT-scaffolded prompt (default)
            prompt = PromptFactory.mmlu_pro_prompt(question, choices_dict, allowed_letters)
            input_ids, attention_mask, cot_start_idx = build_input_with_cot(
                tokenizer, prompt, torch.device('cpu')
            )
            
            # Build controller kwargs for structured generation
            controller_kwargs = {
                'mode': 'mcq',
                'phrase_text': 'MCQ ANSWER:',
                'allowed_letters': allowed_letters,
                'blocked_special_ids': get_blocked_special_ids(tokenizer),
                'min_cot_tokens': 24,
                'require_reason': True,
            }
        
        return input_ids, attention_mask, cot_start_idx, controller_kwargs
    
    def parse_pred(self, raw_decoded: str, ex: Example) -> Tuple[Optional[str], str]:
        """Parse MCQ answer from output."""
        # Get allowed letters from example
        choices = getattr(ex, "choices", [])
        allowed_letters = [chr(65 + i) for i in range(len(choices))] if choices else None
        
        return extract_mcq_answer(raw_decoded, allowed_letters)
    
    def gold_target(self, ex: Example) -> str:
        """Return gold letter."""
        gold = getattr(ex, "correct_answer", "")
        return str(gold).strip().upper()
    
    def compare(self, pred: str, gold: str) -> bool:
        """Case-insensitive letter comparison."""
        return pred.upper() == gold.upper()


# Registry of all handlers
HANDLERS: Dict[str, BaseHandler] = {
    "arc": ARCHandler(),
    "gsm8k": GSM8KHandler(),
    "mmlu_pro": MMLUProHandler(),
}


def get_handler(dataset_name: str) -> BaseHandler:
    """
    Get handler for dataset name.
    
    Args:
        dataset_name: Name of dataset (case-insensitive)
    
    Returns:
        Handler instance
    
    Raises:
        ValueError: If dataset not supported
    """
    name = dataset_name.lower().strip()
    if name not in HANDLERS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: {list(HANDLERS.keys())}"
        )
    return HANDLERS[name]


def compute_locality_mask_post_generation(
    decoded_text: str,
    prompt_len: int,
    gen_token_ids: List[int],
    gen_decoded_str: str,
    locality: str,
    tokenizer: PreTrainedTokenizerBase,
    K_answer: int = 16,
    K_cot: Optional[int] = None,
    max_mask_tokens: int = 256,
) -> Tuple[torch.BoolTensor, Dict[str, Any]]:
    """
    Compute locality mask after generation completes using robust answer phrase detection.
    
    This function:
    1. Uses find_answer_span() for robust char-level detection
    2. Maps char spans to token spans via tokenizer offsets
    3. Builds windowed masks with K padding
    4. Enforces max_mask_tokens cap
    5. Returns diagnostic info
    
    Args:
        decoded_text: Full decoded text (prompt + generated)
        prompt_len: Length of prompt in tokens
        gen_token_ids: ONLY the generated token IDs (no prompt)
        gen_decoded_str: ONLY the generated text segment
        locality: One of "cot", "answer", "all"
        tokenizer: HuggingFace tokenizer for mapping
        K_answer: Window padding for answer locality (tokens on each side)
        K_cot: If provided, take only last K_cot tokens of CoT span
        max_mask_tokens: Maximum number of tokens to mask (enforced by dropping earliest)
    
    Returns:
        Tuple of (mask, diag_dict):
        - mask: Boolean tensor [total_len] in absolute decode coordinates
        - diag_dict with prompt_len, cot_span, answer_span, masked_count
    
    Example:
        >>> mask, diag = compute_locality_mask_post_generation(
        ...     full_text, prompt_len, gen_ids, gen_text, "cot", tokenizer
        ... )
        >>> print(f"[diag-locality] masked {diag['masked_count']} of {len(gen_ids)} gen tokens")
    """
    from utils.parse_answers import find_answer_span
    from hf_model_wrapper import token_spans_from_char_span
    import os
    
    total_len = prompt_len + len(gen_token_ids)
    
    # Step 1: Find answer phrase in decoded text (char-level)
    answer_result = find_answer_span(decoded_text)
    answer_char_span = answer_result['char_span'] if answer_result['found'] else None
    
    # Step 2: Map char span to token span (in generated segment coordinates)
    answer_span_gen = None
    if answer_char_span is not None and answer_result['found']:
        # Need to find offset of gen_decoded_str within decoded_text
        gen_start_char = decoded_text.find(gen_decoded_str)
        if gen_start_char >= 0:
            # Adjust char span to gen segment coordinates
            ans_start_char = answer_char_span[0] - gen_start_char
            ans_end_char = answer_char_span[1] - gen_start_char
            
            if 0 <= ans_start_char < len(gen_decoded_str):
                # Map to token indices
                tok_span = token_spans_from_char_span(
                    gen_decoded_str, tokenizer, (ans_start_char, min(ans_end_char, len(gen_decoded_str)))
                )
                if tok_span is not None:
                    answer_span_gen = tok_span  # (start_tok, end_tok) in gen coords
    
    # Step 3: Define CoT span (in gen coordinates)
    if answer_span_gen is not None:
        cot_span_gen = (0, answer_span_gen[0])  # From start of gen to answer phrase
    else:
        cot_span_gen = (0, len(gen_token_ids))  # Until end if no answer found
    
    # Step 4: Build boolean mask for generated tokens
    gen_mask = [False] * len(gen_token_ids)
    
    if locality == "answer":
        if answer_span_gen is not None:
            # Windowed mask centered on answer phrase with K_answer padding
            ans_start, ans_end = answer_span_gen
            ans_center = (ans_start + ans_end) // 2
            ans_len = ans_end - ans_start
            
            # Window: [center - K_answer, center + K_answer + ans_len]
            window_start = max(0, ans_start - K_answer)
            window_end = min(len(gen_token_ids), ans_end + K_answer)
            
            for i in range(window_start, window_end):
                gen_mask[i] = True
        else:
            # Fallback: mask last 8 tokens only
            fallback_start = max(0, len(gen_token_ids) - 8)
            for i in range(fallback_start, len(gen_token_ids)):
                gen_mask[i] = True
            
            if os.environ.get("DEBUG_LOCALITY", "0") == "1" or os.environ.get("WRV_DEBUG_LOCALITY", "0") == "1":
                print("[warn] answer locality requested but answer span not found; masking last 8 tokens only")
    
    elif locality == "cot":
        # Mask CoT span
        cot_start, cot_end = cot_span_gen
        if K_cot is not None and K_cot > 0:
            # Take only last K_cot tokens of CoT
            cot_start = max(cot_start, cot_end - K_cot)
        
        for i in range(cot_start, cot_end):
            gen_mask[i] = True
    
    elif locality == "all":
        # Mask all generated tokens
        gen_mask = [True] * len(gen_token_ids)
    
    else:
        raise ValueError(f"Unknown locality: {locality}")
    
    # Step 5: Enforce max_mask_tokens by dropping earliest True positions
    masked_count = sum(gen_mask)
    if masked_count > max_mask_tokens:
        # Find True positions
        true_positions = [i for i, m in enumerate(gen_mask) if m]
        # Keep only last max_mask_tokens
        positions_to_keep = set(true_positions[-max_mask_tokens:])
        gen_mask = [i in positions_to_keep for i in range(len(gen_token_ids))]
        masked_count = max_mask_tokens
    
    # Step 6: Build full mask in absolute decode coordinates
    full_mask = torch.zeros(total_len, dtype=torch.bool)
    full_mask[prompt_len:prompt_len + len(gen_token_ids)] = torch.tensor(gen_mask)
    
    # Step 7: Build diagnostic dict
    cot_span_abs = (prompt_len + cot_span_gen[0], prompt_len + cot_span_gen[1]) if cot_span_gen else None
    answer_span_abs = (prompt_len + answer_span_gen[0], prompt_len + answer_span_gen[1]) if answer_span_gen else None
    
    diag = {
        "prompt_len": prompt_len,
        "cot_span": cot_span_abs,
        "answer_span": answer_span_abs,
        "masked_count": masked_count,
    }
    
    # Debug logging
    if os.environ.get("DEBUG_LOCALITY", "0") == "1" or os.environ.get("WRV_DEBUG_LOCALITY", "0") == "1":
        print(f"[diag-locality] prompt_len={prompt_len} locality={locality}")
        if cot_span_abs:
            cot_len = cot_span_abs[1] - cot_span_abs[0]
            print(f"  cot_span=[{cot_span_abs[0]}, {cot_span_abs[1]}) len={cot_len}")
        if answer_span_abs:
            ans_len = answer_span_abs[1] - answer_span_abs[0]
            print(f"  answer_span=[{answer_span_abs[0]}, {answer_span_abs[1]}) len={ans_len}")
        print(f"  final_mask: {masked_count} tokens marked True out of {len(gen_token_ids)}")
    
    # Safety guard
    if masked_count == 0 and os.environ.get("DEBUG_LOCALITY", "0") == "1":
        print(f"[warn] final mask is empty for locality={locality}, intervention will be skipped")
    
    return full_mask, diag
