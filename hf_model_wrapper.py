from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessor,
    LogitsProcessorList,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


ModuleSelector = Union[
    str,
    Sequence[str],
    Callable[[str, torch.nn.Module], bool],
]

HookCallable = Callable[..., None]
HookType = Union["forward", "forward_pre", "backward"]


def token_spans_from_char_span(
    gen_decoded_str: str,
    tokenizer: PreTrainedTokenizerBase,
    char_span: Tuple[int, int]
) -> Optional[Tuple[int, int]]:
    """Map character span to token span within generated segment."""
    start_char, end_char = char_span
    
    # Validate inputs
    if start_char < 0 or end_char > len(gen_decoded_str) or start_char >= end_char:
        return None
    
    # Re-tokenize with offsets (fast tokenizer required)
    try:
        encoding = tokenizer(gen_decoded_str, add_special_tokens=False, return_offsets_mapping=True)
    except Exception:
        # Fallback if tokenizer doesn't support offsets
        return None
    
    if 'offset_mapping' not in encoding:
        return None
    
    offsets = encoding['offset_mapping']
    
    # Find token indices that overlap with char span
    start_tok_idx = None
    end_tok_idx = None
    
    for tok_idx, (tok_start, tok_end) in enumerate(offsets):
        # Find first token that overlaps with start_char
        if start_tok_idx is None and tok_end > start_char:
            start_tok_idx = tok_idx
        
        # Find last token that overlaps with end_char
        if tok_start < end_char:
            end_tok_idx = tok_idx + 1  # Exclusive end
    
    if start_tok_idx is None or end_tok_idx is None:
        return None
    
    return (start_tok_idx, end_tok_idx)


def get_chat_template(model_name_or_path: str):
    """
    Get chat template builder function for specific model family.
    
    Returns a function that builds properly formatted prompts for:
    - Mistral: [INST] ... [/INST]
    - Qwen: <|im_start|>system/user/assistant<|im_end|>
    - LLaMA: Raw prompt with markers
    
    Args:
        model_name_or_path: Model identifier (e.g., "mistralai/Mistral-7B-Instruct-v0.3")
    
    Returns:
        Function that takes (prompt: str, with_cot_marker: bool) and returns formatted string
    
    Example:
        >>> builder = get_chat_template("mistralai/Mistral-7B-Instruct-v0.3")
        >>> formatted = builder("What is 2+2?", with_cot_marker=True)
        >>> # Returns: "[INST] What is 2+2?\n<cot>\n [/INST]"
    """
    name = model_name_or_path.lower()
    
    if "mistral" in name:
        # Mistral Instruct: [INST] ... [/INST]
        # Place <cot> marker right after the user content so it ends up in the generation segment
        def build(prompt: str, with_cot_marker: bool = True) -> str:
            user = prompt + ("\n<cot>\n" if with_cot_marker else "\n")
            return f"[INST] {user} [/INST]"
        return build
    
    elif "qwen" in name:
        # Qwen2.5: uses chat template with roles
        # Keep it simple & explicit to avoid HF auto-template surprises
        def build(prompt: str, with_cot_marker: bool = True) -> str:
            sys = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            user = "<|im_start|>user\n" + prompt + ("\n<cot>\n" if with_cot_marker else "\n") + "<|im_end|>\n"
            asst = "<|im_start|>assistant\n"
            return sys + user + asst  # model will complete assistant
        return build
    
    else:
        # LLaMA family: many repos use llama-2-style chat; keep raw prompt with <cot> line
        def build(prompt: str, with_cot_marker: bool = True) -> str:
            return prompt + ("\n<cot>\n" if with_cot_marker else "\n")
        return build


@dataclass
class HFModelConfig:
    """
    Configuration container for loading HuggingFace causal decoder models.

    Attributes:
        model_name: HuggingFace repository ID or local path.
        revision: Optional git revision, branch, or commit.
        dtype: Torch dtype to load the model in. Accepts torch.dtype instances or
            string aliases such as "float16", "bfloat16", "auto".
        device: Target device. Use "auto" to rely on Accelerate device mapping.
            Can be values like "cpu", "cuda", "cuda:0", "mps".
        trust_remote_code: Whether to allow custom model code from the repo.
        use_fast_tokenizer: Prefer fast tokenizer implementation when available.
        cache_dir: Optional directory to store/download model artifacts.
        special_tokens: List of special tokens to add to the tokenizer (e.g., ["<cot>", "</cot>", "<answer>"]).
        init_special_tokens_with_avg: If True, initialize new token embeddings with average of existing embeddings.
        tokenizer_kwargs: Extra keyword arguments forwarded to AutoTokenizer.
        model_kwargs: Extra keyword arguments forwarded to AutoModelForCausalLM.
    """

    model_name: str
    revision: Optional[str] = None
    dtype: Optional[Union[str, torch.dtype]] = "auto"
    device: Optional[str] = "auto"
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = True
    cache_dir: Optional[Union[str, Path]] = None
    special_tokens: Optional[List[str]] = None
    init_special_tokens_with_avg: bool = True
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


class HFModelWrapper:
    """
    Convenience wrapper for loading HuggingFace causal decoder models with unified
    access to the tokenizer, model, and hooking utilities.

    Example
    -------

    ::

        from hf_model_wrapper import HFModelConfig, HFModelWrapper

        config = HFModelConfig(
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            dtype="bfloat16",
            device="cuda",  # or "auto" to use accelerate device mapping
        )

        wrapper = HFModelWrapper(config).load()

        # Run generation
        output = wrapper.generate(
            "Explain the difference between supervised and unsupervised learning.",
            max_new_tokens=128,
        )
        print(output)

        # Register forward hook to capture hidden activations
        activations = {}

        def capture(name):
            def _hook(module, inp, out):
                activations[name] = out

            return _hook

        hook_handle = wrapper.register_hook(
            module_selector=lambda name, module: name.endswith("mlp"),
            hook_fn=capture("mlp_output"),
            hook_type="forward",
            hook_group="mlp_capture",
        )

        with torch.no_grad():
            wrapper.model(**wrapper.tokenize("Hello, world!"))

        # Inspect captured activations
        tensor = activations["mlp_output"]

        # Remove hooks when done
        wrapper.remove_hooks("mlp_capture")

    Supported models include, but are not limited to:
        - mistralai/Mistral-7B-Instruct-v0.3
        - Qwen/Qwen2.5-7B-Instruct
        - meta-llama/Llama-3.1-8B-Instruct
    """

    def __init__(self, config: HFModelConfig):
        self.config = config
        self._auto_config: Optional[AutoConfig] = None
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._model: Optional[PreTrainedModel] = None
        self._hook_handles: Dict[str, List[torch.utils.hooks.RemovableHandle]] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        self._ensure_loaded()
        assert self._tokenizer is not None
        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        self._ensure_loaded()
        assert self._model is not None
        return self._model

    @property
    def config_object(self) -> AutoConfig:
        self._ensure_loaded()
        assert self._auto_config is not None
        return self._auto_config

    @property
    def primary_device(self) -> torch.device:
        self._ensure_loaded()
        assert self._model is not None
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def devices(self) -> List[torch.device]:
        self._ensure_loaded()
        assert self._model is not None
        devices = {param.device for param in self._model.parameters()}
        return sorted(devices, key=str)

    def load(self, *, force_reload: bool = False) -> "HFModelWrapper":
        """
        Materialize tokenizer/model pair according to the provided configuration.

        Args:
            force_reload: If True, re-download/reload the model even if cached.

        Returns:
            Self for fluent chaining.
        """
        if not force_reload and self._model is not None and self._tokenizer is not None:
            return self

        dtype = self._resolve_dtype(self.config.dtype)
        cache_dir = str(self.config.cache_dir) if self.config.cache_dir else None

        tokenizer_kwargs = dict(self.config.tokenizer_kwargs)
        tokenizer_kwargs.setdefault("use_fast", self.config.use_fast_tokenizer)
        if cache_dir:
            tokenizer_kwargs.setdefault("cache_dir", cache_dir)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            revision=self.config.revision,
            trust_remote_code=self.config.trust_remote_code,
            **tokenizer_kwargs,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Add special tokens if specified (with robust checking)
        num_added_tokens = 0
        if self.config.special_tokens:
            # Check which tokens are already present
            existing_vocab = self._tokenizer.get_added_vocab()
            tokens_to_add = [t for t in self.config.special_tokens if t not in existing_vocab]
            
            if tokens_to_add:
                special_tokens_dict = {"additional_special_tokens": tokens_to_add}
                num_added_tokens = self._tokenizer.add_special_tokens(special_tokens_dict)
                print(f"[model-load] Added {num_added_tokens} special tokens: {tokens_to_add}")
            else:
                print(f"[model-load] All special tokens already present: {self.config.special_tokens}")

        model_kwargs = dict(self.config.model_kwargs)
        if cache_dir:
            model_kwargs.setdefault("cache_dir", cache_dir)
        if dtype is not None:
            model_kwargs.setdefault("torch_dtype", dtype)
        model_kwargs.setdefault("device_map", self._resolve_device_map(self.config.device))
        model_kwargs.setdefault("trust_remote_code", self.config.trust_remote_code)

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            revision=self.config.revision,
            **model_kwargs,
        )
        self._model.eval()

        # Load config first so we can check vocab_size
        self._auto_config = AutoConfig.from_pretrained(
            self.config.model_name,
            revision=self.config.revision,
            trust_remote_code=self.config.trust_remote_code,
            cache_dir=cache_dir,
        )

        if num_added_tokens > 0:
            new_vocab_size = len(self._tokenizer)
            config_vocab_size = self._auto_config.vocab_size
            
            if new_vocab_size > config_vocab_size:
                # Need to expand embedding table
                self._model.resize_token_embeddings(new_vocab_size)
                print(f"[model-load] Resized embeddings: {config_vocab_size} -> {new_vocab_size}")
            else:
                # Special tokens fit within existing vocab_size - no resize needed
                print(f"[model-load] Special tokens fit within existing vocab ({new_vocab_size} <= {config_vocab_size}), no resize needed")
            
            if self.config.init_special_tokens_with_avg:
                with torch.no_grad():
                    emb = self._model.get_input_embeddings().weight
                    avg = emb.mean(dim=0, keepdim=True)
                    new_token_ids = self._tokenizer.convert_tokens_to_ids(self.config.special_tokens)
                    emb[new_token_ids] = avg

        if model_kwargs["device_map"] is None and self.config.device not in (None, "auto"):
            self._model.to(self._resolve_device(self.config.device))

        # Config already loaded above
        return self

    def unload(self) -> None:
        """Delete model/tokenizer to free memory."""
        self.remove_hooks()
        self._model = None
        self._tokenizer = None
        self._auto_config = None
        torch.cuda.empty_cache()

    def tokenize(
        self,
        text: Union[str, Sequence[str]],
        *,
        return_tensors: str = "pt",
        padding: Union[bool, str] = True,
        **tokenizer_kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text into model inputs."""
        tokens = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            **tokenizer_kwargs,
        )
        return {key: value.to(self.primary_device) for key, value in tokens.items()}

    def detokenize(self, token_ids: Union[List[int], torch.Tensor], **decode_kwargs: Any) -> str:
        """Convert token IDs back into text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, **decode_kwargs)

    def get_special_token_ids(self) -> Dict[str, int]:
        """
        Get the token IDs for all added special tokens.
        
        Returns:
            Dictionary mapping special token strings to their token IDs.
        """
        self._ensure_loaded()
        if not self.config.special_tokens:
            return {}
        
        return {
            token: self._tokenizer.convert_tokens_to_ids(token)
            for token in self.config.special_tokens
        }

    def generate(
        self,
        prompt: Union[str, Sequence[str]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        return_text: bool = True,
        skip_special_tokens: bool = True,
        **generate_kwargs: Any,
    ) -> Union[str, List[str], torch.Tensor]:
        """
        Run autoregressive generation with the wrapped model.

        Args:
            prompt: Input text or batch of texts.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (ignored for greedy decoding).
            return_text: If True, decode to text. Otherwise return token IDs tensor.
            skip_special_tokens: Whether to remove special tokens when decoding.
            generate_kwargs: Additional args forwarded to `model.generate`.

        Returns:
            Generated text or tensor of token IDs depending on `return_text`.
        """
        self._ensure_loaded()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self.primary_device) for key, value in inputs.items()}

        with torch.no_grad():
            sequences = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **generate_kwargs,
            )

        if return_text:
            texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)
            if isinstance(prompt, str):
                return texts[0]
            return texts
        return sequences

    def generate_until_final_answer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        max_new_tokens: int = 256,
        stop_on_final_answer: bool = True,
        repetition_penalty: Optional[float] = None,
        **generate_kwargs: Any,
    ) -> Any:
        """
        Generate with proper stopping criteria and parameters for MCQ tasks.
        
        This method is designed for use with Step 1 generation where:
        - input_ids has <cot> token as the last token
        - We want to stop when "Final answer: X" appears
        - We use greedy decoding (temperature=0.0)
        
        Args:
            input_ids: Input token IDs [1, L] with <cot> as last token
            attention_mask: Attention mask [1, L]
            max_new_tokens: Maximum tokens to generate
            stop_on_final_answer: Whether to stop on "Final answer: X" pattern
            repetition_penalty: Optional repetition penalty (default: 1.0)
            generate_kwargs: Additional args forwarded to model.generate
        
        Returns:
            GenerateDecoderOnlyOutput with sequences and other generation info
        
        Example:
            >>> input_ids, attention_mask, cot_id = build_mcq_chat_input(...)
            >>> outputs = wrapper.generate_until_final_answer(
            ...     input_ids, attention_mask, max_new_tokens=256
            ... )
            >>> gen_text = wrapper.tokenizer.decode(
            ...     outputs.sequences[0, input_ids.shape[1]:], 
            ...     skip_special_tokens=True
            ... )
        """
        self._ensure_loaded()
        
        # Set up stopping criteria
        from transformers import StoppingCriteriaList
        stopping_criteria = None
        if stop_on_final_answer:
            stopping_criteria = StoppingCriteriaList([StopOnFinalAnswer(self.tokenizer)])
        
        # Prepare generation parameters
        gen_params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": None,  # Greedy decoding
            "return_dict_in_generate": True,
            **generate_kwargs,
        }
        
        if repetition_penalty is not None:
            gen_params["repetition_penalty"] = repetition_penalty
        
        if stopping_criteria is not None:
            gen_params["stopping_criteria"] = stopping_criteria
        
        outputs = self.model.generate(**gen_params)
        return outputs

    def generate_with_chat_template(
        self,
        user_question: str,
        answer_block_prefix: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        **generate_kwargs: Any,
    ) -> str:
        """
        Generate with chat template and prewritten answer block prefix.
        
        This method:
        1. Uses tokenizer chat template with system/user/assistant messages
        2. Assistant message contains prewritten answer prefix (e.g., "<cot>\n" + "<answer>\nMCQ ANSWER: ")
        3. Uses simple substring stopper for "</answer>"
        4. Post-fixes reason header and suffix if missing
        5. Does NOT use heavy FSM controller
        
        Args:
            user_question: The user's question/prompt
            answer_block_prefix: Prewritten answer prefix for assistant
                (e.g., "<cot>\nThink step by step.\n</cot>\n<answer>\nMCQ ANSWER: ")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            generate_kwargs: Additional args forwarded to model.generate
        
        Returns:
            Complete generated text (including prompt and answer block)
        
        Example:
            >>> from utils.handlers import build_answer_block
            >>> answer_prefix, reason_header, answer_suffix = build_answer_block("mcq")
            >>> cot_block = "<cot>\nThink step by step.\n</cot>\n"
            >>> full_prefix = cot_block + answer_prefix
            >>> result = wrapper.generate_with_chat_template(
            ...     "Question: What is 2+2?\nChoices: A) 3 B) 4 C) 5 D) 6",
            ...     full_prefix,
            ...     max_new_tokens=256
            ... )
        """
        self._ensure_loaded()
        
        # Build messages for chat template
        messages = [
            {"role": "system", "content": "You are a careful problem solver."},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": answer_block_prefix},
        ]
        
        # Apply chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # We already have assistant content
            )
        except Exception:
            # Fallback if no chat template
            prompt = user_question + "\n" + answer_block_prefix
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False)
        input_ids = inputs["input_ids"].to(self.primary_device)
        attention_mask = inputs["attention_mask"].to(self.primary_device)
        
        # Set up simple stopping criteria for </answer>
        stopper = StopOnStrings(["</answer>"], self.tokenizer)
        stopping_criteria = StoppingCriteriaList([stopper])
        
        # Prepare generation parameters
        gen_params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.95,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "stopping_criteria": stopping_criteria,
        }
        
        # Add any additional generate kwargs (filter out invalid keys)
        for key, value in generate_kwargs.items():
            if key not in gen_params:
                gen_params[key] = value
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**gen_params)
        
        # Decode full output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Post-process: ensure reason header and suffix are present
        if "\nREASON: " not in generated_text:
            # Insert reason header before </answer> if present, or at end
            if "</answer>" in generated_text:
                generated_text = generated_text.replace("</answer>", "\nREASON: \n</answer>")
            else:
                generated_text += "\nREASON: "
        
        if "</answer>" not in generated_text:
            # Append closing tag
            generated_text += "\n</answer>"
        
        return generated_text
    
    def generate_direct(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        max_new_tokens: int = 256,
        **generate_kwargs: Any,
    ) -> Any:
        """
        Generate free-form text without structured output enforcement.
        
        This is used for Phase B "direct" mode experiments where we test
        whether injecting the reasoning subspace can induce reasoning behavior
        without explicit CoT scaffolding.
        
        Uses DETERMINISTIC greedy decoding for evaluation consistency.
        
        Args:
            input_ids: Input token IDs [1, L]
            attention_mask: Attention mask [1, L]
            max_new_tokens: Maximum tokens to generate
            generate_kwargs: Additional args forwarded to model.generate
        
        Returns:
            GenerateDecoderOnlyOutput with sequences and other generation info
        
        Example:
            >>> input_ids, attention_mask = build_input_direct(tokenizer, prompt, device)
            >>> outputs = wrapper.generate_direct(
            ...     input_ids, attention_mask,
            ...     max_new_tokens=256
            ... )
        """
        self._ensure_loaded()
        
        # Prepare generation parameters for DETERMINISTIC evaluation
        gen_params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,               # DETERMINISTIC: no sampling
            "temperature": None,               # Must be None when do_sample=False
            "top_p": None,                     # Ignored when do_sample=False
            "top_k": None,                     # Ignored when do_sample=False
            "num_beams": 1,                    # Greedy decode
            "return_dict_in_generate": True,
            "use_cache": True,                 # Enable KV cache for efficiency
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Add any additional valid generate kwargs
        for key, value in generate_kwargs.items():
            gen_params[key] = value
        
        outputs = self.model.generate(**gen_params)
        return outputs
    
    def generate_structured(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        controller_config: Dict[str, Any],
        max_new_tokens: int = 256,
        **generate_kwargs: Any,
    ) -> Any:
        """
        Generate with structured output format enforcement (MCQ, Numeric, or Labelset).
        
        This method uses the unified FSM controller to enforce:
        1. Free-form CoT reasoning (>= min_cot_tokens)
        2. Structured answer line: "\nMCQ ANSWER: X" or "\nNUM ANSWER: <number>" or "\nCLS ANSWER: <label>"
        3. Structured reason line: "\nREASON: ..."
        
        Always uses DETERMINISTIC greedy decoding for evaluation consistency.
        
        Args:
            input_ids: Input token IDs [1, L]
            attention_mask: Attention mask [1, L]
            controller_config: Dictionary containing controller configuration:
                - mode: Either 'mcq', 'numeric', or 'labelset'
                - phrase_text: Text of answer phrase (e.g., "MCQ ANSWER:")
                - min_cot_tokens: Minimum CoT tokens before allowing answer
                - allowed_letters: For MCQ - list of valid letters
                - label_token_sequences: For labelset - list of tokenized labels
                - blocked_special_ids: Set of token IDs to block
                - require_reason: Whether to require REASON: phrase
                - max_answer_tokens: Max tokens for numeric answers
            max_new_tokens: Maximum tokens to generate
            generate_kwargs: Additional args forwarded to model.generate (e.g., pad_token_id)
        
        Returns:
            GenerateDecoderOnlyOutput with sequences and other generation info
        
        Example:
            >>> from utils.format_control import get_blocked_special_ids
            >>> controller_config = {
            ...     'mode': 'mcq',
            ...     'phrase_text': 'MCQ ANSWER:',
            ...     'allowed_letters': ["A","B","C","D"],
            ...     'blocked_special_ids': get_blocked_special_ids(wrapper.tokenizer),
            ...     'min_cot_tokens': 24,
            ...     'require_reason': True,
            ... }
            >>> outputs = wrapper.generate_structured(
            ...     input_ids, attention_mask,
            ...     controller_config=controller_config,
            ...     max_new_tokens=256
            ... )
        """
        self._ensure_loaded()
        
        # Import format control tools
        from utils.format_control import (
            UnifiedFormatController, 
            StopWhenStructuredTailComplete,
            get_blocked_special_ids
        )
        
        # Extract controller parameters
        mode = controller_config.get('mode')
        if mode is None:
            raise ValueError("controller_config must contain 'mode' key")
        
        # Get blocked special IDs if not provided
        blocked_special_ids = controller_config.get('blocked_special_ids')
        if blocked_special_ids is None:
            blocked_special_ids = get_blocked_special_ids(self.tokenizer)
        
        # Build controller with all config parameters
        controller = UnifiedFormatController(
            tok=self.tokenizer,
            mode=mode,
            min_cot_tokens=controller_config.get('min_cot_tokens', 24),
            phrase_text=controller_config.get('phrase_text'),
            allowed_letters=controller_config.get('allowed_letters'),
            label_token_sequences=controller_config.get('label_token_sequences'),
            blocked_special_ids=blocked_special_ids,
            require_reason=controller_config.get('require_reason', True),
            max_answer_tokens=controller_config.get('max_answer_tokens', 32),
        )
        
        # Set up logits processor and stopping criteria
        logits_processor = LogitsProcessorList([controller])
        
        stopping_criteria = StoppingCriteriaList([
            StopWhenStructuredTailComplete(self.tokenizer, controller)
        ])
        
        # Prepare generation parameters for DETERMINISTIC evaluation
        # Critical: ensure reproducible results by removing all sampling
        gen_params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,               # DETERMINISTIC: no sampling
            "temperature": None,               # Must be None when do_sample=False
            "top_p": None,                     # Ignored when do_sample=False
            "top_k": None,                     # Ignored when do_sample=False
            "num_beams": 1,                    # Greedy decode
            "return_dict_in_generate": True,
            "use_cache": True,                 # Enable KV cache for efficiency
            "logits_processor": logits_processor,
            "stopping_criteria": stopping_criteria,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Add any additional valid generate kwargs (but filter out controller params)
        CONTROLLER_KEYS = {
            'mode', 'phrase_text', 'min_cot_tokens', 'allowed_letters',
            'label_token_sequences', 'blocked_special_ids', 'require_reason',
            'max_answer_tokens'
        }
        for key, value in generate_kwargs.items():
            if key not in CONTROLLER_KEYS:
                gen_params[key] = value
        
        outputs = self.model.generate(**gen_params)
        return outputs

    # -------------------------------------------------------------------------
    # Hook utilities
    # -------------------------------------------------------------------------
    def list_modules(self) -> List[Tuple[str, torch.nn.Module]]:
        """Return a list of (qualified_name, module) pairs for the model."""
        self._ensure_loaded()
        return list(self.model.named_modules())

    def register_hook(
        self,
        module_selector: ModuleSelector,
        hook_fn: HookCallable,
        *,
        hook_type: HookType = "forward",
        hook_group: Optional[str] = None,
    ) -> List[torch.utils.hooks.RemovableHandle]:
        """
        Attach hooks to modules selected by `module_selector`.

        Args:
            module_selector: Module name, sequence of names, or predicate function.
            hook_fn: Callable executed during the hook. Signature depends on hook type.
            hook_type: "forward", "forward_pre", or "backward".
            hook_group: Optional identifier to group handles for bulk removal.

        Returns:
            List of removable handles corresponding to registered hooks.
        """
        self._ensure_loaded()
        modules = self._resolve_modules(module_selector)
        if not modules:
            raise ValueError("module_selector did not match any modules.")

        handles: List[torch.utils.hooks.RemovableHandle] = []
        for name, module in modules:
            if hook_type == "forward":
                handle = module.register_forward_hook(hook_fn, with_kwargs=True)
            elif hook_type == "forward_pre":
                handle = module.register_forward_pre_hook(hook_fn, with_kwargs=True)
            elif hook_type == "backward":
                handle = module.register_full_backward_hook(hook_fn)
            else:
                raise ValueError(f"Unsupported hook_type: {hook_type}")
            handles.append(handle)

        group_key = hook_group or f"{hook_type}:{id(hook_fn)}"
        self._hook_handles.setdefault(group_key, []).extend(handles)
        return handles

    def remove_hooks(self, hook_group: Optional[str] = None) -> int:
        """
        Remove registered hooks.

        Args:
            hook_group: Identifier returned by `register_hook`. If None, remove all hooks.

        Returns:
            Number of hooks removed.
        """
        if hook_group is not None:
            targets = [hook_group] if hook_group in self._hook_handles else []
        else:
            targets = list(self._hook_handles.keys())

        count = 0
        for group in targets:
            handles = self._hook_handles.pop(group, [])
            for handle in handles:
                handle.remove()
                count += 1
        return count

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model and tokenizer not loaded. Call `load()` first.")

    def _resolve_dtype(self, dtype: Optional[Union[str, torch.dtype]]) -> Optional[torch.dtype]:
        if dtype is None or dtype == "auto":
            return None
        if isinstance(dtype, torch.dtype):
            return dtype
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float64": torch.float64,
            "fp64": torch.float64,
        }
        key = str(dtype).lower()
        if key not in mapping:
            raise ValueError(f"Unsupported dtype value: {dtype}")
        return mapping[key]

    def _resolve_device_map(self, device: Optional[str]) -> Optional[str]:
        if device in (None, "auto"):
            return "auto"
        return None

    def _resolve_device(self, device: Optional[str]) -> torch.device:
        if device is None or device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if torch_device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch_device

    def _resolve_modules(self, selector: ModuleSelector) -> List[Tuple[str, torch.nn.Module]]:
        modules = list(self.model.named_modules())
        if isinstance(selector, str):
            return [(name, module) for name, module in modules if name == selector]
        if isinstance(selector, Sequence):
            names = set(selector)
            return [(name, module) for name, module in modules if name in names]
        if callable(selector):
            return [(name, module) for name, module in modules if selector(name, module)]
        raise TypeError("module_selector must be a str, sequence of str, or callable.")


# -------------------------------------------------------------------------
# Shared utilities for Phase A generation
# -------------------------------------------------------------------------

def decode_both(tokenizer: PreTrainedTokenizerBase, token_ids) -> Tuple[str, str]:
    """
    Decode token IDs both with and without special tokens.
    
    Args:
        tokenizer: HuggingFace tokenizer
        token_ids: Token IDs to decode (list, tensor, or array)
    
    Returns:
        Tuple of (raw_text, clean_text) where:
        - raw_text includes special tokens like <cot>, </cot>, <answer>
        - clean_text has special tokens stripped
    
    Example:
        >>> raw, clean = decode_both(tokenizer, gen_ids)
        >>> # raw might be: "</cot>\n<answer> B\n"
        >>> # clean might be: "\nB\n"
    """
    raw = tokenizer.decode(token_ids, skip_special_tokens=False)
    clean = tokenizer.decode(token_ids, skip_special_tokens=True)
    return raw, clean


def _letter_token_set(tok: PreTrainedTokenizerBase, letter: str) -> set:
    """
    Collect single-token variants for a letter (A, B, C, or D).
    
    Handles different tokenization patterns like "A", " A", "\nA" that may
    exist as single tokens depending on the tokenizer.
    
    Args:
        tok: HuggingFace tokenizer
        letter: Single letter string ("A", "B", "C", or "D")
    
    Returns:
        Set of token IDs that represent this letter as a single token
    
    Example:
        >>> tok = wrapper.tokenizer
        >>> a_tokens = _letter_token_set(tok, "A")
        >>> # Might return {32, 65, 362} for different variants
    """
    s = set()
    for prefix in ["", " ", "\n"]:
        ids = tok.encode(prefix + letter, add_special_tokens=False)
        if len(ids) == 1:
            s.add(ids[0])
    return s


class ReasonAnswerController(LogitsProcessor):
    """
    FSM-based logits processor to enforce CoT + answer format.
    
    This processor ensures the model generates:
    1. Free-form reasoning inside <cot>...</cot>
    2. Exactly the sequence: </cot> <answer> <letter>
    3. Strong bias to newline/EOS after the letter
    
    States:
    - 0: Inside CoT (free generation, but suppress early closing if min_cot_tokens not met)
    - 1: Just saw </cot>, must emit <answer> next
    - 2: Just saw <answer>, must emit A/B/C/D (single token), then bias to newline/EOS
    
    Example:
        >>> tok = wrapper.tokenizer
        >>> special_ids = wrapper.get_special_token_ids()
        >>> controller = ReasonAnswerController(
        ...     tok, special_ids["<cot>"], special_ids["</cot>"], special_ids["<answer>"]
        ... )
        >>> procs = LogitsProcessorList([controller])
        >>> outputs = model.generate(..., logits_processor=procs)
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        cot_start_id: int,
        cot_end_id: int,
        answer_id: int,
        min_cot_tokens: int = 8,
    ):
        """
        Initialize the controller.
        
        Args:
            tokenizer: HuggingFace tokenizer
            cot_start_id: Token ID for <cot>
            cot_end_id: Token ID for </cot>
            answer_id: Token ID for <answer>
            min_cot_tokens: Minimum reasoning tokens before allowing </cot>
        """
        self.tokenizer = tokenizer
        self.cot_start_id = cot_start_id
        self.cot_end_id = cot_end_id
        self.answer_id = answer_id
        self.min_cot_tokens = int(min_cot_tokens)
        
        # Build allowed letter token sets once
        self.letter_tokens = set()
        for letter in ["A", "B", "C", "D"]:
            self.letter_tokens.update(_letter_token_set(tokenizer, letter))
        
        # Newline and EOS tokens
        self.newline_ids = set(tokenizer.encode("\n", add_special_tokens=False))
        self.eos_id = tokenizer.eos_token_id
        
        # State tracking per batch index
        self.state = {}  # batch_idx -> dict
    
    def _get_state(self, batch_idx: int) -> Dict:
        """Get or initialize state for a batch index."""
        if batch_idx not in self.state:
            self.state[batch_idx] = {
                "in_cot": False,
                "cot_tokens": 0,
                "phase": 0,
                "took_letter": False,
            }
        return self.state[batch_idx]
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Process logits to enforce format.
        
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
            
            # Track CoT span and state transitions
            if last_token == self.cot_start_id:
                st["in_cot"] = True
                st["cot_tokens"] = 0
                st["phase"] = 0
            elif st["in_cot"] and last_token not in (self.cot_start_id, self.cot_end_id):
                st["cot_tokens"] += 1
            
            if last_token == self.cot_end_id and st["in_cot"]:
                st["in_cot"] = False
                st["phase"] = 1  # Require <answer> next
            
            # Apply phase-specific constraints
            if st["phase"] == 0:
                # Inside CoT or before </cot>
                if st["cot_tokens"] < self.min_cot_tokens:
                    # Suppress early closing
                    for tid in (self.cot_end_id, self.answer_id):
                        scores[b, tid] = scores[b, tid] - 500.0
                # else: allow natural </cot>, no constraints
            
            elif st["phase"] == 1:
                # Must emit <answer> next
                mask = torch.full_like(scores[b], -1e9)
                mask[self.answer_id] = 0.0
                scores[b] = scores[b] + mask
                st["phase"] = 2  # Next call will handle letter
            
            elif st["phase"] == 2:
                # Require A/B/C/D (single-token variants)
                if not st["took_letter"]:
                    allow = list(self.letter_tokens)
                    mask = torch.full_like(scores[b], -1e9)
                    if allow:
                        mask[allow] = 0.0
                    scores[b] = scores[b] + mask
                    
                    if last_token in self.letter_tokens:
                        st["took_letter"] = True
                else:
                    # After letter, bias strongly to newline/EOS
                    forbid = torch.ones_like(scores[b], dtype=torch.bool)
                    allowed_tokens = list(self.newline_ids) + ([self.eos_id] if self.eos_id is not None else [])
                    for tid in allowed_tokens:
                        forbid[tid] = False
                    scores[b, forbid] = -1e9
        
        return scores


class StopOnStrings(StoppingCriteria):
    """
    Simple substring-based stopping criteria.
    
    Stops generation when any of the specified stop strings appears in the decoded text.
    
    Example:
        >>> tokenizer = wrapper.tokenizer
        >>> stopper = StopOnStrings(["</answer>"], tokenizer)
        >>> stopping_criteria = StoppingCriteriaList([stopper])
        >>> outputs = model.generate(..., stopping_criteria=stopping_criteria)
    """
    
    def __init__(self, stop_strings: List[str], tokenizer: PreTrainedTokenizerBase):
        """
        Args:
            stop_strings: List of strings that will trigger stopping
            tokenizer: HuggingFace tokenizer for decoding tokens
        """
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.prev_text = ""
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Check if any stop string appears in decoded text.
        
        Args:
            input_ids: Generated token IDs so far [batch_size, seq_len]
            scores: Model output scores (unused)
            **kwargs: Additional arguments (unused)
        
        Returns:
            True if generation should stop, False otherwise
        """
        # Decode only the newly generated tail (fast enough for our lengths)
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        if any(s in text for s in self.stop_strings):
            return True
        return False


class InputIdsTap(StoppingCriteria):
    """
    Stopping criteria that taps input_ids into hook manager each generation step.
    
    This never actually stops generation - it just allows the hook manager to track
    the current input_ids for persistent CoT gating. Always returns False.
    
    Example:
        >>> from multi_hook_manager import HookManager
        >>> hook_manager = HookManager()
        >>> tap = InputIdsTap(hook_manager)
        >>> stopping_criteria = StoppingCriteriaList([tap])
        >>> outputs = model.generate(..., stopping_criteria=stopping_criteria)
    """
    
    def __init__(self, hook_manager):
        """
        Args:
            hook_manager: HookManager instance to update with current input_ids
        """
        self.hook_manager = hook_manager
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Update hook manager with current input_ids and never stop generation.
        
        Args:
            input_ids: Generated token IDs so far [batch_size, seq_len]
            scores: Model output scores (unused)
            **kwargs: Additional arguments (unused)
        
        Returns:
            Always False (never stops generation)
        """
        self.hook_manager.set_current_input_ids(input_ids)
        return False


class StopOnAnswerPrefix(StoppingCriteria):
    """
    Stopping criteria that halts generation when the answer prefix appears.
    
    Uses the canonical ANSWER_SENTINEL_TEXT from format_control to detect answer boundary.
    
    Example:
        >>> from utils.format_control import get_answer_sentinel_text
        >>> tokenizer = wrapper.tokenizer
        >>> stopping_criteria = StoppingCriteriaList([StopOnAnswerPrefix(tokenizer)])
        >>> outputs = model.generate(..., stopping_criteria=stopping_criteria)
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        """
        Args:
            tokenizer: HuggingFace tokenizer for decoding tokens.
        """
        from utils.format_control import get_answer_sentinel_text
        self.tokenizer = tokenizer
        self.phrase_text = get_answer_sentinel_text()
        self.phrase_ids = tokenizer(self.phrase_text, add_special_tokens=False).input_ids
        self.window = []
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Check if generation should stop based on phrase match.
        
        Args:
            input_ids: Generated token IDs so far [batch_size, seq_len].
            scores: Model output scores (unused).
            **kwargs: Additional arguments (unused).
            
        Returns:
            True if generation should stop, False otherwise.
        """
        # Append the last generated token id for windowed matching
        last_id = input_ids[0, -1].item()
        self.window.append(last_id)
        if len(self.window) > len(self.phrase_ids):
            self.window.pop(0)
        return self.window == self.phrase_ids


class BoundaryMonitor(StoppingCriteria):
    """
    Monitor generation boundaries and notify MultiHookManager on every decode step.
    
    This class tracks when the answer phrase appears and updates the hook manager's
    boundary state, allowing it to flip masks dynamically during generation.
    
    Never actually stops generation - always returns False. Its purpose is to
    provide live boundary updates to the hook manager.
    
    Example:
        >>> from utils.format_control import get_answer_sentinel_text
        >>> tokenizer = wrapper.tokenizer
        >>> monitor = BoundaryMonitor(tokenizer, multi_hook_manager)
        >>> stopping_criteria = StoppingCriteriaList([monitor])
        >>> outputs = model.generate(..., stopping_criteria=stopping_criteria)
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizerBase, multi_hook_manager):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            multi_hook_manager: MultiHookManager instance to notify of boundaries
        """
        from utils.format_control import get_answer_sentinel_text
        self.tokenizer = tokenizer
        self.mhm = multi_hook_manager
        self.phrase_ids = tokenizer(get_answer_sentinel_text(), add_special_tokens=False).input_ids
        self.window = []
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Update hook manager with current decode step and never stop generation.
        
        Args:
            input_ids: Generated token IDs so far [batch_size, seq_len]
            scores: Model output scores (unused)
            **kwargs: Additional arguments (unused)
            
        Returns:
            Always False (never stops generation)
        """
        last_id = input_ids[0, -1].item()
        self.window.append(last_id)
        if len(self.window) > len(self.phrase_ids):
            self.window.pop(0)
        
        # Notify hook manager on every decode step
        step_idx = input_ids.shape[1] - 1  # decode index
        self.mhm.update_boundaries(step_idx, last_id)
        
        # Never stop generation
        return False


def build_input_with_cot(
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    model_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Build chat-formatted input with <cot> token ID appended AFTER chat template.
    
    This function:
    1. Applies the model's chat template with add_generation_prompt=True
    2. **Then** explicitly appends the <cot> token ID as the last token
    3. Creates an attention mask (all ones, no padding)
    4. Returns inputs ready for structured generation
    
    This ensures generation starts immediately after <cot>, not after assistant header.
    
    Args:
        tokenizer: HuggingFace tokenizer with chat template support
        prompt_text: The formatted prompt text (should NOT end with <cot>)
        model_device: Target device for tensors
    
    Returns:
        Tuple of (input_ids, attention_mask, cot_start_idx):
            - input_ids: Token IDs tensor [1, seq_len] with <cot> as last token
            - attention_mask: Attention mask tensor [1, seq_len] (all ones)
            - cot_start_idx: Index position of <cot> token (for masking)
    
    Example:
        >>> wrapper = HFModelWrapper(config).load()
        >>> prompt = "Question: What is 2+2?\\nChoices: A) 3 B) 4 C) 5 D) 6"
        >>> input_ids, attn_mask, cot_idx = build_input_with_cot(
        ...     wrapper.tokenizer, prompt, wrapper.primary_device
        ... )
        >>> # Generation now starts immediately after <cot>
    """
    # Remove <cot> from prompt if present (we'll append as token ID)
    prompt_text = prompt_text.rstrip()
    if prompt_text.endswith("<cot>"):
        prompt_text = prompt_text[:-5].rstrip()
    
    messages = [
        {"role": "system", "content": "Follow the format exactly. Provide your reasoning, then answer, then explanation."},
        {"role": "user", "content": prompt_text}
    ]
    
    try:
        # Apply chat template with generation prompt (adds assistant header)
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model_device)
    except Exception:
        # Fallback for models without chat template
        input_ids = tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"].to(model_device)
    
    # Get <cot> token ID
    cot_id = tokenizer.convert_tokens_to_ids("<cot>")
    if cot_id is None or cot_id == tokenizer.unk_token_id:
        raise RuntimeError("Tokenizer is missing <cot> special token.")
    
    # Append <cot> token ID AFTER chat template
    cot_tail = torch.tensor([[cot_id]], dtype=torch.long, device=model_device)
    input_ids = torch.cat([input_ids, cot_tail], dim=1)
    
    # Create attention mask
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_device)
    
    # Return <cot> position (last index)
    cot_start_idx = input_ids.shape[1] - 1
    
    return input_ids, attention_mask, cot_start_idx


def build_input_direct(
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    model_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build chat-formatted input WITHOUT appending any special tokens for direct prompting.
    
    This is used for Phase B experiments where we want to test whether injecting
    the reasoning subspace can induce reasoning behavior without explicit CoT scaffolding.
    
    Args:
        tokenizer: HuggingFace tokenizer with chat template support
        prompt_text: The formatted prompt text (direct question without CoT instructions)
        model_device: Target device for tensors
    
    Returns:
        Tuple of (input_ids, attention_mask):
            - input_ids: Token IDs tensor [1, seq_len]
            - attention_mask: Attention mask tensor [1, seq_len] (all ones)
    
    Example:
        >>> wrapper = HFModelWrapper(config).load()
        >>> prompt = "Question: What is 2+2?\\nA) 3 B) 4 C) 5 D) 6\\nAnswer:"
        >>> input_ids, attn_mask = build_input_direct(
        ...     wrapper.tokenizer, prompt, wrapper.primary_device
        ... )
        >>> # Generation starts after prompt, no <cot> token
    """
    messages = [
        {"role": "system", "content": "Answer directly."},
        {"role": "user", "content": prompt_text}
    ]
    
    try:
        # Apply chat template with generation prompt
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model_device)
    except Exception:
        # Fallback for models without chat template
        input_ids = tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"].to(model_device)
    
    # Create attention mask
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_device)
    
    return input_ids, attention_mask


def build_mcq_chat_input(
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    model_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Build chat-formatted input with <cot> token ID forced to be the last token.
    
    DEPRECATED: Use build_input_with_cot() instead for structured generation.
    
    This function:
    1. Applies the model's chat template to format the prompt
    2. Force-appends the <cot> token ID to ensure it's the last token before generation
    3. Creates an attention mask (all ones, no padding)
    4. Returns inputs ready for model.generate() along with CoT start index
    
    Args:
        tokenizer: HuggingFace tokenizer with chat template support.
        prompt_text: The formatted prompt text. Should NOT end with <cot> as a string
                    (it will be appended as a token ID).
        model_device: Target device for tensors.
    
    Returns:
        Tuple of (input_ids, attention_mask, cot_start_idx):
            - input_ids: Token IDs tensor [1, seq_len] with <cot> as last token
            - attention_mask: Attention mask tensor [1, seq_len] (all ones)
            - cot_start_idx: Index position of <cot> token (for masking generated tokens after it)
    
    Example:
        >>> wrapper = HFModelWrapper(config).load()
        >>> prompt = "Question: What is 2+2?\\nChoices: A) 3 B) 4 C) 5 D) 6"
        >>> input_ids, attn_mask, cot_start_idx = build_mcq_chat_input(
        ...     wrapper.tokenizer, prompt, wrapper.primary_device
        ... )
        >>> # input_ids now ends with <cot> token ID at position cot_start_idx
        >>> pooler.set_cot_start_idx(cot_start_idx)
        >>> outputs = wrapper.model.generate(input_ids=input_ids, attention_mask=attn_mask, ...)
    """
    return build_input_with_cot(tokenizer, prompt_text, model_device)
