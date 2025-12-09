"""
Multi-layer hook management for activation collection and causal intervention.

This module provides tools for:
1. Collecting pooled activations from multiple layers during generation
2. Applying causal interventions (add/lesion) to model activations
3. Offline screening of layers for reasoning vector identification
4. Managing current input_ids for persistent CoT tracking
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


class HookManager:
    """
    Simple container for managing state during generation with hooks.
    
    Used to track current input_ids for persistent CoT gating in interventions.
    Can be extended with additional state tracking as needed.
    
    Example:
        >>> hook_manager = HookManager()
        >>> # During generation, update with current input_ids
        >>> hook_manager.set_current_input_ids(input_ids)
        >>> # In hook, check current tokens
        >>> ids = hook_manager.get_current_input_ids()
    """
    
    def __init__(self):
        """Initialize hook manager with empty state."""
        self._current_input_ids: Optional[torch.Tensor] = None
    
    def set_current_input_ids(self, input_ids_tensor: torch.Tensor) -> None:
        """
        Update the current input_ids tensor.
        
        Args:
            input_ids_tensor: Token IDs tensor, typically shape [batch, seq_len]
        """
        self._current_input_ids = input_ids_tensor
    
    def get_current_input_ids(self) -> Optional[torch.Tensor]:
        """
        Get the current input_ids tensor.
        
        Returns:
            Current input_ids tensor or None if not set
        """
        return self._current_input_ids


def detect_layer_path(model: nn.Module) -> str:
    """
    Detect the correct layer path for a given model architecture.
    
    This helper function attempts to find the correct attribute path to access
    transformer layers for different model families (LLaMA, Mistral, Qwen, etc.).
    
    Args:
        model: The transformer model
    
    Returns:
        String path to access layers (e.g., "model.layers" or "transformer.h")
    
    Example:
        >>> wrapper = HFModelWrapper(config).load()
        >>> layer_path = detect_layer_path(wrapper.model)
        >>> pooler = MultiLayerPooler(wrapper.model, layers=[10, 20], hidden_size=4096, layer_path=layer_path)
    """
    # Common paths to check
    common_paths = [
        "model.layers",      # LLaMA, Mistral, Qwen2
        "transformer.h",     # GPT-2, GPT-J
        "gpt_neox.layers",   # GPT-NeoX
        "transformer.layers", # Some Qwen variants
        "model.decoder.layers", # Some encoder-decoder models
    ]
    
    for path in common_paths:
        try:
            parts = path.split(".")
            module = model
            for part in parts:
                module = getattr(module, part)
            # Check if it's indexable (list-like)
            if hasattr(module, "__getitem__") and len(module) > 0:
                return path
        except (AttributeError, TypeError):
            continue
    
    # If no common path works, try to find any ModuleList
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            # Verify it looks like transformer layers by checking first element
            if hasattr(module[0], "self_attn") or hasattr(module[0], "attention"):
                return name
    
    # Default fallback
    return "model.layers"


def build_mask_from_ids(
    input_ids: torch.LongTensor,
    cot_start_id: int,
    cot_end_id: Optional[int] = None,
    answer_id: Optional[int] = None,
    include_anchors: bool = False,
) -> torch.BoolTensor:
    """
    Build a per-sequence boolean mask for CoT (Chain-of-Thought) tokens based on special token IDs.
    
    The mask marks tokens as True from the position after the **last** <cot> up to (but not including):
    - The first </cot> token after that last <cot>, if cot_end_id is provided and found
    - Otherwise, the first <answer> token after that last <cot>, if answer_id is provided and found
    - Otherwise, the end of the sequence (fail-open behavior)
    
    Args:
        input_ids: Token IDs tensor of shape [B, L] or [L]
        cot_start_id: ID of the CoT start token (e.g., <cot>)
        cot_end_id: Optional ID of the CoT end token (e.g., </cot>)
        answer_id: Optional ID of the answer token (e.g., <answer>)
        include_anchors: If True, include the <cot> and </cot> tokens themselves in the mask.
                        If False (default), exclude them and only mask content between them.
    
    Returns:
        Boolean mask of shape [B, L] where True indicates tokens inside the CoT window.
        If cot_start_id is not found in a sequence, that row will be all False.
    
    Example:
        >>> input_ids = torch.tensor([[1, 2, 100, 3, 4, 101, 5]])  # 100=<cot>, 101=</cot>
        >>> mask = build_mask_from_ids(input_ids, cot_start_id=100, cot_end_id=101)
        >>> mask
        tensor([[False, False, False, True, True, False, False]])
        >>> mask_with_anchors = build_mask_from_ids(input_ids, cot_start_id=100, cot_end_id=101, include_anchors=True)
        >>> mask_with_anchors
        tensor([[False, False, True, True, True, True, False]])
    """
    # Handle 1D input by adding batch dimension
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    
    for batch_idx in range(batch_size):
        # Find start position (LAST occurrence of cot_start_id)
        start_positions = (input_ids[batch_idx] == cot_start_id).nonzero(as_tuple=True)[0]
        
        if len(start_positions) == 0:
            # No start token found, leave this row as all False
            continue
        
        # Determine start position based on include_anchors
        start_anchor = start_positions[-1].item()  # Position of the LAST <cot>
        start_pos = start_anchor if include_anchors else start_anchor + 1
        
        # Find end position
        end_pos = seq_len  # Default to end of sequence (fail-open)
        
        # First try to find cot_end_id AFTER the last <cot>
        if cot_end_id is not None:
            end_positions = (input_ids[batch_idx, start_anchor + 1:] == cot_end_id).nonzero(as_tuple=True)[0]
            if len(end_positions) > 0:
                end_anchor = start_anchor + 1 + end_positions[0].item()
                end_pos = end_anchor + 1 if include_anchors else end_anchor
        
        # If no cot_end_id found, try answer_id AFTER the last <cot>
        if end_pos == seq_len and answer_id is not None:
            answer_positions = (input_ids[batch_idx, start_anchor + 1:] == answer_id).nonzero(as_tuple=True)[0]
            if len(answer_positions) > 0:
                end_pos = start_anchor + 1 + answer_positions[0].item()
        
        # Set mask from start_pos to end_pos (exclusive of end_pos)
        if start_pos < end_pos:
            mask[batch_idx, start_pos:end_pos] = True
    
    if squeeze_output:
        mask = mask.squeeze(0)
    
    return mask


def _label_counts(y: np.ndarray) -> Counter:
    y_arr = np.asarray(y)
    return Counter(y_arr.tolist())

def _min_class_count(y: np.ndarray) -> int:
    c = _label_counts(y)
    return min(c.values()) if len(c) > 0 else 0

def _can_stratify(y: np.ndarray, n_splits: int = 2) -> bool:
    """
    True iff there are at least 2 classes AND every class has >= 2 samples
    AND we can actually form 'n_splits' non-empty folds per class.
    """
    y_arr = np.asarray(y)
    uniq = np.unique(y_arr)
    if len(uniq) < 2:
        return False
    counts = Counter(y_arr.tolist())
    if min(counts.values()) < 2:
        return False
    # Ensure each class can appear in all folds if using CV with n_splits
    return all(v >= n_splits for v in counts.values())

def _safe_train_val_split(indices: np.ndarray,
                          y: np.ndarray,
                          test_size: float,
                          random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """
    If classes are too small for stratification, fall back to non-stratified split.
    If dataset is too tiny to form a validation set, put everything in train and leave val empty.
    """
    n = len(indices)
    if n < 3 or int(round(test_size * n)) == 0:
        # Too small to split: train = all, val = empty
        print("⚠️  [screen] Dataset too small for a validation split; using all samples for train, val=0.")
        return indices, np.array([], dtype=indices.dtype)

    if _can_stratify(y, n_splits=2):
        tr_idx, va_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        return tr_idx, va_idx

    # Fallback: non-stratified split
    counts = _label_counts(y)
    print(f"⚠️  [screen] Disabling stratification (class counts: {dict(counts)}). "
          f"Using non-stratified split.")
    tr_idx, va_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=None
    )
    return tr_idx, va_idx

def _choose_cv(y: np.ndarray,
               n_splits: int,
               shuffle: bool,
               random_state: int):
    """
    Return a CV splitter. Prefer StratifiedKFold when feasible; otherwise KFold.
    Also reduce n_splits if dataset is too small.
    """
    n = len(y)
    if n_splits > n:
        n_splits = n  # avoid impossible CV
    if n_splits < 2:
        # Degenerate: no CV; use a single split dummy (handled by caller if needed)
        n_splits = 2 if n >= 2 else 2  # keep API; caller can ignore

    if _can_stratify(y, n_splits=n_splits):
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    else:
        print("⚠️  [screen] Using KFold (stratification not possible with current label distribution).")
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


class MultiLayerPooler:
    """
    Collect and pool activations from multiple transformer layers during generation.

    This class registers forward hooks on specified layers to capture hidden states,
    then pools them over a specified CoT (Chain-of-Thought) token window.
    
    Supports dynamic masking during generation where mask is rebuilt based on
    start_idx or token IDs at each forward pass.

    Example
    -------
    ::

        pooler = MultiLayerPooler(
            model=wrapper.model,
            layers=[16, 20, 24, 28],
            hidden_size=4096
        )

        # Option 1: Set fixed mask for static sequences
        cot_mask = torch.tensor([False, False, True, True, True, False])
        pooler.set_cot_window(cot_mask)

        # Option 2: Set dynamic start index for generation
        pooler.set_cot_start_idx(prompt_length)  # Pools from prompt_length onwards
        
        # Option 3: Dynamic token-based masking
        pooler.set_dynamic_cot_by_ids(cot_start_id, cot_end_id, answer_id)

        # Run generation (hooks will capture activations)
        with torch.no_grad():
            outputs = model.generate(...)

        # Get pooled activations per layer
        pooled = pooler.pooled()  # {layer_idx: np.array of shape [D]}

        pooler.close()
    """

    def __init__(
        self,
        model: nn.Module,
        layers: List[int],
        hidden_size: int,
        layer_path: str = "model.layers",
    ):
        """
        Initialize multi-layer pooler.

        Args:
            model: The transformer model (e.g., wrapper.model)
            layers: List of layer indices to monitor
            hidden_size: Hidden dimension size (D)
            layer_path: Attribute path to access layers (e.g., "model.layers")
        """
        self.layers = layers
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.hidden_size = hidden_size
        self.layer_path = layer_path
        
        # Masking strategies
        self.collect_mask: Optional[torch.Tensor] = None  # Static mask
        self.cot_start_idx: Optional[int] = None  # Dynamic start index
        self.cot_start_id: Optional[int] = None  # Dynamic token-based
        self.cot_end_id: Optional[int] = None
        self.answer_id: Optional[int] = None
        
        self.reset_buffers()

        # Register hooks on each layer
        for layer_idx in layers:
            layer_module = self._get_layer_module(model, layer_idx)
            handle = layer_module.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(handle)

    def _get_layer_module(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Navigate to the specific layer module."""
        parts = self.layer_path.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)
        return module[layer_idx]

    def reset_buffers(self) -> None:
        """Reset activation buffers for a new generation pass."""
        self.sum: Dict[int, Optional[torch.Tensor]] = {layer: None for layer in self.layers}
        self.count: Dict[int, int] = {layer: 0 for layer in self.layers}
        self._last_mask_count: Dict[int, int] = {layer: 0 for layer in self.layers}

    def set_cot_window(self, mask: torch.Tensor) -> None:
        """
        Set static boolean mask indicating which tokens to pool over.

        Args:
            mask: Boolean tensor of shape [seq_len] where True indicates
                  tokens to include in pooling (e.g., reasoning tokens)
        """
        self.collect_mask = mask
        self.cot_start_idx = None
        self.cot_start_id = None
        
    def set_cot_start_idx(self, start_idx: int) -> None:
        """
        Set dynamic masking based on start index.
        
        During generation, mask will be rebuilt as: mask = torch.arange(L) >= start_idx
        This is suitable when you know the prompt length and want to pool all generated tokens.
        
        Args:
            start_idx: Position to start masking from (e.g., prompt length)
        """
        self.cot_start_idx = start_idx
        self.collect_mask = None
        self.cot_start_id = None

    def set_dynamic_cot_by_ids(
        self,
        cot_start_id: int,
        cot_end_id: Optional[int] = None,
        answer_id: Optional[int] = None,
    ) -> None:
        """
        Set dynamic token-based masking for generation.
        
        The mask will be rebuilt at each forward pass by searching for special tokens.
        Note: This mode requires access to input_ids during forward pass, which may not
        work with all generation methods. Prefer set_cot_start_idx() for generation.
        
        Args:
            cot_start_id: ID of CoT start token
            cot_end_id: Optional ID of CoT end token
            answer_id: Optional ID of answer token
        """
        self.cot_start_id = cot_start_id
        self.cot_end_id = cot_end_id
        self.answer_id = answer_id
        self.collect_mask = None
        self.cot_start_idx = None

    def set_cot_window_by_ids(
        self,
        input_ids: torch.LongTensor,
        cot_start_id: int,
        cot_end_id: Optional[int] = None,
        answer_id: Optional[int] = None,
    ) -> None:
        """
        Set the CoT window using special token IDs from input_ids.
        
        This is a convenience method that builds the mask from token IDs and
        calls set_cot_window() with the result.
        
        Args:
            input_ids: Token IDs tensor of shape [B, L] or [L]
            cot_start_id: ID of the CoT start token (e.g., <cot>)
            cot_end_id: Optional ID of the CoT end token (e.g., </cot>)
            answer_id: Optional ID of the answer token (e.g., <answer>)
        
        Example:
            >>> # Get special token IDs from wrapper
            >>> token_ids = wrapper.get_special_token_ids()
            >>> # Tokenize prompt
            >>> inputs = wrapper.tokenize(prompt, padding=False)
            >>> # Set CoT window
            >>> pooler.set_cot_window_by_ids(
            ...     inputs["input_ids"],
            ...     cot_start_id=token_ids["<cot>"],
            ...     cot_end_id=token_ids["</cot>"],
            ...     answer_id=token_ids.get("<answer>")
            ... )
        """
        mask = build_mask_from_ids(input_ids, cot_start_id, cot_end_id, answer_id)
        # Handle batch dimension - if input was batched, take first sequence
        if mask.dim() == 2:
            mask = mask[0]
        self.set_cot_window(mask)

    def _make_hook(self, layer_idx: int):
        """Create a forward hook for the specified layer."""

        def hook(module: nn.Module, inputs: Tuple, output: torch.Tensor) -> None:
            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]  # [B, L, D]
            else:
                hidden_states = output  # [B, L, D]
            
            # Determine mask based on current strategy
            current_mask = None
            
            if self.collect_mask is not None:
                # Static mask - ensure it matches current sequence length
                L = hidden_states.shape[1]
                if len(self.collect_mask) == L:
                    current_mask = self.collect_mask
                elif len(self.collect_mask) < L:
                    # Pad mask with False for new tokens  
                    pad_size = L - len(self.collect_mask)
                    current_mask = torch.cat([
                        self.collect_mask,
                        torch.zeros(pad_size, dtype=torch.bool, device=self.collect_mask.device)
                    ])
                else:
                    # Mask longer than sequence - truncate
                    current_mask = self.collect_mask[:L]
                    
            elif self.cot_start_idx is not None:
                # Dynamic mask from start index: mask all tokens AFTER start_idx (exclude the <cot> token itself)
                L = hidden_states.shape[1]
                current_mask = torch.arange(L, device=hidden_states.device) > self.cot_start_idx
                
            # Skip if no mask available
            if current_mask is None:
                return

            # Assume batch size of 1 during generation
            masked_hidden = hidden_states[0, current_mask, :]  # [|T|, D]

            # Sum over selected tokens
            token_sum = masked_hidden.sum(dim=0)  # [D]

            # Accumulate
            if self.sum[layer_idx] is None:
                self.sum[layer_idx] = token_sum
            else:
                self.sum[layer_idx] = self.sum[layer_idx] + token_sum

            self.count[layer_idx] += current_mask.sum().item()
            self._last_mask_count[layer_idx] = current_mask.sum().item()

        return hook

    def pooled(self) -> Dict[int, np.ndarray]:
        """
        Get mean-pooled activations for each layer.

        Returns:
            Dictionary mapping layer index to pooled vector [D]
        """
        result = {}
        for layer_idx in self.layers:
            if self.sum[layer_idx] is not None and self.count[layer_idx] > 0:
                mean_vector = self.sum[layer_idx] / max(self.count[layer_idx], 1)
                result[layer_idx] = mean_vector.detach().cpu().numpy()
            else:
                result[layer_idx] = np.zeros(self.hidden_size, dtype=np.float32)
        return result

    def get_last_mask_count_summary(self) -> int:
        """
        Return the average masked token count across all layers from the last forward pass.
        
        This reflects the actual number of tokens that were masked and pooled during
        the most recent generation step.
        
        Returns:
            Average masked count across layers (as int)
        """
        if not self.layers:
            return 0
        counts = [self._last_mask_count.get(layer, 0) for layer in self.layers]
        return int(sum(counts) / max(1, len(counts)))

    def close(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class MultiLayerEditor:
    """
    Apply causal interventions to multiple transformer layers.

    This class modifies hidden states during forward passes by adding or
    subtracting projections onto learned subspaces.

    Example
    -------
    ::

        # Assume U_dict contains projection matrices per layer
        # U_dict = {layer_idx: np.array of shape [D, d]}
        editor = MultiLayerEditor(
            model=wrapper.model,
            layer_to_US={24: U_24, 28: U_28},
            layer_to_alpha={24: 0.5, 28: 0.5}
        )

        # Set which tokens to edit
        cot_mask = torch.tensor([False, False, True, True, True, False])
        editor.set_cot_mask(cot_mask)

        # Run generation with intervention active
        with torch.no_grad():
            outputs = model.generate(...)

        editor.close()
    """

    def __init__(
        self,
        model: nn.Module,
        layer_to_US: Dict[int, np.ndarray],
        layer_to_alpha: Dict[int, float],
        layer_path: str = "model.layers",
        use_lesion: bool = False,
    ):
        """
        Initialize multi-layer editor.

        Args:
            model: The transformer model
            layer_to_US: Dict mapping layer index to projection matrix U [D, d]
            layer_to_alpha: Dict mapping layer index to intervention strength
            layer_path: Attribute path to access layers
            use_lesion: If True, subtract projection (lesion). If False, add (boost)
        """
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.params: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.layer_path = layer_path
        self.use_lesion = use_lesion
        self.cot_mask: Optional[torch.Tensor] = None
        self.cot_start_idx: Optional[int] = None

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Store U matrices directly for memory-efficient low-rank projection
        # proj = (h @ U) @ U.T instead of h @ (U @ U.T)
        for layer_idx, U_np in layer_to_US.items():
            U = torch.tensor(U_np, device=device, dtype=dtype)  # [D, d]
            alpha = torch.tensor(layer_to_alpha[layer_idx], device=device, dtype=dtype)
            self.params[layer_idx] = (U, alpha)

            # Register hook
            layer_module = self._get_layer_module(model, layer_idx)
            self.handles.append(layer_module.register_forward_hook(self._make_hook(layer_idx)))

    def _get_layer_module(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Navigate to the specific layer module."""
        parts = self.layer_path.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)
        return module[layer_idx]

    def set_cot_mask(self, mask_bool_1D: torch.Tensor) -> None:
        """
        Set boolean mask indicating which tokens to edit.

        Args:
            mask_bool_1D: Boolean tensor [seq_len] where True = apply edit
        """
        self.cot_mask = mask_bool_1D

    def set_cot_window_by_ids(
        self,
        input_ids: torch.LongTensor,
        cot_start_id: int,
        cot_end_id: Optional[int] = None,
        answer_id: Optional[int] = None,
    ) -> None:
        """
        Set the CoT mask using special token IDs from input_ids.
        
        This is a convenience method that builds the mask from token IDs and
        calls set_cot_mask() with the result.
        
        Args:
            input_ids: Token IDs tensor of shape [B, L] or [L]
            cot_start_id: ID of the CoT start token (e.g., <cot>)
            cot_end_id: Optional ID of the CoT end token (e.g., </cot>)
            answer_id: Optional ID of the answer token (e.g., <answer>)
        
        Example:
            >>> # Get special token IDs from wrapper
            >>> token_ids = wrapper.get_special_token_ids()
            >>> # Tokenize prompt
            >>> inputs = wrapper.tokenize(prompt, padding=False)
            >>> # Set CoT mask
            >>> editor.set_cot_window_by_ids(
            ...     inputs["input_ids"],
            ...     cot_start_id=token_ids["<cot>"],
            ...     cot_end_id=token_ids["</cot>"],
            ...     answer_id=token_ids.get("<answer>")
            ... )
        """
        mask = build_mask_from_ids(input_ids, cot_start_id, cot_end_id, answer_id)
        # Handle batch dimension - if input was batched, take first sequence
        if mask.dim() == 2:
            mask = mask[0]
        self.set_cot_mask(mask)

    def set_cot_start_idx(self, start_idx: int) -> None:
        """
        Set dynamic masking based on start index.
        
        Args:
            start_idx: Position to start masking from (e.g., prompt length)
        """
        self.cot_start_idx = start_idx
        self.cot_mask = None

    def _make_hook(self, layer_idx: int):
        """Create editing hook for the specified layer."""

        def hook(module: nn.Module, inputs: Tuple, output: torch.Tensor) -> torch.Tensor:
            U, alpha = self.params[layer_idx]

            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]  # [B, L, D]
                rest = output[1:]
            else:
                hidden_states = output  # [B, L, D]
                rest = None

            # Determine mask based on current strategy
            current_mask = None
            L = hidden_states.shape[1]
            
            if self.cot_mask is not None:
                # Static mask - handle dynamic length
                if len(self.cot_mask) == L:
                    current_mask = self.cot_mask.to(hidden_states.device)
                elif len(self.cot_mask) < L:
                    # Pad mask for new tokens
                    pad_size = L - len(self.cot_mask)
                    current_mask = torch.cat([
                        self.cot_mask,
                        torch.zeros(pad_size, dtype=torch.bool, device=self.cot_mask.device)
                    ]).to(hidden_states.device)
                else:
                    # Truncate mask
                    current_mask = self.cot_mask[:L].to(hidden_states.device)
                    
            elif self.cot_start_idx is not None:
                # Dynamic mask from start index: mask all tokens AFTER start_idx (exclude the <cot> token itself)
                current_mask = torch.arange(L, device=hidden_states.device) > self.cot_start_idx
                
            if current_mask is None:
                return output

            # Apply low-rank projection: proj = (h @ U) @ U.T
            # This is memory-efficient: O(|T|*D*d + |T|*d*D) vs O(|T|*D*D)
            masked_h = hidden_states[:, current_mask, :]  # [B, |T|, D]
            
            # Low-rank projection in two steps
            h_proj_subspace = masked_h @ U  # [B, |T|, d]
            projection = h_proj_subspace @ U.t()  # [B, |T|, D]

            if self.use_lesion:
                # Lesion: subtract projection
                hidden_states[:, current_mask, :] = masked_h - alpha * projection
            else:
                # Add: boost the subspace
                hidden_states[:, current_mask, :] = masked_h + alpha * projection

            if rest is not None:
                return (hidden_states,) + rest
            return hidden_states

        return hook

    def close(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class MultiHookManager:
    """
    Manage causal interventions across multiple layers with live CoT/answer boundary detection.
    
    This class handles:
    1. Registering forward hooks on specified layers
    2. Tracking generation boundaries (CoT start, answer start) live during decoding
    3. Applying locality-aware masking (cot, answer, all) with dynamic updates
    4. Applying interventions (add/lesion/rescue) with configurable strength
    
    Example:
        >>> from utils.format_control import get_answer_sentinel_text
        >>> mhm = MultiHookManager(
        ...     model=wrapper.model, 
        ...     tokenizer=wrapper.tokenizer,
        ...     layer_to_directions={24: {'type': 'u', 'vec': u_tensor}},
        ...     device=wrapper.primary_device
        ... )
        >>> mhm.set_locality("cot")
        >>> mhm.set_intervention_params(mode="add", alpha=1.0)
        >>> # During generation, call update_boundaries() from BoundaryMonitor
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        layer_to_directions: Dict[int, Dict],
        device: torch.device,
        layer_path: str = "model.layers",
    ):
        """
        Initialize MultiHookManager.
        
        Args:
            model: The transformer model
            tokenizer: HuggingFace tokenizer for phrase detection
            layer_to_directions: Dict mapping layer_idx to direction info:
                {'type': 'u', 'vec': tensor} for single direction
                {'type': 'U', 'basis': tensor} for subspace
            device: Target device for tensors
            layer_path: Attribute path to access layers
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer_to_directions = layer_to_directions
        self.device = device
        self.layer_path = layer_path
        
        # Boundary tracking
        from utils.format_control import get_answer_sentinel_text
        self.answer_phrase_ids = tokenizer(get_answer_sentinel_text(), add_special_tokens=False).input_ids
        self.phrase_window = []
        self.cot_started_at = 0  # decode begins right after <cot>
        self.answer_started_at = None
        
        # Locality and intervention params
        self.locality = "all"  # "all", "cot", "answer"
        self.current_mask = 1.0  # scalar mask value (0.0 or 1.0)
        
        # Intervention parameters
        self.mode = "add"  # "add", "lesion", "rescue"
        self.alpha = 1.0
        self.gamma = 1.0
        self.beta = 1.0
        self.add_mode = "proj"  # "proj" or "constant"
        
        # Hook handles
        self.hook_handles = []
        
        # Register hooks on all layers
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on all layers in layer_to_directions."""
        parts = self.layer_path.split(".")
        layers_module = self.model
        for part in parts:
            layers_module = getattr(layers_module, part)
        
        for layer_idx in self.layer_to_directions.keys():
            target_layer = layers_module[layer_idx]
            handle = target_layer.register_forward_hook(self._make_hook(layer_idx))
            self.hook_handles.append(handle)
    
    def set_locality(self, mode: str):
        """
        Set locality mode for interventions.
        
        Args:
            mode: One of "all", "cot", "answer"
        """
        assert mode in ("all", "cot", "answer"), f"Invalid locality: {mode}"
        self.locality = mode
    
    def set_intervention_params(
        self,
        mode: str,
        alpha: float = 1.0,
        gamma: float = 1.0,
        beta: float = 1.0,
        add_mode: str = "proj",
    ):
        """
        Set intervention parameters.
        
        Args:
            mode: One of "add", "lesion", "rescue"
            alpha: Strength for add mode
            gamma: Strength for lesion/rescue mode
            beta: Beta parameter for rescue mode
            add_mode: "proj" or "constant" for add mode
        """
        assert mode in ("add", "lesion", "rescue"), f"Invalid mode: {mode}"
        self.mode = mode
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.add_mode = add_mode
    
    def set_cot_start(self, idx: int):
        """Set the decode index where CoT starts (typically 0 after <cot> token)."""
        self.cot_started_at = idx
    
    def reset_for_new_example(self):
        """
        Reset all state for a new example.
        
        Call this before each generation to ensure fresh state.
        Critical for deterministic evaluation.
        """
        self.phrase_window = []
        self.cot_started_at = 0
        self.answer_started_at = None
        self.current_mask = 1.0
        print("[MHM] State reset for new example")
    
    def set_answer_start(self, idx: int):
        """Set the decode index where answer starts (detected by phrase match)."""
        self.answer_started_at = idx
        print(f"[MHM] Answer boundary detected at decode_idx={idx}")
    
    def update_boundaries(self, decode_idx: int, token_id: int):
        """
        Update boundary tracking and recompute mask.
        
        Called every decode step by BoundaryMonitor.
        
        Args:
            decode_idx: Current decode step index
            token_id: Token ID just generated
        """
        # Update phrase matching window
        self.phrase_window.append(token_id)
        if len(self.phrase_window) > len(self.answer_phrase_ids):
            self.phrase_window.pop(0)
        
        # Check for answer phrase match (first time only)
        if self.answer_started_at is None and self.phrase_window == self.answer_phrase_ids:
            # Answer phrase just completed
            # The phrase started at (decode_idx - len(phrase) + 1)
            self.set_answer_start(decode_idx - len(self.answer_phrase_ids) + 1)
        
        # Recompute mask based on locality and boundaries
        self.current_mask = self._compute_step_mask(decode_idx)
    
    def _compute_step_mask(self, decode_idx: int) -> float:
        """
        Compute mask value for current decode step based on locality.
        
        Returns:
            1.0 if intervention should be applied, 0.0 otherwise
        """
        if self.locality == "all":
            return 1.0
        
        elif self.locality == "cot":
            # ON until answer starts
            if self.answer_started_at is None:
                return 1.0
            else:
                return 1.0 if decode_idx < self.answer_started_at else 0.0
        
        elif self.locality == "answer":
            # OFF until answer starts
            if self.answer_started_at is None:
                return 0.0
            else:
                return 1.0 if decode_idx >= self.answer_started_at else 0.0
        
        return 1.0
    
    def _make_hook(self, layer_idx: int):
        """Create intervention hook for specified layer."""
        
        def hook(module, inputs, output):
            # Extract hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]  # [B, T, H]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None
            
            B, T, H = hidden_states.shape
            
            # Only intervene on decode steps (T=1), not prefill
            if T != 1:
                return (hidden_states,) + rest if rest is not None else hidden_states
            
            # Apply locality mask
            if self.current_mask == 0.0:
                # Skip intervention
                return (hidden_states,) + rest if rest is not None else hidden_states
            
            # Get direction for this layer
            dir_info = self.layer_to_directions[layer_idx]
            
            # Create mask tensor
            mask_2d = torch.ones(B, T, dtype=torch.bool, device=hidden_states.device)
            
            # Apply intervention based on mode
            if self.mode == "add":
                from utils.hooks import apply_add
                if dir_info['type'] == 'u':
                    u = dir_info['vec']
                    U = None
                else:
                    u = None
                    U = dir_info['basis']
                
                hidden_states = apply_add(
                    hidden_states,
                    U=U,
                    u=u,
                    alpha=self.alpha * self.current_mask,  # Scale by mask
                    add_mode=self.add_mode,
                    mask=mask_2d,
                    debug_label=f"L{layer_idx}_mhm",
                    debug_every=50,  # Less verbose
                )
            
            elif self.mode == "lesion":
                from utils.hooks import apply_lesion
                if dir_info['type'] == 'u':
                    u = dir_info['vec']
                    U = None
                else:
                    u = None
                    U = dir_info['basis']
                
                hidden_states = apply_lesion(
                    hidden_states,
                    U=U,
                    u=u,
                    gamma=self.gamma * self.current_mask,
                    mask=mask_2d,
                )
            
            elif self.mode == "rescue":
                from utils.hooks import apply_rescue
                if dir_info['type'] == 'u':
                    u = dir_info['vec']
                    U = None
                else:
                    u = None
                    U = dir_info['basis']
                
                hidden_states = apply_rescue(
                    hidden_states,
                    U=U,
                    u=u,
                    gamma=self.gamma * self.current_mask,
                    beta=self.beta,
                    mask=mask_2d,
                )
            
            # Repack output
            if rest is not None:
                return (hidden_states,) + rest
            return hidden_states
        
        return hook
    
    def get_state_summary(self) -> dict:
        """
        Get current state summary for diagnostics.
        
        Returns:
            Dict with boundary information and current mask state
        """
        return {
            "locality": self.locality,
            "cot_started_at": self.cot_started_at,
            "answer_started_at": self.answer_started_at,
            "current_mask": self.current_mask,
            "mode": self.mode,
            "alpha": self.alpha,
        }
    
    def should_skip_example(self, locality: str, answer_found: bool) -> bool:
        """
        Determine if example should be skipped based on locality and answer detection.
        
        Args:
            locality: Target locality ("cot", "answer", "all")
            answer_found: Whether answer was successfully extracted
        
        Returns:
            True if example should be skipped, False otherwise
        """
        if locality in ("answer", "all") and not answer_found:
            print(f"[MHM] Skipping example: locality={locality} but answer_found=False")
            return True
        return False
    
    def close(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()


class OfflineLayerScreener:
    """
    Screen layers using collected activations to identify reasoning subspaces.

    This implements Phase A: offline screening without re-generation.

    Example
    -------
    ::

        # Assume X is [n_traces, n_layers, D] and y is [n_traces]
        screener = OfflineLayerScreener(
            X=pooled_activations,
            y=labels,
            layer_indices=[16, 20, 24, 28]
        )

        results = screener.screen_all_layers()
        ranked = screener.rank_layers(results)

        print("Top layers:", ranked[:3])
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        layer_indices: List[int],
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialize layer screener.

        Args:
            X: Pooled activations [n_traces, n_layers, D]
            y: Binary labels [n_traces] (1=correct, 0=incorrect)
            layer_indices: List of layer indices corresponding to X's 2nd dimension
            test_size: Fraction of data for validation
            random_state: Random seed for reproducibility
        """
        self.X = X  # [n, L, D]
        self.y = y  # [n]
        self.layer_indices = layer_indices
        self.test_size = test_size
        self.random_state = random_state

        # Split data
        all_idx = np.arange(len(y))
        self.train_idx, self.val_idx = _safe_train_val_split(
            indices=all_idx,
            y=np.asarray(y),
            test_size=self.test_size,
            random_state=self.random_state,
        )

    def compute_delta_mu_norm(self, layer_idx: int) -> float:
        """
        Compute ||μ⁺ - μ⁻||₂ for a layer.
        
        Returns 0.0 if either class has no examples (avoids empty mean).

        Args:
            layer_idx: Index in layer_indices list

        Returns:
            L2 norm of mean difference, or 0.0 if degenerate
        """
        X_layer = self.X[:, layer_idx, :]  # [n, D]
        mask_pos = self.y == 1
        mask_neg = self.y == 0

        # Guard against empty masks
        if not mask_pos.any() or not mask_neg.any():
            return 0.0

        mu_pos = X_layer[mask_pos].mean(axis=0)
        mu_neg = X_layer[mask_neg].mean(axis=0)

        return float(np.linalg.norm(mu_pos - mu_neg))

    def train_l1_probe_cv(
        self,
        layer_idx: int,
        C: float = 1.0,
        max_iter: int = 10000,
        n_splits: int = 3,
        class_weight: Optional[str] = "balanced",
    ) -> Dict[str, Union[float, Optional[LogisticRegression], Optional[np.ndarray]]]:
        """
        Train L1-regularized logistic regression probe with stratified k-fold CV.
        
        Uses standardization, threshold tuning for balanced accuracy, and reports
        mean±std for AUC, accuracy, and AUPRC across folds.
        
        IMPORTANT: Also returns the scaler.scale_ vector so that probe weights
        can be properly de-standardized when extracting directions. Without this,
        the extracted direction is in standardized-feature space, not original 
        hidden-state space, which can significantly distort interventions.

        Args:
            layer_idx: Index in layer_indices list
            C: Inverse regularization strength
            max_iter: Maximum iterations
            n_splits: Number of CV folds
            class_weight: Class weighting strategy ("balanced" or None)

        Returns:
            Dictionary with CV metrics:
                - mean_auc, std_auc: AUC mean and std
                - mean_acc, std_acc: Accuracy (at tuned threshold) mean and std
                - mean_auprc, std_auprc: AUPRC mean and std
                - probe_model: Trained model on full data (for extracting directions)
                - scaler_scale: np.ndarray of shape [D] - the per-feature std used for scaling.
                                Use this to de-standardize probe weights: w_orig = w / scaler_scale
                Returns NaN metrics if degenerate labels.
        """
        X_layer = self.X[:, layer_idx, :]
        y = self.y

        # Check for degenerate labels
        if len(np.unique(y)) < 2:
            return {
                "mean_auc": np.nan,
                "std_auc": np.nan,
                "mean_acc": np.nan,
                "std_acc": np.nan,
                "mean_auprc": np.nan,
                "std_auprc": np.nan,
                "probe_model": None,
                "scaler_scale": None,
            }

        # Stratified K-Fold CV (or KFold fallback)
        kfold = _choose_cv(y=np.asarray(y), n_splits=n_splits, shuffle=True, random_state=42)
        
        aucs, accs, auprcs = [], [], []
        
        for train_idx, val_idx in kfold.split(X_layer, y):
            X_train, X_val = X_layer[train_idx], X_layer[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Check fold has both classes
            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                continue
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train L1 logistic regression
            try:
                model = LogisticRegression(
                    penalty="l1",
                    C=C,
                    solver="liblinear",  # Better for L1 penalty
                    max_iter=max_iter,
                    random_state=42,
                    class_weight=class_weight,
                )
                model.fit(X_train_scaled, y_train)
            except (ValueError, Exception):
                # Skip fold if training fails
                continue
            
            # Predict probabilities
            try:
                y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            except (ValueError, IndexError):
                continue
            
            # Compute AUC and AUPRC
            try:
                auc = roc_auc_score(y_val, y_pred_proba)
                auprc = average_precision_score(y_val, y_pred_proba)
            except ValueError:
                continue
            
            # Tune threshold for balanced accuracy
            thresholds = np.linspace(0, 1, 101)
            best_thresh = 0.5
            best_ba = 0.0
            
            for thresh in thresholds:
                y_pred = (y_pred_proba >= thresh).astype(int)
                try:
                    ba = balanced_accuracy_score(y_val, y_pred)
                    if ba > best_ba:
                        best_ba = ba
                        best_thresh = thresh
                except ValueError:
                    continue
            
            # Accuracy at tuned threshold
            y_pred_tuned = (y_pred_proba >= best_thresh).astype(int)
            acc_tuned = accuracy_score(y_val, y_pred_tuned)
            
            aucs.append(auc)
            accs.append(acc_tuned)
            auprcs.append(auprc)
        
        # Return NaN if no successful folds
        if len(aucs) == 0:
            return {
                "mean_auc": np.nan,
                "std_auc": np.nan,
                "mean_acc": np.nan,
                "std_acc": np.nan,
                "mean_auprc": np.nan,
                "std_auprc": np.nan,
                "probe_model": None,
                "scaler_scale": None,
            }
        
        # Train final model on full data for direction extraction
        scaler_full = StandardScaler()
        X_scaled_full = scaler_full.fit_transform(X_layer)
        
        try:
            probe_model = LogisticRegression(
                penalty="l1",
                C=C,
                solver="liblinear",
                max_iter=max_iter,
                random_state=42,
                class_weight=class_weight,
            )
            probe_model.fit(X_scaled_full, y)
        except (ValueError, Exception):
            probe_model = None
        
        # Store scaler.scale_ for de-standardization when extracting directions
        # This is CRITICAL: probe weights are in standardized space, so we need
        # to divide by scale_ to get weights in original hidden-state space.
        scaler_scale = scaler_full.scale_.astype(np.float32)
        
        return {
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "mean_auprc": float(np.mean(auprcs)),
            "std_auprc": float(np.std(auprcs)),
            "probe_model": probe_model,
            "scaler_scale": scaler_scale,
        }

    def train_l1_probe(
        self,
        layer_idx: int,
        C: float = 1.0,
        max_iter: int = 1000,
    ) -> Tuple[float, float, Optional[LogisticRegression]]:
        """
        Train L1-regularized logistic regression probe (legacy single-split method).
        
        DEPRECATED: Use train_l1_probe_cv() for more robust CV-based evaluation.
        
        Returns NaN metrics if training/validation split has fewer than 2 classes.

        Args:
            layer_idx: Index in layer_indices list
            C: Inverse regularization strength
            max_iter: Maximum iterations

        Returns:
            Tuple of (val_auc, val_acc, trained_model).
            Returns (np.nan, np.nan, None) if degenerate labels.
        """
        X_layer = self.X[:, layer_idx, :]

        X_train = X_layer[self.train_idx]
        y_train = self.y[self.train_idx]
        X_val = X_layer[self.val_idx]
        y_val = self.y[self.val_idx]

        # Check for degenerate labels in train or val split
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            return np.nan, np.nan, None

        model = LogisticRegression(
            penalty="l1",
            C=C,
            solver="saga",
            max_iter=max_iter,
            random_state=self.random_state,
        )
        
        try:
            model.fit(X_train, y_train)
        except ValueError:
            # Fallback for rare sklearn errors with degenerate data
            return np.nan, np.nan, None

        # Validation metrics
        try:
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)

            val_auc = roc_auc_score(y_val, y_pred_proba)
            val_acc = float((y_pred == y_val).mean())
        except (ValueError, IndexError):
            # Handle edge cases in prediction
            return np.nan, np.nan, None

        return float(val_auc), val_acc, model

    def extract_top_directions(
        self,
        layer_idx: int,
        probe_model: LogisticRegression,
        k: int = 128,
        method: str = "dense_normalized",
        scaler_scale: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract intervention directions from probe weights.
        
        CRITICAL FIX: When a StandardScaler was used to train the probe, the weights
        are in standardized-feature space. To get the correct direction in the original
        hidden-state space, we must de-standardize: w_orig = w / scaler_scale.
        
        Without this correction, the extracted direction is skewed toward low-variance
        dimensions and can point largely in the wrong direction.

        Args:
            layer_idx: Index in layer_indices list
            probe_model: Trained logistic regression model
            k: Number of directions to extract (ignored for "dense_normalized")
            method: Extraction method:
                - "dense_normalized": Single dense direction (probe weight normalized). Best for d=1.
                - "top_k_sparse": Top-k one-hot basis vectors (original method). Memory-heavy.
                - "top_k_dense": Top-k features as columns with their weights.
            scaler_scale: Optional array of shape [D] from StandardScaler.scale_.
                         If provided, weights are de-standardized before normalization.
                         STRONGLY RECOMMENDED when probe was trained on scaled data.

        Returns:
            Matrix U of shape:
            - [1, D] for "dense_normalized" (row-major, for compatibility with hooks)
            - [k, D] for "top_k_sparse" or "top_k_dense" (row-major)
            
        Note:
            The returned U is in row-major format [k, D] where k is the number of
            directions. This is the expected format for utils/hooks.py functions.
        """
        weights = probe_model.coef_[0].copy()  # [D], copy to avoid modifying original
        D = len(weights)
        
        # DE-STANDARDIZE: Convert weights from standardized space to original space
        # This is CRITICAL for correct intervention directions.
        # In standardized space: z = (x - μ) / σ
        # Probe learns: decision = w_z · z = w_z · (x - μ) / σ
        # To express in original space: w_x = w_z / σ
        if scaler_scale is not None:
            # Guard against division by zero for constant features
            safe_scale = np.where(scaler_scale > 1e-8, scaler_scale, 1.0)
            weights = weights / safe_scale
            print(f"[extract_top_directions] De-standardized weights for layer {layer_idx}")

        if method == "dense_normalized":
            # Return normalized probe weight as single direction (best for interventions)
            w_norm = weights / (np.linalg.norm(weights) + 1e-8)
            # Return as row-major [1, D] for compatibility with hooks
            return w_norm[np.newaxis, :].astype(np.float32)  # [1, D]
            
        elif method == "top_k_dense":
            # Top-k features with their actual weights (more interpretable)
            top_indices = np.argsort(np.abs(weights))[-k:]
            # Build row-major U [k, D]
            U = np.zeros((k, D), dtype=np.float32)
            for i, idx in enumerate(top_indices):
                U[i, idx] = weights[idx]
            # Normalize each row
            row_norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
            U = U / row_norms
            # Orthonormalize rows using QR decomposition for proper projection
            U = self._orthonormalize_rows(U)
            return U
            
        elif method == "top_k_sparse":
            # Original method: one-hot basis vectors
            top_indices = np.argsort(np.abs(weights))[-k:]
            # Build row-major U [k, D]
            U = np.zeros((k, D), dtype=np.float32)
            for i, idx in enumerate(top_indices):
                U[i, idx] = 1.0
            # These are already orthonormal (one-hot vectors)
            return U
            
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'dense_normalized', 'top_k_dense', 'top_k_sparse'")
    
    def _orthonormalize_rows(self, U: np.ndarray) -> np.ndarray:
        """
        Orthonormalize rows of U using QR decomposition.
        
        For multi-dimensional subspaces, proper projection requires orthonormal basis.
        P_S h = U^T (U U^T)^{-1} U h, but if U is orthonormal, this simplifies to
        P_S h = U^T U h.
        
        Args:
            U: Matrix of shape [k, D] with k <= D
        
        Returns:
            Orthonormalized matrix of shape [k, D] where rows are orthonormal
        """
        k, D = U.shape
        if k == 1:
            # Single direction, just normalize
            norm = np.linalg.norm(U[0]) + 1e-8
            return U / norm
        
        if k > D:
            # More directions than dimensions - truncate
            U = U[:D]
            k = D
        
        # QR decomposition of U^T gives Q [D, k] with orthonormal columns
        # We want rows of result to be orthonormal, so take Q^T
        try:
            Q, R = np.linalg.qr(U.T, mode='reduced')  # Q is [D, k], R is [k, k]
            U_ortho = Q.T  # [k, D] with orthonormal rows
            return U_ortho.astype(np.float32)
        except np.linalg.LinAlgError:
            # Fallback: just normalize rows if QR fails
            row_norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
            return (U / row_norms).astype(np.float32)

    def screen_all_layers(
        self,
        C: float = 1.0,
        max_iter: int = 10000,
        use_cv: bool = True,
        n_splits: int = 3,
        class_weight: Optional[str] = "balanced",
    ) -> Dict[int, Dict[str, Union[float, np.ndarray]]]:
        """
        Screen all layers and compute metrics.

        Args:
            C: L1 regularization strength
            max_iter: Max iterations for probe training
            use_cv: If True, use CV-based training; if False, use legacy single-split
            n_splits: Number of CV folds (if use_cv=True)
            class_weight: Class weighting strategy

        Returns:
            Dictionary mapping layer index to metrics dict containing:
                - delta_mu_norm: float
                - mean_auc, std_auc: CV metrics (if use_cv=True)
                - mean_acc, std_acc: CV metrics (if use_cv=True)
                - mean_auprc, std_auprc: CV metrics (if use_cv=True)
                - val_auc, val_acc: Legacy single-split metrics (if use_cv=False)
                - probe_model: LogisticRegression
                - scaler_scale: np.ndarray (if use_cv=True) - for de-standardizing probe weights
        """
        results = {}

        for i, layer_id in enumerate(self.layer_indices):
            delta_mu = self.compute_delta_mu_norm(i)
            
            if use_cv:
                cv_results = self.train_l1_probe_cv(
                    i, C=C, max_iter=max_iter, n_splits=n_splits, class_weight=class_weight
                )
                results[layer_id] = {
                    "delta_mu_norm": delta_mu,
                    "mean_auc": cv_results["mean_auc"],
                    "std_auc": cv_results["std_auc"],
                    "mean_acc": cv_results["mean_acc"],
                    "std_acc": cv_results["std_acc"],
                    "mean_auprc": cv_results["mean_auprc"],
                    "std_auprc": cv_results["std_auprc"],
                    "probe_model": cv_results["probe_model"],
                    "scaler_scale": cv_results["scaler_scale"],  # For de-standardization
                }
            else:
                # Legacy single-split
                val_auc, val_acc, probe = self.train_l1_probe(i, C=C, max_iter=max_iter)
                results[layer_id] = {
                    "delta_mu_norm": delta_mu,
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                    "probe_model": probe,
                    "scaler_scale": None,  # Legacy path doesn't have scaler
                }

        return results

    def rank_layers(
        self,
        results: Dict[int, Dict[str, Union[float, np.ndarray]]],
        auc_weight: float = 0.6,
        delta_weight: float = 0.4,
        use_cv: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        Rank layers by weighted combination of metrics.
        
        Layers with NaN metrics are pushed to the bottom of the ranking.

        Args:
            results: Output from screen_all_layers()
            auc_weight: Weight for validation AUC (or mean_auc if use_cv=True)
            delta_weight: Weight for delta-mu norm
            use_cv: If True, rank by mean_auc; if False, rank by val_auc

        Returns:
            List of (layer_idx, score) tuples sorted by score (descending).
            Layers with NaN scores appear at the end with score=-inf.
        """
        scores = []

        # Determine which AUC metric to use
        auc_key = "mean_auc" if use_cv else "val_auc"

        # Collect metrics, handling NaNs
        aucs = []
        deltas = []
        for layer in self.layer_indices:
            auc = results[layer].get(auc_key, np.nan)
            delta = results[layer]["delta_mu_norm"]
            # Replace NaN with 0 for normalization purposes
            aucs.append(0.0 if np.isnan(auc) else auc)
            deltas.append(0.0 if np.isnan(delta) else delta)
        
        aucs = np.array(aucs)
        deltas = np.array(deltas)

        # Compute max values for normalization (ignoring NaNs)
        auc_max = np.nanmax(aucs) if np.any(aucs > 0) else 1.0
        delta_max = np.nanmax(deltas) if np.any(deltas > 0) else 1.0

        for i, layer_id in enumerate(self.layer_indices):
            auc_val = results[layer_id].get(auc_key, np.nan)
            delta_val = results[layer_id]["delta_mu_norm"]
            
            # If either metric is NaN, assign -inf score to push to bottom
            if np.isnan(auc_val) or np.isnan(delta_val):
                score = float('-inf')
            else:
                norm_auc = auc_val / auc_max
                norm_delta = delta_val / delta_max
                score = auc_weight * norm_auc + delta_weight * norm_delta
            
            scores.append((layer_id, score))

        # Sort by score descending (NaN/-inf will be at the end)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
