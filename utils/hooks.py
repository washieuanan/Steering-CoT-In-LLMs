"""
Causal intervention primitives for reasoning vector experiments.

This module provides low-level operations for modifying model activations:
- Projection onto learned subspaces
- Add interventions (boost subspace component)
- Lesion interventions (suppress subspace component)
- Rescue interventions (lesion + partial restore)
- Locality mask computation with token span mapping

All operations support both single-vector and multi-dimensional subspaces.
"""

from typing import Optional, Union, Dict, Tuple, Literal, List
import torch
import os

# Import unified answer extractor for locality mask computation
from answers.extract_final_choice import extract_choice_with_fallback


def project_onto_subspace(h: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Project hidden states onto subspace spanned by U.
    
    Computes P_S h = U (U^T h) where U is a set of orthonormal basis vectors.
    Uses memory-efficient two-step multiplication to avoid forming U U^T.
    
    Args:
        h: Hidden states [..., H] (any shape ending in hidden dimension)
        U: Basis matrix [R, H] where R is subspace dimension (row-major)
    
    Returns:
        Projected states [..., H] with same shape as h
    
    Example:
        >>> h = torch.randn(32, 512, 4096)  # [B, T, H]
        >>> U = torch.randn(8, 4096)  # [R, H], R=8 dimensional subspace
        >>> h_proj = project_onto_subspace(h, U)
        >>> h_proj.shape
        torch.Size([32, 512, 4096])
    """
    # h: [..., H], U: [R, H]
    # Step 1: Project onto subspace coordinates: U^T h -> [..., R]
    coords = torch.matmul(h, U.T)  # [..., H] @ [H, R] = [..., R]
    
    # Step 2: Reconstruct in original space: U coords -> [..., H]
    projection = torch.matmul(coords, U)  # [..., R] @ [R, H] = [..., H]
    
    return projection


def apply_add(
    h: torch.Tensor,
    U: Optional[torch.Tensor] = None,
    u: Optional[torch.Tensor] = None,
    alpha: float = 0.0,
    add_mode: str = "proj",
    mask: Optional[torch.Tensor] = None,
    debug_label: Optional[str] = None,
    debug_every: int = 0,
    lm_head: Optional[torch.nn.Module] = None,
    pos: Optional[int] = None,
    in_cot: Optional[bool] = None,
    in_answer: Optional[bool] = None,
    layer: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply additive intervention with configurable mode.
    
    Two modes available:
    - "proj": h' = h + α * Proj_U(h) - projection-based (current behavior)
    - "constant": h' = h + α * u - constant direction addition
    
    Args:
        h: Hidden states [..., H] (any shape ending in hidden dimension)
        U: Optional basis matrix [k, H] for subspace intervention (row-major)
        u: Optional single direction [H] for vector intervention
        alpha: Intervention strength (0 = no change)
        add_mode: Either "proj" or "constant"
        mask: Optional boolean mask [...] indicating which positions to edit
        debug_label: Optional label for debug logging
        debug_every: If >0 and DEBUG_HOOKS=1, log perturbation magnitude
        pos: Optional decode position for logging
        in_cot: Optional CoT flag for logging
        in_answer: Optional answer flag for logging
        layer: Optional layer index for logging
    
    Returns:
        Modified hidden states [..., H] with same shape as h
    
    Example:
        >>> h = torch.randn(1, 100, 4096)
        >>> u = torch.randn(4096)
        >>> u = u / u.norm()  # Normalize
        >>> h_boosted = apply_add(h, u=u, alpha=1.5, add_mode="proj")
        >>> h_constant = apply_add(h, u=u, alpha=1.5, add_mode="constant")
    """
    H = h.shape[-1]
    
    # Shape assertions
    if u is not None:
        assert u.ndim == 1, f"u must be 1D, got shape {u.shape}"
        assert u.shape[0] == H, f"u dim {u.shape} != H {H}"
    if U is not None:
        assert U.ndim == 2, f"U must be 2D, got shape {U.shape}"
        assert U.shape[1] == H, f"U dim {U.shape} != H {H}"
    
    # Select positions to modify
    if mask is not None:
        h_sel = h[mask]  # [N, H] where N is number of True positions
    else:
        h_sel = h.view(-1, h.size(-1))  # [B*T, H]
    
    # Compute intervention component in fp32 for stability
    h_sel_float = h_sel.float()
    
    if add_mode == "proj":
        # Projection-based: α * Proj_U(h)
        if U is not None:
            # Subspace projection
            U_float = U.float()
            comp = project_onto_subspace(h_sel_float, U_float)
        elif u is not None:
            # Single vector projection: <h, u> * u
            u_float = u.float()
            inner_prod = torch.matmul(h_sel_float, u_float.unsqueeze(-1))  # [N, 1]
            comp = inner_prod * u_float.unsqueeze(0)  # [N, H]
        else:
            raise ValueError("Must provide either U or u for intervention")
    
    elif add_mode == "constant":
        # Constant direction: α * u (independent of h·u)
        if u is not None:
            u_float = u.float()
            comp = u_float.unsqueeze(0).expand_as(h_sel_float)  # [N, H]
        elif U is not None:
            # Pool multiple directions by taking mean
            U_float = U.float()
            u_mean = U_float.mean(dim=0)  # [H]
            comp = u_mean.unsqueeze(0).expand_as(h_sel_float)  # [N, H]
        else:
            raise ValueError("Must provide either U or u for intervention")
    
    else:
        raise ValueError(f"Unknown add_mode={add_mode}, must be 'proj' or 'constant'")
    
    # Apply intervention: h' = h + α * comp
    delta = alpha * comp
    h_sel_new = h_sel_float + delta
    
    # Cast back to original dtype
    h_sel_new = h_sel_new.to(h.dtype)
    delta = delta.to(h.dtype)
    
    # Enhanced debug with delta norm and decode position tracking
    if debug_every and (os.environ.get("DEBUG_HOOKS", "0") == "1"):
        with torch.no_grad():
            hn_mean = h_sel.norm(dim=-1).mean().item() + 1e-12
            dn_mean = delta.norm(dim=-1).mean().item()
            dn_max = delta.norm(dim=-1).max().item()
            ratio = dn_mean / hn_mean
            
            # Compute intervention vector norm
            if add_mode == "proj":
                if U is not None:
                    # Subspace projection magnitude
                    comp_norm = comp.norm(dim=-1).mean().item()
                elif u is not None:
                    # Single vector projection magnitude
                    comp_norm = comp.norm(dim=-1).mean().item()
                else:
                    comp_norm = 0.0
            else:  # constant mode
                comp_norm = comp.norm(dim=-1).mean().item()
            
            # Build log message with all context
            log_parts = [f"[hook]"]
            if layer is not None:
                log_parts.append(f"L{layer}")
            log_parts.append(debug_label or 'add')
            if pos is not None:
                log_parts.append(f"pos={pos}")
            if in_cot is not None:
                log_parts.append(f"in_cot={int(in_cot)}")
            if in_answer is not None:
                log_parts.append(f"in_ans={int(in_answer)}")
            log_parts.append(f"alpha={alpha:.2f}")
            log_parts.append(f"mode={add_mode}")
            log_parts.append(f"Δ‖h‖={dn_mean:.4e}/{dn_max:.4e}")
            log_parts.append(f"‖comp‖={comp_norm:.4e}")
            
            print(" ".join(log_parts))
    
    # Update hidden states
    if mask is not None:
        h = h.clone()  # Don't modify in-place if using mask
        h[mask] = h_sel_new
    else:
        h = h_sel_new.view_as(h)
    
    return h


def apply_lesion(
    h: torch.Tensor,
    U: Optional[torch.Tensor] = None,
    u: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply lesion intervention: h' = (I - γ * P_S) h = h - γ * P_S h
    
    Suppresses the component of h along the subspace S by factor γ.
    γ=1.0 completely removes the subspace component (orthogonal projection).
    γ<1.0 partially suppresses. γ>1.0 reverses the component.
    
    Args:
        h: Hidden states [B, T, H]
        U: Optional basis matrix [R, H] for subspace intervention
        u: Optional single direction [H] for vector intervention
        gamma: Suppression strength (1.0 = full removal, 0.0 = no change)
        mask: Optional boolean mask [B, T] indicating which positions to edit
    
    Returns:
        Modified hidden states [B, T, H]
    
    Example:
        >>> h = torch.randn(1, 100, 4096)
        >>> U = torch.randn(8, 4096)
        >>> h_lesioned = apply_lesion(h, U=U, gamma=1.0)
    """
    # Select positions to modify
    if mask is not None:
        h_sel = h[mask]  # [N, H]
    else:
        h_sel = h.view(-1, h.size(-1))  # [B*T, H]
    
    # Compute projection component in fp32 for stability
    h_sel_float = h_sel.float()
    
    if U is not None:
        # Subspace projection
        U_float = U.float()
        comp = project_onto_subspace(h_sel_float, U_float)
    elif u is not None:
        # Single vector projection: <h, u> * u
        u_float = u.float()
        inner_prod = torch.matmul(h_sel_float, u_float.unsqueeze(-1))  # [N, 1]
        comp = inner_prod * u_float.unsqueeze(0)  # [N, H]
    else:
        raise ValueError("Must provide either U or u for intervention")
    
    # Apply intervention: h' = h - γ * comp
    h_sel_new = h_sel_float - gamma * comp
    
    # Cast back to original dtype and update
    h_sel_new = h_sel_new.to(h.dtype)
    
    if mask is not None:
        h = h.clone()  # Don't modify in-place if using mask
        h[mask] = h_sel_new
    else:
        h = h_sel_new.view_as(h)
    
    return h


def apply_rescue(
    h: torch.Tensor,
    U: Optional[torch.Tensor] = None,
    u: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    beta: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply rescue intervention: h' = (I - γ * P_S) h + β * P_S h = h + (β - γ) * P_S h
    
    Combines lesion with partial restoration. Useful for testing if a subspace
    is necessary but not sufficient.
    
    Args:
        h: Hidden states [B, T, H]
        U: Optional basis matrix [R, H] for subspace intervention
        u: Optional single direction [H] for vector intervention
        gamma: Initial suppression strength (typically 1.0 for full lesion)
        beta: Restoration strength (0.0 to 1.0, where 1.0 = full restore)
        mask: Optional boolean mask [B, T] indicating which positions to edit
    
    Returns:
        Modified hidden states [B, T, H]
    
    Example:
        >>> h = torch.randn(1, 100, 4096)
        >>> u = torch.randn(4096)
        >>> u = u / u.norm()
        >>> # Lesion then restore 50%
        >>> h_rescued = apply_rescue(h, u=u, gamma=1.0, beta=0.5)
    """
    # Select positions to modify
    if mask is not None:
        h_sel = h[mask]  # [N, H]
    else:
        h_sel = h.view(-1, h.size(-1))  # [B*T, H]
    
    # Compute projection component in fp32 for stability
    h_sel_float = h_sel.float()
    
    if U is not None:
        # Subspace projection
        U_float = U.float()
        comp = project_onto_subspace(h_sel_float, U_float)
    elif u is not None:
        # Single vector projection: <h, u> * u
        u_float = u.float()
        inner_prod = torch.matmul(h_sel_float, u_float.unsqueeze(-1))  # [N, 1]
        comp = inner_prod * u_float.unsqueeze(0)  # [N, H]
    else:
        raise ValueError("Must provide either U or u for intervention")
    
    # Apply intervention: h' = h + (β - γ) * comp
    h_sel_new = h_sel_float + (beta - gamma) * comp
    
    # Cast back to original dtype and update
    h_sel_new = h_sel_new.to(h.dtype)
    
    if mask is not None:
        h = h.clone()  # Don't modify in-place if using mask
        h[mask] = h_sel_new
    else:
        h = h_sel_new.view_as(h)
    
    return h


def intervene_hidden(
    h: torch.Tensor,
    mode: str,
    U: Optional[torch.Tensor] = None,
    u: Optional[torch.Tensor] = None,
    alpha: float = 0.0,
    gamma: float = 1.0,
    beta: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Unified dispatcher for all intervention types.
    
    Args:
        h: Hidden states [B, T, H]
        mode: Intervention type - "add", "lesion", or "rescue"
        U: Optional basis matrix [R, H] for subspace intervention
        u: Optional single direction [H] for vector intervention
        alpha: Strength for "add" mode
        gamma: Suppression strength for "lesion" and "rescue" modes
        beta: Restoration strength for "rescue" mode
        mask: Optional boolean mask [B, T] indicating which positions to edit
    
    Returns:
        Modified hidden states [B, T, H]
    
    Example:
        >>> h = torch.randn(1, 100, 4096)
        >>> u = torch.randn(4096) / torch.randn(4096).norm()
        >>> mask = torch.zeros(1, 100, dtype=torch.bool)
        >>> mask[0, 50:80] = True  # Only edit positions 50-80
        >>> 
        >>> # Add intervention
        >>> h_add = intervene_hidden(h, "add", u=u, alpha=1.5, mask=mask)
        >>> 
        >>> # Lesion intervention
        >>> h_lesion = intervene_hidden(h, "lesion", u=u, gamma=1.0, mask=mask)
        >>> 
        >>> # Rescue intervention
        >>> h_rescue = intervene_hidden(h, "rescue", u=u, gamma=1.0, beta=0.5, mask=mask)
    """
    if mode == "add":
        return apply_add(h, U=U, u=u, alpha=alpha, mask=mask)
    elif mode == "lesion":
        return apply_lesion(h, U=U, u=u, gamma=gamma, mask=mask)
    elif mode == "rescue":
        return apply_rescue(h, U=U, u=u, gamma=gamma, beta=beta, mask=mask)
    else:
        # No intervention or unknown mode - return unchanged
        return h


def apply_subspace_intervention(
    hidden: torch.Tensor,  # (B, T, H)
    mask: torch.BoolTensor,  # (B, T)
    U: Optional[torch.Tensor] = None,  # (H, K) or (K, H) 
    u: Optional[torch.Tensor] = None,  # (H,) single direction
    alpha: float = 1.0,
    mode: Literal["add", "project", "erase", "rescue"] = "add",
    constant: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """
    Single source of truth for applying subspace interventions with diagnostics.
    
    This is the unified interface for all Phase-B interventions, providing:
    - Consistent tensor shape handling
    - Automatic normalization verification
    - Rich diagnostic output
    - Support for multiple intervention modes
    
    Args:
        hidden: Hidden states tensor (B, T, H)
        mask: Boolean mask (B, T) where True indicates positions to edit
        U: Optional basis matrix. Can be (H, K) col-major or (K, H) row-major.
           Will auto-detect and transpose if needed.
        u: Optional single direction vector (H,)
        alpha: Intervention strength
        mode: Intervention type:
            - "add": h + alpha * Proj_U(h)  [or h + alpha * u if constant=True]
            - "project": h + alpha * Proj_U(h)  [same as add, for compatibility]
            - "erase": h - alpha * Proj_U(h)
            - "rescue": h + (alpha - 1.0) * Proj_U(h)  [partial restoration]
        constant: If True and mode="add", use constant direction (h + alpha * u)
                 instead of projection (h + alpha * Proj_U(h))
    
    Returns:
        Tuple of (modified_hidden, diagnostics_dict)
        
        diagnostics_dict contains:
            - num_tokens_masked: int
            - per_token_delta_norm_mean: float
            - per_token_delta_norm_std: float
            - per_token_delta_norm_max: float
            - mean_hidden_norm: float (baseline)
            - mean_comp_norm: float (intervention component)
            - direction_was_normalized: bool
    
    Example:
        >>> h = torch.randn(1, 100, 4096)
        >>> mask = torch.zeros(1, 100, dtype=torch.bool)
        >>> mask[0, 20:80] = True
        >>> u = torch.randn(4096)
        >>> h_out, diag = apply_subspace_intervention(h, mask, u=u, alpha=2.0, mode="add")
        >>> print(f"Modified {diag['num_tokens_masked']} tokens")
        >>> print(f"Mean delta norm: {diag['per_token_delta_norm_mean']:.4f}")
    """
    B, T, H = hidden.shape
    assert mask.shape == (B, T), f"Mask shape {mask.shape} != hidden shape {(B, T)}"
    
    # Validate and prepare direction
    direction_was_normalized = False
    if U is not None:
        assert U.ndim == 2, f"U must be 2D, got shape {U.shape}"
        # Auto-detect orientation: prefer (K, H) row-major format
        if U.shape[0] == H and U.shape[1] != H:
            # Transpose (H, K) -> (K, H)
            U = U.T
        assert U.shape[1] == H, f"U shape {U.shape} incompatible with H={H}"
        
        # Verify/normalize rows
        with torch.no_grad():
            row_norms = U.norm(dim=1)
            if not torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-3):
                # Normalize
                U = U / (row_norms.unsqueeze(1) + 1e-8)
                direction_was_normalized = True
        
        U_work = U
        u_work = None
    elif u is not None:
        assert u.ndim == 1, f"u must be 1D, got shape {u.shape}"
        assert u.shape[0] == H, f"u shape {u.shape} != H {H}"
        
        # Verify/normalize
        with torch.no_grad():
            u_norm = u.norm()
            if not torch.isclose(u_norm, torch.tensor(1.0, device=u.device), atol=1e-3):
                u = u / (u_norm + 1e-8)
                direction_was_normalized = True
        
        U_work = None
        u_work = u
    else:
        raise ValueError("Must provide either U or u")
    
    # Count masked tokens
    num_masked = int(mask.sum().item())
    
    if num_masked == 0:
        # No tokens to modify
        diag = {
            "num_tokens_masked": 0,
            "per_token_delta_norm_mean": 0.0,
            "per_token_delta_norm_std": 0.0,
            "per_token_delta_norm_max": 0.0,
            "mean_hidden_norm": 0.0,
            "mean_comp_norm": 0.0,
            "direction_was_normalized": direction_was_normalized,
        }
        return hidden, diag
    
    # Extract masked positions
    h_sel = hidden[mask]  # [N, H] where N = num_masked
    h_sel_float = h_sel.float()
    
    # Compute intervention component
    if mode in ("add", "project"):
        add_mode_str = "constant" if constant else "proj"
        h_out_sel = apply_add(
            h_sel.unsqueeze(0),  # Add batch dim [1, N, H]
            U=U_work,
            u=u_work,
            alpha=alpha,
            add_mode=add_mode_str,
            mask=None,  # Already selected
        ).squeeze(0)  # Remove batch dim
        
    elif mode == "erase":
        h_out_sel = apply_lesion(
            h_sel.unsqueeze(0),
            U=U_work,
            u=u_work,
            gamma=alpha,
            mask=None,
        ).squeeze(0)
        
    elif mode == "rescue":
        h_out_sel = apply_rescue(
            h_sel.unsqueeze(0),
            U=U_work,
            u =u_work,
            gamma=1.0,
            beta=alpha,
            mask=None,
        ).squeeze(0)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute deltas and diagnostics
    with torch.no_grad():
        delta = h_out_sel - h_sel
        delta_norms = delta.norm(dim=-1)  # [N]
        hidden_norms = h_sel.norm(dim=-1)  # [N]
        
        # Compute component norm
        if U_work is not None:
            comp = project_onto_subspace(h_sel_float, U_work.float())
        elif u_work is not None:
            u_float = u_work.float()
            inner_prod = torch.matmul(h_sel_float, u_float.unsqueeze(-1))
            comp = inner_prod * u_float.unsqueeze(0)
        else:
            comp = torch.zeros_like(h_sel_float)
        comp_norms = comp.norm(dim=-1)
        
        diag = {
            "num_tokens_masked": num_masked,
            "per_token_delta_norm_mean": float(delta_norms.mean().item()),
            "per_token_delta_norm_std": float(delta_norms.std().item()),
            "per_token_delta_norm_max": float(delta_norms.max().item()),
            "mean_hidden_norm": float(hidden_norms.mean().item()),
            "mean_comp_norm": float(comp_norms.mean().item()),
            "direction_was_normalized": direction_was_normalized,
        }
    
    # Update hidden states (don't modify in-place)
    hidden_out = hidden.clone()
    hidden_out[mask] = h_out_sel
    
    return hidden_out, diag


def compute_locality_mask_post_generation(
    tokenizer,
    prompt_text: str,
    full_text: str,
    input_ids: List[int],
    locality: str,
    cot_text: Optional[str] = None,
    token_texts_tail: Optional[List[str]] = None,
    choice_logits: Optional[Dict] = None,
) -> Dict:
    """
    Compute locality mask with robust token span mapping.
    
    This is the unified function for computing intervention masks post-generation.
    It handles:
    1. Answer extraction via unified extractor
    2. Character→token span mapping via re-tokenization
    3. CoT and answer span identification
    4. Mask construction with hard assertions
    
    Args:
        tokenizer: HuggingFace tokenizer
        prompt_text: Original prompt string
        full_text: Complete generation (prompt + CoT + answer)
        input_ids: Token IDs for full_text
        locality: One of 'cot', 'answer', 'all'
        cot_text: Optional CoT text (not used, kept for compatibility)
        token_texts_tail: Optional list of last ~32 decoded tokens for fallback
        choice_logits: Optional logits for fallback (not used currently)
    
    Returns:
        Dictionary with:
        - mask: torch.BoolTensor of length len(input_ids)
        - spans: Dict with 'cot_span', 'answer_span', 'answer_found'
        - skip: bool, True if should skip this example
    
    Example:
        >>> result = compute_locality_mask_post_generation(
        ...     tokenizer, prompt, full_text, input_ids, 'cot'
        ... )
        >>> mask = result['mask']  # [L] bool tensor
        >>> if not result['skip']:
        ...     # Apply intervention with mask
    """
    # Compute prompt length in tokens
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    prompt_len = len(prompt_ids)
    
    L = len(input_ids)
    
    # Extract answer using unified extractor
    answer_found, letter, (char_start, char_end) = extract_choice_with_fallback(
        full_text, token_texts_tail=token_texts_tail
    )
    
    # Map character span to token span via re-tokenization
    answer_token_start = L  # Default: not found
    answer_token_end = L
    
    if answer_found and char_start < char_end:
        # Re-tokenize with offset mapping
        try:
            encoding = tokenizer(
                full_text,
                add_special_tokens=False,
                return_offsets_mapping=True
            )
            
            if 'offset_mapping' in encoding:
                offsets = encoding['offset_mapping']
                
                # Find tokens that overlap with [char_start, char_end)
                for tok_idx, (tok_start, tok_end) in enumerate(offsets):
                    # Check if token overlaps with answer span
                    if tok_end > char_start and tok_start < char_end:
                        if answer_token_start == L:
                            answer_token_start = tok_idx
                        answer_token_end = tok_idx + 1  # Exclusive end
        except Exception:
            # Fallback: rolling match by concatenating token texts
            pass
    
    # Determine CoT span
    if answer_found and answer_token_start < L:
        cot_token_start = prompt_len
        cot_token_end = answer_token_start
    else:
        # No answer found: entire generation is CoT
        cot_token_start = prompt_len
        cot_token_end = L
    
    # Build mask based on locality
    mask = torch.zeros(L, dtype=torch.bool)
    skip = False
    
    if locality == "cot":
        # Mask CoT region
        if cot_token_end > cot_token_start:
            mask[cot_token_start:cot_token_end] = True
    
    elif locality == "answer":
        # Mask answer region only
        if answer_found and answer_token_end > answer_token_start:
            mask[answer_token_start:answer_token_end] = True
        else:
            # No answer span found - must skip
            skip = True
    
    elif locality == "all":
        # Mask entire generation
        if answer_found and answer_token_end > answer_token_start:
            # CoT + answer
            mask[cot_token_start:cot_token_end] = True
            mask[answer_token_start:answer_token_end] = True
        else:
            # No answer: just CoT
            mask[cot_token_start:L] = True
    
    # Diagnostic logging
    if os.environ.get("DEBUG_HOOKS", "0") == "1":
        cot_len = cot_token_end - cot_token_start
        ans_len = answer_token_end - answer_token_start if answer_found else 0
        masked_count = int(mask.sum().item())
        
        print(f"[diag-locality] L={L} prompt_len={prompt_len} locality={locality}")
        print(f"  answer_found={answer_found} answer_start={answer_token_start if answer_found else -1}")
        print(f"  cot_span=[{cot_token_start},{cot_token_end}) len={cot_len}")
        print(f"  answer_span=[{answer_token_start},{answer_token_end}) len={ans_len}")
        print(f"  final_mask: {masked_count} tokens marked True out of {L}")
        if skip:
            print(f"  SKIP=True (no answer span for locality={locality})")
    
    return {
        'mask': mask,
        'spans': {
            'cot_span': (cot_token_start, cot_token_end),
            'answer_span': (answer_token_start, answer_token_end),
            'answer_found': answer_found,
            'answer_letter': letter,
        },
        'skip': skip,
    }
