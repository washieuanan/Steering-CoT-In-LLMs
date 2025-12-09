"""
Phase B Step 1: Interventions & Data Collection with Pre-CoT vs Post-CoT Locality

This script implements causal interventions on reasoning vectors with locality control:
1. Loads Phase A artifacts (directions, metrics, generations)
2. Runs paired baseline vs intervention generations
3. Sweeps intervention strength (α) across unified grid
4. Supports four modes: add, lesion, rescue, random
5. Supports two locality modes: cot (pre-CoT injection), answer (post-CoT injection)
6. Inline GPT grading for reasoning quality assessment
7. Saves paired outputs with parsing and correctness labels

Modes:
- add: Positive alpha sweep (0.5, 1, 2) to amplify reasoning
- lesion: Remove reasoning subspace (γ=1.0) to test necessity
- rescue: Remove then restore (γ=1.0, β=1.0) to test causal triad
- random: Control experiment using random orthogonal subspace

Locality:
- cot: Inject during CoT generation (before answer phrase appears)
- answer: Inject during answer generation (after answer phrase appears)

Example usage:
    python -m experiments.phase_b_runner \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --phase_a_run results/phase_a/Llama-3.1-8B-Instruct__gpt_graded/arc \
        --datasets arc \
        --split test \
        --num_examples 50 \
        --topk_layers 3 \
        --modes add \
        --alpha_grid "0.5,1,2" \
        --locality cot \
        --prompt_mode cot \
        --judge_model gpt-4.1-mini \
        --seed 42 \
        --out_dir results/phase_b
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from hf_model_wrapper import HFModelConfig, HFModelWrapper, BoundaryMonitor
from dataset_loaders import load_dataset_by_name
from multi_hook_manager import detect_layer_path, MultiHookManager
from utils import get_handler
from utils.reasoning_grader import grade_reasoning


def _ensure_rowspace_kH_and_normalize(U_or_u: np.ndarray, H_expected: int) -> np.ndarray:
    """
    Ensure direction has correct shape and is L2-normalized.
    
    Args:
        U_or_u: Direction vector or subspace matrix
        H_expected: Expected hidden dimension
    
    Returns:
        Normalized direction(s) in shape (H,) for vector or (k, H) for subspace
    """
    X = U_or_u
    
    # Handle 1D vector
    if X.ndim == 1:
        assert X.shape[0] == H_expected, f"Direction dim {X.shape} != hidden {H_expected}"
        # Normalize
        n = np.linalg.norm(X) + 1e-12
        X = X / n
        return X
    
    # Handle 2D subspace
    elif X.ndim == 2:
        # Expect (k, H) row-major format
        if X.shape[0] == H_expected and X.shape[1] != H_expected:
            # Transpose if we loaded (H, k)
            X = X.T
        assert X.shape[1] == H_expected, f"U dim {X.shape} incompatible with H={H_expected}"
        # Normalize rows
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X = X / norms
        return X
    
    else:
        raise ValueError(f"Bad direction shape {X.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase B Step 1: Causal interventions with pre-CoT vs post-CoT locality"
    )

    # Model and data
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--phase_a_run", type=str, required=True,
                        help="Path to Phase A output dir (e.g., results/phase_a/MODEL__TAG/dataset/)")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated datasets (optional if phase_a_run points to specific dataset)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_examples", type=int, default=200)
    parser.add_argument("--few_shot", type=int, default=0)

    # Layer selection
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (e.g., '31,29,21')")
    parser.add_argument("--topk_layers", type=int, default=0,
                        help="If >0, use top-k layers from Phase A metrics (ignores --layers)")

    # Intervention parameters
    parser.add_argument("--modes", type=str, default="add",
                        help="Comma-separated modes: add, lesion, rescue, random")
    parser.add_argument("--alpha_grid", type=str, default="0.5,1,2",
                        help="Comma-separated alpha values for 'add' and 'random' modes (positive only)")
    parser.add_argument("--add_mode", type=str, choices=["proj", "constant"], default="constant",
                        help="How to construct the add vector: 'constant' uses α*u directly (RECOMMENDED), 'proj' uses α*Proj_U(h)")
    
    # Locality mode for interventions (KEY PARAMETER)
    parser.add_argument("--locality", type=str, choices=["all", "cot", "answer"], default="cot",
                        help="Where to apply interventions: 'cot' (during CoT, before answer phrase), "
                             "'answer' (after answer phrase), 'all' (every decode step)")
    
    # Lesion/Rescue parameters
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Lesion strength (γ) - fraction of projection to remove")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Rescue re-add strength (β) - fraction of projection to re-add after lesion")
    
    # Prompt mode - MUST be 'cot' for pre-CoT vs post-CoT experiments
    parser.add_argument("--prompt_mode", type=str, choices=["cot", "direct", "phase_b"], default="cot",
                        help="'cot': CoT-scaffolded prompt (REQUIRED for locality experiments), "
                             "'direct': bare question, "
                             "'phase_b': answer format but NO CoT instruction")

    # GPT grading (primary accuracy measure)
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini",
                        help="OpenAI model for GPT grading (primary accuracy measure)")

    # Generation parameters
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--out_dir", type=str, default="results/phase_b")
    parser.add_argument("--phase_b_tag", type=str, default=None,
                        help="Optional tag for Phase B run. If provided, uses this instead of auto-generating "
                             "a timestamp. All modes (add, lesion, rescue, random) will share the same directory.")
    
    # Random subspace control
    parser.add_argument("--use_random_subspace", action="store_true",
                        help="Use random orthogonal subspace instead of learned directions for ALL modes. "
                             "This serves as a control experiment to verify effects are specific to learned direction.")

    return parser.parse_args()


def load_layer_directions(npz_path: Path, device: torch.device) -> Dict[int, Dict[str, Any]]:
    """
    Load direction vectors/subspaces from Phase A screening.
    
    Returns dict[layer_idx -> {'type': str, 'vec': Tensor or 'basis': Tensor, ...}]
    """
    data = np.load(npz_path, allow_pickle=True)
    layer_to_dir = {}
    
    for key in data.files:
        if not key.startswith('L'):
            continue
        
        layer_idx = int(key[1:])
        arr = data[key]  # Shape could be [H], [H,1], [R,H], etc.
        
        # Determine direction type and normalize
        if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
            # Single direction vector
            vec = arr.flatten().astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            layer_to_dir[layer_idx] = {
                'type': 'u',
                'vec': torch.from_numpy(vec).to(device)
            }
        else:
            # Multi-dimensional subspace [R, H]
            U = arr.astype(np.float32)
            # Normalize each row
            norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
            U = U / norms
            layer_to_dir[layer_idx] = {
                'type': 'U',
                'basis': torch.from_numpy(U).to(device)
            }
    
    return layer_to_dir


def create_random_subspace(
    layer_to_dir: Dict[int, Dict[str, Any]],
    device: torch.device,
    seed: int = 42
) -> Dict[int, Dict[str, Any]]:
    """
    Create random orthogonal subspace with same dimensions as learned subspace.
    
    This serves as a control to verify that effects are specific to the
    learned reasoning direction, not just any random perturbation.
    """
    rng = np.random.RandomState(seed)
    random_dirs = {}
    
    for layer_idx, dir_info in layer_to_dir.items():
        if dir_info['type'] == 'u':
            # Single vector: create random unit vector
            vec = dir_info['vec'].cpu().numpy()
            H = vec.shape[0]
            random_vec = rng.randn(H).astype(np.float32)
            random_vec = random_vec / (np.linalg.norm(random_vec) + 1e-8)
            random_dirs[layer_idx] = {
                'type': 'u',
                'vec': torch.from_numpy(random_vec).to(device)
            }
        else:
            # Subspace: create random orthogonal basis with same rank
            basis = dir_info['basis'].cpu().numpy()
            k, H = basis.shape
            # Generate random matrix and orthonormalize
            random_mat = rng.randn(k, H).astype(np.float32)
            Q, _ = np.linalg.qr(random_mat.T)
            random_basis = Q.T[:k]  # Take first k rows
            random_dirs[layer_idx] = {
                'type': 'U',
                'basis': torch.from_numpy(random_basis).to(device)
            }
    
    return random_dirs


def load_phase_a_artifacts(
    phase_a_run: Path,
    topk_layers: int = 0
) -> Tuple[Dict[int, Dict], List[int], pd.DataFrame, Dict]:
    """
    Load Phase A outputs: directions, layer ranking, generations, config.
    Always uses dataset-specific directions and metrics.
    
    Returns:
        Tuple of (layer_directions, target_layers, generations_df, run_config)
    """
    # Load dataset-specific directions
    directions_path = phase_a_run / "screening" / "layer_to_U.npz"
    if not directions_path.exists():
        raise FileNotFoundError(f"No directions found at {directions_path}")
    print(f"Loading directions from {directions_path}")
    
    # Load directions (device will be set when loading model)
    layer_to_dir = load_layer_directions(directions_path, torch.device('cpu'))
    
    # DIAGNOSTIC: Verify direction norms and shapes
    print(f"\n[diag-directions] Loaded {len(layer_to_dir)} layers from {directions_path}")
    for layer_idx in sorted(layer_to_dir.keys())[:5]:  # Show first 5 layers
        dir_info = layer_to_dir[layer_idx]
        if dir_info['type'] == 'u':
            vec = dir_info['vec']
            norm = float(vec.norm().item())
            print(f"  Layer {layer_idx}: type=u, shape={tuple(vec.shape)}, ||u||={norm:.6f}")
        else:
            basis = dir_info['basis']
            fnorm = float(torch.norm(basis).item())
            col0_norm = float(basis[0].norm().item()) if basis.shape[0] > 0 else 0.0
            print(f"  Layer {layer_idx}: type=U, shape={tuple(basis.shape)}, ||U||_F={fnorm:.6f}, ||U[0]||={col0_norm:.6f}")
    if len(layer_to_dir) > 5:
        print(f"  ... ({len(layer_to_dir) - 5} more layers)")
    print()
    
    # Determine target layers
    if topk_layers > 0:
        # Use dataset-specific metrics
        metrics_path = phase_a_run / "screening" / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"No metrics found at {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Extract ranked layers (format: [[layer, score, ...], ...])
        ranked = metrics.get('ranked_layers', [])
        target_layers = [int(item[0]) for item in ranked[:topk_layers]]
        
        print(f"Selected top-{topk_layers} layers: {target_layers}")
    else:
        # Use all available layers from directions
        target_layers = sorted(layer_to_dir.keys())
        print(f"Using all available layers: {target_layers}")
    
    # Load generations for prompts
    csv_path = phase_a_run / "generations.csv"
    csv_gz_path = phase_a_run / "generations.csv.gz"
    
    if csv_gz_path.exists():
        with gzip.open(csv_gz_path, 'rt', encoding='utf-8') as f:
            generations_df = pd.read_csv(f)
    elif csv_path.exists():
        generations_df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No generations CSV found in {phase_a_run}")
    
    # Load run config
    run_config_path = phase_a_run / "run_config.json"
    if run_config_path.exists():
        with open(run_config_path, 'r') as f:
            run_config = json.load(f)
    else:
        run_config = {}
    
    return layer_to_dir, target_layers, generations_df, run_config


def grade_with_retry(
    question: str,
    gold_answer: str,
    model_output: str,
    judge_model: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Dict[str, Any]:
    """
    Call GPT grading with exponential backoff retry logic.
    """
    for attempt in range(max_retries):
        try:
            result = grade_reasoning(
                question=question,
                gold_answer=gold_answer,
                model_output=model_output,
                judge_model=judge_model,
            )
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"    [GPT retry {attempt+1}/{max_retries}] {e}, waiting {delay:.1f}s...")
                time.sleep(delay)
            else:
                print(f"    [GPT failed] {e}")
                return {
                    "reasoning_correct": False,
                    "answer_correct": False,
                    "raw_response": f"ERROR: {e}",
                }


def run_generation_with_mhm(
    wrapper: HFModelWrapper,
    handler,
    example,
    multi_hook_manager: Optional[MultiHookManager],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Run a single generation with MultiHookManager for locality-aware intervention.
    
    Args:
        wrapper: HFModelWrapper instance
        handler: Dataset handler
        example: Dataset example
        multi_hook_manager: MultiHookManager instance (None for baseline)
        args: Command line arguments
    
    Returns:
        Dict with output text, parsing results, timing.
    """
    tokenizer = wrapper.tokenizer
    model = wrapper.model
    device = wrapper.primary_device
    
    # Build prompt using handler (MUST be CoT mode for locality experiments)
    input_ids, attention_mask, cot_start_idx, controller_kwargs = handler.build_prompt(
        tokenizer, example, prompt_mode=args.prompt_mode
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    prompt_len = input_ids.shape[1]
    
    # Reset MultiHookManager state for this example
    if multi_hook_manager is not None:
        multi_hook_manager.reset_for_new_example()
        multi_hook_manager.set_cot_start(0)  # Generation starts immediately after prompt
    
    # Set up stopping criteria with BoundaryMonitor if using MHM
    from transformers import StoppingCriteriaList
    stopping_criteria = None
    if multi_hook_manager is not None:
        boundary_monitor = BoundaryMonitor(tokenizer, multi_hook_manager)
        stopping_criteria = StoppingCriteriaList([boundary_monitor])
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        # Use structured generation for CoT mode
        if args.prompt_mode == "cot" and 'mode' in controller_kwargs and controller_kwargs['mode'] in ('mcq', 'numeric'):
            # Build controller config
            controller_config = {
                'mode': controller_kwargs.get('mode', 'mcq'),
                'phrase_text': controller_kwargs.get('phrase_text'),
                'min_cot_tokens': controller_kwargs.get('min_cot_tokens', 24),
                'allowed_letters': controller_kwargs.get('allowed_letters'),
                'blocked_special_ids': controller_kwargs.get('blocked_special_ids'),
                'require_reason': controller_kwargs.get('require_reason', True),
                'max_answer_tokens': controller_kwargs.get('max_answer_tokens', 32),
            }
            
            outputs = wrapper.generate_structured(
                input_ids=input_ids,
                attention_mask=attention_mask,
                controller_config=controller_config,
                max_new_tokens=args.max_new_tokens,
                stopping_criteria=stopping_criteria,
            )
        else:
            # Fallback to direct generation
            outputs = wrapper.generate_direct(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                stopping_criteria=stopping_criteria,
            )
    
    gen_time_ms = int((time.perf_counter() - start_time) * 1000)
    
    seq = outputs.sequences[0]
    gen_ids = seq[prompt_len:]
    n_tokens = len(gen_ids)
    
    # Decode
    text_raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
    text_clean = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    # Parse answer
    pred, status = handler.parse_pred(text_raw, example)
    gold = handler.gold_target(example)
    correct = handler.compare(pred, gold) if pred else False
    
    # Get MHM state for diagnostics
    mhm_state = None
    if multi_hook_manager is not None:
        mhm_state = multi_hook_manager.get_state_summary()
    
    return {
        'text_raw': text_raw,
        'text_clean': text_clean,
        'pred': pred,
        'status': status,
        'gold': gold,
        'correct': correct,
        'gen_time_ms': gen_time_ms,
        'n_tokens': n_tokens,
        'mhm_state': mhm_state,
    }


def run_paired_generation(
    wrapper: HFModelWrapper,
    handler,
    example,
    layer_idx: int,
    alpha: float,
    layer_to_dir: Dict[int, Dict],
    args: argparse.Namespace,
    intervention_mode: str = "add",
    gamma: float = 1.0,
    beta: float = 1.0,
) -> Dict[str, Any]:
    """
    Run paired baseline and intervention generation with MultiHookManager.
    
    Returns dict with baseline and intervention outputs, GPT grades.
    """
    device = wrapper.primary_device
    
    # ===== BASELINE (no intervention) =====
    baseline = run_generation_with_mhm(
        wrapper=wrapper,
        handler=handler,
        example=example,
        multi_hook_manager=None,  # No hooks for baseline
        args=args,
    )
    
    # ===== INTERVENTION (with MultiHookManager) =====
    # Build directions dict for this layer only
    single_layer_dir = {layer_idx: layer_to_dir[layer_idx]}
    
    # Create MultiHookManager
    mhm = MultiHookManager(
        model=wrapper.model,
        tokenizer=wrapper.tokenizer,
        layer_to_directions=single_layer_dir,
        device=device,
        layer_path=detect_layer_path(wrapper.model),
    )
    
    # Configure locality
    mhm.set_locality(args.locality)
    
    # Configure intervention parameters
    mhm.set_intervention_params(
        mode=intervention_mode,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        add_mode=args.add_mode,
    )
    
    try:
        intv = run_generation_with_mhm(
            wrapper=wrapper,
            handler=handler,
            example=example,
            multi_hook_manager=mhm,
            args=args,
        )
    finally:
        mhm.close()
    
    # Get question for GPT grading
    question = (
        getattr(example, "question", None)
        or getattr(example, "problem", None)
        or str(example)
    )
    
    # GPT grading
    baseline_gpt = grade_with_retry(
        question=question,
        gold_answer=baseline['gold'],
        model_output=baseline['text_clean'],
        judge_model=args.judge_model,
    )
    intv_gpt = grade_with_retry(
        question=question,
        gold_answer=intv['gold'],
        model_output=intv['text_clean'],
        judge_model=args.judge_model,
    )
    
    # Build result dict
    task_type = handler.task_type(example)
    allowed_letters = None
    
    if task_type == 'mcq':
        choices = getattr(example, 'choices', [])
        allowed_letters = ",".join([chr(65 + i) for i in range(len(choices))])
    
    result = {
        'example_id': getattr(example, 'example_id', ''),
        'task_type': task_type,
        'prompt_mode': args.prompt_mode,
        'locality': args.locality,
        'gold': baseline['gold'],
        'allowed_letters': allowed_letters if task_type == 'mcq' else '',
        
        # Baseline results - GPT grades (PRIMARY)
        'baseline_answer_correct': baseline_gpt['answer_correct'],
        'baseline_reasoning_correct': baseline_gpt['reasoning_correct'],
        # Baseline results - Regex parsing (SUPPLEMENTARY)
        'baseline_correct_regex': int(baseline['correct']),
        'baseline_pred_regex': baseline['pred'] if baseline['pred'] else '',
        'baseline_text_raw': baseline['text_raw'],
        'gen_ms_base': baseline['gen_time_ms'],
        'n_gen_tokens_base': baseline['n_tokens'],
        
        # Intervention results - GPT grades (PRIMARY)
        'intv_answer_correct': intv_gpt['answer_correct'],
        'intv_reasoning_correct': intv_gpt['reasoning_correct'],
        # Intervention results - Regex parsing (SUPPLEMENTARY)
        'intv_correct_regex': int(intv['correct']),
        'intv_pred_regex': intv['pred'] if intv['pred'] else '',
        'intv_text_raw': intv['text_raw'],
        'gen_ms_intv': intv['gen_time_ms'],
        'n_gen_tokens_intv': intv['n_tokens'],
        
        # MHM diagnostics
        'intv_answer_detected_at': intv['mhm_state']['answer_started_at'] if intv['mhm_state'] else None,
        
        # Config
        'mode': intervention_mode,
        'layer': layer_idx,
        'alpha': alpha,
        'gamma': gamma,
        'beta': beta,
    }
    
    return result


def run_rescue_triplet(
    wrapper: HFModelWrapper,
    handler,
    example,
    layer_idx: int,
    gamma: float,
    beta: float,
    layer_to_dir: Dict[int, Dict],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Run triplet: baseline, lesion, rescue for causal triad test.
    """
    device = wrapper.primary_device
    
    # ===== BASELINE (no intervention) =====
    baseline = run_generation_with_mhm(
        wrapper=wrapper,
        handler=handler,
        example=example,
        multi_hook_manager=None,
        args=args,
    )
    
    # Build directions dict for this layer only
    single_layer_dir = {layer_idx: layer_to_dir[layer_idx]}
    
    # ===== LESION =====
    mhm_lesion = MultiHookManager(
        model=wrapper.model,
        tokenizer=wrapper.tokenizer,
        layer_to_directions=single_layer_dir,
        device=device,
        layer_path=detect_layer_path(wrapper.model),
    )
    mhm_lesion.set_locality(args.locality)
    mhm_lesion.set_intervention_params(mode="lesion", gamma=gamma)
    
    try:
        lesion = run_generation_with_mhm(
            wrapper=wrapper,
            handler=handler,
            example=example,
            multi_hook_manager=mhm_lesion,
            args=args,
        )
    finally:
        mhm_lesion.close()
    
    # ===== RESCUE =====
    mhm_rescue = MultiHookManager(
        model=wrapper.model,
        tokenizer=wrapper.tokenizer,
        layer_to_directions=single_layer_dir,
        device=device,
        layer_path=detect_layer_path(wrapper.model),
    )
    mhm_rescue.set_locality(args.locality)
    mhm_rescue.set_intervention_params(mode="rescue", gamma=gamma, beta=beta)
    
    try:
        rescue = run_generation_with_mhm(
            wrapper=wrapper,
            handler=handler,
            example=example,
            multi_hook_manager=mhm_rescue,
            args=args,
        )
    finally:
        mhm_rescue.close()
    
    # Get question for GPT grading
    question = (
        getattr(example, "question", None)
        or getattr(example, "problem", None)
        or str(example)
    )
    
    # GPT grading
    baseline_gpt = grade_with_retry(question, baseline['gold'], baseline['text_clean'], args.judge_model)
    lesion_gpt = grade_with_retry(question, lesion['gold'], lesion['text_clean'], args.judge_model)
    rescue_gpt = grade_with_retry(question, rescue['gold'], rescue['text_clean'], args.judge_model)
    
    # Build result dict
    task_type = handler.task_type(example)
    allowed_letters = None
    if task_type == 'mcq':
        choices = getattr(example, 'choices', [])
        allowed_letters = ",".join([chr(65 + i) for i in range(len(choices))])
    
    result = {
        'example_id': getattr(example, 'example_id', ''),
        'task_type': task_type,
        'prompt_mode': args.prompt_mode,
        'locality': args.locality,
        'gold': baseline['gold'],
        'allowed_letters': allowed_letters if task_type == 'mcq' else '',
        
        # Baseline
        'baseline_answer_correct': baseline_gpt['answer_correct'],
        'baseline_reasoning_correct': baseline_gpt['reasoning_correct'],
        'baseline_correct_regex': int(baseline['correct']),
        'baseline_pred_regex': baseline['pred'] if baseline['pred'] else '',
        'baseline_text_raw': baseline['text_raw'],
        'gen_ms_base': baseline['gen_time_ms'],
        'n_gen_tokens_base': baseline['n_tokens'],
        
        # Lesion
        'lesion_answer_correct': lesion_gpt['answer_correct'],
        'lesion_reasoning_correct': lesion_gpt['reasoning_correct'],
        'lesion_correct_regex': int(lesion['correct']),
        'lesion_pred_regex': lesion['pred'] if lesion['pred'] else '',
        'lesion_text_raw': lesion['text_raw'],
        'gen_ms_lesion': lesion['gen_time_ms'],
        'n_gen_tokens_lesion': lesion['n_tokens'],
        
        # Rescue
        'rescue_answer_correct': rescue_gpt['answer_correct'],
        'rescue_reasoning_correct': rescue_gpt['reasoning_correct'],
        'rescue_correct_regex': int(rescue['correct']),
        'rescue_pred_regex': rescue['pred'] if rescue['pred'] else '',
        'rescue_text_raw': rescue['text_raw'],
        'gen_ms_rescue': rescue['gen_time_ms'],
        'n_gen_tokens_rescue': rescue['n_tokens'],
        
        # Config
        'mode': 'rescue',
        'layer': layer_idx,
        'gamma': gamma,
        'beta': beta,
    }
    
    return result


def main() -> None:
    args = parse_args()
    
    # Validate prompt_mode for locality experiments
    if args.locality in ("cot", "answer") and args.prompt_mode != "cot":
        print(f"WARNING: locality='{args.locality}' requires prompt_mode='cot' for proper boundary detection!")
        print(f"         Current prompt_mode='{args.prompt_mode}'. Switching to 'cot'.")
        args.prompt_mode = "cot"
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Parse phase_a_run path
    phase_a_run = Path(args.phase_a_run)
    if not phase_a_run.exists():
        raise ValueError(f"Phase A run directory does not exist: {phase_a_run}")
    
    # Load Phase A artifacts
    print(f"Loading Phase A artifacts from {phase_a_run}")
    layer_to_dir, target_layers, generations_df, phase_a_config = load_phase_a_artifacts(
        phase_a_run, args.topk_layers
    )
    
    # Override layers if specified
    if args.layers and args.topk_layers == 0:
        target_layers = [int(x.strip()) for x in args.layers.split(',')]
        print(f"Using specified layers: {target_layers}")
    
    # Determine dataset
    if args.datasets:
        datasets = [s.strip() for s in args.datasets.split(',')]
    else:
        # Infer from phase_a_run path or generations_df
        if 'dataset' in generations_df.columns:
            datasets = [generations_df['dataset'].iloc[0]]
        else:
            datasets = [phase_a_run.name]
    
    dataset_name = datasets[0]
    print(f"Dataset: {dataset_name}")
    
    # Load model
    print(f"Loading model: {args.model_name}")
    special_tokens = ["<cot>", "</cot>", "<answer>"]
    cfg = HFModelConfig(
        model_name=args.model_name,
        dtype=args.dtype,
        device=args.device,
        special_tokens=special_tokens,
        init_special_tokens_with_avg=True,
    )
    wrapper = HFModelWrapper(cfg).load()
    
    # Move directions to model device
    device = wrapper.primary_device
    for layer_idx in layer_to_dir:
        if 'vec' in layer_to_dir[layer_idx]:
            layer_to_dir[layer_idx]['vec'] = layer_to_dir[layer_idx]['vec'].to(device)
        if 'basis' in layer_to_dir[layer_idx]:
            layer_to_dir[layer_idx]['basis'] = layer_to_dir[layer_idx]['basis'].to(device)
    
    # Create random subspace for control experiments
    random_dirs = create_random_subspace(layer_to_dir, device, args.seed)
    
    # If --use_random_subspace is set, replace learned directions with random ones
    # This applies to ALL modes (add, lesion, rescue) for proper control experiment
    if args.use_random_subspace:
        print(f"\n[RANDOM SUBSPACE CONTROL] Using random orthogonal subspace instead of learned directions")
        print(f"  This applies to ALL intervention modes (add, lesion, rescue)")
        layer_to_dir = random_dirs  # Replace learned directions with random ones
    
    # Load dataset and handler
    loader = load_dataset_by_name(dataset_name, split=args.split, seed=args.seed)
    handler = get_handler(dataset_name)
    
    # Parse modes and alpha grid
    modes = [m.strip() for m in args.modes.split(',')]
    alpha_values = [float(x.strip()) for x in args.alpha_grid.split(',')]
    
    # Set up output directory with locality in name
    model_short_name = Path(args.model_name).name.replace("/", "-")
    
    # Use provided phase_b_tag or generate timestamp-based tag
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.phase_b_tag:
        tag = args.phase_b_tag
        print(f"Using provided phase_b_tag: {tag}")
    else:
        tag = f"{args.locality}_locality_{timestamp}"
    
    out_root = Path(args.out_dir) / f"{model_short_name}__{tag}" / dataset_name
    out_root.mkdir(parents=True, exist_ok=True)
    runs_dir = out_root / "runs"
    runs_dir.mkdir(exist_ok=True)
    
    # Load existing run_config.json and merge if it exists (to track all modes run)
    run_config_path = out_root / 'run_config.json'
    if run_config_path.exists():
        with open(run_config_path, 'r') as f:
            existing_config = json.load(f)
        # Merge modes: add new modes to existing list
        existing_modes = existing_config.get('modes_completed', existing_config.get('modes', []))
        all_modes = list(set(existing_modes + modes))
        existing_config['modes_completed'] = all_modes
        existing_config['last_updated'] = timestamp
        # Update with current run params (in case they changed)
        existing_config['modes'] = modes  # Current run modes
        run_config = existing_config
    else:
        run_config = {
            'model_name': args.model_name,
            'phase_a_run': str(phase_a_run),
            'dataset': dataset_name,
            'split': args.split,
            'num_examples': args.num_examples,
            'target_layers': target_layers,
            'modes': modes,
            'modes_completed': modes,  # Track all modes that have been run
            'alpha_values': alpha_values,
            'add_mode': args.add_mode,
            'locality': args.locality,  # KEY: cot or answer
            'use_random_subspace': args.use_random_subspace,  # Control experiment flag
            'gamma': args.gamma,
            'beta': args.beta,
            'prompt_mode': args.prompt_mode,
            'judge_model': args.judge_model,
            'grading_mode': 'gpt_primary',
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'seed': args.seed,
            'timestamp': timestamp,
            'last_updated': timestamp,
        }
    run_config_path.write_text(json.dumps(run_config, indent=2))
    
    # Load existing grid.csv if it exists (to append new results)
    grid_path = out_root / 'grid.csv'
    if grid_path.exists():
        existing_grid_df = pd.read_csv(grid_path)
        grid_results = existing_grid_df.to_dict('records')
        print(f"Loaded existing grid.csv with {len(grid_results)} rows")
    else:
        grid_results = []
    
    # Load existing rescue_summary.csv if it exists (to append new results)
    rescue_path = out_root / 'rescue_summary.csv'
    if rescue_path.exists():
        existing_rescue_df = pd.read_csv(rescue_path)
        rescue_results = existing_rescue_df.to_dict('records')
        print(f"Loaded existing rescue_summary.csv with {len(rescue_results)} rows")
    else:
        rescue_results = []
    
    print(f"\n{'='*60}")
    print(f"Phase B Experiments - Pre-CoT vs Post-CoT Locality")
    print(f"{'='*60}")
    print(f"Locality: {args.locality} (inject during {args.locality} phase)")
    print(f"Prompt mode: {args.prompt_mode}")
    print(f"Modes: {modes}")
    print(f"Layers: {target_layers}")
    print(f"Alpha grid: {alpha_values}")
    print(f"GPT grading (PRIMARY): {args.judge_model}")
    print(f"{'='*60}\n")
    
    num_examples = min(args.num_examples, len(loader))
    
    for mode in modes:
        print(f"\n[Mode: {mode}]")
        
        if mode == "add":
            # Alpha sweep experiment (positive values only)
            for layer_idx in target_layers:
                if layer_idx not in layer_to_dir:
                    print(f"  Warning: No direction found for layer {layer_idx}, skipping")
                    continue
                
                for alpha in alpha_values:
                    config_label = f"add_L{layer_idx}_A{alpha}_{args.locality}"
                    print(f"\n  [{config_label}]")
                    
                    paired_rows = []
                    for idx in tqdm(range(num_examples), desc=f"    Generating"):
                        example = loader.get_example(idx)
                        
                        try:
                            result = run_paired_generation(
                                wrapper=wrapper,
                                handler=handler,
                                example=example,
                                layer_idx=layer_idx,
                                alpha=alpha,
                                layer_to_dir=layer_to_dir,
                                args=args,
                                intervention_mode="add",
                            )
                            paired_rows.append(result)
                        except Exception as e:
                            print(f"      Error on example {idx}: {e}")
                            continue
                    
                    # Save paired CSV
                    csv_path = runs_dir / f"paired_{config_label}.csv.gz"
                    df = pd.DataFrame(paired_rows)
                    with gzip.open(csv_path, 'wt', encoding='utf-8') as f:
                        df.to_csv(f, index=False, quoting=csv.QUOTE_ALL)
                    
                    print(f"    Saved {len(df)} examples to {csv_path.name}")
                    
                    # Compute grid summary stats
                    n = len(df)
                    if n > 0:
                        acc_base_answer = df['baseline_answer_correct'].mean()
                        acc_intv_answer = df['intv_answer_correct'].mean()
                        delta_answer = acc_intv_answer - acc_base_answer
                        
                        acc_base_reasoning = df['baseline_reasoning_correct'].mean()
                        acc_intv_reasoning = df['intv_reasoning_correct'].mean()
                        delta_reasoning = acc_intv_reasoning - acc_base_reasoning
                        
                        answer_wrong_to_right = int(((df['baseline_answer_correct'] == False) & (df['intv_answer_correct'] == True)).sum())
                        answer_right_to_wrong = int(((df['baseline_answer_correct'] == True) & (df['intv_answer_correct'] == False)).sum())
                        answer_net_gain = answer_wrong_to_right - answer_right_to_wrong
                        
                        acc_base_regex = df['baseline_correct_regex'].mean()
                        acc_intv_regex = df['intv_correct_regex'].mean()
                    else:
                        acc_base_answer = acc_intv_answer = delta_answer = 0.0
                        acc_base_reasoning = acc_intv_reasoning = delta_reasoning = 0.0
                        answer_wrong_to_right = answer_right_to_wrong = answer_net_gain = 0
                        acc_base_regex = acc_intv_regex = 0.0
                    
                    grid_row = {
                        'mode': mode,
                        'locality': args.locality,
                        'layer': layer_idx,
                        'alpha': alpha,
                        'n': n,
                        'acc_base_answer': acc_base_answer,
                        'acc_intv_answer': acc_intv_answer,
                        'delta_answer': delta_answer,
                        'answer_wrong_to_right': answer_wrong_to_right,
                        'answer_right_to_wrong': answer_right_to_wrong,
                        'answer_net_gain': answer_net_gain,
                        'acc_base_reasoning': acc_base_reasoning,
                        'acc_intv_reasoning': acc_intv_reasoning,
                        'delta_reasoning': delta_reasoning,
                        'acc_base_regex': acc_base_regex,
                        'acc_intv_regex': acc_intv_regex,
                    }
                    grid_results.append(grid_row)
                    
                    print(f"    [GPT Answer]    Baseline: {acc_base_answer:.3f}, Intv: {acc_intv_answer:.3f}, Δ: {delta_answer:+.3f}")
                    print(f"    [GPT Reasoning] Baseline: {acc_base_reasoning:.3f}, Intv: {acc_intv_reasoning:.3f}, Δ: {delta_reasoning:+.3f}")
                    print(f"    Answer flips:    {answer_wrong_to_right}→right, {answer_right_to_wrong}→wrong, net={answer_net_gain:+d}")
        
        elif mode == "lesion":
            # Lesion experiment (necessity test)
            for layer_idx in target_layers:
                if layer_idx not in layer_to_dir:
                    print(f"  Warning: No direction found for layer {layer_idx}, skipping")
                    continue
                
                config_label = f"lesion_L{layer_idx}_G{args.gamma}_{args.locality}"
                print(f"\n  [{config_label}]")
                
                lesion_rows = []
                for idx in tqdm(range(num_examples), desc=f"    Generating"):
                    example = loader.get_example(idx)
                    
                    try:
                        result = run_paired_generation(
                            wrapper=wrapper,
                            handler=handler,
                            example=example,
                            layer_idx=layer_idx,
                            alpha=0.0,  # Not used for lesion
                            layer_to_dir=layer_to_dir,
                            args=args,
                            intervention_mode="lesion",
                            gamma=args.gamma,
                        )
                        lesion_rows.append(result)
                    except Exception as e:
                        print(f"      Error on example {idx}: {e}")
                        continue
                
                # Save lesion CSV
                csv_path = runs_dir / f"lesion_{config_label}.csv.gz"
                df = pd.DataFrame(lesion_rows)
                with gzip.open(csv_path, 'wt', encoding='utf-8') as f:
                    df.to_csv(f, index=False, quoting=csv.QUOTE_ALL)
                
                print(f"    Saved {len(df)} examples to {csv_path.name}")
                
                # Compute lesion summary stats
                n = len(df)
                if n > 0:
                    acc_base_answer = df['baseline_answer_correct'].mean()
                    acc_lesion_answer = df['intv_answer_correct'].mean()
                    delta_answer = acc_lesion_answer - acc_base_answer
                    
                    acc_base_reasoning = df['baseline_reasoning_correct'].mean()
                    acc_lesion_reasoning = df['intv_reasoning_correct'].mean()
                    delta_reasoning = acc_lesion_reasoning - acc_base_reasoning
                    
                    answer_wrong_to_right = int(((df['baseline_answer_correct'] == False) & (df['intv_answer_correct'] == True)).sum())
                    answer_right_to_wrong = int(((df['baseline_answer_correct'] == True) & (df['intv_answer_correct'] == False)).sum())
                    answer_net_gain = answer_wrong_to_right - answer_right_to_wrong
                    
                    acc_base_regex = df['baseline_correct_regex'].mean()
                    acc_lesion_regex = df['intv_correct_regex'].mean()
                else:
                    acc_base_answer = acc_lesion_answer = delta_answer = 0.0
                    acc_base_reasoning = acc_lesion_reasoning = delta_reasoning = 0.0
                    answer_wrong_to_right = answer_right_to_wrong = answer_net_gain = 0
                    acc_base_regex = acc_lesion_regex = 0.0
                
                grid_row = {
                    'mode': 'lesion',
                    'locality': args.locality,
                    'layer': layer_idx,
                    'gamma': args.gamma,
                    'n': n,
                    'acc_base_answer': acc_base_answer,
                    'acc_intv_answer': acc_lesion_answer,
                    'delta_answer': delta_answer,
                    'answer_wrong_to_right': answer_wrong_to_right,
                    'answer_right_to_wrong': answer_right_to_wrong,
                    'answer_net_gain': answer_net_gain,
                    'acc_base_reasoning': acc_base_reasoning,
                    'acc_intv_reasoning': acc_lesion_reasoning,
                    'delta_reasoning': delta_reasoning,
                    'acc_base_regex': acc_base_regex,
                    'acc_intv_regex': acc_lesion_regex,
                }
                grid_results.append(grid_row)
                
                print(f"    [GPT Answer]    Baseline: {acc_base_answer:.3f}, Lesion: {acc_lesion_answer:.3f}, Δ: {delta_answer:+.3f}")
                print(f"    [GPT Reasoning] Baseline: {acc_base_reasoning:.3f}, Lesion: {acc_lesion_reasoning:.3f}, Δ: {delta_reasoning:+.3f}")
                if delta_answer < -0.05:
                    print(f"    ✓ Necessity signal: Δ={delta_answer:+.3f} (lesion degrades performance)")
        
        elif mode == "rescue":
            # Rescue experiment (full causal triad)
            for layer_idx in target_layers:
                if layer_idx not in layer_to_dir:
                    print(f"  Warning: No direction found for layer {layer_idx}, skipping")
                    continue
                
                config_label = f"rescue_L{layer_idx}_G{args.gamma}_B{args.beta}_{args.locality}"
                print(f"\n  [{config_label}]")
                
                rescue_rows = []
                for idx in tqdm(range(num_examples), desc=f"    Generating"):
                    example = loader.get_example(idx)
                    
                    try:
                        result = run_rescue_triplet(
                            wrapper=wrapper,
                            handler=handler,
                            example=example,
                            layer_idx=layer_idx,
                            gamma=args.gamma,
                            beta=args.beta,
                            layer_to_dir=layer_to_dir,
                            args=args,
                        )
                        rescue_rows.append(result)
                    except Exception as e:
                        print(f"      Error on example {idx}: {e}")
                        continue
                
                # Save rescue CSV
                csv_path = runs_dir / f"rescue_{config_label}.csv.gz"
                df = pd.DataFrame(rescue_rows)
                with gzip.open(csv_path, 'wt', encoding='utf-8') as f:
                    df.to_csv(f, index=False, quoting=csv.QUOTE_ALL)
                
                print(f"    Saved {len(df)} examples to {csv_path.name}")
                
                # Compute rescue summary stats
                n = len(df)
                if n > 0:
                    acc_base_answer = df['baseline_answer_correct'].mean()
                    acc_lesion_answer = df['lesion_answer_correct'].mean()
                    acc_rescue_answer = df['rescue_answer_correct'].mean()
                    
                    delta_lesion_answer = acc_lesion_answer - acc_base_answer
                    recovery_answer = acc_rescue_answer - acc_lesion_answer
                    
                    acc_base_reasoning = df['baseline_reasoning_correct'].mean()
                    acc_lesion_reasoning = df['lesion_reasoning_correct'].mean()
                    acc_rescue_reasoning = df['rescue_reasoning_correct'].mean()
                    
                    full_recovery = int(((df['baseline_answer_correct'] == True) & (df['lesion_answer_correct'] == False) & (df['rescue_answer_correct'] == True)).sum())
                else:
                    acc_base_answer = acc_lesion_answer = acc_rescue_answer = 0.0
                    delta_lesion_answer = recovery_answer = 0.0
                    acc_base_reasoning = acc_lesion_reasoning = acc_rescue_reasoning = 0.0
                    full_recovery = 0
                
                rescue_row = {
                    'mode': 'rescue',
                    'locality': args.locality,
                    'layer': layer_idx,
                    'gamma': args.gamma,
                    'beta': args.beta,
                    'n': n,
                    'acc_base_answer': acc_base_answer,
                    'acc_lesion_answer': acc_lesion_answer,
                    'acc_rescue_answer': acc_rescue_answer,
                    'delta_lesion_answer': delta_lesion_answer,
                    'recovery_answer': recovery_answer,
                    'acc_base_reasoning': acc_base_reasoning,
                    'acc_lesion_reasoning': acc_lesion_reasoning,
                    'acc_rescue_reasoning': acc_rescue_reasoning,
                    'full_recovery': full_recovery,
                }
                rescue_results.append(rescue_row)
                
                print(f"    [GPT Answer]    Base: {acc_base_answer:.3f}, Lesion: {acc_lesion_answer:.3f}, Rescue: {acc_rescue_answer:.3f}")
                print(f"    Δ_lesion={delta_lesion_answer:+.3f}, Recovery={recovery_answer:+.3f}, Full recovery: {full_recovery}")
                if delta_lesion_answer < -0.05 and recovery_answer > 0.03:
                    print(f"    ✓ Causal evidence: lesion degrades, rescue recovers")
        
        elif mode == "random":
            # Random subspace control experiment
            for layer_idx in target_layers:
                if layer_idx not in random_dirs:
                    print(f"  Warning: No random direction for layer {layer_idx}, skipping")
                    continue
                
                for alpha in alpha_values:
                    config_label = f"random_L{layer_idx}_A{alpha}_{args.locality}"
                    print(f"\n  [{config_label}]")
                    
                    paired_rows = []
                    for idx in tqdm(range(num_examples), desc=f"    Generating"):
                        example = loader.get_example(idx)
                        
                        try:
                            result = run_paired_generation(
                                wrapper=wrapper,
                                handler=handler,
                                example=example,
                                layer_idx=layer_idx,
                                alpha=alpha,
                                layer_to_dir=random_dirs,  # Use random directions
                                args=args,
                                intervention_mode="add",
                            )
                            result['mode'] = 'random'
                            paired_rows.append(result)
                        except Exception as e:
                            print(f"      Error on example {idx}: {e}")
                            continue
                    
                    # Save paired CSV
                    csv_path = runs_dir / f"paired_{config_label}.csv.gz"
                    df = pd.DataFrame(paired_rows)
                    with gzip.open(csv_path, 'wt', encoding='utf-8') as f:
                        df.to_csv(f, index=False, quoting=csv.QUOTE_ALL)
                    
                    print(f"    Saved {len(df)} examples to {csv_path.name}")
                    
                    # Compute grid summary stats
                    n = len(df)
                    if n > 0:
                        acc_base_answer = df['baseline_answer_correct'].mean()
                        acc_intv_answer = df['intv_answer_correct'].mean()
                        delta_answer = acc_intv_answer - acc_base_answer
                        
                        acc_base_reasoning = df['baseline_reasoning_correct'].mean()
                        acc_intv_reasoning = df['intv_reasoning_correct'].mean()
                        delta_reasoning = acc_intv_reasoning - acc_base_reasoning
                        
                        answer_wrong_to_right = int(((df['baseline_answer_correct'] == False) & (df['intv_answer_correct'] == True)).sum())
                        answer_right_to_wrong = int(((df['baseline_answer_correct'] == True) & (df['intv_answer_correct'] == False)).sum())
                        answer_net_gain = answer_wrong_to_right - answer_right_to_wrong
                        
                        acc_base_regex = df['baseline_correct_regex'].mean()
                        acc_intv_regex = df['intv_correct_regex'].mean()
                    else:
                        acc_base_answer = acc_intv_answer = delta_answer = 0.0
                        acc_base_reasoning = acc_intv_reasoning = delta_reasoning = 0.0
                        answer_wrong_to_right = answer_right_to_wrong = answer_net_gain = 0
                        acc_base_regex = acc_intv_regex = 0.0
                    
                    grid_row = {
                        'mode': 'random',
                        'locality': args.locality,
                        'layer': layer_idx,
                        'alpha': alpha,
                        'n': n,
                        'acc_base_answer': acc_base_answer,
                        'acc_intv_answer': acc_intv_answer,
                        'delta_answer': delta_answer,
                        'answer_wrong_to_right': answer_wrong_to_right,
                        'answer_right_to_wrong': answer_right_to_wrong,
                        'answer_net_gain': answer_net_gain,
                        'acc_base_reasoning': acc_base_reasoning,
                        'acc_intv_reasoning': acc_intv_reasoning,
                        'delta_reasoning': delta_reasoning,
                        'acc_base_regex': acc_base_regex,
                        'acc_intv_regex': acc_intv_regex,
                    }
                    grid_results.append(grid_row)
                    
                    print(f"    [GPT Answer]    Baseline: {acc_base_answer:.3f}, Intv: {acc_intv_answer:.3f}, Δ: {delta_answer:+.3f}")
                    print(f"    [GPT Reasoning] Baseline: {acc_base_reasoning:.3f}, Intv: {acc_intv_reasoning:.3f}, Δ: {delta_reasoning:+.3f}")
                    print(f"    Random control: expect near-zero delta (Δ={delta_answer:+.3f})")
    
    # Save grid summary
    if grid_results:
        grid_df = pd.DataFrame(grid_results)
        grid_path = out_root / 'grid.csv'
        grid_df.to_csv(grid_path, index=False)
        print(f"\n{'='*60}")
        print(f"Saved grid summary ({len(grid_df)} configs) to {grid_path}")
    
    # Save rescue summary
    if rescue_results:
        rescue_df = pd.DataFrame(rescue_results)
        rescue_path = out_root / 'rescue_summary.csv'
        rescue_df.to_csv(rescue_path, index=False)
        print(f"Saved rescue summary ({len(rescue_df)} configs) to {rescue_path}")
    
    print(f"\n{'='*60}")
    print(f"Phase B Step 1 completed!")
    print(f"Locality: {args.locality} (inject during {args.locality} phase)")
    print(f"Outputs saved to: {out_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
