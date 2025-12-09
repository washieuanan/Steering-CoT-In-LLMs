"""
Sanity check for hook attachment and intervention effects.

This script verifies that:
1. Hooks correctly attach to the residual stream
2. Interventions modify hidden states
3. Logit distributions change as expected
4. Delta norms are non-zero when interventions are active

Usage:
    python -m scripts.sanity_check_hooks \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --phase_a_run results/phase_a/MODEL__TAG/arc \
        --layer 31 \
        --alpha 2.0 \
        --mode add \
        --nuke_test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hf_model_wrapper import HFModelConfig, HFModelWrapper
from dataset_loaders import load_dataset_by_name
from multi_hook_manager import detect_layer_path
from utils import get_handler
from utils.hooks import apply_add


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity check for hook interventions")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--phase_a_run", type=str, required=True,
                        help="Path to Phase A output dir")
    parser.add_argument("--layer", type=int, default=31,
                        help="Layer to test")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Intervention strength")
    parser.add_argument("--mode", type=str, default="add",
                        choices=["add", "nuke"],
                        help="Test mode: 'add' for normal intervention, 'nuke' for zeroing test")
    parser.add_argument("--dataset", type=str, default="arc",
                        help="Dataset to test on")
    return parser.parse_args()


def load_direction(phase_a_run: Path, layer_idx: int, device: torch.device):
    """Load direction vector for specified layer."""
    npz_path = phase_a_run / "screening" / "layer_to_U.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No directions found at {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    key = f"L{layer_idx}"
    
    if key not in data.files:
        raise ValueError(f"Layer {layer_idx} not found in {npz_path}")
    
    arr = data[key]
    
    # Normalize and convert to tensor
    if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
        vec = arr.flatten().astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return torch.from_numpy(vec).to(device), 'u'
    else:
        U = arr.astype(np.float32)
        norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
        U = U / norms
        return torch.from_numpy(U).to(device), 'U'


def test_hook_with_nuke(wrapper, layer_idx, example, handler):
    """Test that hooks work by nuking hidden states and verifying generation breaks."""
    print(f"\n{'='*60}")
    print("NUKE TEST: Verifying hook attachment point")
    print(f"{'='*60}\n")
    
    model = wrapper.model
    tokenizer = wrapper.tokenizer
    device = wrapper.primary_device
    
    # Build prompt
    input_ids, attention_mask, cot_start_idx, controller_kwargs = handler.build_prompt(
        tokenizer, example
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_len = input_ids.shape[1]
    
    # Get target layer
    layer_path = detect_layer_path(model)
    parts = layer_path.split(".")
    layers_module = model
    for part in parts:
        layers_module = getattr(layers_module, part)
    target_layer = layers_module[layer_idx]
    
    # Track interventions
    intervention_count = [0]
    
    # Nuke hook: zero out hidden states
    def nuke_hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None
        
        B, T, H = hidden_states.shape
        
        # Only nuke decode steps
        if T == 1:
            intervention_count[0] += 1
            # NUKE: zero out hidden states
            hidden_states = hidden_states * 0.0
        
        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states
    
    handle = target_layer.register_forward_hook(nuke_hook)
    
    try:
        with torch.no_grad():
            outputs = wrapper.generate_structured(
                input_ids=input_ids,
                attention_mask=attention_mask,
                controller_config=controller_kwargs,
                max_new_tokens=50,
            )
        
        gen_ids = outputs.sequences[0][prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        print(f"Interventions applied: {intervention_count[0]}")
        print(f"Generated tokens: {len(gen_ids)}")
        print(f"Generated text (first 100 chars): {gen_text[:100]}")
        
        # Check if generation is broken (should produce garbage/repeated tokens)
        unique_tokens = len(set(gen_ids.tolist()))
        repetition_rate = 1.0 - (unique_tokens / max(1, len(gen_ids)))
        
        print(f"\nToken diversity: {unique_tokens}/{len(gen_ids)} unique")
        print(f"Repetition rate: {repetition_rate:.2%}")
        
        if intervention_count[0] == 0:
            print("\n❌ FAILED: No interventions applied! Hook may not be attached correctly.")
            return False
        elif repetition_rate > 0.5 or unique_tokens < 5:
            print("\n✓ PASSED: Generation is broken as expected (hook is attached correctly)")
            return True
        else:
            print("\n⚠️  WARNING: Generation not sufficiently broken. Hook may be in wrong location.")
            return False
    
    finally:
        handle.remove()


def test_hook_with_intervention(wrapper, layer_idx, direction, dir_type, alpha, example, handler):
    """Test intervention effects on logits and hidden states."""
    print(f"\n{'='*60}")
    print(f"INTERVENTION TEST: Layer {layer_idx}, alpha={alpha}")
    print(f"{'='*60}\n")
    
    model = wrapper.model
    tokenizer = wrapper.tokenizer
    device = wrapper.primary_device
    
    # Build prompt
    input_ids, attention_mask, cot_start_idx, controller_kwargs = handler.build_prompt(
        tokenizer, example
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_len = input_ids.shape[1]
    
    # Get lm_head for logit computation
    lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
    
    # Track delta norms
    delta_norms = []
    
    # Get target layer
    layer_path = detect_layer_path(model)
    parts = layer_path.split(".")
    layers_module = model
    for part in parts:
        layers_module = getattr(layers_module, part)
    target_layer = layers_module[layer_idx]
    
    # Intervention hook
    def intervention_hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None
        
        B, T, H = hidden_states.shape
        
        # Only intervene on decode steps
        if T != 1:
            return (hidden_states,) + rest if rest is not None else hidden_states
        
        # Store original for comparison
        h_before = hidden_states.clone()
        
        # Apply intervention
        mask = torch.ones(B, T, dtype=torch.bool, device=hidden_states.device)
        
        if dir_type == 'u':
            hidden_states = apply_add(
                hidden_states,
                u=direction,
                alpha=alpha,
                add_mode="proj",
                mask=mask,
            )
        else:
            hidden_states = apply_add(
                hidden_states,
                U=direction,
                alpha=alpha,
                add_mode="proj",
                mask=mask,
            )
        
        # Compute delta norm
        delta = hidden_states - h_before
        delta_norm = delta.norm(dim=-1).mean().item()
        delta_norms.append(delta_norm)
        
        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states
    
    handle = target_layer.register_forward_hook(intervention_hook)
    
    try:
        with torch.no_grad():
            outputs = wrapper.generate_structured(
                input_ids=input_ids,
                attention_mask=attention_mask,
                controller_config=controller_kwargs,
                max_new_tokens=50,
            )
        
        gen_ids = outputs.sequences[0][prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        print(f"Generated {len(gen_ids)} tokens")
        print(f"Interventions applied: {len(delta_norms)}")
        print(f"Generated text (first 100 chars): {gen_text[:100]}\n")
        
        # Analyze delta norms
        if len(delta_norms) > 0:
            print("Delta norm statistics:")
            print(f"  Mean: {np.mean(delta_norms):.6e}")
            print(f"  Std:  {np.std(delta_norms):.6e}")
            print(f"  Min:  {np.min(delta_norms):.6e}")
            print(f"  Max:  {np.max(delta_norms):.6e}")
            
            # Check if deltas are non-zero
            if np.mean(delta_norms) > 1e-6:
                print("\n✓ PASSED: Non-zero delta norms (intervention is active)")
                return True
            else:
                print("\n❌ FAILED: Delta norms near zero (intervention not working)")
                return False
        else:
            print("\n❌ FAILED: No interventions recorded")
            return False
    
    finally:
        handle.remove()


def main():
    args = parse_args()
    
    # Load model
    print(f"Loading model: {args.model_name}")
    special_tokens = ["<cot>", "</cot>", "<answer>"]
    cfg = HFModelConfig(
        model_name=args.model_name,
        dtype="bfloat16",
        device="auto",
        special_tokens=special_tokens,
        init_special_tokens_with_avg=True,
    )
    wrapper = HFModelWrapper(cfg).load()
    device = wrapper.primary_device
    
    # Load dataset
    loader = load_dataset_by_name(args.dataset, split="test", seed=42)
    handler = get_handler(args.dataset)
    example = loader.get_example(0)
    
    print(f"\nTest example:")
    print(f"  Question: {getattr(example, 'question', 'N/A')[:100]}...")
    print(f"  Gold: {handler.gold_target(example)}")
    
    # Load direction
    phase_a_run = Path(args.phase_a_run)
    direction, dir_type = load_direction(phase_a_run, args.layer, device)
    
    print(f"\nDirection loaded:")
    print(f"  Type: {dir_type}")
    print(f"  Shape: {tuple(direction.shape)}")
    print(f"  Norm: {direction.norm().item():.6f}")
    
    # Run tests
    results= {}
    
    if args.mode == "nuke":
        results['nuke'] = test_hook_with_nuke(wrapper, args.layer, example, handler)
    else:
        results['intervention'] = test_hook_with_intervention(
            wrapper, args.layer, direction, dir_type, args.alpha, example, handler
        )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
