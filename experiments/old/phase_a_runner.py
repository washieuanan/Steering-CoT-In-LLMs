from __future__ import annotations

"""
Phase A runner: collect pooled activations across layers and screen for
candidate reasoning subspaces.

Workflow per dataset:
1) Load model with special CoT tokens
2) Iterate N examples → build prompt with `<cot>` and no answer
3) Pool hidden states over generated tokens (mask starts at prompt length)
4) Generate model output and label correctness via dataset loader
5) Save pooled activations X, labels y, and metadata to NPZ
6) Run OfflineLayerScreener to compute metrics and extract directions; save

This script is intentionally conservative: single-GPU, batch size = 1,
deterministic seeds where applicable. It saves artifacts so analysis can be
done offline without re-running models.
"""

import argparse
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from hf_model_wrapper import HFModelConfig, HFModelWrapper
from dataset_loaders import load_dataset_by_name, BaseDatasetLoader
from multi_hook_manager import (
    MultiLayerPooler,
    OfflineLayerScreener,
    detect_layer_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase A: activation pooling and screening")

    # Model/config
    parser.add_argument("--model_name", type=str, required=True, help="HF repo id or local path")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "fp16", "half", "bfloat16", "bf16", "float32", "fp32"])  # noqa: E501
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--special_tokens", type=str, default="<cot>,</cot>,<answer>")

    # Datasets
    parser.add_argument("--datasets", type=str, required=True, help="Comma-separated: arc,gsm8k,mmlu_pro")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_examples", type=int, default=100, help="Max examples per dataset")
    parser.add_argument("--few_shot", type=int, default=0, help="Optional # few-shot examples to prepend")

    # Layers
    parser.add_argument("--layer_start", type=int, default=8, help="Inclusive start layer index")
    parser.add_argument("--layer_end", type=int, default=None, help="Exclusive end layer index (default: all)")
    parser.add_argument("--layers", type=str, default=None, help="Explicit comma-separated layer indices (overrides start/end)")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Output
    parser.add_argument("--output_dir", type=str, default="results/phase_a")
    parser.add_argument("--tag", type=str, default=None, help="Optional run tag for folder name")

    # Screening
    parser.add_argument("--probe_C", type=float, default=1.0)
    parser.add_argument("--probe_max_iter", type=int, default=1000)
    parser.add_argument("--topk_layers", type=int, default=8, help="Number of layers to shortlist")

    return parser.parse_args()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_layers(model, args: argparse.Namespace) -> List[int]:
    # Determine number of layers from model config
    try:
        num_layers = int(getattr(model.config, "num_hidden_layers"))
    except Exception:
        # Fallback: try common paths
        path = detect_layer_path(model)
        module = model
        for part in path.split("."):
            module = getattr(module, part)
        num_layers = len(module)

    if args.layers:
        return [int(x) for x in args.layers.split(",")]

    layer_end = num_layers if args.layer_end is None else min(args.layer_end, num_layers)
    return list(range(args.layer_start, layer_end))


class StopOnFinalAnswer(StoppingCriteria):
    """Stop generation when we see 'Final answer: X' on its own line."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.buffer = ""
        self.pattern = re.compile(r"(?im)^\s*Final answer:\s*[ABCD]\s*$")
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Decode the last generated token and add to buffer
        self.buffer += self.tokenizer.decode(input_ids[0, -1:], skip_special_tokens=False)
        
        # Check if the last complete line matches our pattern
        last_line = self.buffer.splitlines()[-1] if self.buffer else ""
        return bool(self.pattern.search(last_line))


def build_chat_input_with_cot(
    tokenizer,
    prompt_text: str,
    model_device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Build chat-formatted input with <cot> token ID forced to be the last token.
    
    Returns:
        Tuple of (input_ids, attention_mask, cot_id) all on model_device
    """
    # Remove <cot> from the end of prompt_text since we'll append it as a token ID
    prompt_text = prompt_text.rstrip()
    if prompt_text.endswith("<cot>"):
        prompt_text = prompt_text[:-5].rstrip()
    
    messages = [
        {"role": "system", "content": "Follow the format exactly. Do not add extra text after the final line."},
        {"role": "user", "content": prompt_text}
    ]
    
    try:
        # Try chat template (for Llama 3.1, Qwen2.5, Mistral)
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model_device)
    except Exception:
        # Fallback for models without chat template
        input_ids = tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"].to(model_device)
    
    # Force-append <cot> token ID to ensure it's the last token
    cot_id = tokenizer.convert_tokens_to_ids("<cot>")
    if cot_id is not None and cot_id != tokenizer.unk_token_id:
        # Check if <cot> is already the last token
        if input_ids[0, -1].item() != cot_id:
            # Append the <cot> token ID explicitly
            cot_tail = torch.tensor([[cot_id]], dtype=torch.long, device=input_ids.device)
            input_ids = torch.cat([input_ids, cot_tail], dim=1)
    else:
        cot_id = None  # <cot> not recognized as a special token
    
    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Create attention mask (no padding, all 1s)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model_device)
    
    return input_ids, attention_mask, cot_id


def collect_for_dataset(
    dataset_name: str,
    loader: BaseDatasetLoader,
    wrapper: HFModelWrapper,
    layers: List[int],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    tokenizer = wrapper.tokenizer
    model = wrapper.model

    # Hidden size
    hidden_size = int(getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0)))
    if hidden_size <= 0:
        raise RuntimeError("Could not infer hidden size from model.config")

    layer_path = detect_layer_path(model)
    pooler = MultiLayerPooler(model=model, layers=layers, hidden_size=hidden_size, layer_path=layer_path)

    # For labels, ids, and prompt lengths
    y: List[int] = []
    example_ids: List[str] = []
    prompt_lens: List[int] = []

    # Pre-select few-shot examples if requested
    fs_examples = loader.get_random_examples(args.few_shot) if args.few_shot > 0 else None

    X_list: List[np.ndarray] = []

    total = min(args.num_examples, len(loader))
    for idx in range(total):
        print(f"\n--- Example {idx + 1}/{total} (ID: {dataset_name}) ---")
        pooler.reset_buffers()

        example = loader.get_example(idx)
        prompt_text = loader.format_prompt(example, include_cot=True, few_shot_examples=fs_examples)

        # Build chat input with <cot> token ID forced at the end
        input_ids, attention_mask, cot_id = build_chat_input_with_cot(
            tokenizer, prompt_text, model.device
        )
        
        prompt_len = input_ids.shape[1]
        pooler.set_cot_start_idx(prompt_len)
        prompt_lens.append(prompt_len)
        print(f"Prompt length: {prompt_len} tokens")
        
        # Debug: verify <cot> is the last token
        if idx == 0:  # Only print once per dataset
            last_10_ids = input_ids[0, -10:].tolist()
            last_10_tokens = tokenizer.convert_ids_to_tokens(last_10_ids)
            print(f"DEBUG last-10 input ids: {last_10_ids}")
            print(f"DEBUG last-10 input toks: {last_10_tokens}")
            print(f"DEBUG <cot> token id: {cot_id}")
            if cot_id is not None:
                print(f"DEBUG last token is <cot>: {input_ids[0, -1].item() == cot_id}")

        # Set up stopping criteria for MCQ (stop on "Final answer: X")
        stopping_criteria = StoppingCriteriaList([StopOnFinalAnswer(tokenizer)])

        # Generate with proper inputs and stopping
        print("Generating...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature if args.temperature > 0.0 else None,
                do_sample=(args.temperature > 0.0),
                stopping_criteria=stopping_criteria,
                return_dict_in_generate=True,
            )
        
        # Extract only the generated tokens (not the prompt)
        seq = outputs.sequences[0]
        gen_ids = seq[prompt_len:]
        
        # Debug view with special tokens visible
        debug_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        print(f"\nGenerated tokens (with specials):\n{debug_text}")
        
        # Normal scoring without special tokens
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Pooled activations: dict[layer -> (D,)]
        pooled = pooler.pooled()
        X_vecs = [pooled[layer] for layer in layers]
        X_mat = np.stack(X_vecs, axis=0)  # [L, D]
        X_list.append(X_mat)

        # Label correctness
        correct = 1 if loader.check_answer(example, text) else 0
        y.append(correct)
        example_ids.append(example.example_id)

        # Print output and correctness
        print(f"\nModel Output (without specials):\n{text}")
        print(f"\nCorrect: {'✓ YES' if correct else '✗ NO'}")
        print("-" * 80)

    # [N, L, D]
    X = np.stack(X_list, axis=0) if len(X_list) > 0 else np.zeros((0, len(layers), hidden_size), dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int64)

    pooler.close()
    return X, y_arr, example_ids, prompt_lens


def run_screening(
    X: np.ndarray,
    y: np.ndarray,
    layers: List[int],
    C: float,
    max_iter: int,
    topk_layers: int,
) -> Dict:
    screener = OfflineLayerScreener(X=X, y=y, layer_indices=layers)
    results = screener.screen_all_layers(C=C, max_iter=max_iter)

    ranked = screener.rank_layers(results)
    shortlist = [layer for layer, _ in ranked[:topk_layers]]

    # Also extract one dense direction per shortlisted layer
    layer_to_U: Dict[int, np.ndarray] = {}
    for i, layer in enumerate(layers):
        if layer not in shortlist:
            continue
        probe = results[layer]["probe_model"]
        U = screener.extract_top_directions(i, probe, k=1, method="dense_normalized")  # [D,1]
        layer_to_U[layer] = U.astype(np.float32)

    # Convert results to lightweight JSON-serializable metrics
    metrics = {
        int(layer): {
            "delta_mu_norm": float(res["delta_mu_norm"]),
            "val_auc": float(res["val_auc"]),
            "val_acc": float(res["val_acc"]),
        }
        for layer, res in results.items()
    }

    return {
        "ranked_layers": ranked,
        "shortlist": shortlist,
        "metrics": metrics,
        "layer_to_U": layer_to_U,
    }


def main() -> None:
    args = parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    special_tokens = [s.strip() for s in args.special_tokens.split(",") if s.strip()]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Model
    cfg = HFModelConfig(
        model_name=args.model_name,
        revision=args.revision,
        dtype=args.dtype,
        device=args.device,
        special_tokens=special_tokens,
        init_special_tokens_with_avg=True,
    )
    wrapper = HFModelWrapper(cfg).load()

    # Layers to monitor
    layers = build_layers(wrapper.model, args)

    # Output root
    tag = args.tag or timestamp
    root = Path(args.output_dir) / (Path(args.model_name).name.replace("/", "-") + f"__{tag}")
    ensure_dir(root)

    # Save run config
    run_cfg = {
        "model_name": args.model_name,
        "revision": args.revision,
        "dtype": args.dtype,
        "device": args.device,
        "special_tokens": special_tokens,
        "datasets": args.datasets,
        "split": args.split,
        "num_examples": args.num_examples,
        "few_shot": args.few_shot,
        "layers": layers,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "probe_C": args.probe_C,
        "probe_max_iter": args.probe_max_iter,
        "topk_layers": args.topk_layers,
    }
    (root / "run_config.json").write_text(json.dumps(run_cfg, indent=2))

    # Iterate datasets
    for ds_name in [s.strip() for s in args.datasets.split(",") if s.strip()]:
        print(f"\n=== Dataset: {ds_name} ===")
        ds_dir = root / ds_name
        ensure_dir(ds_dir)

        # Load dataset
        loader = load_dataset_by_name(ds_name, split=args.split)

        # Collect activations
        X, y, example_ids, prompt_lens = collect_for_dataset(ds_name, loader, wrapper, layers, args)
        
        # Print label histogram
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"\nCollected X shape: {X.shape}")
        print(f"Label histogram: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
        print(f"Positives: {int(y.sum())}/{len(y)}")
        
        # Check for degenerate labels (only one class)
        if len(unique_labels) < 2:
            print(f"\n⚠️  WARNING: Only one class present in labels: {unique_labels}")
            print("Skipping screening for this dataset (cannot train probe with single class)")
            
            # Log some debug samples
            print("\nDebug samples (first 5 examples):")
            for i in range(min(5, len(example_ids))):
                example = loader.get_example(i)
                print(f"  [{i}] ID: {example_ids[i]}")
                print(f"      Gold: {example.correct_answer}")
                print(f"      Label: {y[i]}")

        # Save pooled activations
        pooled_path = ds_dir / "pooled.npz"
        np.savez_compressed(
            pooled_path,
            X=X,
            y=y,
            layers=np.asarray(layers, dtype=np.int32),
            example_ids=np.asarray(example_ids, dtype=object),
            special_tokens=np.asarray(special_tokens, dtype=object),
            prompt_lengths=np.asarray(prompt_lens, dtype=np.int32),
        )

        # Screening and ranking (skip if degenerate labels)
        screen_dir = ds_dir / "screening"
        ensure_dir(screen_dir)
        
        if len(unique_labels) < 2:
            # Save degenerate label marker
            (screen_dir / "metrics.json").write_text(json.dumps({
                "degenerate_labels": True,
                "note": "only one class present; screening skipped",
                "unique_labels": unique_labels.tolist(),
                "label_counts": counts.tolist(),
            }, indent=2))
            print(f"Saved degenerate label marker for {ds_name}")
        else:
            # Normal screening
            screen = run_screening(
                X=X,
                y=y,
                layers=layers,
                C=args.probe_C,
                max_iter=args.probe_max_iter,
                topk_layers=args.topk_layers,
            )

            # Save metrics and directions
            (screen_dir / "metrics.json").write_text(json.dumps({
                "ranked_layers": [(int(l), float(s)) for l, s in screen["ranked_layers"]],
                "shortlist": [int(l) for l in screen["shortlist"]],
                "metrics": screen["metrics"],
            }, indent=2))

            # Save U matrices in one NPZ
            if len(screen["layer_to_U"]) > 0:
                np.savez_compressed(
                    screen_dir / "layer_to_U.npz",
                    **{f"L{layer}": U for layer, U in screen["layer_to_U"].items()},
                )

    print("\nPhase A completed. Artifacts saved under:", str(root))


if __name__ == "__main__":
    main()
