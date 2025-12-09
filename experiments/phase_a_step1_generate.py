from __future__ import annotations

"""
Phase A Step 1: Generation-only runner

This script runs model generation on reasoning datasets and saves:
1. Pooled activations (pooled.npz)
2. Raw generation outputs with metadata (generations.csv)
3. Run configuration (run_config.json)

No parsing or scoring is done in this step - that happens in Step 2.
"""

import argparse
import csv
import gzip
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import StoppingCriteriaList, LogitsProcessorList

from hf_model_wrapper import (
    HFModelConfig,
    HFModelWrapper,
    decode_both,
)
from dataset_loaders import load_dataset_by_name, DatasetExample
from multi_hook_manager import MultiLayerPooler, detect_layer_path
from utils import get_handler
from utils.parse_answers import get_answer_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase A Step 1: Generation and activation collection"
    )

    # Model config
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "fp16", "half", "bfloat16", "bf16", "float32", "fp32"],
    )
    parser.add_argument("--device", type=str, default="auto")

    # Datasets
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated: arc,gsm8k,mmlu_pro",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--few_shot", type=int, default=0)

    # Layers
    parser.add_argument("--layer_start", type=int, default=8)
    parser.add_argument("--layer_end", type=int, default=None)
    parser.add_argument("--layers", type=str, default=None)

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--stop_on_final_answer", action="store_true", default=True)
    parser.add_argument("--min_cot_tokens", type=int, default=24, help="Minimum CoT tokens before allowing answer")
    parser.add_argument(
        "--task_type_override",
        type=str,
        default="auto",
        choices=["auto", "mcq", "numeric"],
        help="Override task type detection (default: auto-detect from dataset name)",
    )

    # Output
    parser.add_argument("--output_dir", type=str, default="results/phase_a")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--csv_gzip", action="store_true")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_layers(model, args: argparse.Namespace) -> List[int]:
    """Determine which layers to monitor."""
    try:
        num_layers = int(getattr(model.config, "num_hidden_layers"))
    except Exception:
        path = detect_layer_path(model)
        module = model
        for part in path.split("."):
            module = getattr(module, part)
        num_layers = len(module)

    if args.layers:
        return [int(x) for x in args.layers.split(",")]

    layer_end = (
        num_layers if args.layer_end is None else min(args.layer_end, num_layers)
    )
    return list(range(args.layer_start, layer_end))


def collect_generations(
    dataset_name: str,
    loader,
    wrapper: HFModelWrapper,
    layers: List[int],
    args: argparse.Namespace,
    run_id: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate outputs and collect activations for a dataset.
    
    Returns:
        Tuple of (pooled_activations, generations_dataframe)
    """
    tokenizer = wrapper.tokenizer
    model = wrapper.model

    # Get hidden size
    hidden_size = int(
        getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0))
    )
    if hidden_size <= 0:
        raise RuntimeError("Could not infer hidden size from model.config")

    # Get handler for this dataset
    handler = get_handler(dataset_name)
    
    # Storage
    X_list: List[np.ndarray] = []
    rows: List[Dict] = []

    total = min(args.num_examples, len(loader))

    for idx in range(total):
        print(f"\n--- Example {idx + 1}/{total} ---")

        # Get example
        example = loader.get_example(idx)
        
        # Use handler to build prompt and get controller config
        input_ids, attention_mask, cot_start_idx, controller_kwargs = handler.build_prompt(
            tokenizer, example
        )
        
        # Move tensors to correct device
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        
        # Override min_cot_tokens if specified
        if args.min_cot_tokens != 24:  # 24 is handler default
            controller_kwargs['min_cot_tokens'] = args.min_cot_tokens
        
        task_type = handler.task_type(example)
        print(f"  Task type: {task_type}")
        if task_type == "mcq" and 'allowed_letters' in controller_kwargs:
            print(f"  Allowed letters: {controller_kwargs['allowed_letters']}")
        elif task_type == "labelset":
            labels = getattr(example, 'labels', None) or getattr(example, 'choices', [])
            if labels:
                print(f"  Labels: {labels}")

        prompt_len = input_ids.shape[1]
        n_input_tokens = prompt_len

        # Generate with structured format enforcement
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = wrapper.generate_structured(
                input_ids=input_ids,
                attention_mask=attention_mask,
                controller_config=controller_kwargs,
                max_new_tokens=args.max_new_tokens,
            )

        end_time = time.perf_counter()
        total_time_ms = int((end_time - start_time) * 1000)

        # Extract generated tokens only
        seq = outputs.sequences[0]
        gen_ids = seq[prompt_len:]
        n_generated_tokens = len(gen_ids)

        # Decode with and without special tokens using unified helper
        gen_text_raw, gen_text = decode_both(tokenizer, gen_ids)

        # Use handler to parse prediction
        extracted_pred, parse_status = handler.parse_pred(gen_text_raw, example)
        
        answer_found = (extracted_pred is not None)
        
        # Check if stopped on regex (simple heuristic)
        stopped_on_regex = int("Final answer:" in gen_text.lower() or "<answer>" in gen_text_raw.lower())

        # Post-generation pooling: run full forward pass to get hidden states
        # Build full sequence (prompt + generated tokens)
        full_ids = torch.cat([input_ids, gen_ids.unsqueeze(0)], dim=1)
        full_attention_mask = torch.ones_like(full_ids, dtype=torch.long, device=model.device)
        
        # Single forward pass with hidden states extraction (no cache for accurate layer-wise states)
        with torch.no_grad():
            forward_outputs = model(
                input_ids=full_ids,
                attention_mask=full_attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True
            )
        
        # Build CoT mask using token-based search for structured answer markers
        # For new format: mask from prompt_len to first "MCQ ANSWER:" or "NUM ANSWER:" or "CLS ANSWER:"
        full_ids_1d = full_ids[0]  # [L]
        
        # Start position: end of prompt (beginning of generation)
        start_pos = prompt_len - 1
        
        # End position: find first occurrence of answer phrase tokens
        # Search for all three answer phrase types
        mcq_phrase_ids = tokenizer.encode("MCQ ANSWER:", add_special_tokens=False)
        num_phrase_ids = tokenizer.encode("NUM ANSWER:", add_special_tokens=False)
        cls_phrase_ids = tokenizer.encode("CLS ANSWER:", add_special_tokens=False)
        
        end_pos = len(full_ids_1d)  # Default to end of sequence
        
        # Search for MCQ ANSWER: phrase
        for i in range(prompt_len, len(full_ids_1d) - len(mcq_phrase_ids) + 1):
            if torch.all(full_ids_1d[i:i+len(mcq_phrase_ids)] == torch.tensor(mcq_phrase_ids, device=full_ids.device)):
                end_pos = i
                break
        
        # Search for NUM ANSWER: phrase (if MCQ not found or found later)
        for i in range(prompt_len, len(full_ids_1d) - len(num_phrase_ids) + 1):
            if torch.all(full_ids_1d[i:i+len(num_phrase_ids)] == torch.tensor(num_phrase_ids, device=full_ids.device)):
                if i < end_pos:
                    end_pos = i
                break
        
        # Search for CLS ANSWER: phrase (if others not found or found later)
        for i in range(prompt_len, len(full_ids_1d) - len(cls_phrase_ids) + 1):
            if torch.all(full_ids_1d[i:i+len(cls_phrase_ids)] == torch.tensor(cls_phrase_ids, device=full_ids.device)):
                if i < end_pos:
                    end_pos = i
                break
        
        # Build mask: from (start_pos + 1) to end_pos (exclusive)
        # This masks only the CoT reasoning, excluding prompt and answer sections
        L_full = full_ids.shape[1]
        mask = (torch.arange(L_full, device=full_ids.device) > start_pos) & \
               (torch.arange(L_full, device=full_ids.device) < end_pos)
        
        masked_count = int(mask.sum().item())
        
        # Extract and pool for each target layer
        pooled = {}
        
        for layer_idx in layers:
            # hidden_states[0] is embeddings, layers[i] maps to hidden_states[i+1]
            hidden = forward_outputs.hidden_states[layer_idx + 1]  # [B, L_full, D]
            if masked_count > 0:
                pooled_vec = hidden[0, mask].mean(dim=0)  # [D]
            else:
                pooled_vec = torch.zeros(hidden.size(-1), device=hidden.device)
            pooled[layer_idx] = pooled_vec.float().detach().cpu().numpy()
        
        # Build pooled activation matrix
        X_vecs = [pooled[layer] for layer in layers]
        X_mat = np.stack(X_vecs, axis=0)  # [L, D]
        X_list.append(X_mat)
        
        # masked_token_count now reflects actual masked tokens
        masked_token_count = masked_count

        # Build CSV row with handler metadata
        row = {
            "run_id": run_id,
            "dataset": dataset_name,
            "split": args.split,
            "example_id": idx,  # 0-indexed, aligned with X array
            "model_name": args.model_name,
            "seed": args.seed,
            "task_type": task_type,
            "prompt_len": prompt_len,
            "max_new_tokens": args.max_new_tokens,
            "gold_label": example.correct_answer,
            "gen_text_raw": gen_text_raw,
            "gen_text": gen_text,
            "extracted_pred": extracted_pred if extracted_pred else "",
            "parse_status": parse_status,
            "stopped_on_regex": stopped_on_regex,
            "n_input_tokens": n_input_tokens,
            "n_generated_tokens": n_generated_tokens,
            "total_time_ms": total_time_ms,
            "masked_token_count": masked_token_count,
        }
        
        # Add mode-specific metadata
        if task_type == "mcq":
            row["choices_count"] = len(getattr(example, "choices", []))
        elif task_type == "labelset":
            labels = getattr(example, "labels", None) or getattr(example, "choices", [])
            row["labels_json"] = json.dumps(labels) if labels else "[]"
        
        rows.append(row)

        # Debug logging for answer block structure
        has_answer_open = "<answer>" in gen_text_raw
        has_answer_close = "</answer>" in gen_text_raw
        print(f"Generated {n_generated_tokens} tokens in {total_time_ms}ms")
        print(f"Masked token count: {masked_token_count}")
        if masked_token_count == 0:
            print("⚠️  WARNING: masked_token_count is 0!")
        
        # Check answer block structure
        print(f"Answer block: <answer>={has_answer_open}, </answer>={has_answer_close}")
        if has_answer_open:
            # Extract content inside <answer>...</answer>
            import re
            answer_match = re.search(r'<answer>(.*?)(?:</answer>|$)', gen_text_raw, re.DOTALL | re.IGNORECASE)
            if answer_match:
                answer_content = answer_match.group(1)
                # Show first 120 chars inside answer block
                preview = answer_content[:120] if len(answer_content) > 120 else answer_content
                print(f"Inside <answer> (first 120 chars): {repr(preview)}")
        
        # Parser results
        print(f"Parser: status={parse_status}, extracted={repr(extracted_pred)}, gold={repr(example.correct_answer)}")
        
        if answer_found:
            print(f"✓ Answer extracted successfully")
            # Show last 120 chars of raw output with repr to see newlines
            tail = gen_text_raw[-120:] if len(gen_text_raw) > 120 else gen_text_raw
            print(f"   Tail (last 120 chars): {repr(tail)}")
        else:
            print(f"⚠️  No answer found! Status: {parse_status}")
            # Show context and repr to debug newline issues
            context = get_answer_context(gen_text_raw, max_lines=3)
            print(f"   Context around answer area:")
            print(f"   <<<SNIP")
            for line in context.split('\n'):
                print(f"   {line}")
            print(f"   SNIP>>>")
            # Show last 120 chars with repr
            tail = gen_text_raw[-120:] if len(gen_text_raw) > 120 else gen_text_raw
            print(f"   Tail (last 120 chars): {repr(tail)}")
        
        print(f"Output preview: {gen_text[:200]}...")

    # Build X array [N, L, D]
    X = (
        np.stack(X_list, axis=0)
        if len(X_list) > 0
        else np.zeros((0, len(layers), hidden_size), dtype=np.float32)
    )

    # Build DataFrame
    df = pd.DataFrame(rows)

    return X, df


def main() -> None:
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Prepare special tokens
    special_tokens = ["<cot>", "</cot>", "<answer>"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.tag or timestamp

    # Load model
    print(f"Loading model: {args.model_name}")
    cfg = HFModelConfig(
        model_name=args.model_name,
        revision=args.revision,
        dtype=args.dtype,
        device=args.device,
        special_tokens=special_tokens,
        init_special_tokens_with_avg=True,
    )
    wrapper = HFModelWrapper(cfg).load()

    # Determine layers
    layers = build_layers(wrapper.model, args)
    print(f"Monitoring layers: {layers}")

    # Output directory
    tag = args.tag or timestamp
    model_short_name = Path(args.model_name).name.replace("/", "-")
    root = Path(args.output_dir) / f"{model_short_name}__{tag}"
    ensure_dir(root)

    # Get model metadata
    hidden_size = int(
        getattr(
            wrapper.model.config,
            "hidden_size",
            getattr(wrapper.model.config, "n_embd", 0),
        )
    )
    layer_path = detect_layer_path(wrapper.model)
    tokenizer_class = type(wrapper.tokenizer).__name__

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
        "layer_path": layer_path,
        "hidden_size": hidden_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "stop_on_final_answer": args.stop_on_final_answer,
        "seed": args.seed,
        "tokenizer_class": tokenizer_class,
        "run_id": run_id,
        "timestamp": timestamp,
    }
    (root / "run_config.json").write_text(json.dumps(run_cfg, indent=2))
    print(f"Saved run config to {root / 'run_config.json'}")

    # Process each dataset
    for ds_name in [s.strip() for s in args.datasets.split(",") if s.strip()]:
        print(f"\n{'='*60}\nDataset: {ds_name}\n{'='*60}")

        ds_dir = root / ds_name
        ensure_dir(ds_dir)

        # Load dataset
        loader = load_dataset_by_name(ds_name, split=args.split, seed=args.seed)
        print(f"Loaded {len(loader)} examples from {ds_name}")

        # Collect generations and activations
        X, df = collect_generations(
            ds_name, loader, wrapper, layers, args, run_id
        )

        print(f"\nCollected X shape: {X.shape}")
        print(f"Generated {len(df)} examples")

        # Save pooled activations
        pooled_path = ds_dir / "pooled.npz"
        special_token_ids = wrapper.get_special_token_ids()
        np.savez_compressed(
            pooled_path,
            X=X,
            layers=np.asarray(layers, dtype=np.int32),
            example_ids=np.arange(len(df), dtype=np.int32),
            special_tokens=np.asarray(
                [special_token_ids.get(tok) for tok in special_tokens],
                dtype=np.int32,
            ),
            model_name=args.model_name,
            layer_path=layer_path,
            hidden_size=hidden_size,
        )
        print(f"Saved pooled activations to {pooled_path}")

        # Save generations CSV
        csv_path = ds_dir / "generations.csv"
        if args.csv_gzip:
            csv_path = ds_dir / "generations.csv.gz"
            with gzip.open(csv_path, "wt", encoding="utf-8") as f:
                df.to_csv(f, index=False, quoting=csv.QUOTE_ALL)
        else:
            df.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL)

        print(f"Saved generations to {csv_path}")

    print(f"\n{'='*60}")
    print(f"Phase A Step 1 completed!")
    print(f"Artifacts saved under: {root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
