from __future__ import annotations

"""
Phase A Step 2: Offline parsing and screening

This script reads Step 1 outputs and performs:
1. Strict parsing/labeling of model outputs
2. Offline screening with L1 logistic probes per layer
3. Ranking and direction extraction

No model forward pass in this step - purely offline analysis.
"""

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple



import numpy as np
import pandas as pd

from multi_hook_manager import OfflineLayerScreener
from utils import get_handler
from dataset_loaders import load_dataset_by_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase A Step 2: Offline parsing and screening"
    )

    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to Step 1 output directory (e.g., results/phase_a/model__tag/dataset/)",
    )

    # Screening params
    parser.add_argument("--probe_C", type=float, default=1.0)
    parser.add_argument("--probe_max_iter", type=int, default=1000)
    parser.add_argument("--topk_layers", type=int, default=8)
    parser.add_argument(
        "--direction_method",
        type=str,
        default="dense_normalized",
        choices=["dense_normalized", "top_k_dense", "top_k_sparse"],
    )
    parser.add_argument("--save_suffix", type=str, default=None)
    parser.add_argument(
        "--numeric_tol",
        type=float,
        default=1e-6,
        help="Tolerance for numeric answer comparison (default: 1e-6)",
    )
    parser.add_argument(
        "--use_gpt_labels",
        action="store_true",
        help=(
            "If set, use reasoning_correct_judge from "
            "generations_with_judgments.csv as labels y instead of "
            "handler-based answer correctness. Run "
            "scripts/annotate_reasoning_with_gpt.py first."
        ),
    )

    return parser.parse_args()



def parse_and_label(
    df: pd.DataFrame, 
    handler, 
    dataset_name: str = "unknown", 
    split: str = "test", 
    numeric_tol: float = 1e-6
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Parse model outputs and create binary labels using handler.
    
    Args:
        df: DataFrame from Step 1 with gen_text_raw and gen_text columns
        handler: Dataset handler instance
        dataset_name: Name of dataset (for loading examples)
        split: Dataset split (for loading examples)
        numeric_tol: Tolerance for numeric answer comparison (default: 1e-6)
    
    Returns:
        Tuple of (updated_dataframe, labels_array)
    """
    # Load dataset to get examples for parsing
    loader = load_dataset_by_name(dataset_name, split=split)
    
    extracted_preds = []
    parse_statuses = []
    labels = []

    for idx, row in df.iterrows():
        # Get raw text for parsing
        gen_text_raw = row.get("gen_text_raw", "")
        
        # Get example for context
        example_id = int(row.get("example_id", idx))
        example = loader.get_example(example_id)
        
        # Use handler to parse prediction
        extracted, status = handler.parse_pred(gen_text_raw, example)
        
        extracted_preds.append(extracted if extracted else "")
        parse_statuses.append(status)

        # Get gold target from handler
        gold = handler.gold_target(example)
        
        # Use handler to compare
        if extracted is None:
            label = 0  # No match = incorrect
        else:
            # Special handling for numeric tolerance
            if handler.name == "gsm8k":
                label = int(handler.compare(extracted, gold, tolerance=numeric_tol))
            else:
                label = int(handler.compare(extracted, gold))
        
        labels.append(label)

    # Update DataFrame
    df = df.copy()
    df["extracted_pred"] = extracted_preds
    df["parse_status"] = parse_statuses
    df["y"] = labels

    return df, np.array(labels, dtype=np.int64)


def run_screening(
    X: np.ndarray,
    y: np.ndarray,
    layers: list,
    C: float,
    max_iter: int,
    topk_layers: int,
    direction_method: str,
    n_splits: int = 3,
    class_weight: Optional[str] = "balanced",
) -> Dict:
    """
    Run offline screening on pooled activations with stratified k-fold CV.
    
    Returns:
        Dictionary with ranked_layers, shortlist, metrics, and layer_to_U
    """
    screener = OfflineLayerScreener(X=X, y=y, layer_indices=layers)
    results = screener.screen_all_layers(
        C=C, max_iter=max_iter, use_cv=True, n_splits=n_splits, class_weight=class_weight
    )

    ranked = screener.rank_layers(results, use_cv=True)
    shortlist = [layer for layer, _ in ranked[:topk_layers]]

    # Extract directions for shortlisted layers
    # CRITICAL FIX: Pass scaler_scale for de-standardization of probe weights
    layer_to_U: Dict[int, np.ndarray] = {}
    for i, layer in enumerate(layers):
        if layer not in shortlist:
            continue
        probe = results[layer]["probe_model"]
        if probe is None:
            # Degenerate case - skip
            continue
        
        # Get scaler_scale for de-standardization (critical for correct direction)
        scaler_scale = results[layer].get("scaler_scale", None)
        
        U = screener.extract_top_directions(
            i, probe, k=1, method=direction_method, scaler_scale=scaler_scale
        )
        layer_to_U[layer] = U.astype(np.float32)

    # Convert results to JSON-serializable metrics (CV-based)
    metrics = {
        int(layer): {
            "delta_mu_norm": float(res["delta_mu_norm"]),
            "mean_auc": float(res["mean_auc"]) if not np.isnan(res["mean_auc"]) else None,
            "std_auc": float(res["std_auc"]) if not np.isnan(res["std_auc"]) else None,
            "mean_acc": float(res["mean_acc"]) if not np.isnan(res["mean_acc"]) else None,
            "std_acc": float(res["std_acc"]) if not np.isnan(res["std_acc"]) else None,
            "mean_auprc": float(res["mean_auprc"]) if not np.isnan(res["mean_auprc"]) else None,
            "std_auprc": float(res["std_auprc"]) if not np.isnan(res["std_auprc"]) else None,
        }
        for layer, res in results.items()
    }

    return {
        "ranked_layers": ranked,
        "shortlist": shortlist,
        "metrics": metrics,
        "layer_to_U": layer_to_U,
    }


def parse_model_tag_dataset(run_dir: Path) -> Tuple[str, str, str]:
    """
    Parse model_dir, tag, and dataset from run_dir path.
    
    Expected format: results/phase_a/<MODEL_DIR>__<TAG>/<DATASET>/
    
    Args:
        run_dir: Path to run directory
    
    Returns:
        Tuple of (model_dir, tag, dataset)
    """
    # Get dataset (last component before any trailing slash)
    parts = run_dir.parts
    dataset = parts[-1] if parts[-1] != "screening" else parts[-2]
    
    # Get model__tag (parent of dataset or grandparent if we're in screening/)
    if parts[-1] == "screening":
        model_tag_part = parts[-3]
    else:
        model_tag_part = parts[-2]
    
    # Split on "__" to get model_dir and tag
    if "__" in model_tag_part:
        model_dir, tag = model_tag_part.split("__", 1)
    else:
        # Fallback if no __ separator found
        model_dir = model_tag_part
        tag = "unknown"
    
    return model_dir, tag, dataset


def write_metrics_flat_csv(
    screen_results: Dict,
    run_dir: Path,
    layers: list,
    n_examples: int,
    n_pos: int,
    n_neg: int,
) -> None:
    """
    Write a flat CSV with per-layer metrics for easier aggregation.
    
    Schema: model_dir, tag, dataset, layer_idx, auc, std_auc, acc, std_acc, 
            auprc, std_auprc, delta_mu, score, n_examples, n_pos, n_neg
    
    Args:
        screen_results: Screening results dict
        run_dir: Path to run directory
        layers: List of layer indices
        n_examples: Total number of examples
        n_pos: Number of positive examples
        n_neg: Number of negative examples
    """
    # Parse metadata from path
    model_dir, tag, dataset = parse_model_tag_dataset(run_dir)
    
    # Build rows
    rows = []
    for layer, score in screen_results["ranked_layers"]:
        metrics = screen_results["metrics"][layer]
        row = {
            "model_dir": model_dir,
            "tag": tag,
            "dataset": dataset,
            "layer_idx": int(layer),
            "auc": metrics["mean_auc"] if metrics["mean_auc"] is not None else np.nan,
            "std_auc": metrics["std_auc"] if metrics["std_auc"] is not None else np.nan,
            "acc": metrics["mean_acc"] if metrics["mean_acc"] is not None else np.nan,
            "std_acc": metrics["std_acc"] if metrics["std_acc"] is not None else np.nan,
            "auprc": metrics["mean_auprc"] if metrics["mean_auprc"] is not None else np.nan,
            "std_auprc": metrics["std_auprc"] if metrics["std_auprc"] is not None else np.nan,
            "delta_mu": metrics["delta_mu_norm"],
            "score": score,
            "n_examples": n_examples,
            "n_pos": n_pos,
            "n_neg": n_neg,
        }
        rows.append(row)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(rows)
    screen_dir = run_dir / "screening"
    screen_dir.mkdir(exist_ok=True)
    csv_path = screen_dir / "metrics_flat.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved flat metrics to {csv_path}")


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise ValueError(f"Run directory does not exist: {run_dir}")

    print(f"Loading Step 1 artifacts from: {run_dir}")

    # Load generations CSV
    csv_path = run_dir / "generations.csv"
    csv_gz_path = run_dir / "generations.csv.gz"

    if csv_gz_path.exists():
        print(f"Loading {csv_gz_path}")
        with gzip.open(csv_gz_path, "rt", encoding="utf-8") as f:
            df = pd.read_csv(f)
    elif csv_path.exists():
        print(f"Loading {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No generations CSV found in {run_dir}")

    # Load pooled activations
    pooled_path = run_dir / "pooled.npz"
    if not pooled_path.exists():
        raise FileNotFoundError(f"No pooled.npz found in {run_dir}")

    print(f"Loading {pooled_path}")
    pooled_data = np.load(pooled_path, allow_pickle=True)
    X = pooled_data["X"]  # [N, L, D]
    layers = pooled_data["layers"].tolist()

    print(f"Loaded X shape: {X.shape}")
    print(f"Layers: {layers}")
    print(f"Loaded {len(df)} examples")

    # Get dataset info and handler
    dataset_name = df["dataset"].iloc[0] if "dataset" in df.columns else "unknown"
    split = df["split"].iloc[0] if "split" in df.columns else "test"
    
    # Get handler for this dataset (still used for gold-target normalization, etc.)
    handler = get_handler(dataset_name)
    print(f"Dataset: {dataset_name}, Split: {split}, Handler: {handler.name}")

    # Choose label source: GPT judgments vs handler-based parsing
    if args.use_gpt_labels:
        gpt_csv = run_dir / "generations_with_judgments.csv"
        if not gpt_csv.exists():
            raise FileNotFoundError(
                f"--use_gpt_labels set, but {gpt_csv} not found. "
                "Run scripts/annotate_reasoning_with_gpt.py first."
            )
        print(f"\nLoading GPT judgments from {gpt_csv}")
        df = pd.read_csv(gpt_csv)

        if "reasoning_correct_judge" not in df.columns:
            raise ValueError(
                "Expected 'reasoning_correct_judge' column in "
                "generations_with_judgments.csv when --use_gpt_labels is set."
            )

        # Use GPT's reasoning correctness as the binary label y
        y = df["reasoning_correct_judge"].astype(int).to_numpy()
        print("Using GPT-based reasoning_correct_judge as labels y.")
    else:
        # Default path: parse and label using handler's answer correctness
        print("\nParsing and labeling...")
        df, y = parse_and_label(
            df, handler, dataset_name=dataset_name,
            split=split, numeric_tol=args.numeric_tol
        )

    # Print label statistics
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"Label histogram: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
    print(f"Positives: {int(y.sum())}/{len(y)}")

    # Check for degenerate labels
    is_degenerate = len(unique_labels) < 2
    if is_degenerate:
        print(f"\n⚠️  WARNING: Only one class present: {unique_labels}")
        print("Screening will be skipped (cannot train probe with single class)")

        # Log debug samples
        print("\nDebug samples (first 5):")
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            print(f"  [{i}] Gold: {row['gold_label']}, Pred: {row['extracted_pred']}, Status: {row['parse_status']}")

    # Save labeled CSV and labels
    labeled_csv_path = run_dir / "generations_labeled.csv"
    print(f"\nSaving labeled generations to {labeled_csv_path}")
    df.to_csv(labeled_csv_path, index=False)

    labels_path = run_dir / "labels.npy"
    print(f"Saving labels to {labels_path}")
    np.save(labels_path, y)

    # Screening
    screen_dir = run_dir / "screening"
    screen_dir.mkdir(exist_ok=True)

    if is_degenerate:
        # Save degenerate marker
        metrics_path = screen_dir / "metrics.json"
        print(f"\nSaving degenerate label marker to {metrics_path}")
        degenerate_info = {
            "degenerate_labels": True,
            "note": "only one class present; screening skipped",
            "unique_labels": unique_labels.tolist(),
            "label_counts": counts.tolist(),
        }
        metrics_path.write_text(json.dumps(degenerate_info, indent=2))
    else:
        # Normal screening
        print("\nRunning offline screening...")
        screen_results = run_screening(
            X=X,
            y=y,
            layers=layers,
            C=args.probe_C,
            max_iter=args.probe_max_iter,
            topk_layers=args.topk_layers,
            direction_method=args.direction_method,
        )

        # Save metrics
        metrics_path = screen_dir / "metrics.json"
        print(f"\nSaving metrics to {metrics_path}")
        
        metrics_output = {
            "ranked_layers": [
                [int(layer), float(score), 
                 float(screen_results["metrics"][layer]["mean_auc"]) if screen_results["metrics"][layer]["mean_auc"] is not None else None,
                 float(screen_results["metrics"][layer]["std_auc"]) if screen_results["metrics"][layer]["std_auc"] is not None else None,
                 float(screen_results["metrics"][layer]["mean_acc"]) if screen_results["metrics"][layer]["mean_acc"] is not None else None,
                 float(screen_results["metrics"][layer]["std_acc"]) if screen_results["metrics"][layer]["std_acc"] is not None else None,
                 float(screen_results["metrics"][layer]["mean_auprc"]) if screen_results["metrics"][layer]["mean_auprc"] is not None else None,
                 float(screen_results["metrics"][layer]["std_auprc"]) if screen_results["metrics"][layer]["std_auprc"] is not None else None,
                 float(screen_results["metrics"][layer]["delta_mu_norm"])]
                for layer, score in screen_results["ranked_layers"]
            ],
            "shortlist": [int(l) for l in screen_results["shortlist"]],
            "degenerate_labels": False,
        }
        metrics_path.write_text(json.dumps(metrics_output, indent=2))

        # Save directions
        if len(screen_results["layer_to_U"]) > 0:
            directions_path = screen_dir / "layer_to_U.npz"
            print(f"Saving directions to {directions_path}")
            np.savez_compressed(
                directions_path,
                **{f"L{layer}": U for layer, U in screen_results["layer_to_U"].items()},
            )
            print(f"Saved directions for {len(screen_results['layer_to_U'])} layers")

        # Save flat CSV for aggregation
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        write_metrics_flat_csv(
            screen_results=screen_results,
            run_dir=run_dir,
            layers=layers,
            n_examples=len(y),
            n_pos=n_pos,
            n_neg=n_neg,
        )

        # Print top layers with CV metrics
        print("\nTop ranked layers (CV metrics):")
        for layer, score in screen_results["ranked_layers"][:5]:
            m = screen_results["metrics"][layer]
            auc_str = f"{m['mean_auc']:.3f}±{m['std_auc']:.3f}" if m['mean_auc'] is not None else 'N/A'
            acc_str = f"{m['mean_acc']:.3f}±{m['std_acc']:.3f}" if m['mean_acc'] is not None else 'N/A'
            auprc_str = f"{m['mean_auprc']:.3f}±{m['std_auprc']:.3f}" if m['mean_auprc'] is not None else 'N/A'
            print(f"  Layer {layer}: score={score:.4f}, AUC={auc_str}, Acc={acc_str}, AUPRC={auprc_str}, Δμ={m['delta_mu_norm']:.2f}")

    print(f"\n{'='*60}")
    print(f"Phase A Step 2 completed!")
    print(f"Artifacts saved under: {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
