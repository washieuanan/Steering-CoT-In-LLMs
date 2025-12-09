"""
Aggregation utilities for Phase A screening results.

This module provides functions for:
1. Discovering completed Phase A runs
2. Loading and parsing metrics from various formats
3. Aggregating metrics across datasets per model
4. Building consensus directions via sign-aligned averaging
5. Computing locality masks for Phase-B interventions
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.parse_answers import locate_answer_span


def compute_locality_mask_post_generation(
    full_ids: List[int],
    prompt_len: int,
    tokenizer,
    locality: str,
    answer_phrase: Optional[str] = None
) -> Tuple[List[bool], Tuple[int, int, bool]]:
    """
    Compute locality mask for Phase-B interventions (legacy wrapper).
    
    This is a compatibility wrapper that calls the new implementation in utils.hooks.
    For new code, prefer calling hooks.compute_locality_mask_post_generation directly.
    
    Args:
        full_ids: Complete token ID sequence [prompt + generated]
        prompt_len: Length of prompt portion
        tokenizer: HuggingFace tokenizer for phrase detection
        locality: One of "cot", "answer", "all"
        answer_phrase: Optional explicit answer phrase (e.g., "MCQ ANSWER:")
    
    Returns:
        Tuple of (mask, answer_span_info):
        - mask: Boolean list of length len(full_ids), True = intervene here
        - answer_span_info: (start, end, found) from answer extraction
    
    Example:
        >>> full_ids = [1, 2, 3, 100, 4, 5, 6, 200, 7, 8]  # 100=<cot>, 200=answer_phrase
        >>> mask, span = compute_locality_mask_post_generation(
        ...     full_ids, prompt_len=3, tokenizer=tok, locality="cot"
        ... )
        >>> # mask will be [F, F, F, F, T, T, T, F, F, F] (True for CoT tokens 4,5,6)
    """
    # Delegate to the new implementation in hooks.py
    # We need to reconstruct the full_text from full_ids
    full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
    prompt_text = tokenizer.decode(full_ids[:prompt_len], skip_special_tokens=False)
    
    from utils.hooks import compute_locality_mask_post_generation as compute_mask_new
    
    result = compute_mask_new(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        full_text=full_text,
        input_ids=full_ids,
        locality=locality,
        cot_text=None,
        token_texts_tail=None,
        choice_logits=None,
    )
    
    # Convert torch mask to list
    mask_list = result['mask'].tolist()
    
    # Extract span info
    spans = result['spans']
    answer_span = spans['answer_span']
    found = spans['answer_found']
    
    return mask_list, (answer_span[0], answer_span[1], found)


# Schema for empty metrics DataFrames
EMPTY_METRICS_COLUMNS = [
    "layer_idx",    # join key
    "score",        # overall screening score if present
    "auc",
    "acc",
    "auprc",
    "delta_mu",     # or Δμ in some files; normalize to delta_mu
    "dataset",
    "model",
]


def _empty_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EMPTY_METRICS_COLUMNS)


def _normalize_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_metrics_df()
    # Standardize column names we might see
    rename_map = {
        "layer": "layer_idx",
        "layer_index": "layer_idx",
        "Δμ": "delta_mu",
        "delta-μ": "delta_mu",
        "DeltaMu": "delta_mu",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    # Ensure join key exists; if not, we cannot use these rows
    if "layer_idx" not in df.columns:
        # keep schema for safe outer merges later
        base = _empty_metrics_df()
        # carry over any known columns
        for c in df.columns:
            if c in base.columns:
                base[c] = df[c]
        return base
    # Ensure all expected columns exist
    for c in EMPTY_METRICS_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    # Make sure types are merge-friendly
    try:
        df["layer_idx"] = pd.to_numeric(df["layer_idx"], errors="coerce").astype("Int64")
    except Exception:
        pass
    return df[EMPTY_METRICS_COLUMNS]


def _safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    return np.nanmean(arr)


def discover_runs(
    results_root: Path,
    tag: str,
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Path]]:
    """
    Discover all completed Phase A runs matching the given criteria.
    
    Args:
        results_root: Root directory for Phase A results
        tag: Tag to match (e.g., "nov_ten_arc_only")
        models: Optional list of model directory prefixes to include
        datasets: Optional list of datasets to include
    
    Returns:
        Nested dict: {model_dir: {dataset: path_to_screening_dir}}
        Example: {"Llama-3.1-8B-Instruct": {"arc": Path(...), "gsm8k": Path(...)}}
    """
    results_root = Path(results_root)
    if not results_root.exists():
        warnings.warn(f"Results root does not exist: {results_root}")
        return {}
    
    runs = {}
    
    # Find all directories matching *__<tag>
    for model_tag_dir in results_root.iterdir():
        if not model_tag_dir.is_dir():
            continue
        
        # Check if this matches our tag
        if not model_tag_dir.name.endswith(f"__{tag}"):
            continue
        
        # Extract model_dir
        model_dir = model_tag_dir.name.rsplit("__", 1)[0]
        
        # Check if this model is in our filter
        if models is not None:
            if model_dir not in models:
                continue
        
        # Find dataset subdirectories
        runs[model_dir] = {}
        for dataset_dir in model_tag_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            
            # Check if this dataset is in our filter
            if datasets is not None:
                if dataset_name not in datasets:
                    continue
            
            # Check if screening directory exists
            screening_dir = dataset_dir / "screening"
            if not screening_dir.exists():
                continue
            
            # Check if metrics exist
            metrics_json = screening_dir / "metrics.json"
            metrics_csv = screening_dir / "metrics_flat.csv"
            if not metrics_json.exists() and not metrics_csv.exists():
                continue
            
            runs[model_dir][dataset_name] = screening_dir
    
    return runs


def parse_model_tag_from_path(path: Path) -> Tuple[str, str]:
    """
    Parse model_dir and tag from a path.
    
    Expected format: .../results/phase_a/<MODEL_DIR>__<TAG>/...
    
    Args:
        path: Path containing model__tag
    
    Returns:
        Tuple of (model_dir, tag)
    """
    # Find the part with __
    for part in path.parts:
        if "__" in part:
            model_dir, tag = part.split("__", 1)
            return model_dir, tag
    
    return "unknown", "unknown"


def load_metrics_json(json_path: Path) -> pd.DataFrame:
    """
    Load metrics from JSON format and convert to flat DataFrame.
    
    Handles both old and new JSON formats:
    - New: {"ranked_layers": [[layer, score, auc, std_auc, ...], ...], "degenerate_labels": bool}
    - Old: {"metrics": {layer: {...}}, ...}
    
    Args:
        json_path: Path to metrics.json
    
    Returns:
        DataFrame with columns: layer_idx, auc, std_auc, acc, std_acc, 
                                auprc, std_auprc, delta_mu, score
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception:
        warnings.warn(f"Failed to read {json_path}, returning empty metrics DataFrame")
        return _empty_metrics_df()

    # If we marked degenerate labels earlier, or nothing useful is present, return empty schema
    if not data or data.get("degenerate", False) or data.get("degenerate_labels", False) or len(data.get("layers", [])) == 0:
        # Check for ranked_layers too
        if "ranked_layers" not in data and "metrics" not in data:
            warnings.warn(f"Degenerate labels in {json_path}, returning empty DataFrame")
            return _empty_metrics_df()
    
    rows = []
    
    # New format: ranked_layers list
    if "ranked_layers" in data:
        for entry in data["ranked_layers"]:
            if len(entry) == 9:
                # Full format: [layer, score, auc, std_auc, acc, std_acc, auprc, std_auprc, delta_mu]
                layer, score, auc, std_auc, acc, std_acc, auprc, std_auprc, delta_mu = entry
                row = {
                    "layer_idx": int(layer),
                    "auc": auc,
                    "std_auc": std_auc,
                    "acc": acc,
                    "std_acc": std_acc,
                    "auprc": auprc,
                    "std_auprc": std_auprc,
                    "delta_mu": delta_mu,
                    "score": score,
                    "dataset": data.get("dataset"),
                    "model": data.get("model"),
                }
            else:
                # Partial format: just extract what we have
                layer = int(entry[0])
                score = entry[1] if len(entry) > 1 else np.nan
                row = {
                    "layer_idx": layer,
                    "auc": entry[2] if len(entry) > 2 else np.nan,
                    "std_auc": entry[3] if len(entry) > 3 else np.nan,
                    "acc": entry[4] if len(entry) > 4 else np.nan,
                    "std_acc": entry[5] if len(entry) > 5 else np.nan,
                    "auprc": entry[6] if len(entry) > 6 else np.nan,
                    "std_auprc": entry[7] if len(entry) > 7 else np.nan,
                    "delta_mu": entry[8] if len(entry) > 8 else np.nan,
                    "score": score,
                    "dataset": data.get("dataset"),
                    "model": data.get("model"),
                }
            rows.append(row)
    
    # Old format: metrics dict
    elif "metrics" in data:
        for layer_str, metrics in data["metrics"].items():
            layer = int(layer_str)
            row = {
                "layer_idx": layer,
                "auc": metrics.get("mean_auc", np.nan),
                "std_auc": metrics.get("std_auc", np.nan),
                "acc": metrics.get("mean_acc", np.nan),
                "std_acc": metrics.get("std_acc", np.nan),
                "auprc": metrics.get("mean_auprc", np.nan),
                "std_auprc": metrics.get("std_auprc", np.nan),
                "delta_mu": metrics.get("delta_mu_norm", np.nan),
                "score": metrics.get("score", np.nan),
                "dataset": data.get("dataset"),
                "model": data.get("model"),
            }
            rows.append(row)
    
    if not rows:
        return _empty_metrics_df()
    
    df = pd.DataFrame(rows)
    return _normalize_metrics_df(df)


def load_metrics_flat_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load metrics from CSV format with resilient error handling.
    
    Args:
        csv_path: Path to metrics_flat.csv
    
    Returns:
        DataFrame with normalized schema
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        warnings.warn(f"Failed to read {csv_path}, returning empty metrics DataFrame")
        return _empty_metrics_df()
    return _normalize_metrics_df(df)


def load_metrics_for_run(screening_dir: Path) -> pd.DataFrame:
    """
    Load metrics for a single run, preferring CSV over JSON.
    
    Args:
        screening_dir: Path to screening directory
    
    Returns:
        DataFrame with per-layer metrics
    """
    csv_path = screening_dir / "metrics_flat.csv"
    json_path = screening_dir / "metrics.json"
    
    # Prefer CSV
    if csv_path.exists():
        df = load_metrics_flat_csv(csv_path)
        if not df.empty and "layer_idx" in df.columns:
            return df
        else:
            warnings.warn(f"CSV missing required data in {csv_path}, falling back to JSON")
    
    # Fallback to JSON
    if json_path.exists():
        return load_metrics_json(json_path)
    
    warnings.warn(f"No metrics found in {screening_dir}")
    return _empty_metrics_df()


def compute_rank_aggregation(
    dfs: List[pd.DataFrame],
    dataset_names: List[str],
    primary_metric: str = "auc",
) -> pd.DataFrame:
    """
    Compute rank-based aggregation across datasets.
    
    For each dataset, ranks layers by primary_metric (descending).
    Then computes:
    - agg_rank_median: Median rank across datasets
    - agg_borda: Borda count (sum of L_max + 1 - rank)
    
    Args:
        dfs: List of DataFrames, one per dataset
        dataset_names: List of dataset names (for debugging)
        primary_metric: Metric to rank by
    
    Returns:
        DataFrame with columns: layer_idx, agg_rank_median, agg_borda
    """
    all_layers = set()
    for df in dfs:
        all_layers.update(df["layer_idx"].values)
    
    layer_ranks = {layer: [] for layer in all_layers}
    
    for df, dataset in zip(dfs, dataset_names):
        if df.empty or primary_metric not in df.columns:
            continue
        
        # Rank within this dataset (descending: higher metric = better = lower rank)
        df = df.copy()
        df["rank"] = df[primary_metric].rank(ascending=False, method="average")
        
        L_max = len(df)
        
        for _, row in df.iterrows():
            layer = int(row["layer_idx"])
            rank = row["rank"]
            layer_ranks[layer].append((rank, L_max))
    
    # Aggregate
    results = []
    for layer in sorted(all_layers):
        ranks = layer_ranks[layer]
        if not ranks:
            continue
        
        rank_values = [r for r, _ in ranks]
        borda_values = [L_max + 1 - r for r, L_max in ranks]
        
        results.append({
            "layer_idx": int(layer),
            "agg_rank_median": float(np.median(rank_values)),
            "agg_borda": float(np.sum(borda_values)),
        })
    
    return pd.DataFrame(results)


def compute_z_score_aggregation(
    dfs: List[pd.DataFrame],
    dataset_names: List[str],
    primary_metric: str = "auc",
    weight_by: str = "n_examples",
) -> pd.DataFrame:
    """
    Compute z-score standardized aggregation across datasets.
    
    Within each dataset, z-scores the primary_metric, then computes
    weighted mean across datasets.
    
    Args:
        dfs: List of DataFrames, one per dataset
        dataset_names: List of dataset names
        primary_metric: Metric to aggregate
        weight_by: Weighting scheme: "n_examples", "min_posneg", or "none"
    
    Returns:
        DataFrame with columns: layer_idx, agg_z_mean_weighted
    """
    all_layers = set()
    for df in dfs:
        all_layers.update(df["layer_idx"].values)
    
    layer_z_values = {layer: [] for layer in all_layers}
    layer_weights = {layer: [] for layer in all_layers}
    
    for df, dataset in zip(dfs, dataset_names):
        if df.empty or primary_metric not in df.columns:
            continue
        
        df = df.copy()
        
        # Compute z-scores (handle zero variance)
        values = df[primary_metric].values
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values, ddof=0)
        
        if std_val == 0 or np.isnan(std_val):
            df["z"] = 0.0
        else:
            df["z"] = (values - mean_val) / std_val
        
        # Determine weights
        if weight_by == "n_examples":
            weight_col = "n_examples"
        elif weight_by == "min_posneg":
            # Prefer min(n_pos, n_neg) if available
            if "n_pos" in df.columns and "n_neg" in df.columns:
                df["weight"] = df[["n_pos", "n_neg"]].min(axis=1)
                weight_col = "weight"
            elif "n_examples" in df.columns:
                weight_col = "n_examples"
            else:
                weight_col = None
        else:  # "none"
            weight_col = None
        
        for _, row in df.iterrows():
            layer = int(row["layer_idx"])
            z = row["z"]
            
            if not np.isnan(z):
                layer_z_values[layer].append(z)
                
                if weight_col is not None and weight_col in df.columns:
                    w = row[weight_col]
                    layer_weights[layer].append(w if not np.isnan(w) else 1.0)
                else:
                    layer_weights[layer].append(1.0)
    
    # Aggregate
    results = []
    for layer in sorted(all_layers):
        z_vals = layer_z_values[layer]
        weights = layer_weights[layer]
        
        if not z_vals:
            continue
        
        z_vals = np.array(z_vals)
        weights = np.array(weights)
        
        # Weighted mean
        z_mean = np.average(z_vals, weights=weights)
        
        results.append({
            "layer_idx": int(layer),
            "agg_z_mean_weighted": float(z_mean),
        })
    
    return pd.DataFrame(results)


def compute_stability_selection(
    dfs: List[pd.DataFrame],
    dataset_names: List[str],
    primary_metric: str = "auc",
    topk: int = 8,
) -> pd.DataFrame:
    """
    Compute stability selection: how often each layer appears in top-k.
    
    Args:
        dfs: List of DataFrames, one per dataset
        dataset_names: List of dataset names
        primary_metric: Metric to rank by
        topk: Number of top layers to consider per dataset
    
    Returns:
        DataFrame with columns: layer_idx, topk_hits, topk_hit_rate
    """
    all_layers = set()
    for df in dfs:
        all_layers.update(df["layer_idx"].values)
    
    layer_hits = {layer: 0 for layer in all_layers}
    layer_datasets = {layer: 0 for layer in all_layers}
    
    for df, dataset in zip(dfs, dataset_names):
        if df.empty or primary_metric not in df.columns:
            continue
        
        # Mark all layers as having data from this dataset
        for layer in df["layer_idx"].values:
            layer_datasets[int(layer)] += 1
        
        # Get top-k by primary metric
        df_sorted = df.sort_values(primary_metric, ascending=False)
        top_k_layers = df_sorted.head(topk)["layer_idx"].values
        
        for layer in top_k_layers:
            layer_hits[int(layer)] += 1
    
    # Compute hit rates
    results = []
    for layer in sorted(all_layers):
        hits = layer_hits[layer]
        n_datasets = layer_datasets[layer]
        hit_rate = hits / n_datasets if n_datasets > 0 else 0.0
        
        results.append({
            "layer_idx": int(layer),
            "topk_hits": hits,
            "topk_hit_rate": float(hit_rate),
        })
    
    return pd.DataFrame(results)


def load_layer_direction(npz_path: Path, layer_idx: int) -> Optional[np.ndarray]:
    """
    Load direction vector for a specific layer from NPZ file.
    
    Handles different key formats: "L24", "24", 24, "layer_24"
    
    Args:
        npz_path: Path to layer_to_U.npz
        layer_idx: Layer index to load
    
    Returns:
        Direction vector [D] or [D, 1], or None if not found
    """
    if not npz_path.exists():
        return None
    
    data = np.load(npz_path)
    
    # Try different key formats
    possible_keys = [
        f"L{layer_idx}",
        str(layer_idx),
        layer_idx,
        f"layer_{layer_idx}",
    ]
    
    for key in possible_keys:
        if key in data:
            vec = data[key]
            # Ensure 1D
            if vec.ndim == 2 and vec.shape[1] == 1:
                vec = vec.squeeze(1)
            return vec
    
    return None


def build_consensus_directions(
    screening_dirs: List[Path],
    layer_indices: List[int],
    dataset_names: Optional[List[str]] = None,
) -> Dict[int, np.ndarray]:
    """
    Build consensus directions via sign-aligned averaging.
    
    For each layer:
    1. Load all available direction vectors from datasets
    2. Align signs using cosine similarity with running mean
    3. Average and L2-normalize
    
    Args:
        screening_dirs: List of paths to screening directories
        layer_indices: List of layer indices to process
        dataset_names: Optional list of dataset names for logging
    
    Returns:
        Dict mapping layer_idx to consensus direction [D]
    """
    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(screening_dirs))]
    
    consensus = {}
    
    for layer_idx in layer_indices:
        vectors = []
        sources = []
        
        # Load vectors from all datasets
        for screening_dir, dataset in zip(screening_dirs, dataset_names):
            npz_path = screening_dir / "layer_to_U.npz"
            vec = load_layer_direction(npz_path, layer_idx)
            
            if vec is not None:
                vectors.append(vec)
                sources.append(dataset)
        
        # Need at least 2 vectors for consensus
        if len(vectors) < 2:
            if len(vectors) == 1:
                warnings.warn(
                    f"Layer {layer_idx}: Only 1 vector available from {sources}, "
                    "need >=2 for consensus. Skipping."
                )
            continue
        
        # Sign-aligned averaging
        vectors = [v.astype(np.float64) for v in vectors]
        
        # Initialize with first vector
        running_mean = vectors[0].copy()
        aligned_vectors = [vectors[0]]
        
        # Align remaining vectors
        for i, vec in enumerate(vectors[1:], 1):
            # Compute cosine similarity with running mean
            cos_sim = np.dot(vec, running_mean) / (
                np.linalg.norm(vec) * np.linalg.norm(running_mean) + 1e-10
            )
            
            # Flip if negative
            if cos_sim < 0:
                vec = -vec
            
            aligned_vectors.append(vec)
            # Update running mean
            running_mean = np.mean(aligned_vectors, axis=0)
        
        # Final average and normalize
        consensus_vec = np.mean(aligned_vectors, axis=0)
        consensus_vec = consensus_vec / (np.linalg.norm(consensus_vec) + 1e-10)
        
        consensus[layer_idx] = consensus_vec.astype(np.float32)
    
    return consensus


def aggregate_per_dataset_metrics(
    dfs: List[pd.DataFrame],
    dataset_names: List[str],
    primary_metric: str = "auc",
) -> pd.DataFrame:
    """
    Compute per-layer statistics across datasets.
    
    Args:
        dfs: List of DataFrames, one per dataset
        dataset_names: List of dataset names
        primary_metric: Metric to summarize
    
    Returns:
        DataFrame with columns: layer_idx, num_datasets, primary_metric_mean,
                                primary_metric_std, datasets_present, per_dataset_metrics
    """
    all_layers = set()
    for df in dfs:
        all_layers.update(df["layer_idx"].values)
    
    results = []
    
    for layer in sorted(all_layers):
        values = []
        present_datasets = []
        per_dataset = {}
        
        for df, dataset in zip(dfs, dataset_names):
            if df.empty or primary_metric not in df.columns:
                continue
            
            layer_rows = df[df["layer_idx"] == layer]
            if len(layer_rows) > 0:
                val = layer_rows.iloc[0][primary_metric]
                if not np.isnan(val):
                    values.append(val)
                    present_datasets.append(dataset)
                    per_dataset[dataset] = float(val)
        
        if values:
            results.append({
                "layer_idx": int(layer),
                "num_datasets": len(values),
                "primary_metric_mean": float(np.mean(values)),
                "primary_metric_std": float(np.std(values)),
                "datasets_present": json.dumps(present_datasets),
                "per_dataset_metrics": json.dumps(per_dataset),
            })
    
    return pd.DataFrame(results)
