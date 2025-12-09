"""
Phase A Aggregation: Combine per-dataset screening results

This script aggregates Phase A Step 2 screening results across multiple datasets
per model to produce:
1. Robust layer rankings using rank aggregation, z-score normalization, and stability selection
2. A recommended shortlist of layers for Phase B
3. (Optional) Consensus directions via sign-aligned averaging

Usage:
    python -m experiments.phase_a_aggregate \
        --results_root results/phase_a \
        --tag nov_ten_arc_only \
        --models "Mistral-7B-Instruct-v0.3,Qwen2.5-7B-Instruct" \
        --datasets "arc,gsm8k" \
        --topk 8 \
        --primary_metric auc \
        --weight_by n_examples \
        --save_consensus_directions
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.agg_utils import (
    discover_runs,
    load_metrics_for_run,
    compute_rank_aggregation,
    compute_z_score_aggregation,
    compute_stability_selection,
    aggregate_per_dataset_metrics,
    build_consensus_directions,
)

logger = logging.getLogger(__name__)


def _schema_empty_df():
    # minimal schema with join key, so outer merges never KeyError
    return pd.DataFrame(columns=["layer_idx"])


def _safe_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if right is None or right.empty or "layer_idx" not in right.columns:
        logger.warning("Skipping merge: right side empty or missing 'layer_idx'")
        return left
    if left is None or left.empty:
        # still need 'layer_idx' to exist
        if "layer_idx" not in right.columns:
            return _schema_empty_df()
        return right
    if "layer_idx" not in left.columns:
        # bring left onto schema to allow outer merge
        left = left.copy()
        left["layer_idx"] = pd.Series(dtype="Int64")
    return left.merge(right, on="layer_idx", how="outer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase A Aggregation: Combine per-dataset screening results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--results_root",
        type=str,
        default="results/phase_a",
        help="Root directory for Phase A results (default: results/phase_a)",
    )

    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Tag to filter runs (e.g., 'nov_ten_arc_only')",
    )

    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of model dir prefixes to include (e.g., 'Llama-3.1-8B-Instruct,Mistral-7B-Instruct-v0.3'). If omitted, discovers all models with the tag.",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of datasets to include (e.g., 'arc,gsm8k,mmlu_pro'). If omitted, discovers all available datasets.",
    )

    parser.add_argument(
        "--primary_metric",
        type=str,
        default="auc",
        choices=["auc", "acc", "auprc", "delta_mu", "score"],
        help="Primary metric for ranking (default: auc)",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        help="Number of top layers for stability selection (default: 8)",
    )

    parser.add_argument(
        "--weight_by",
        type=str,
        default="n_examples",
        choices=["n_examples", "min_posneg", "none"],
        help="Weighting for z-score aggregation: 'n_examples', 'min_posneg', or 'none' (default: n_examples)",
    )

    parser.add_argument(
        "--save_consensus_directions",
        action="store_true",
        help="If set, build and save consensus directions for selected layers",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: <results_root>/<MODEL_DIR>__<TAG>/aggregation/)",
    )

    return parser.parse_args()


def select_top_layers(
    summary_df: pd.DataFrame,
    topk: int,
    min_hit_rate: float = 0.5,
) -> List[int]:
    """
    Select top layers using the composite selection rule.
    
    Rule: Keep layers with topk_hit_rate >= min_hit_rate AND
          among the best topk layers by agg_borda (breaking ties by agg_rank_median, then agg_z_mean_weighted)
    
    Args:
        summary_df: Summary DataFrame with aggregation metrics
        topk: Number of layers to select
        min_hit_rate: Minimum hit rate threshold
    
    Returns:
        List of selected layer indices
    """
    # Filter by hit rate
    candidates = summary_df[summary_df["topk_hit_rate"] >= min_hit_rate].copy()
    
    if len(candidates) == 0:
        print(f"⚠️  Warning: No layers meet hit_rate >= {min_hit_rate}, relaxing to all layers")
        candidates = summary_df.copy()
    
    # Sort by Borda (desc), then rank_median (asc), then z_mean (desc)
    candidates = candidates.sort_values(
        by=["agg_borda", "agg_rank_median", "agg_z_mean_weighted"],
        ascending=[False, True, False],
    )
    
    # Take top-k
    selected = candidates.head(topk)["layer_idx"].tolist()
    
    return selected


def format_layer_table(summary_df: pd.DataFrame, layer_indices: List[int]) -> str:
    """
    Format a table of layer statistics.
    
    Args:
        summary_df: Summary DataFrame
        layer_indices: List of layer indices to include
    
    Returns:
        Formatted table string
    """
    subset = summary_df[summary_df["layer_idx"].isin(layer_indices)].copy()
    
    # Sort by layer index
    subset = subset.sort_values("layer_idx")
    
    lines = []
    lines.append("Layer | Rank_Med | Borda  | Z_Weighted | Hit_Rate | Datasets")
    lines.append("------|----------|--------|------------|----------|----------")
    
    for _, row in subset.iterrows():
        layer = int(row["layer_idx"])
        rank_med = row["agg_rank_median"]
        borda = row["agg_borda"]
        z_mean = row["agg_z_mean_weighted"]
        hit_rate = row["topk_hit_rate"]
        n_datasets = int(row["num_datasets"])
        
        lines.append(
            f" {layer:4d} | {rank_med:8.2f} | {borda:6.1f} | {z_mean:10.3f} | {hit_rate:8.3f} | {n_datasets:8d}"
        )
    
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    results_root = Path(args.results_root)
    
    # Parse model and dataset filters
    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    
    datasets = None
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    
    print("="*70)
    print("Phase A Aggregation")
    print("="*70)
    print(f"Results root: {results_root}")
    print(f"Tag: {args.tag}")
    print(f"Models filter: {models if models else 'all'}")
    print(f"Datasets filter: {datasets if datasets else 'all'}")
    print(f"Primary metric: {args.primary_metric}")
    print(f"Top-k: {args.topk}")
    print(f"Weighting: {args.weight_by}")
    print(f"Build consensus directions: {args.save_consensus_directions}")
    print()

    # Discover runs
    print("Discovering runs...")
    runs = discover_runs(
        results_root=results_root,
        tag=args.tag,
        models=models,
        datasets=datasets,
    )
    
    if not runs:
        print("❌ No runs found matching the criteria!")
        return
    
    print(f"Found {len(runs)} model(s):")
    for model_dir in sorted(runs.keys()):
        dataset_list = sorted(runs[model_dir].keys())
        print(f"  • {model_dir}: {len(dataset_list)} dataset(s) - {dataset_list}")
    print()

    # Process each model
    for model_dir in sorted(runs.keys()):
        print("="*70)
        print(f"Processing: {model_dir}")
        print("="*70)
        
        dataset_dict = runs[model_dir]
        dataset_names = sorted(dataset_dict.keys())
        screening_dirs = [dataset_dict[d] for d in dataset_names]
        
        print(f"Datasets: {dataset_names}")
        print(f"Number of datasets: {len(dataset_names)}")
        
        # Load metrics for all datasets
        print("\nLoading metrics...")
        dfs = []
        for dataset, screening_dir in zip(dataset_names, screening_dirs):
            df = load_metrics_for_run(screening_dir)
            if not df.empty and "layer_idx" in df.columns:
                dfs.append(df)
                print(f"  ✓ {dataset}: {len(df)} layers")
            else:
                logger.warning("Degenerate/empty metrics for %s/%s — continuing without contributing to aggregation", model_dir, dataset)
                print(f"  ✗ {dataset}: no metrics found")
        
        if not dfs:
            print(f"⚠️  No valid metrics found for {model_dir}, skipping\n")
            continue
        
        # Compute aggregations
        print("\nComputing aggregations...")
        
        # 1. Rank-based
        rank_df = compute_rank_aggregation(
            dfs=dfs,
            dataset_names=dataset_names,
            primary_metric=args.primary_metric,
        )
        print(f"  ✓ Rank aggregation: {len(rank_df)} layers")
        
        # 2. Z-score
        zscore_df = compute_z_score_aggregation(
            dfs=dfs,
            dataset_names=dataset_names,
            primary_metric=args.primary_metric,
            weight_by=args.weight_by,
        )
        print(f"  ✓ Z-score aggregation: {len(zscore_df)} layers")
        
        # 3. Stability selection
        stability_df = compute_stability_selection(
            dfs=dfs,
            dataset_names=dataset_names,
            primary_metric=args.primary_metric,
            topk=args.topk,
        )
        print(f"  ✓ Stability selection: {len(stability_df)} layers")
        
        # 4. Per-dataset stats
        stats_df = aggregate_per_dataset_metrics(
            dfs=dfs,
            dataset_names=dataset_names,
            primary_metric=args.primary_metric,
        )
        print(f"  ✓ Per-dataset statistics: {len(stats_df)} layers")
        
        # Merge all aggregations
        print("\nMerging aggregations...")
        summary_df = rank_df
        summary_df = _safe_merge(summary_df, zscore_df)
        summary_df = _safe_merge(summary_df, stability_df)
        summary_df = _safe_merge(summary_df, stats_df)
        
        # Fill NaN values
        summary_df = summary_df.fillna({
            "agg_rank_median": np.inf,
            "agg_borda": 0.0,
            "agg_z_mean_weighted": 0.0,
            "topk_hits": 0,
            "topk_hit_rate": 0.0,
            "num_datasets": 0,
            "primary_metric_mean": np.nan,
            "primary_metric_std": np.nan,
        })
        
        print(f"Summary DataFrame: {len(summary_df)} layers")
        
        # Select top layers
        print("\nSelecting top layers...")
        selected_layers = select_top_layers(
            summary_df=summary_df,
            topk=args.topk,
            min_hit_rate=0.5,
        )
        print(f"Selected {len(selected_layers)} layers: {selected_layers}")
        
        # Prepare output directory
        if args.out_dir:
            out_dir = Path(args.out_dir)
        else:
            model_tag_dir = results_root / f"{model_dir}__{args.tag}"
            out_dir = model_tag_dir / "aggregation"
        
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput directory: {out_dir}")
        
        # Save summary CSV
        csv_path = out_dir / "summary_layers.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved {csv_path}")
        
        # Save consensus top layers text file
        txt_path = out_dir / "consensus_top_layers.txt"
        with open(txt_path, "w") as f:
            f.write(f"Recommended Phase B layers for {model_dir}: {selected_layers}\n\n")
            f.write(f"Selection criteria: topk_hit_rate >= 0.5 AND top-{args.topk} by agg_borda\n")
            f.write(f"(Breaking ties: agg_rank_median, then agg_z_mean_weighted)\n\n")
            f.write(format_layer_table(summary_df, selected_layers))
            f.write("\n")
        print(f"  ✓ Saved {txt_path}")
        
        # Print top 10 layers
        print("\nTop 10 layers by selection criterion:")
        top10_candidates = summary_df.copy()
        top10_candidates = top10_candidates.sort_values(
            by=["agg_borda", "agg_rank_median", "agg_z_mean_weighted"],
            ascending=[False, True, False],
        ).head(10)
        print(format_layer_table(summary_df, top10_candidates["layer_idx"].tolist()))
        
        # Build consensus directions if requested
        if args.save_consensus_directions:
            print("\nBuilding consensus directions...")
            
            if len(dataset_names) < 2:
                print(f"  ⚠️  Warning: Only {len(dataset_names)} dataset(s) available, need >=2 for consensus")
                print("     Skipping consensus direction building")
            else:
                consensus_dirs = build_consensus_directions(
                    screening_dirs=screening_dirs,
                    layer_indices=selected_layers,
                    dataset_names=dataset_names,
                )
                
                if consensus_dirs:
                    npz_path = out_dir / "layer_to_U_consensus.npz"
                    np.savez_compressed(npz_path, **{str(k): v for k, v in consensus_dirs.items()})
                    print(f"  ✓ Saved consensus directions for {len(consensus_dirs)} layers to {npz_path}")
                else:
                    print("  ⚠️  No consensus directions could be built (need >=2 vectors per layer)")
        
        print()
    
    print("="*70)
    print("Phase A Aggregation completed!")
    print("="*70)


if __name__ == "__main__":
    main()
