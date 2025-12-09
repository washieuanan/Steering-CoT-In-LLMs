"""
Metrics Computation for Intervention Analysis

This module provides functions for computing various metrics from Phase B
intervention experiment results.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ==============================================================================
# Basic Accuracy Metrics
# ==============================================================================

def compute_accuracy(correct_col: pd.Series) -> float:
    """Compute accuracy from a boolean column.
    
    Args:
        correct_col: Series of boolean values
        
    Returns:
        Accuracy as float between 0 and 1
    """
    # Handle string 'True'/'False' values
    if correct_col.dtype == object:
        correct_col = correct_col.map({'True': True, 'False': False, True: True, False: False})
    return correct_col.astype(bool).mean()


def compute_baseline_accuracy(df: pd.DataFrame, metric: str = 'answer') -> float:
    """Compute baseline accuracy.
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        
    Returns:
        Baseline accuracy
    """
    col = f'baseline_{metric}_correct'
    if col in df.columns:
        return compute_accuracy(df[col])
    return np.nan


def compute_intervention_accuracy(df: pd.DataFrame, metric: str = 'answer') -> float:
    """Compute intervention accuracy.
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        
    Returns:
        Intervention accuracy
    """
    col = f'intv_{metric}_correct'
    if col in df.columns:
        return compute_accuracy(df[col])
    return np.nan


def compute_delta_accuracy(df: pd.DataFrame, metric: str = 'answer') -> float:
    """Compute delta (intervention - baseline) accuracy.
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        
    Returns:
        Delta accuracy
    """
    base_acc = compute_baseline_accuracy(df, metric)
    intv_acc = compute_intervention_accuracy(df, metric)
    return intv_acc - base_acc


# ==============================================================================
# Flip Analysis
# ==============================================================================

def _to_bool_series(col: pd.Series) -> pd.Series:
    """Convert column to boolean series."""
    if col.dtype == object:
        return col.map({'True': True, 'False': False, True: True, False: False}).astype(bool)
    return col.astype(bool)


def count_wrong_to_right(df: pd.DataFrame, metric: str = 'answer') -> int:
    """Count examples that flipped from wrong (baseline) to right (intervention).
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        
    Returns:
        Number of wrong→right flips
    """
    base_col = f'baseline_{metric}_correct'
    intv_col = f'intv_{metric}_correct'
    
    if base_col not in df.columns or intv_col not in df.columns:
        return 0
    
    base_correct = _to_bool_series(df[base_col])
    intv_correct = _to_bool_series(df[intv_col])
    
    return int((~base_correct & intv_correct).sum())


def count_right_to_wrong(df: pd.DataFrame, metric: str = 'answer') -> int:
    """Count examples that flipped from right (baseline) to wrong (intervention).
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        
    Returns:
        Number of right→wrong flips
    """
    base_col = f'baseline_{metric}_correct'
    intv_col = f'intv_{metric}_correct'
    
    if base_col not in df.columns or intv_col not in df.columns:
        return 0
    
    base_correct = _to_bool_series(df[base_col])
    intv_correct = _to_bool_series(df[intv_col])
    
    return int((base_correct & ~intv_correct).sum())


def compute_net_gain(df: pd.DataFrame, metric: str = 'answer') -> int:
    """Compute net gain (wrong→right - right→wrong).
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        
    Returns:
        Net gain (positive = more fixes than breaks)
    """
    return count_wrong_to_right(df, metric) - count_right_to_wrong(df, metric)


def compute_flip_rate(df: pd.DataFrame, metric: str = 'answer') -> float:
    """Compute flip rate (fraction of examples that changed).
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        
    Returns:
        Flip rate as fraction
    """
    base_col = f'baseline_{metric}_correct'
    intv_col = f'intv_{metric}_correct'
    
    if base_col not in df.columns or intv_col not in df.columns:
        return np.nan
    
    base_correct = _to_bool_series(df[base_col])
    intv_correct = _to_bool_series(df[intv_col])
    
    flipped = (base_correct != intv_correct)
    return flipped.mean()


def compute_flip_counts(df: pd.DataFrame, metric: str = 'answer') -> Dict[str, int]:
    """Compute all flip-related counts.
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        
    Returns:
        Dictionary with flip counts
    """
    return {
        'wrong_to_right': count_wrong_to_right(df, metric),
        'right_to_wrong': count_right_to_wrong(df, metric),
        'net_gain': compute_net_gain(df, metric),
        'total_flips': count_wrong_to_right(df, metric) + count_right_to_wrong(df, metric)
    }


# ==============================================================================
# Aggregation Functions
# ==============================================================================

def aggregate_by_config(df: pd.DataFrame,
                        group_cols: List[str] = ['mode', 'layer', 'alpha'],
                        metric: str = 'answer') -> pd.DataFrame:
    """Aggregate metrics by experiment configuration.
    
    Args:
        df: DataFrame with experiment results
        group_cols: Columns to group by
        metric: 'answer' or 'reasoning'
        
    Returns:
        DataFrame with aggregated metrics per configuration
    """
    # Filter to only include existing group columns
    group_cols = [c for c in group_cols if c in df.columns]
    
    if not group_cols:
        return pd.DataFrame()
    
    results = []
    
    for name, group in df.groupby(group_cols):
        if not isinstance(name, tuple):
            name = (name,)
        
        row = dict(zip(group_cols, name))
        row['n'] = len(group)
        row['acc_base'] = compute_baseline_accuracy(group, metric)
        row['acc_intv'] = compute_intervention_accuracy(group, metric)
        row['delta'] = compute_delta_accuracy(group, metric)
        
        flip_counts = compute_flip_counts(group, metric)
        row.update(flip_counts)
        
        results.append(row)
    
    return pd.DataFrame(results)


def aggregate_by_mode_alpha(df: pd.DataFrame, metric: str = 'answer') -> pd.DataFrame:
    """Aggregate metrics by mode and alpha.
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        
    Returns:
        DataFrame with aggregated metrics
    """
    return aggregate_by_config(df, ['mode', 'alpha'], metric)


def aggregate_by_layer(df: pd.DataFrame, metric: str = 'answer') -> pd.DataFrame:
    """Aggregate metrics by layer.
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        
    Returns:
        DataFrame with aggregated metrics
    """
    return aggregate_by_config(df, ['layer'], metric)


def aggregate_by_mode_layer_alpha(df: pd.DataFrame, metric: str = 'answer') -> pd.DataFrame:
    """Aggregate metrics by mode, layer, and alpha.
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        
    Returns:
        DataFrame with aggregated metrics
    """
    return aggregate_by_config(df, ['mode', 'layer', 'alpha'], metric)


# ==============================================================================
# Summary Statistics
# ==============================================================================

def compute_summary(df: pd.DataFrame) -> Dict:
    """Compute comprehensive summary statistics.
    
    Args:
        df: DataFrame with experiment results
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'n_total': len(df),
        'n_unique_examples': df['example_id'].nunique() if 'example_id' in df.columns else None,
    }
    
    # Baseline accuracy
    summary['baseline_accuracy_answer'] = compute_baseline_accuracy(df, 'answer')
    summary['baseline_accuracy_reasoning'] = compute_baseline_accuracy(df, 'reasoning')
    
    # Per-mode summaries
    if 'mode' in df.columns:
        for mode in df['mode'].unique():
            mode_df = df[df['mode'] == mode]
            prefix = f'{mode}_'
            summary[f'{prefix}n'] = len(mode_df)
            summary[f'{prefix}delta_answer'] = compute_delta_accuracy(mode_df, 'answer')
            summary[f'{prefix}delta_reasoning'] = compute_delta_accuracy(mode_df, 'reasoning')
            
            flip_counts = compute_flip_counts(mode_df, 'answer')
            for key, value in flip_counts.items():
                summary[f'{prefix}{key}'] = value
    
    return summary


def compute_per_alpha_summary(df: pd.DataFrame, mode: str = 'add') -> pd.DataFrame:
    """Compute summary statistics per alpha value for a specific mode.
    
    Args:
        df: DataFrame with experiment results
        mode: Intervention mode to analyze
        
    Returns:
        DataFrame with per-alpha statistics
    """
    if 'mode' in df.columns:
        df = df[df['mode'] == mode]
    
    if 'alpha' not in df.columns:
        return pd.DataFrame()
    
    results = []
    
    for alpha in sorted(df['alpha'].unique()):
        alpha_df = df[df['alpha'] == alpha]
        
        row = {
            'alpha': alpha,
            'n': len(alpha_df),
            'acc_base_answer': compute_baseline_accuracy(alpha_df, 'answer'),
            'acc_intv_answer': compute_intervention_accuracy(alpha_df, 'answer'),
            'delta_answer': compute_delta_accuracy(alpha_df, 'answer'),
            'acc_base_reasoning': compute_baseline_accuracy(alpha_df, 'reasoning'),
            'acc_intv_reasoning': compute_intervention_accuracy(alpha_df, 'reasoning'),
            'delta_reasoning': compute_delta_accuracy(alpha_df, 'reasoning'),
        }
        
        flip_counts = compute_flip_counts(alpha_df, 'answer')
        row.update({f'answer_{k}': v for k, v in flip_counts.items()})
        
        results.append(row)
    
    return pd.DataFrame(results)


def compute_per_layer_summary(df: pd.DataFrame, mode: str = 'add') -> pd.DataFrame:
    """Compute summary statistics per layer for a specific mode.
    
    Args:
        df: DataFrame with experiment results
        mode: Intervention mode to analyze
        
    Returns:
        DataFrame with per-layer statistics
    """
    if 'mode' in df.columns:
        df = df[df['mode'] == mode]
    
    if 'layer' not in df.columns:
        return pd.DataFrame()
    
    results = []
    
    for layer in sorted(df['layer'].unique()):
        layer_df = df[df['layer'] == layer]
        
        row = {
            'layer': layer,
            'n': len(layer_df),
            'acc_base_answer': compute_baseline_accuracy(layer_df, 'answer'),
            'acc_intv_answer': compute_intervention_accuracy(layer_df, 'answer'),
            'delta_answer': compute_delta_accuracy(layer_df, 'answer'),
            'acc_base_reasoning': compute_baseline_accuracy(layer_df, 'reasoning'),
            'acc_intv_reasoning': compute_intervention_accuracy(layer_df, 'reasoning'),
            'delta_reasoning': compute_delta_accuracy(layer_df, 'reasoning'),
        }
        
        flip_counts = compute_flip_counts(layer_df, 'answer')
        row.update({f'answer_{k}': v for k, v in flip_counts.items()})
        
        results.append(row)
    
    return pd.DataFrame(results)


# ==============================================================================
# Comparison Metrics
# ==============================================================================

def compare_modes(df: pd.DataFrame, 
                  mode1: str = 'add', 
                  mode2: str = 'random',
                  metric: str = 'answer') -> pd.DataFrame:
    """Compare metrics between two modes (e.g., add vs random).
    
    Args:
        df: DataFrame with experiment results
        mode1: First mode (typically 'add')
        mode2: Second mode (typically 'random')
        metric: 'answer' or 'reasoning'
        
    Returns:
        DataFrame with comparison metrics per alpha/layer
    """
    if 'mode' not in df.columns:
        return pd.DataFrame()
    
    mode1_df = df[df['mode'] == mode1]
    mode2_df = df[df['mode'] == mode2]
    
    results = []
    
    # Group by alpha if available
    group_cols = []
    if 'alpha' in df.columns:
        group_cols.append('alpha')
    if 'layer' in df.columns:
        group_cols.append('layer')
    
    if not group_cols:
        return pd.DataFrame()
    
    for name, group1 in mode1_df.groupby(group_cols):
        if not isinstance(name, tuple):
            name = (name,)
        
        # Find matching group in mode2
        mask = pd.Series([True] * len(mode2_df))
        for col, val in zip(group_cols, name):
            mask &= (mode2_df[col] == val)
        
        group2 = mode2_df[mask]
        
        if len(group2) == 0:
            continue
        
        row = dict(zip(group_cols, name))
        row[f'{mode1}_delta'] = compute_delta_accuracy(group1, metric)
        row[f'{mode2}_delta'] = compute_delta_accuracy(group2, metric)
        row['difference'] = row[f'{mode1}_delta'] - row[f'{mode2}_delta']
        row[f'{mode1}_n'] = len(group1)
        row[f'{mode2}_n'] = len(group2)
        
        results.append(row)
    
    return pd.DataFrame(results)


def compute_specificity(df: pd.DataFrame, metric: str = 'answer') -> float:
    """Compute specificity: difference between add and random mode effects.
    
    Specificity = mean(delta_add) - mean(delta_random)
    Positive value indicates learned direction is more effective than random.
    
    Args:
        df: DataFrame with experiment results
        metric: 'answer' or 'reasoning'
        
    Returns:
        Specificity score (positive = add is better)
    """
    if 'mode' not in df.columns:
        return np.nan
    
    add_df = df[df['mode'] == 'add']
    random_df = df[df['mode'] == 'random']
    
    if len(add_df) == 0 or len(random_df) == 0:
        return np.nan
    
    add_delta = compute_delta_accuracy(add_df, metric)
    random_delta = compute_delta_accuracy(random_df, metric)
    
    return add_delta - random_delta


# ==============================================================================
# Rescue/Lesion Metrics
# ==============================================================================

def compute_lesion_effect(df: pd.DataFrame, metric: str = 'answer') -> Dict:
    """Compute effect of lesion intervention.
    
    Args:
        df: DataFrame with lesion experiment results
        metric: 'answer' or 'reasoning'
        
    Returns:
        Dictionary with lesion effect metrics
    """
    if 'mode' in df.columns:
        df = df[df['mode'] == 'lesion']
    
    return {
        'baseline_acc': compute_baseline_accuracy(df, metric),
        'lesion_acc': compute_intervention_accuracy(df, metric),
        'delta': compute_delta_accuracy(df, metric),
        'n': len(df)
    }


def compute_rescue_recovery(df: pd.DataFrame, metric: str = 'answer') -> Dict:
    """Compute rescue recovery metrics.
    
    For rescue experiments, we compare:
    - Baseline (no intervention)
    - Lesion effect
    - Rescue (restored)
    
    Args:
        df: DataFrame with rescue experiment results
        metric: 'answer' or 'reasoning'
        
    Returns:
        Dictionary with rescue metrics
    """
    if 'mode' in df.columns:
        df = df[df['mode'] == 'rescue']
    
    return {
        'baseline_acc': compute_baseline_accuracy(df, metric),
        'rescue_acc': compute_intervention_accuracy(df, metric),
        'delta': compute_delta_accuracy(df, metric),
        'n': len(df)
    }


# ==============================================================================
# Locality Comparison
# ==============================================================================

def compare_localities(df: pd.DataFrame,
                       locality1: str = 'cot',
                       locality2: str = 'answer',
                       metric: str = 'answer') -> pd.DataFrame:
    """Compare metrics between two locality types.
    
    Args:
        df: DataFrame with experiment results from multiple localities
        locality1: First locality type
        locality2: Second locality type
        metric: 'answer' or 'reasoning'
        
    Returns:
        DataFrame with comparison per mode/alpha/layer
    """
    if 'locality' not in df.columns:
        return pd.DataFrame()
    
    loc1_df = df[df['locality'] == locality1]
    loc2_df = df[df['locality'] == locality2]
    
    if len(loc1_df) == 0 or len(loc2_df) == 0:
        print(f"Warning: One or both localities have no data")
        print(f"  {locality1}: {len(loc1_df)} rows")
        print(f"  {locality2}: {len(loc2_df)} rows")
        return pd.DataFrame()
    
    results = []
    
    # Group by mode, alpha, layer if available
    group_cols = []
    if 'mode' in df.columns:
        group_cols.append('mode')
    if 'alpha' in df.columns:
        group_cols.append('alpha')
    if 'layer' in df.columns:
        group_cols.append('layer')
    
    if not group_cols:
        # Simple comparison
        row = {
            f'{locality1}_delta': compute_delta_accuracy(loc1_df, metric),
            f'{locality2}_delta': compute_delta_accuracy(loc2_df, metric),
        }
        row['difference'] = row[f'{locality1}_delta'] - row[f'{locality2}_delta']
        return pd.DataFrame([row])
    
    for name, group1 in loc1_df.groupby(group_cols):
        if not isinstance(name, tuple):
            name = (name,)
        
        # Find matching group in locality2
        mask = pd.Series([True] * len(loc2_df))
        for col, val in zip(group_cols, name):
            mask &= (loc2_df[col] == val)
        
        group2 = loc2_df[mask]
        
        if len(group2) == 0:
            continue
        
        row = dict(zip(group_cols, name))
        row[f'{locality1}_delta'] = compute_delta_accuracy(group1, metric)
        row[f'{locality2}_delta'] = compute_delta_accuracy(group2, metric)
        row['difference'] = row[f'{locality1}_delta'] - row[f'{locality2}_delta']
        row[f'{locality1}_n'] = len(group1)
        row[f'{locality2}_n'] = len(group2)
        
        results.append(row)
    
    return pd.DataFrame(results)


# ==============================================================================
# Table Generation
# ==============================================================================

def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a publication-ready summary table.
    
    Args:
        df: DataFrame with experiment results
        
    Returns:
        Formatted summary table
    """
    # Aggregate by mode and alpha
    agg_df = aggregate_by_mode_alpha(df, 'answer')
    
    if len(agg_df) == 0:
        return pd.DataFrame()
    
    # Format columns
    agg_df['acc_base'] = agg_df['acc_base'].apply(lambda x: f"{x:.1%}")
    agg_df['acc_intv'] = agg_df['acc_intv'].apply(lambda x: f"{x:.1%}")
    agg_df['delta'] = agg_df['delta'].apply(lambda x: f"{x:+.1%}")
    
    return agg_df


def generate_flip_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a flip analysis table.
    
    Args:
        df: DataFrame with experiment results
        
    Returns:
        Formatted flip analysis table
    """
    results = []
    
    for mode in df['mode'].unique() if 'mode' in df.columns else [None]:
        mode_df = df if mode is None else df[df['mode'] == mode]
        
        flip_counts = compute_flip_counts(mode_df, 'answer')
        flip_counts['mode'] = mode or 'all'
        flip_counts['flip_rate'] = compute_flip_rate(mode_df, 'answer')
        
        results.append(flip_counts)
    
    return pd.DataFrame(results)
