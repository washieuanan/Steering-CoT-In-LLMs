"""
Data Loading Utilities for Intervention Analysis

This module provides functions for loading and preprocessing Phase B intervention
results from .csv.gz files.
"""

import gzip
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


# ==============================================================================
# Core Loading Functions
# ==============================================================================

def load_csv_gz(path: Path) -> pd.DataFrame:
    """Load a gzipped CSV file.
    
    Args:
        path: Path to the .csv.gz file
        
    Returns:
        DataFrame with the CSV contents
    """
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return pd.read_csv(f)


def parse_filename(filename: str) -> Dict[str, Union[str, float, int]]:
    """Parse experiment parameters from filename.
    
    Filename patterns:
        - paired_add_L25_A1.0_cot.csv.gz
        - paired_random_L26_A0.5_cot.csv.gz
        - lesion_lesion_L27_G1.0_cot.csv.gz
        - rescue_rescue_L25_G1.0_B1.0_cot.csv.gz
    
    Args:
        filename: Name of the CSV file
        
    Returns:
        Dictionary with extracted parameters
    """
    params = {}
    
    # Remove extension
    name = filename.replace('.csv.gz', '').replace('.csv', '')
    
    # Extract experiment type and mode
    if name.startswith('paired_'):
        parts = name.split('_')
        params['experiment_type'] = 'paired'
        params['mode'] = parts[1]  # add, random
    elif name.startswith('lesion_'):
        params['experiment_type'] = 'lesion'
        params['mode'] = 'lesion'
    elif name.startswith('rescue_'):
        params['experiment_type'] = 'rescue'
        params['mode'] = 'rescue'
    else:
        params['experiment_type'] = 'unknown'
        params['mode'] = 'unknown'
    
    # Extract layer
    layer_match = re.search(r'L(\d+)', name)
    if layer_match:
        params['layer'] = int(layer_match.group(1))
    
    # Extract alpha
    alpha_match = re.search(r'A([\d.]+)', name)
    if alpha_match:
        params['alpha'] = float(alpha_match.group(1))
    
    # Extract gamma (for lesion/rescue)
    gamma_match = re.search(r'G([\d.]+)', name)
    if gamma_match:
        params['gamma'] = float(gamma_match.group(1))
    
    # Extract beta (for rescue)
    beta_match = re.search(r'B([\d.]+)', name)
    if beta_match:
        params['beta'] = float(beta_match.group(1))
    
    # Extract locality (last part before extension)
    locality_match = re.search(r'_(cot|answer|full)\.csv', filename)
    if locality_match:
        params['locality'] = locality_match.group(1)
    
    return params


def load_run_config(dataset_dir: Path) -> Optional[Dict]:
    """Load run_config.json from a dataset directory.
    
    Args:
        dataset_dir: Path to the dataset directory
        
    Returns:
        Dictionary with run configuration or None
    """
    config_path = dataset_dir / 'run_config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def load_single_run(csv_path: Path, include_raw_text: bool = False) -> pd.DataFrame:
    """Load a single experiment run from a CSV file.
    
    Args:
        csv_path: Path to the .csv.gz file
        include_raw_text: Whether to include raw text columns
        
    Returns:
        DataFrame with experiment results and metadata
    """
    df = load_csv_gz(csv_path)
    
    # Parse filename for parameters
    params = parse_filename(csv_path.name)
    
    # Add parameters as columns
    for key, value in params.items():
        if key not in df.columns:
            df[key] = value
    
    # Add source file info
    df['source_file'] = csv_path.name
    
    # Drop raw text columns if not needed (they're large)
    if not include_raw_text:
        text_cols = ['baseline_text_raw', 'intv_text_raw']
        df = df.drop(columns=[c for c in text_cols if c in df.columns], errors='ignore')
    
    return df


# ==============================================================================
# Directory Loading Functions
# ==============================================================================

def load_runs_directory(runs_dir: Path, include_raw_text: bool = False) -> pd.DataFrame:
    """Load all CSV files from a runs/ directory.
    
    Args:
        runs_dir: Path to the runs/ directory
        include_raw_text: Whether to include raw text columns
        
    Returns:
        Concatenated DataFrame with all runs
    """
    all_dfs = []
    
    for csv_file in sorted(runs_dir.glob('*.csv.gz')):
        try:
            df = load_single_run(csv_file, include_raw_text)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not all_dfs:
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)


def load_dataset_results(dataset_dir: Path, include_raw_text: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """Load all results for a single dataset.
    
    Args:
        dataset_dir: Path to the dataset directory (e.g., vm_results/.../mmlu_pro/)
        include_raw_text: Whether to include raw text columns
        
    Returns:
        Tuple of (DataFrame with all runs, run_config dict)
    """
    runs_dir = dataset_dir / 'runs'
    
    if not runs_dir.exists():
        return pd.DataFrame(), {}
    
    df = load_runs_directory(runs_dir, include_raw_text)
    config = load_run_config(dataset_dir) or {}
    
    # Add dataset name
    df['dataset'] = dataset_dir.name
    
    return df, config


def load_experiment_results(experiment_dir: Path, include_raw_text: bool = False) -> pd.DataFrame:
    """Load all results from an experiment directory.
    
    An experiment directory contains subdirectories for each dataset.
    
    Args:
        experiment_dir: Path to the experiment directory 
                       (e.g., vm_results/Qwen2.5-7B-Instruct__cot_locality_20251201/)
        include_raw_text: Whether to include raw text columns
        
    Returns:
        DataFrame with all runs across datasets
    """
    all_dfs = []
    
    for dataset_dir in sorted(experiment_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
            continue
        
        df, config = load_dataset_results(dataset_dir, include_raw_text)
        
        if len(df) > 0:
            # Extract model name from experiment dir name
            parts = experiment_dir.name.split('__')
            df['model'] = parts[0] if parts else 'unknown'
            df['experiment_name'] = experiment_dir.name
            all_dfs.append(df)
    
    if not all_dfs:
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)


def load_all_runs(results_root: Union[str, Path], include_raw_text: bool = False) -> pd.DataFrame:
    """Load all experiment results from a results root directory.
    
    Args:
        results_root: Path to results directory (e.g., 'vm_results' or 'results/phase_b')
        include_raw_text: Whether to include raw text columns
        
    Returns:
        DataFrame with all runs across all experiments
    """
    results_root = Path(results_root)
    
    if not results_root.exists():
        raise ValueError(f"Results directory does not exist: {results_root}")
    
    all_dfs = []
    
    for experiment_dir in sorted(results_root.iterdir()):
        if not experiment_dir.is_dir() or experiment_dir.name.startswith('.'):
            continue
        
        df = load_experiment_results(experiment_dir, include_raw_text)
        
        if len(df) > 0:
            all_dfs.append(df)
    
    if not all_dfs:
        raise ValueError(f"No experiment data found in {results_root}")
    
    return pd.concat(all_dfs, ignore_index=True)


# ==============================================================================
# Filtering Functions
# ==============================================================================

def filter_by_locality(df: pd.DataFrame, locality: str) -> pd.DataFrame:
    """Filter DataFrame by locality type.
    
    Args:
        df: Input DataFrame
        locality: 'cot', 'answer', or 'full'
        
    Returns:
        Filtered DataFrame
    """
    if 'locality' in df.columns:
        return df[df['locality'] == locality].copy()
    return df


def filter_by_mode(df: pd.DataFrame, modes: List[str]) -> pd.DataFrame:
    """Filter DataFrame by intervention mode(s).
    
    Args:
        df: Input DataFrame
        modes: List of modes to include (e.g., ['add', 'random'])
        
    Returns:
        Filtered DataFrame
    """
    if 'mode' in df.columns:
        return df[df['mode'].isin(modes)].copy()
    return df


def filter_by_layer(df: pd.DataFrame, layers: List[int]) -> pd.DataFrame:
    """Filter DataFrame by target layer(s).
    
    Args:
        df: Input DataFrame
        layers: List of layer numbers to include
        
    Returns:
        Filtered DataFrame
    """
    if 'layer' in df.columns:
        return df[df['layer'].isin(layers)].copy()
    return df


def filter_by_alpha(df: pd.DataFrame, 
                    alpha_min: Optional[float] = None,
                    alpha_max: Optional[float] = None) -> pd.DataFrame:
    """Filter DataFrame by alpha range.
    
    Args:
        df: Input DataFrame
        alpha_min: Minimum alpha value (inclusive)
        alpha_max: Maximum alpha value (inclusive)
        
    Returns:
        Filtered DataFrame
    """
    if 'alpha' not in df.columns:
        return df
    
    mask = pd.Series([True] * len(df))
    
    if alpha_min is not None:
        mask &= df['alpha'] >= alpha_min
    if alpha_max is not None:
        mask &= df['alpha'] <= alpha_max
    
    return df[mask].copy()


def get_paired_experiments(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only paired (add/random) experiments.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Filtered DataFrame
    """
    return filter_by_mode(df, ['add', 'random'])


def get_lesion_experiments(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only lesion experiments.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Filtered DataFrame
    """
    return filter_by_mode(df, ['lesion'])


def get_rescue_experiments(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only rescue experiments.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Filtered DataFrame
    """
    return filter_by_mode(df, ['rescue'])


# ==============================================================================
# Pre-computed Summary Loading
# ==============================================================================

def load_grid_csv(dataset_dir: Path) -> Optional[pd.DataFrame]:
    """Load pre-computed grid.csv summary.
    
    Args:
        dataset_dir: Path to the dataset directory
        
    Returns:
        DataFrame with grid summary or None
    """
    grid_path = dataset_dir / 'grid.csv'
    if grid_path.exists():
        return pd.read_csv(grid_path)
    return None


def load_rescue_summary(dataset_dir: Path) -> Optional[pd.DataFrame]:
    """Load pre-computed rescue_summary.csv.
    
    Args:
        dataset_dir: Path to the dataset directory
        
    Returns:
        DataFrame with rescue summary or None
    """
    rescue_path = dataset_dir / 'rescue_summary.csv'
    if rescue_path.exists():
        return pd.read_csv(rescue_path)
    return None


def load_all_grid_summaries(results_root: Union[str, Path]) -> pd.DataFrame:
    """Load all pre-computed grid.csv files.
    
    Args:
        results_root: Path to results directory
        
    Returns:
        Concatenated DataFrame with all grid summaries
    """
    results_root = Path(results_root)
    all_dfs = []
    
    for experiment_dir in sorted(results_root.iterdir()):
        if not experiment_dir.is_dir() or experiment_dir.name.startswith('.'):
            continue
        
        parts = experiment_dir.name.split('__')
        model_name = parts[0] if parts else 'unknown'
        
        for dataset_dir in sorted(experiment_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            
            grid_df = load_grid_csv(dataset_dir)
            if grid_df is not None:
                grid_df['model'] = model_name
                grid_df['dataset'] = dataset_dir.name
                grid_df['experiment_name'] = experiment_dir.name
                all_dfs.append(grid_df)
    
    if not all_dfs:
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)


def load_all_rescue_summaries(results_root: Union[str, Path]) -> pd.DataFrame:
    """Load all pre-computed rescue_summary.csv files.
    
    Args:
        results_root: Path to results directory
        
    Returns:
        Concatenated DataFrame with all rescue summaries
    """
    results_root = Path(results_root)
    all_dfs = []
    
    for experiment_dir in sorted(results_root.iterdir()):
        if not experiment_dir.is_dir() or experiment_dir.name.startswith('.'):
            continue
        
        parts = experiment_dir.name.split('__')
        model_name = parts[0] if parts else 'unknown'
        
        for dataset_dir in sorted(experiment_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            
            rescue_df = load_rescue_summary(dataset_dir)
            if rescue_df is not None:
                rescue_df['model'] = model_name
                rescue_df['dataset'] = dataset_dir.name
                rescue_df['experiment_name'] = experiment_dir.name
                all_dfs.append(rescue_df)
    
    if not all_dfs:
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)


# ==============================================================================
# Utility Functions
# ==============================================================================

def get_unique_values(df: pd.DataFrame, column: str) -> List:
    """Get unique values for a column.
    
    Args:
        df: Input DataFrame
        column: Column name
        
    Returns:
        List of unique values
    """
    if column in df.columns:
        return sorted(df[column].unique().tolist())
    return []


def describe_data(df: pd.DataFrame) -> Dict:
    """Get a summary description of the loaded data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with data summary
    """
    return {
        'n_rows': len(df),
        'n_examples': df['example_id'].nunique() if 'example_id' in df.columns else None,
        'models': get_unique_values(df, 'model'),
        'datasets': get_unique_values(df, 'dataset'),
        'modes': get_unique_values(df, 'mode'),
        'layers': get_unique_values(df, 'layer'),
        'alphas': get_unique_values(df, 'alpha'),
        'localities': get_unique_values(df, 'locality'),
        'columns': df.columns.tolist()
    }


def print_data_summary(df: pd.DataFrame):
    """Print a formatted summary of the loaded data.
    
    Args:
        df: Input DataFrame
    """
    summary = describe_data(df)
    
    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Total rows: {summary['n_rows']}")
    print(f"Unique examples: {summary['n_examples']}")
    print(f"Models: {summary['models']}")
    print(f"Datasets: {summary['datasets']}")
    print(f"Modes: {summary['modes']}")
    print(f"Layers: {summary['layers']}")
    print(f"Alpha values: {summary['alphas']}")
    print(f"Localities: {summary['localities']}")
    print("=" * 60)
