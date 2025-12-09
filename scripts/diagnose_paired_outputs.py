#!/usr/bin/env python3
"""
Diagnostic script to verify paired baseline and intervention outputs actually differ.

This implements Step 1 from the diagnostic plan: prove the evaluator isn't reading
the same output twice.

Usage:
    python scripts/diagnose_paired_outputs.py <path_to_paired_csv_gz>
    
Example:
    python scripts/diagnose_paired_outputs.py results/phase_b/Llama-3.1-8B-Instruct__intv_20231111/arc/runs/paired_add_L31_A-4.0_locall.csv.gz
"""

import argparse
import gzip
import sys
from pathlib import Path

import pandas as pd


def analyze_paired_outputs(csv_path: Path) -> None:
    """Analyze a paired CSV file to check if outputs actually differ."""
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC: Paired Output Analysis")
    print(f"{'='*60}")
    print(f"File: {csv_path}")
    
    # Load data
    try:
        if csv_path.suffix == '.gz':
            with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f)
        else:
            df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"\nERROR: Failed to load file: {e}")
        return
    
    print(f"\nTotal examples: {len(df)}")
    
    # Check required columns
    required_cols = ['baseline_text_raw', 'intv_text_raw', 'baseline_pred', 'intv_pred',
                     'baseline_correct', 'intv_correct']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\nERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    print(f"\n{'='*60}")
    print(f"TEXT-LEVEL COMPARISON (Raw)")
    print(f"{'='*60}")
    
    # Check if raw texts are identical
    identical_raw = (df['baseline_text_raw'] == df['intv_text_raw'])
    n_identical_raw = identical_raw.sum()
    n_different_raw = (~identical_raw).sum()
    
    print(f"Identical raw texts:  {n_identical_raw:4d} ({100*n_identical_raw/len(df):5.1f}%)")
    print(f"Different raw texts:  {n_different_raw:4d} ({100*n_different_raw/len(df):5.1f}%)")
    
    if n_identical_raw == len(df):
        print("\n⚠️  WARNING: ALL raw texts are identical!")
        print("   This suggests the intervention is not changing the outputs.")
    
    print(f"\n{'='*60}")
    print(f"PREDICTION-LEVEL COMPARISON")
    print(f"{'='*60}")
    
    # Check if predictions are identical
    identical_pred = (df['baseline_pred'] == df['intv_pred'])
    n_identical_pred = identical_pred.sum()
    n_different_pred = (~identical_pred).sum()
    
    print(f"Identical predictions: {n_identical_pred:4d} ({100*n_identical_pred/len(df):5.1f}%)")
    print(f"Different predictions: {n_different_pred:4d} ({100*n_different_pred/len(df):5.1f}%)")
    
    if n_identical_pred == len(df):
        print("\n⚠️  WARNING: ALL predictions are identical!")
        print("   This suggests the intervention is not affecting model outputs.")
    
    print(f"\n{'='*60}")
    print(f"CORRECTNESS-LEVEL COMPARISON")
    print(f"{'='*60}")
    
    # Accuracy comparison
    acc_base = df['baseline_correct'].mean()
    acc_intv = df['intv_correct'].mean()
    
    print(f"Baseline accuracy:     {acc_base:.3f}")
    print(f"Intervention accuracy: {acc_intv:.3f}")
    print(f"Delta accuracy:        {acc_intv - acc_base:+.3f}")
    
    # Flip analysis
    wrong_to_right = ((df['baseline_correct'] == 0) & (df['intv_correct'] == 1)).sum()
    right_to_wrong = ((df['baseline_correct'] == 1) & (df['intv_correct'] == 0)).sum()
    
    print(f"\nFlips:")
    print(f"  Wrong → Right: {wrong_to_right:4d}")
    print(f"  Right → Wrong: {right_to_wrong:4d}")
    print(f"  Net gain:      {wrong_to_right - right_to_wrong:+4d}")
    
    if wrong_to_right == 0 and right_to_wrong == 0:
        print("\n⚠️  WARNING: ZERO flips detected!")
        print("   This is a strong signal that interventions are not working.")
    
    print(f"\n{'='*60}")
    print(f"SAMPLE COMPARISONS (First 3 examples)")
    print(f"{'='*60}")
    
    # Show first 3 examples with differences
    if n_different_raw > 0:
        diff_indices = df[~identical_raw].head(3).index
        print("\nExamples with DIFFERENT outputs:\n")
    else:
        diff_indices = df.head(3).index
        print("\nFirst 3 examples (all identical):\n")
    
    for i, idx in enumerate(diff_indices):
        row = df.loc[idx]
        print(f"Example {i+1} (row {idx}):")
        print(f"  Gold: {row.get('gold', 'N/A')}")
        print(f"  Baseline pred: {row['baseline_pred']}")
        print(f"  Intv pred:     {row['intv_pred']}")
        print(f"  Baseline correct: {row['baseline_correct']}")
        print(f"  Intv correct:     {row['intv_correct']}")
        
        # Show text difference
        base_text = str(row['baseline_text_raw'])[:100]
        intv_text = str(row['intv_text_raw'])[:100]
        
        if base_text == intv_text:
            print(f"  Texts: IDENTICAL (first 100 chars)")
            print(f"    [{base_text}...]")
        else:
            print(f"  Baseline text (first 100 chars): [{base_text}...]")
            print(f"  Intv text (first 100 chars):     [{intv_text}...]")
        print()
    
    print(f"{'='*60}")
    print(f"CHARACTER-LEVEL DIFF ANALYSIS")
    print(f"{'='*60}")
    
    # Character-level difference stats
    char_diffs = []
    for _, row in df.iterrows():
        base = str(row['baseline_text_raw'])
        intv = str(row['intv_text_raw'])
        
        # Simple edit distance approximation (count different chars)
        min_len = min(len(base), len(intv))
        n_diff = sum(b != i for b, i in zip(base[:min_len], intv[:min_len]))
        n_diff += abs(len(base) - len(intv))  # Add length difference
        
        char_diffs.append(n_diff)
    
    df['char_diff'] = char_diffs
    
    n_zero_diff = (df['char_diff'] == 0).sum()
    print(f"\nExamples with 0 character differences: {n_zero_diff} ({100*n_zero_diff/len(df):.1f}%)")
    print(f"Mean character difference: {df['char_diff'].mean():.1f}")
    print(f"Median character difference: {df['char_diff'].median():.1f}")
    print(f"Max character difference: {df['char_diff'].max()}")
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    
    # Determine root cause
    if n_identical_raw == len(df):
        print("\n🔴 ISSUE CONFIRMED: All outputs are IDENTICAL")
        print("\nPossible causes:")
        print("  1. Hook is not firing (check hook registration)")
        print("  2. Hook fires but doesn't modify hidden states")
        print("  3. Delta being added is zero or near-zero")
        print("  4. Intervention is on wrong tensor (not feeding logits)")
        print("  5. KV-cache is bypassing the intervened path")
        print("\nNext steps:")
        print("  - Run with DEBUG_HOOKS=1 to see hook firing logs")
        print("  - Check if delta norm > 0 in hook diagnostics")
        print("  - Verify KL divergence is non-zero")
        print("  - Check locality masks are correct")
    
    elif n_identical_pred == len(df):
        print("\n🟡 PARTIAL ISSUE: Texts differ but predictions are identical")
        print("\nPossible causes:")
        print("  1. Interventions change non-critical parts of output")
        print("  2. Parsing is extracting same answer despite text changes")
        print("  3. Small text changes don't affect final answer")
        print("\nThis is less critical but still investigate if delta_acc == 0")
    
    elif wrong_to_right == 0 and right_to_wrong == 0:
        print("\n🟡 PARTIAL ISSUE: Predictions differ but no correctness flips")
        print("\nPossible causes:")
        print("  1. Interventions change wrong answers to other wrong answers")
        print("  2. Sample size too small to see flips")
        print("  3. Intervention strength too weak")
        print("\nCheck if increasing |alpha| or changing layers helps")
    
    else:
        print("\n🟢 OUTPUTS ARE DIFFERENT: Intervention is working!")
        print(f"\n   {n_different_raw} examples have different texts")
        print(f"   {n_different_pred} examples have different predictions")
        print(f"   {wrong_to_right + right_to_wrong} examples flipped correctness")
        print("\nThe intervention system appears to be functioning.")
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose paired baseline vs intervention outputs"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to paired CSV file (can be .csv or .csv.gz)"
    )
    
    args = parser.parse_args()
    csv_path = Path(args.csv_path)
    
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)
    
    analyze_paired_outputs(csv_path)


if __name__ == "__main__":
    main()
