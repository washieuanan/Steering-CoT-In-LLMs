"""
Verify subspace directions from Phase A screening.

This script checks that:
1. Direction vectors/subspaces are non-zero
2. Norms are reasonable (not too small/large)
3. For subspaces, check orthonormality
4. Display per-layer statistics

Usage:
    python -m scripts.check_subspace_nz \
        --phase_a_run results/phase_a/MODEL__TAG/arc
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Check subspace directions")
    parser.add_argument("--phase_a_run", type=str, required=True,
                        help="Path to Phase A output dir")
    parser.add_argument("--check_orthonormal", action="store_true",
                        help="Check if subspace columns are orthonormal")
    return parser.parse_args()


def check_orthonormality(U: np.ndarray, tol: float = 1e-3) -> tuple[bool, float]:
    """
    Check if columns of U are orthonormal.
    
    Args:
        U: Matrix [R, H] with R rows (directions)
        tol: Tolerance for orthonormality check
    
    Returns:
        Tuple of (is_orthonormal, max_deviation)
    """
    # Compute Gram matrix: G = U @ U.T
    G = U @ U.T  # [R, R]
    
    # Should be identity matrix
    I = np.eye(G.shape[0])
    deviation = np.abs(G - I)
    max_dev = np.max(deviation)
    
    is_orthonormal = max_dev < tol
    
    return is_orthonormal, max_dev


def analyze_direction(layer_idx: int, arr: np.ndarray, check_ortho: bool = False):
    """Analyze a single direction vector or subspace."""
    print(f"\n{'='*60}")
    print(f"Layer {layer_idx}")
    print(f"{'='*60}")
    
    results = {
        'layer': layer_idx,
        'type': None,
        'shape': tuple(arr.shape),
        'norm': None,
        'is_zero': False,
        'is_orthonormal': None,
        'warnings': []
    }
    
    # Determine type
    if arr.ndim == 1:
        results['type'] = 'vector (1D)'
        vec = arr
        norm = np.linalg.norm(vec)
        results['norm'] = norm
        
        print(f"  Type: Single direction vector")
        print(f"  Shape: {arr.shape}")
        print(f"  ||u||: {norm:.6f}")
        
        # Check for zero vector
        if norm < 1e-8:
            results['is_zero'] = True
            results['warnings'].append("Zero vector!")
            print(f"  ⚠️  WARNING: Near-zero norm!")
        elif norm < 0.5 or norm > 2.0:
            results['warnings'].append(f"Unusual norm: {norm:.6f}")
            print(f"  ⚠️  WARNING: Unusual norm (expected ~1.0)")
        else:
            print(f"  ✓ Norm is reasonable")
        
        # Sample a few values
        print(f"  Sample values: [{vec[0]:.4f}, {vec[1]:.4f}, ..., {vec[-1]:.4f}]")
        
    elif arr.ndim == 2:
        if arr.shape[1] == 1:
            results['type'] = 'vector (2D column)'
            vec = arr.flatten()
            norm = np.linalg.norm(vec)
            results['norm'] = norm
            
            print(f"  Type: Single direction (stored as column)")
            print(f"  Shape: {arr.shape}")
            print(f"  ||u||: {norm:.6f}")
            
            if norm < 1e-8:
                results['is_zero'] = True
                results['warnings'].append("Zero vector!")
                print(f"  ⚠️  WARNING: Near-zero norm!")
            elif norm < 0.5 or norm > 2.0:
                results['warnings'].append(f"Unusual norm: {norm:.6f}")
                print(f"  ⚠️  WARNING: Unusual norm (expected ~1.0)")
            else:
                print(f"  ✓ Norm is reasonable")
        else:
            results['type'] = 'subspace'
            R, H = arr.shape
            
            print(f"  Type: Subspace")
            print(f"  Shape: {arr.shape} (R={R} directions, H={H} dims)")
            
            # Frobenius norm
            fnorm = np.linalg.norm(arr, 'fro')
            results['norm'] = fnorm
            print(f"  ||U||_F: {fnorm:.6f}")
            
            # Check individual row norms
            row_norms = np.linalg.norm(arr, axis=1)
            print(f"  Row norms:")
            print(f"    Mean: {np.mean(row_norms):.6f}")
            print(f"    Std:  {np.std(row_norms):.6f}")
            print(f"    Min:  {np.min(row_norms):.6f}")
            print(f"    Max:  {np.max(row_norms):.6f}")
            
            # Check for zero rows
            zero_rows = np.sum(row_norms < 1e-8)
            if zero_rows > 0:
                results['warnings'].append(f"{zero_rows} zero rows")
                print(f"  ⚠️  WARNING: {zero_rows} rows have near-zero norm!")
            
            # Check for unusual row norms
            unusual_rows = np.sum((row_norms < 0.5) | (row_norms > 2.0))
            if unusual_rows > 0:
                results['warnings'].append(f"{unusual_rows} unusual row norms")
                print(f"  ⚠️  WARNING: {unusual_rows} rows have unusual norms")
            
            # Check orthonormality if requested
            if check_ortho:
                is_ortho, max_dev = check_orthonormality(arr)
                results['is_orthonormal'] = is_ortho
                print(f"  Orthonormality check:")
                print(f"    Max deviation from I: {max_dev:.6e}")
                if is_ortho:
                    print(f"    ✓ Rows are orthonormal")
                else:
                    results['warnings'].append(f"Not orthonormal (dev={max_dev:.2e})")
                    print(f"    ⚠️  WARNING: Rows are not orthonormal")
    else:
        results['type'] = 'unknown'
        results['warnings'].append(f"Unexpected shape: {arr.shape}")
        print(f"  ⚠️  ERROR: Unexpected shape {arr.shape}")
    
    return results


def main():
    args = parse_args()
    
    # Load directions
    phase_a_run = Path(args.phase_a_run)
    npz_path = phase_a_run / "screening" / "layer_to_U.npz"
    
    if not npz_path.exists():
        print(f"❌ ERROR: No directions found at {npz_path}")
        sys.exit(1)
    
    print(f"Loading directions from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    print(f"Found {len(data.files)} entries")
    
    # Extract layer indices
    layer_indices = []
    for key in data.files:
        if key.startswith('L'):
            layer_indices.append(int(key[1:]))
    
    layer_indices.sort()
    print(f"Layers: {layer_indices}")
    
    # Analyze each layer
    all_results = []
    for layer_idx in layer_indices:
        key = f"L{layer_idx}"
        arr = data[key]
        results = analyze_direction(layer_idx, arr, check_ortho=args.check_orthonormal)
        all_results.append(results)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_layers = len(all_results)
    vector_count = sum(1 for r in all_results if 'vector' in r['type'])
    subspace_count = sum(1 for r in all_results if r['type'] == 'subspace')
    zero_count = sum(1 for r in all_results if r['is_zero'])
    warning_count = sum(1 for r in all_results if len(r['warnings']) > 0)
    
    print(f"Total layers: {total_layers}")
    print(f"  Vectors: {vector_count}")
    print(f"  Subspaces: {subspace_count}")
    print(f"  Zero directions: {zero_count}")
    print(f"  Layers with warnings: {warning_count}")
    
    if zero_count > 0:
        print(f"\n❌ FAILED: Found {zero_count} zero directions")
        print("Layers with zero directions:")
        for r in all_results:
            if r['is_zero']:
                print(f"  Layer {r['layer']}")
        sys.exit(1)
    
    if warning_count > 0:
        print(f"\n⚠️  WARNING: {warning_count} layers have issues")
        for r in all_results:
            if len(r['warnings']) > 0:
                print(f"  Layer {r['layer']}:")
                for w in r['warnings']:
                    print(f"    - {w}")
    
    print(f"\n✓ All directions are non-zero")
    
    # Check if norms are reasonable
    reasonable_norms = sum(
        1 for r in all_results 
        if r['norm'] is not None and 0.5 <= r['norm'] <= (2.0 if 'vector' in r['type'] else 10.0)
    )
    
    if reasonable_norms == total_layers:
        print(f"✓ All norms are in reasonable range")
        sys.exit(0)
    else:
        print(f"⚠️  {total_layers - reasonable_norms} layers have unusual norms")
        sys.exit(0)  # Don't fail on unusual norms, just warn


if __name__ == "__main__":
    main()
