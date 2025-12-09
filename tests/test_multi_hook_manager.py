"""
Example usage of multi-hook manager for activation collection and screening.
"""

import numpy as np
import torch

# This is an example showing how to use the multi_hook_manager
# Actual usage would require a loaded model

def example_pooler_usage():
    """
    Example: Collecting pooled activations from multiple layers.
    """
    print("=== MultiLayerPooler Example ===")
    print("""
    from hf_model_wrapper import HFModelConfig, HFModelWrapper
    from multi_hook_manager import MultiLayerPooler

    # Load model
    config = HFModelConfig(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        dtype="bfloat16",
        device="cuda"
    )
    wrapper = HFModelWrapper(config).load()

    # Initialize pooler for layers [16, 20, 24, 28]
    pooler = MultiLayerPooler(
        model=wrapper.model,
        layers=[16, 20, 24, 28],
        hidden_size=4096
    )

    # Tokenize prompt and identify CoT window
    tokens = wrapper.tokenize("Let's think step by step...")
    seq_len = tokens['input_ids'].shape[1]
    
    # Create mask for reasoning tokens (example: tokens 5-20)
    cot_mask = torch.zeros(seq_len, dtype=torch.bool)
    cot_mask[5:20] = True
    pooler.set_cot_window(cot_mask)

    # Run generation
    with torch.no_grad():
        output = wrapper.model.generate(**tokens, max_new_tokens=50)

    # Get pooled activations
    pooled = pooler.pooled()
    # pooled = {16: array([...]), 20: array([...]), ...}

    pooler.close()
    """)


def example_editor_usage():
    """
    Example: Applying causal interventions.
    """
    print("\n=== MultiLayerEditor Example ===")
    print("""
    from multi_hook_manager import MultiLayerEditor
    import numpy as np

    # Assume you have learned projection matrices U from screening
    U_24 = np.random.randn(4096, 128).astype(np.float32)  # Example
    U_28 = np.random.randn(4096, 128).astype(np.float32) 

    # Initialize editor with interventions
    editor = MultiLayerEditor(
        model=wrapper.model,
        layer_to_US={24: U_24, 28: U_28},
        layer_to_alpha={24: 0.5, 28: 0.5},
        use_lesion=False  # False = add, True = subtract
    )

    # Set CoT mask
    cot_mask = torch.zeros(seq_len, dtype=torch.bool)
    cot_mask[5:20] = True
    editor.set_cot_mask(cot_mask)

    # Generate with intervention active
    with torch.no_grad():
        output_intervened = wrapper.model.generate(**tokens, max_new_tokens=50)

    editor.close()
    """)


def example_screener_usage():
    """
    Example: Offline layer screening.
    """
    print("\n=== OfflineLayerScreener Example ===")
    print("""
    from multi_hook_manager import OfflineLayerScreener
    import numpy as np

    # Assume you've collected activations from multiple traces
    # X shape: [n_traces=1000, n_layers=8, hidden_size=4096]
    # y shape: [n_traces=1000] with 0/1 labels (incorrect/correct)

    n_traces = 1000
    n_layers = 8
    D = 4096

    # Simulated data (replace with actual collected activations)
    X = np.random.randn(n_traces, n_layers, D).astype(np.float32)
    y = np.random.randint(0, 2, size=n_traces)

    layer_indices = [16, 18, 20, 22, 24, 26, 28, 30]

    # Initialize screener
    screener = OfflineLayerScreener(
        X=X,
        y=y,
        layer_indices=layer_indices,
        test_size=0.2
    )

    # Screen all layers
    results = screener.screen_all_layers(C=1.0, max_iter=1000)

    # Rank layers by combined metrics
    ranked = screener.rank_layers(results, auc_weight=0.6, delta_weight=0.4)

    print("Top 3 layers:", ranked[:3])
    # Output: [(24, 0.95), (28, 0.89), (26, 0.83)]

    # Extract projection directions from top layer
    top_layer_idx = ranked[0][0]
    layer_array_idx = layer_indices.index(top_layer_idx)
    probe_model = results[top_layer_idx]['probe_model']
    
    U = screener.extract_top_directions(
        layer_idx=layer_array_idx,
        probe_model=probe_model,
        k=128
    )
    # U shape: [D, 128]
    """)


def example_full_pipeline():
    """
    Example: Complete pipeline from collection to intervention.
    """
    print("\n=== Complete Pipeline Example ===")
    print("""
    # Step 1: Collect activations from candidate layers
    pooler = MultiLayerPooler(
        model=wrapper.model,
        layers=[16, 20, 24, 28, 30],
        hidden_size=4096
    )

    all_pooled = []
    all_labels = []

    for prompt, correct_answer in dataset:
        # Set CoT window for this prompt
        cot_mask = identify_cot_tokens(prompt)
        pooler.set_cot_window(cot_mask)
        
        # Generate
        output = wrapper.generate(prompt)
        
        # Get pooled activations
        pooled = pooler.pooled()
        all_pooled.append([pooled[l] for l in [16, 20, 24, 28, 30]])
        
        # Check if answer is correct
        is_correct = check_answer(output, correct_answer)
        all_labels.append(int(is_correct))
        
        pooler.reset_buffers()

    pooler.close()

    # Step 2: Offline screening
    X = np.array(all_pooled)  # [n_traces, n_layers, D]
    y = np.array(all_labels)  # [n_traces]

    screener = OfflineLayerScreener(X, y, layer_indices=[16, 20, 24, 28, 30])
    results = screener.screen_all_layers()
    ranked = screener.rank_layers(results)

    print(f"Best layer: {ranked[0][0]} with score {ranked[0][1]:.3f}")

    # Step 3: Extract directions and apply intervention
    best_layer = ranked[0][0]
    layer_idx = [16, 20, 24, 28, 30].index(best_layer)
    probe = results[best_layer]['probe_model']
    U = screener.extract_top_directions(layer_idx, probe, k=128)

    # Step 4: Test intervention
    editor = MultiLayerEditor(
        model=wrapper.model,
        layer_to_US={best_layer: U},
        layer_to_alpha={best_layer: 0.5}
    )

    # Run on validation set with intervention
    for val_prompt in val_dataset:
        cot_mask = identify_cot_tokens(val_prompt)
        editor.set_cot_mask(cot_mask)
        
        output = wrapper.generate(val_prompt)
        # Analyze improvement...

    editor.close()
    """)


if __name__ == "__main__":
    print("Multi-Hook Manager Usage Examples")
    print("=" * 50)
    example_pooler_usage()
    example_editor_usage()
    example_screener_usage()
    example_full_pipeline()
    print("\n" + "=" * 50)
    print("Note: These are example snippets. Actual usage requires:")
    print("  1. A loaded HuggingFace model via HFModelWrapper")
    print("  2. A dataset with prompts and labels")
    print("  3. Logic to identify CoT token windows")
