# Steering Chain-of-Thought in LLMs

This repository contains code for analyzing and manipulating reasoning behavior in large language models through learned directional interventions.

## Project Structure

### Core Modules

- **`dataset_loaders.py`**: Unified dataset loading interface for ARC, GSM8K, and MMLU-Pro
- **`hf_model_wrapper.py`**: Wrapper for HuggingFace models with generation and hook utilities
- **`multi_hook_manager.py`**: Multi-layer activation intervention system

### Directories

- **`answers/`**: Answer extraction utilities
- **`evaluation/`**: Analysis scripts, metrics, statistical tests, and visualization notebooks
- **`experiments/`**: Phase A (direction learning) and Phase B (causal intervention) runners
- **`scripts/`**: Utility scripts for data processing and analysis
  - `scripts/shell/`: Shell scripts for running experiments
- **`tests/`**: Unit tests for core components
- **`utils/`**: Shared utilities for hooks, handlers, and format control

### Evaluation Outputs

- **`evaluation/outputs/`**: Generated plots, statistics, and analysis results
  - Phase A: Direction learning metrics and visualizations
  - Phase B: Intervention effect analysis
  - Mode comparisons, locality tests, LDA analysis

### Requirements

See `requirements.txt` for dependencies.

## Usage

Run experiments using shell scripts in `scripts/shell/`:
- Phase A: Direction learning across layers
- Phase B: Causal interventions with learned directions

Analyze results using Jupyter notebooks in `evaluation/notebooks/`.

## Authors

Washieu Anan, Joey Dong, Gabriel Gomez
