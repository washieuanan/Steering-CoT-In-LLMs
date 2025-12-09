from __future__ import annotations

"""
Annotate Phase A generations with GPT-based reasoning and answer judgments.

Workflow:
1. Run Phase A Step 1 to produce pooled.npz and generations.csv.
2. Run this script to call an OpenAI judge model on each example:
      QUESTION + GOLD_ANSWER + MODEL_OUTPUT
   and obtain:
      - reasoning_correct_judge (bool)
      - answer_correct_judge (bool)
3. Use --use_gpt_labels in Phase A Step 2 to build subspaces solely from these labels.

API key:
    export OPENAI_API_KEY="sk-..."
"""

import argparse
import gzip
from pathlib import Path

import pandas as pd

from dataset_loaders import load_dataset_by_name
from utils import get_handler
from utils.reasoning_grader import grade_reasoning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate Phase A generations with GPT-based reasoning/answer judgments"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to Phase A Step 1 output dir, e.g. results/phase_a/MODEL__TAG/arc",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4.1-mini",  # or your GPT-5 nano model ID
        help="OpenAI judge model to use",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap for quick testing",
    )
    return parser.parse_args()


def load_generations(run_dir: Path) -> pd.DataFrame:
    """Load generations.csv or generations.csv.gz from a run directory."""
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

    return df


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise ValueError(f"Run directory does not exist: {run_dir}")

    print(f"Annotating generations in: {run_dir}")
    df = load_generations(run_dir)

    # Infer dataset/split for loading original examples
    dataset_name = df["dataset"].iloc[0] if "dataset" in df.columns else run_dir.name
    split = df["split"].iloc[0] if "split" in df.columns else "test"
    print(f"Dataset: {dataset_name}, split: {split}")

    loader = load_dataset_by_name(dataset_name, split=split)
    handler = get_handler(dataset_name)

    reasoning_correct = []
    answer_correct = []
    raw_json = []

    n = len(df)
    max_n = n if args.max_examples is None else min(args.max_examples, n)
    print(f"Total examples: {n}, annotating: {max_n}")

    for i in range(max_n):
        row = df.iloc[i]
        example_id = int(row.get("example_id", i))
        ex = loader.get_example(example_id)

        # Question text: prefer .question/.problem/.prompt, fall back to str(ex)
        question = (
            getattr(ex, "question", None)
            or getattr(ex, "problem", None)
            or getattr(ex, "prompt", None)
            or str(ex)
        )

        # Gold answer using handler normalization
        gold = handler.gold_target(ex)

        # Full model output (raw decoded text)
        model_output = str(row.get("gen_text_raw", row.get("gen_text", "")))

        j = grade_reasoning(
            question=question,
            gold_answer=gold,
            model_output=model_output,
            judge_model=args.judge_model,
        )
        reasoning_correct.append(j["reasoning_correct"])
        answer_correct.append(j["answer_correct"])
        raw_json.append(j["raw_response"])

        if (i + 1) % 20 == 0 or i == max_n - 1:
            print(
                f"  Annotated {i+1}/{max_n} "
                f"(reasoning_correct_judge positives so far: {sum(reasoning_correct)})"
            )

    # Attach columns (examples beyond max_n remain NaN if max_examples was used)
    df = df.copy()
    df.loc[: max_n - 1, "reasoning_correct_judge"] = reasoning_correct
    df.loc[: max_n - 1, "answer_correct_judge"] = answer_correct
    df.loc[: max_n - 1, "judge_raw_json"] = raw_json

    out_path = run_dir / "generations_with_judgments.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved annotated generations with GPT judgments to {out_path}")
    print("You can now run Phase A Step 2 with --use_gpt_labels to build GPT-based subspaces.")


if __name__ == "__main__":
    main()
