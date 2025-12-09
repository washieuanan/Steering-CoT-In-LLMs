"""Dataset loaders for reasoning experiments with unified interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from datasets import load_dataset, Dataset
import random


@dataclass
class DatasetExample:
    """
    Standardized format for dataset examples.
    
    Attributes:
        question: The question/prompt text
        choices: List of answer choices (empty for free-form answers)
        correct_answer: The correct answer (letter for MC, text for free-form)
        metadata: Additional information (difficulty, category, etc.)
        example_id: Unique identifier for this example
        task_type: Type of task - "mcq", "numeric", or "labelset"
        gold_index: Index of correct choice (for mcq/labelset with choices)
        gold_letter: Letter of correct choice (for mcq)
        labels: List of valid label strings (for labelset without choices)
    """
    question: str
    choices: List[str]
    correct_answer: str
    metadata: Dict[str, Any]
    example_id: str
    task_type: str = "mcq"  # Default to mcq
    gold_index: Optional[int] = None
    gold_letter: Optional[str] = None
    labels: Optional[List[str]] = None


class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    
    All dataset loaders should extend this class and implement the required methods.
    This ensures a consistent interface for experimentation.
    """
    
    def __init__(self, split: str = "test", seed: int = 42):
        """
        Initialize the dataset loader.
        
        Args:
            split: Which split to load ("train", "test", "validation")
            seed: Random seed for reproducibility
        """
        self.split = split
        self.seed = seed
        self.dataset: Optional[Dataset] = None
        self.examples: List[DatasetExample] = []
        random.seed(seed)
        
    @abstractmethod
    def load(self) -> BaseDatasetLoader:
        """Load the dataset from HuggingFace and parse into examples."""
        pass
    
    @abstractmethod
    def format_prompt(
        self, 
        example: DatasetExample,
        include_cot: bool = True,
        few_shot_examples: Optional[List[DatasetExample]] = None
    ) -> str:
        """
        Format an example into a prompt for the model.
        
        Args:
            example: The example to format
            include_cot: Whether to include CoT prompting
            few_shot_examples: Optional examples for few-shot prompting
            
        Returns:
            Formatted prompt string
        """
        pass
    
    def check_answer(self, example: DatasetExample, model_output: str) -> bool:
        """
        Check if the model's output matches the correct answer.
        
        Args:
            example: The example with correct answer
            model_output: The model's generated output
            
        Returns:
            True if correct, False otherwise
        """
        # Default strict MCQ implementation
        return self._check_mcq_answer_strict(example, model_output)
    
    def _check_mcq_answer_strict(self, example: DatasetExample, model_output: str) -> bool:
        """
        Strict MCQ answer checker using regex to extract final answer.
        Only accepts properly formatted final answers at the end.
        
        Returns 0 (incorrect) if no valid answer format is found.
        """
        import re
        
        # Regex to match "Final answer: X" or just "X" on its own line at the end
        # Case-insensitive, matches A, B, C, or D
        pattern = r'(?im)^\s*(?:Final answer:\s*)?([ABCD])\s*$'
        
        matches = re.findall(pattern, model_output)
        
        if not matches:
            # No valid answer format found
            return False
        
        # Take the last match (most recent answer)
        extracted_answer = matches[-1].upper()
        correct_answer = example.correct_answer.strip().upper()
        
        return extracted_answer == correct_answer
    
    def get_example(self, index: int) -> DatasetExample:
        """Get a specific example by index."""
        return self.examples[index]
    
    def get_random_examples(self, n: int, exclude_indices: Optional[List[int]] = None) -> List[DatasetExample]:
        """
        Get n random examples, optionally excluding certain indices.
        
        Args:
            n: Number of examples to get
            exclude_indices: Indices to exclude from selection
            
        Returns:
            List of random examples
        """
        if exclude_indices is None:
            exclude_indices = []
        
        available_indices = [i for i in range(len(self.examples)) if i not in exclude_indices]
        selected_indices = random.sample(available_indices, min(n, len(available_indices)))
        return [self.examples[i] for i in selected_indices]
    
    def __len__(self) -> int:
        """Return number of examples in dataset."""
        return len(self.examples)
    
    def __iter__(self):
        """Iterate over examples."""
        return iter(self.examples)


class ARCLoader(BaseDatasetLoader):
    """
    Loader for AI2 ARC (Reasoning Challenge) dataset.
    
    ARC contains science questions with multiple choice answers.
    Two difficulty levels: ARC-Easy and ARC-Challenge.
    """
    
    def __init__(self, split: str = "test", difficulty: str = "ARC-Challenge", seed: int = 42):
        """
        Initialize ARC loader.
        
        Args:
            split: "train", "test", or "validation"
            difficulty: "ARC-Easy" or "ARC-Challenge"
            seed: Random seed
        """
        super().__init__(split, seed)
        self.difficulty = difficulty
        
    def load(self) -> ARCLoader:
        """Load ARC dataset."""
        self.dataset = load_dataset("allenai/ai2_arc", self.difficulty, split=self.split)
        
        for idx, item in enumerate(self.dataset):
            # Parse choices
            choices = item["choices"]["text"]
            choice_labels = item["choices"]["label"]
            
            # Map answer key to choice and find its index
            answer_key = item["answerKey"]
            try:
                gold_index = choice_labels.index(answer_key)
            except ValueError:
                gold_index = None
            
            example = DatasetExample(
                question=item["question"],
                choices=choices,
                correct_answer=answer_key,
                metadata={
                    "difficulty": self.difficulty,
                    "choice_labels": choice_labels,
                },
                example_id=f"arc_{self.difficulty}_{self.split}_{idx}",
                task_type="mcq",
                gold_index=gold_index,
                gold_letter=answer_key,
                labels=None
            )
            self.examples.append(example)
        
        return self
    
    def format_prompt(
        self,
        example: DatasetExample,
        include_cot: bool = True,
        few_shot_examples: Optional[List[DatasetExample]] = None
    ) -> str:
        """Format ARC example as a prompt."""
        prompt = ""
        
        # Add few-shot examples if provided
        if few_shot_examples:
            for fs_example in few_shot_examples:
                prompt += self._format_single_example(fs_example, include_answer=True, include_cot=include_cot)
                prompt += "\n\n"
        
        # Add the actual question
        prompt += self._format_single_example(example, include_answer=False, include_cot=include_cot)
        
        return prompt
    
    def _format_single_example(self, example: DatasetExample, include_answer: bool = False, include_cot: bool = True) -> str:
        """Helper to format a single example with strict MCQ format."""
        choice_labels = example.metadata.get("choice_labels", ["A", "B", "C", "D"])
        
        # Strict MCQ format with CoT instruction (sanitized - no real special tokens in prompt)
        if include_cot and not include_answer:
            prompt = 'You will think inside the tokens "<cot>" and "</cot>" (do not print them here).\n'
            prompt += 'Then output exactly one line: "Final answer: X" where X in {A,B,C,D}.\n'
            prompt += f"Question: {example.question}\n"
            prompt += "Choices: "
            for i, (label, choice) in enumerate(zip(choice_labels, example.choices)):
                if i > 0:
                    prompt += " "
                prompt += f"{label}) {choice}"
        elif include_answer:
            # Few-shot example format
            prompt = 'You will think inside the tokens "<cot>" and "</cot>" (do not print them here).\n'
            prompt += 'Then output exactly one line: "Final answer: X" where X in {A,B,C,D}.\n'
            prompt += f"Question: {example.question}\n"
            prompt += "Choices: "
            for i, (label, choice) in enumerate(zip(choice_labels, example.choices)):
                if i > 0:
                    prompt += " "
                prompt += f"{label}) {choice}"
            prompt += "\n<cot>\n"
            prompt += "[reasoning steps]\n"
            prompt += "</cot>\n"
            prompt += f"Final answer: {example.correct_answer}"
        else:
            # No CoT format (legacy)
            prompt = f"Question: {example.question}\n"
            for label, choice in zip(choice_labels, example.choices):
                prompt += f"{label}. {choice}\n"
            if include_answer:
                prompt += f"Answer: {example.correct_answer}"
        
        return prompt


class GSM8KLoader(BaseDatasetLoader):
    """
    Loader for GSM8K (Grade School Math) dataset.
    
    Contains grade school math word problems requiring multi-step reasoning.
    Free-form numerical answers.
    """
    
    def load(self) -> GSM8KLoader:
        """Load GSM8K dataset."""
        # GSM8K uses "train" and "test" splits
        dataset_split = "train" if self.split == "validation" else self.split
        self.dataset = load_dataset("openai/gsm8k", "main", split=dataset_split)
        
        for idx, item in enumerate(self.dataset):
            # Extract numerical answer from the answer string
            answer_text = item["answer"]
            # Answer format is typically "#### NUMBER"
            numerical_answer = answer_text.split("####")[-1].strip()
            
            example = DatasetExample(
                question=item["question"],
                choices=[],  # Free-form answer
                correct_answer=numerical_answer,
                metadata={
                    "full_answer": answer_text,
                },
                example_id=f"gsm8k_{dataset_split}_{idx}",
                task_type="numeric",
                gold_index=None,
                gold_letter=None,
                labels=None
            )
            self.examples.append(example)
        
        return self
    
    def format_prompt(
        self,
        example: DatasetExample,
        include_cot: bool = True,
        few_shot_examples: Optional[List[DatasetExample]] = None
    ) -> str:
        """Format GSM8K example as a prompt."""
        prompt = ""
        
        # Add few-shot examples if provided
        if few_shot_examples:
            for fs_example in few_shot_examples:
                prompt += self._format_single_example(fs_example, include_answer=True, include_cot=include_cot)
                prompt += "\n\n"
        
        # Add the actual question
        prompt += self._format_single_example(example, include_answer=False, include_cot=include_cot)
        
        return prompt
    
    def _format_single_example(self, example: DatasetExample, include_answer: bool = False, include_cot: bool = True) -> str:
        """Helper to format a single example with strict numeric answer format."""
        if include_cot and not include_answer:
            # Strict format for test examples
            prompt = "You will answer by thinking between <cot> and </cot>.\n"
            prompt += "Then output a single line: \"Final answer: <number>\".\n"
            prompt += f"Question: {example.question}\n"
            prompt += "<cot>"
        elif include_answer:
            # Few-shot example format
            prompt = "You will answer by thinking between <cot> and </cot>.\n"
            prompt += "Then output a single line: \"Final answer: <number>\".\n"
            prompt += f"Question: {example.question}\n"
            prompt += "<cot>\n"
            full_answer = example.metadata.get("full_answer", "")
            if full_answer and "####" in full_answer:
                solution = full_answer.split("####")[0].strip()
                prompt += f"{solution}\n"
            else:
                prompt += "[reasoning steps]\n"
            prompt += "</cot>\n"
            prompt += f"Final answer: {example.correct_answer}"
        else:
            # Legacy no-CoT format
            prompt = f"Question: {example.question}\n"
            if include_answer:
                prompt += f"Answer: {example.correct_answer}"
        
        return prompt
    
    def check_answer(self, example: DatasetExample, model_output: str) -> bool:
        """
        Strict numerical answer checker using regex to extract final answer.
        Only accepts properly formatted final answers at the end.
        
        Returns False (incorrect) if no valid answer format is found.
        """
        import re
        
        # Regex to match "Final answer: <number>" on its own line
        # Matches integers and decimals, with optional thousands separators
        pattern = r'(?i)^\s*Final answer:\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*$'
        
        matches = re.findall(pattern, model_output, re.MULTILINE)
        
        if not matches:
            # No valid answer format found
            return False
        
        # Take the last match (most recent answer)
        extracted_answer = matches[-1].replace(",", "").strip()
        correct_answer = example.correct_answer.replace(",", "").strip()
        
        # Compare as numbers to handle formatting variations
        try:
            extracted_num = float(extracted_answer)
            correct_num = float(correct_answer)
            return abs(extracted_num - correct_num) < 1e-6  # Allow for floating point errors
        except ValueError:
            # Fallback to string comparison
            return extracted_answer == correct_answer


class MMLUProLoader(BaseDatasetLoader):
    """
    Loader for MMLU-Pro (Massive Multitask Language Understanding Pro) dataset.
    
    Enhanced version of MMLU with more challenging questions across many subjects.
    Multiple choice with 4-10 options.
    """
    
    def __init__(self, split: str = "test", category: Optional[str] = None, seed: int = 42):
        """
        Initialize MMLU-Pro loader.
        
        Args:
            split: "test" or "validation"
            category: Optional specific category to load (e.g., "math", "physics")
            seed: Random seed
        """
        super().__init__(split, seed)
        self.category = category
    
    def load(self) -> MMLUProLoader:
        """Load MMLU-Pro dataset."""
        # MMLU-Pro structure: split by test/validation
        dataset_split = "validation" if self.split in ["val", "validation"] else "test"
        self.dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=dataset_split)
        
        for idx, item in enumerate(self.dataset):
            # Filter by category if specified
            if self.category and item.get("category") != self.category:
                continue
            
            # Parse options
            options = item["options"]
            answer_letter = item["answer"]  # Letter (A, B, C, etc.)
            
            # Find gold index - answer is already a letter
            gold_index = ord(answer_letter.upper()) - ord('A') if answer_letter else None
            
            example = DatasetExample(
                question=item["question"],
                choices=options,
                correct_answer=answer_letter,
                metadata={
                    "category": item.get("category", "unknown"),
                    "question_id": item.get("question_id", idx),
                },
                example_id=f"mmlu_pro_{dataset_split}_{idx}",
                task_type="mcq",
                gold_index=gold_index,
                gold_letter=answer_letter,
                labels=None
            )
            self.examples.append(example)
        
        return self
    
    def format_prompt(
        self,
        example: DatasetExample,
        include_cot: bool = True,
        few_shot_examples: Optional[List[DatasetExample]] = None
    ) -> str:
        """Format MMLU-Pro example as a prompt."""
        prompt = ""
        
        # Add few-shot examples if provided
        if few_shot_examples:
            for fs_example in few_shot_examples:
                prompt += self._format_single_example(fs_example, include_answer=True, include_cot=include_cot)
                prompt += "\n\n"
        
        # Add the actual question
        prompt += self._format_single_example(example, include_answer=False, include_cot=include_cot)
        
        return prompt
    
    def _format_single_example(self, example: DatasetExample, include_answer: bool = False, include_cot: bool = True) -> str:
        """Helper to format a single example with strict MCQ format."""
        category = example.metadata.get("category", "General")
        
        # Strict MCQ format with CoT instruction (sanitized - no real special tokens in prompt)
        if include_cot and not include_answer:
            prompt = 'You will think inside the tokens "<cot>" and "</cot>" (do not print them here).\n'
            prompt += 'Then output exactly one line: "Final answer: X" where X in {A,B,C,D}.\n'
            prompt += f"Question: {example.question}\n"
            prompt += "Choices: "
            for i, choice in enumerate(example.choices):
                letter = chr(65 + i)  # A, B, C, D, ...
                if i > 0:
                    prompt += " "
                prompt += f"{letter}) {choice}"
        elif include_answer:
            # Few-shot example format
            prompt = 'You will think inside the tokens "<cot>" and "</cot>" (do not print them here).\n'
            prompt += 'Then output exactly one line: "Final answer: X" where X in {A,B,C,D}.\n'
            prompt += f"Question: {example.question}\n"
            prompt += "Choices: "
            for i, choice in enumerate(example.choices):
                letter = chr(65 + i)  # A, B, C, D, ...
                if i > 0:
                    prompt += " "
                prompt += f"{letter}) {choice}"
            prompt += "\n<cot>\n"
            prompt += "[reasoning steps]\n"
            prompt += "</cot>\n"
            prompt += f"Final answer: {example.correct_answer}"
        else:
            # No CoT format (legacy)
            prompt = f"Category: {category}\n"
            prompt += f"Question: {example.question}\n"
            for i, choice in enumerate(example.choices):
                letter = chr(65 + i)
                prompt += f"{letter}. {choice}\n"
            if include_answer:
                prompt += f"Answer: {example.correct_answer}"
        
        return prompt


# Convenience factory function
def load_dataset_by_name(
    dataset_name: str,
    split: str = "test",
    seed: int = 42,
    **kwargs
) -> BaseDatasetLoader:
    """
    Factory function to load a dataset by name.
    
    Args:
        dataset_name: One of "arc", "gsm8k", "mmlu_pro"
        split: Which split to load
        seed: Random seed
        **kwargs: Additional arguments specific to each loader
        
    Returns:
        Loaded dataset instance
        
    Example:
        >>> loader = load_dataset_by_name("arc", split="test", difficulty="ARC-Challenge")
        >>> loader = load_dataset_by_name("gsm8k", split="train")
    """
    loaders = {
        "arc": ARCLoader,
        "gsm8k": GSM8KLoader,
        "mmlu_pro": MMLUProLoader,
    }
    
    if dataset_name.lower() not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(loaders.keys())}")
    
    loader_class = loaders[dataset_name.lower()]
    return loader_class(split=split, seed=seed, **kwargs).load()
