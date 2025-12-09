from __future__ import annotations

"""
Reasoning and answer quality grading via OpenAI Responses API.

This module provides a wrapper around the OpenAI Python SDK to:
- Take (question, gold_answer, model_output)
- Ask a judge model (gpt-5-nano) to grade:
    * reasoning_correct: whether the reasoning trace is logically sound
    * answer_correct: whether the final answer matches the gold answer
- Return a simple dict with booleans and the raw response from the judge.

API key handling
----------------
We NEVER hard-code the API key.

To authenticate, set the environment variable before running any scripts:

    export OPENAI_API_KEY="sk-..."

The OpenAI client below will automatically pick it up.
"""

from typing import Dict, Any, Optional

import json
import re

from openai import OpenAI

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """
    Lazily construct a shared OpenAI client.

    Relies on the OPENAI_API_KEY environment variable. Do NOT hard-code keys.
    """
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def grade_reasoning(
    question: str,
    gold_answer: str,
    model_output: str,
    *,
    judge_model: str = "gpt-5-nano",
) -> Dict[str, Any]:
    """
    Call OpenAI Responses API to judge reasoning + answer correctness.

    Args:
        question: The original question / problem text shown to the model.
        gold_answer: The canonical correct answer (letter or numeric, normalized).
        model_output: The FULL model output text, including reasoning and answer.
        judge_model: OpenAI model name to use as the grader (default: gpt-5-nano).

    Returns:
        Dict with:
            - reasoning_correct: bool
            - answer_correct: bool
            - raw_response: str (raw text from judge)
    """
    client = _get_client()

    # Build the grading prompt as input
    json_template = '{"reasoning_correct": true or false, "answer_correct": true or false}'
    grading_prompt = f"""You are a strict but fair grader of reasoning.

QUESTION:
{question}

GOLD_ANSWER:
{gold_answer}

MODEL_OUTPUT:
{model_output}

INSTRUCTIONS:
1. Determine if the MODEL_OUTPUT's final answer is logically correct relative to GOLD_ANSWER. This is 'answer_correct'.
2. Determine if the reasoning steps in MODEL_OUTPUT are mostly logically correct and internally consistent, even if the final answer is wrong. This is 'reasoning_correct'.
3. Output exactly JSON with two boolean fields:

{json_template}

Only output the JSON, no other text."""

    # Call the Responses API
    response = client.responses.create(
        model=judge_model,
        input=[
            {"role": "user", "content": grading_prompt}
        ],
        text={
            "format": {"type": "text"},
            "verbosity": "medium"
        },
        reasoning={
            "effort": "medium"
        },
        tools=[],
        store=True,
        include=[
            "reasoning.encrypted_content",
            "web_search_call.action.sources"
        ]
    )

    # Extract the text output from the response
    # The response structure has output_text or we need to find the text block
    content = ""
    if hasattr(response, 'output_text'):
        content = response.output_text
    elif hasattr(response, 'output') and response.output:
        # Iterate through output blocks to find text content
        for block in response.output:
            if hasattr(block, 'type') and block.type == 'message':
                if hasattr(block, 'content') and block.content:
                    for content_block in block.content:
                        if hasattr(content_block, 'type') and content_block.type == 'output_text':
                            content = content_block.text
                            break
                        elif hasattr(content_block, 'text'):
                            content = content_block.text
                            break
            elif hasattr(block, 'content'):
                if isinstance(block.content, str):
                    content = block.content
                elif isinstance(block.content, list):
                    for c in block.content:
                        if hasattr(c, 'text'):
                            content = c.text
                            break
    
    # Also try direct text attribute
    if not content and hasattr(response, 'text'):
        content = response.text

    # Parse JSON from the content
    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(content)
    except Exception:
        # Fallback: default to False if parsing fails
        data = {}

    reasoning_correct = bool(data.get("reasoning_correct", False))
    answer_correct = bool(data.get("answer_correct", False))

    return {
        "reasoning_correct": reasoning_correct,
        "answer_correct": answer_correct,
        "raw_response": content,
    }
