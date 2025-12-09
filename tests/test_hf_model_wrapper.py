from hf_model_wrapper import HFModelConfig, HFModelWrapper

# Test 1: Basic usage without special tokens
print("="*80)
print("TEST 1: Basic generation without special tokens")
print("="*80)
config = HFModelConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    device="auto",
)

wrapper = HFModelWrapper(config).load()

# Run generation
output = wrapper.generate(
    "Explain the difference between supervised and unsupervised learning.",
    max_new_tokens=128,
)
print("Output:")
print(output)
print()

# Test 2: Using special tokens for CoT reasoning
print("="*80)
print("TEST 2: Reasoning with CoT special tokens")
print("="*80)
config_with_special_tokens = HFModelConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    device="auto",
    special_tokens=["<cot>", "</cot>", "<answer>"],
    init_special_tokens_with_avg=True,
)

wrapper_with_tokens = HFModelWrapper(config_with_special_tokens).load()

# Get the special token IDs for building CoT masks
special_token_ids = wrapper_with_tokens.get_special_token_ids()
print("Special token IDs mapping:")
for token, token_id in special_token_ids.items():
    print(f"  {token}: {token_id}")
print(f"Vocabulary size: {len(wrapper_with_tokens.tokenizer)}")
print()

# Create a few-shot prompt to teach the model the CoT format
reasoning_prompt = """Here are examples of how to solve math problems step by step:

Question: What is 125 + 238?
<cot>Let me solve this step by step:
- Start with 125
- Add 238
- 125 + 238 = 363</cot>
<answer>363</answer>

Question: What is 567 - 234?
<cot>Let me solve this step by step:
- Start with 567
- Subtract 234
- 567 - 234 = 333</cot>
<answer>333</answer>

Question: What is 415 + 315? Please solve this step by step
<cot>
"""

print("Prompt (testing if model follows CoT structure):")
print(reasoning_prompt)
print()

# Generate with the model - see if it follows the CoT token structure
print("Generating response...")
output = wrapper_with_tokens.generate(
    reasoning_prompt,
    max_new_tokens=200,
    temperature=0.7,
    skip_special_tokens=False,  # Keep special tokens in output to see if model uses them
)

print("Raw output (with special tokens):")
print(output)
print()

# Also show clean output
clean_output = wrapper_with_tokens.generate(
    reasoning_prompt,
    max_new_tokens=200,
    temperature=0.7,
    skip_special_tokens=True,
)
print("Clean output (special tokens removed):")
print(clean_output)
print()

# Test tokenization with special tokens
test_text = "Question: What is 2+2? <cot>2 plus 2 equals 4</cot> <answer>4</answer>"
tokens = wrapper_with_tokens.tokenize(test_text, padding=False)
print("Example tokenization test:")
print(f"Text: {test_text}")
print(f"Token IDs: {tokens['input_ids'][0].tolist()}")
print(f"Decoded: {wrapper_with_tokens.detokenize(tokens['input_ids'][0], skip_special_tokens=False)}")
