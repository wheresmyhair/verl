"""Phase B pilot: verify Qwen3-1.7B thinking mode + DAPO-Math triggers long CoT.

Checks:
  1. Chat template supports enable_thinking=True
  2. Model outputs <think>...</think>answer format
  3. Response length distribution shows variance (easy < 500, hard > 3K tokens)
  4. Answer is extractable from \\boxed{} after </think>
"""

import os
import re
import json
import pandas as pd
from transformers import AutoTokenizer


def check_chat_template(model_path):
    """Step 1: verify Qwen3 chat template supports enable_thinking=True."""
    tok = AutoTokenizer.from_pretrained(model_path)
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    prompt = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False,
        enable_thinking=True,
    )
    print("=== CHAT TEMPLATE OUTPUT (first 300 chars) ===")
    print(prompt[:300])
    print("...")
    has_think_tag = "<think>" in prompt or "<|think|>" in prompt
    print(f"\n<think> tag in prompt: {has_think_tag}")
    return prompt, has_think_tag


def check_sglang_generation(model_path, num_prompts=4):
    """Step 2/3: generate on DAPO-Math samples, check CoT + answer format."""
    import sglang as sgl

    dapo_df = pd.read_parquet('/home/user/data/dapo-math-4k/train.parquet')
    # Pick first N prompts
    msgs_list = []
    ground_truths = []
    for i in range(num_prompts):
        row = dapo_df.iloc[i]
        msgs_list.append(list(row['prompt']))  # already [{'role':'user','content':...}]
        ground_truths.append(row['reward_model']['ground_truth'])

    tok = AutoTokenizer.from_pretrained(model_path)
    prompts = [
        tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=True)
        for m in msgs_list
    ]

    print(f"\n=== Launching SGLang engine for {model_path} ===")
    llm = sgl.Engine(model_path=model_path, tp_size=1, log_level="warning")
    outputs = llm.generate(
        prompts,
        sampling_params={"max_new_tokens": 8192, "temperature": 1.0, "top_p": 1.0},
    )
    llm.shutdown()

    print(f"\n=== Generation results ({num_prompts} prompts) ===")
    for i, (out, gt) in enumerate(zip(outputs, ground_truths)):
        text = out['text']
        n_tok = out.get('meta_info', {}).get('completion_tokens', len(tok.encode(text)))
        has_think = ('<think>' in text and '</think>' in text) or ('<|think|>' in text)
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        extracted = boxed_match.group(1) if boxed_match else None
        answer_match = extracted == gt if extracted else False
        print(f"\nProblem {i}:")
        print(f"  tokens: {n_tok}")
        print(f"  has <think>: {has_think}")
        print(f"  ground_truth: {gt}")
        print(f"  extracted: {extracted}")
        print(f"  match: {answer_match}")
        # Show last 400 chars
        print(f"  ...last 400 chars: {text[-400:]}")

    token_counts = [o.get('meta_info', {}).get('completion_tokens', 0) for o in outputs]
    print(f"\n=== Token distribution ===")
    print(f"  min: {min(token_counts)}, max: {max(token_counts)}, mean: {sum(token_counts)/len(token_counts):.0f}")
    print(f"  variance {max(token_counts) - min(token_counts)} tokens across {num_prompts} prompts")


if __name__ == "__main__":
    model_path = "Qwen/Qwen3-1.7B"
    # Step 1
    prompt, has_think = check_chat_template(model_path)
    if not has_think:
        print("\n⚠️  <think> tag NOT found in chat template. Thinking mode may not work out of box.")
    # Step 2/3
    check_sglang_generation(model_path, num_prompts=4)
