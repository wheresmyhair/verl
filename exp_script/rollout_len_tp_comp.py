import re
import json
import datasets
from datetime import datetime
from transformers import AutoTokenizer
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs



dataset = datasets.load_dataset("openai/gsm8k", "main")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

instruction_following = 'Let\'s think step by step and output the final answer after "####".'

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

# add a row to each data item that represents a unique id
def make_map_fn(split):
    def process_fn(example, idx):
        question_raw = example.pop("question")

        question = question_raw + " " + instruction_following

        answer_raw = example.pop("answer")
        solution = extract_solution(answer_raw)
        data = {
            "data_source": "openai/gsm8k",
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            },
        }
        return data

    return process_fn

train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
train_dataset_b_all = train_dataset.select(range(1024))
train_dataset_b1 = train_dataset.select(range(256))
train_dataset_b2 = train_dataset.select(range(256, 512))
train_dataset_b3 = train_dataset.select(range(512, 768))
train_dataset_b4 = train_dataset.select(range(768, 1024))
tokenizer = AutoTokenizer.from_pretrained("/eric-verl/ff/models/qwen3-1.7b")
batches = [train_dataset_b1, train_dataset_b2, train_dataset_b3, train_dataset_b4]

sampling_params = {
    "temperature": 1,
    "max_new_tokens": 6000,
    "sampling_seed": 42,
}
if __name__ == "__main__":
    enable_deterministic_inference = False
    for iter_idx in range(5):
        tp_size = 1
        for idx, batch in enumerate(batches):
            sgl_server_args = ServerArgs(
                model_path="/eric-verl/ff/models/qwen3-1.7b",
                mem_fraction_static=0.9,
                tp_size=tp_size,
                enable_deterministic_inference=enable_deterministic_inference,
                attention_backend="fa3",
            )
            engine = Engine(server_args=sgl_server_args)

            print(f"="*10 + f" Iter {iter_idx} Batch {idx} TP {tp_size} Deterministic {enable_deterministic_inference} " + "="*10)
            templated_prompts = [tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) for prompt in batch["prompt"]]
            start_time = datetime.now()
            print(f"Rollout start, start_time={start_time}")
            res = engine.generate(templated_prompts, sampling_params)
            end_time = datetime.now()
            print(f"Rollout end, end_time={end_time}")
            print(f"Rollout time, end_time - start_time={end_time - start_time}")
            print(f"Duration: {(end_time - start_time).total_seconds()} seconds")
            out = [{"input": input, "output": output} for input, output in zip(templated_prompts, res)]
            with open(f"rollout_comp_b{idx}_tp{tp_size}_deterministic{enable_deterministic_inference}_iter{iter_idx}.jsonl", "w") as f:
                for item in out:
                    f.write(json.dumps(item) + "\n")
            
            engine.shutdown()
            
        tp_size = 4
        sgl_server_args = ServerArgs(
            model_path="/eric-verl/ff/models/qwen3-1.7b",
            mem_fraction_static=0.9,
            tp_size=tp_size,
            enable_deterministic_inference=enable_deterministic_inference,
            attention_backend="fa3",
        )
        engine = Engine(server_args=sgl_server_args)

        print(f"="*10 + f" Iter {iter_idx} TP {tp_size} Deterministic {enable_deterministic_inference} " + "="*10)
        templated_prompts = [tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) for prompt in train_dataset_b_all["prompt"]]
        start_time = datetime.now()
        print(f"Rollout start, start_time={start_time}")
        res = engine.generate(templated_prompts, sampling_params)
        end_time = datetime.now()
        print(f"Rollout end, end_time={end_time}")
        print(f"Rollout time, end_time - start_time={end_time - start_time}")
        print(f"Duration: {(end_time - start_time).total_seconds()} seconds")
        out = [{"input": input, "output": output} for input, output in zip(templated_prompts, res)]
        with open(f"rollout_comp_full_tp{tp_size}_deterministic{enable_deterministic_inference}_iter{iter_idx}.jsonl", "w") as f:
            for item in out:
                f.write(json.dumps(item) + "\n")
        
        engine.shutdown()