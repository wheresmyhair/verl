import re
import json
import datasets
from datetime import datetime
from transformers import AutoTokenizer
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs

sampling_params = {
    "temperature": 1,
    "max_new_tokens": 6000,
    "sampling_seed": 42,
}
if __name__ == "__main__":
    enable_deterministic_inference = False
    for iter_idx in range(10):
        for tp_size in [1,2,4]:
            sgl_server_args = ServerArgs(
                model_path="/eric-verl/ff/models/qwen3-1.7b",
                mem_fraction_static=0.9,
                tp_size=tp_size,
                enable_deterministic_inference=enable_deterministic_inference,
                attention_backend="fa3",
            )
            engine = Engine(server_args=sgl_server_args)

            print(f"="*10 + f" Iter {iter_idx} TP {tp_size} Deterministic {enable_deterministic_inference} " + "="*10)
            templated_prompts = json.load(open('/eric-verl/ff/temp/verl/exp_script/long_prompts.json'))
            print(templated_prompts)
            start_time = datetime.now()
            print(f"Rollout start, start_time={start_time}")
            res = engine.generate(templated_prompts, sampling_params)
            end_time = datetime.now()
            print(f"Rollout end, end_time={end_time}")
            print(f"Rollout time, end_time - start_time={end_time - start_time}")
            print(f"Duration: {(end_time - start_time).total_seconds()} seconds")
            out = [{"input": input, "output": output} for input, output in zip(templated_prompts, res)]
            with open(f"rollout_long_tp{tp_size}_deterministic{enable_deterministic_inference}_iter{iter_idx}.jsonl", "w") as f:
                for item in out:
                    f.write(json.dumps(item) + "\n")
            
            engine.shutdown()