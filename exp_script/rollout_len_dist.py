import re
import json
import datasets
from datetime import datetime
from transformers import AutoTokenizer
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs


sampling_params = {
    "temperature": 1,
    "max_new_tokens": 16384,
    "sampling_seed": 42,
}
if __name__ == "__main__":
    enable_deterministic_inference = False
    for iter_idx in range(10):
        for split_idx in [2,3,4]:
            for batch_idx in [0,1,2]:
                with open(f'/eric-verl/ff/temp/verl/exp_script/dist/2tp2_1tp4/in/{split_idx}k_b{batch_idx}.json', 'r') as f:
                    this_data = json.load(f)
                
                if batch_idx == 0:
                    tp_size = 4
                else:
                    tp_size = 2
        
                sgl_server_args = ServerArgs(
                    model_path="/eric-verl/ff/models/qwen3-1.7b",
                    mem_fraction_static=0.9,
                    tp_size=tp_size,
                    enable_deterministic_inference=enable_deterministic_inference,
                    attention_backend="fa3",
                )
                engine = Engine(server_args=sgl_server_args)

                print(f"="*10 + f" Iter {iter_idx} Split {split_idx} Batch {batch_idx} TP {tp_size} Deterministic {enable_deterministic_inference} " + "="*10)
                templated_prompts = [sample["input"] for sample in this_data]
                start_time = datetime.now()
                print(f"Rollout start, start_time={start_time}")
                res = engine.generate(templated_prompts, sampling_params)
                end_time = datetime.now()
                print(f"Rollout end, end_time={end_time}")
                print(f"Rollout time, end_time - start_time={end_time - start_time}")
                print(f"Duration: {(end_time - start_time).total_seconds()} seconds")
                out = [{"input": input, "output": output} for input, output in zip(templated_prompts, res)]
                with open(f"/eric-verl/ff/temp/verl/exp_script/dist/2tp2_1tp4/out/{split_idx}k_b{batch_idx}_tp{tp_size}_deterministic{enable_deterministic_inference}_iter{iter_idx}.json", "w") as f:
                    json.dump(out, f)
                
                engine.shutdown()
                
        
        for batch_idx in range(8):
                with open(f'/eric-verl/ff/temp/verl/exp_script/dist/8tp1/in/{batch_idx}.json', 'r') as f:
                    this_data = json.load(f)
                
                tp_size = 1
        
                sgl_server_args = ServerArgs(
                    model_path="/eric-verl/ff/models/qwen3-1.7b",
                    mem_fraction_static=0.9,
                    tp_size=tp_size,
                    enable_deterministic_inference=enable_deterministic_inference,
                    attention_backend="fa3",
                )
                engine = Engine(server_args=sgl_server_args)

                print(f"="*10 + f" Iter {iter_idx} Batch {batch_idx} TP {tp_size} Deterministic {enable_deterministic_inference} " + "="*10)
                templated_prompts = [sample["input"] for sample in this_data]
                start_time = datetime.now()
                print(f"Rollout start, start_time={start_time}")
                res = engine.generate(templated_prompts, sampling_params)
                end_time = datetime.now()
                print(f"Rollout end, end_time={end_time}")
                print(f"Rollout time, end_time - start_time={end_time - start_time}")
                print(f"Duration: {(end_time - start_time).total_seconds()} seconds")
                out = [{"input": input, "output": output} for input, output in zip(templated_prompts, res)]
                with open(f"/eric-verl/ff/temp/verl/exp_script/dist/8tp1/out/{batch_idx}_tp{tp_size}_deterministic{enable_deterministic_inference}_iter{iter_idx}.json", "w") as f:
                    json.dump(out, f)
                
                engine.shutdown()