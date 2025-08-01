# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

import requests
import tqdm

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

SERVER_BACKEND = "VLLM"
USE_OFFLOAD = True
BASE_URL = "http://localhost:30000"
API_KEY = "EMPTY"
MAX_RETRIES = 3
BASE_DELAY = 2
MAX_WORKERS = 32
MODEL_NAME = "genrm-demo"
GENRM_PROMPT_TEMPLATE = """
The following is a math problem and an AI solution:

[Math Problem]

{problem}

[AI Solution]

{solution}

Your task is to review and critique the solution step by step, and output whether the AI solution is correct.

Please put your final answer (i.e., 'True' or 'False') in \\boxed{{}}.
""".strip()


def vllm_execute_method(task="sleep"):
    assert task in ["sleep", "wake_up"], f"Invalid task: {task}"
    url_root = BASE_URL
    response = requests.post(url_root + "/" + task)
    assert response.status_code == 200


def get_response(problem, solution_str, ground_truth):
    prompt = GENRM_PROMPT_TEMPLATE.format(problem=problem, solution=solution_str)
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(MAX_RETRIES):
        try:
            headers = {"Content-Type": "application/json"}
            chat_url = f"{BASE_URL}/v1/chat/completions"
            data = {"model": MODEL_NAME, "messages": messages}
            output = requests.post(chat_url, headers=headers, json=data, timeout=30)
            response = output.json()["choices"][0]["message"]["content"]
            return response
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print("Exception: ", repr(e))
                delay = BASE_DELAY * (2**attempt)
                print(f"Retrying in {delay} seconds...")
                sleep(delay)
            else:
                print(f"Failed after {MAX_RETRIES} attempts. Error: {e}")

    raise ConnectionRefusedError(f"Failed to run the model for {prompt}!")


def compute_reward(response):
    reward_score = 0.0
    try:
        boxed_result = last_boxed_only_string(response)
        if boxed_result is not None:
            result = remove_boxed(boxed_result)
            reward_score = float(result == "True")
    except Exception as e:
        print(e)
    return reward_score


def compute_score(data_source, solution_str, ground_truth, extra_info, index):
    split = extra_info["split"]
    from verl.utils.reward_score import default_compute_score

    func_rm_score = default_compute_score(data_source, solution_str, ground_truth, extra_info)

    if split == "test":
        return func_rm_score, index
    else:
        problem = extra_info["question"]
        response = get_response(problem, solution_str, ground_truth)
        if response is not None:
            reward_score = compute_reward(response)
        else:
            reward_score = 0.0

        return reward_score, index


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    results = []
    indexes = list(range(len(data_sources)))
    if SERVER_BACKEND == "VLLM" and USE_OFFLOAD:
        vllm_execute_method("wake_up")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for data_source, solution_str, ground_truth, extra_info, index in zip(
            data_sources, solution_strs, ground_truths, extra_infos, indexes, strict=True
        ):
            future = executor.submit(compute_score, data_source, solution_str, ground_truth, extra_info, index)
            time.sleep(0.001 * random.random())
            futures.append(future)

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())
        results = sorted(results, key=lambda x: x[-1], reverse=False)
        results = [result[0] for result in results]

    if SERVER_BACKEND == "VLLM" and USE_OFFLOAD:
        vllm_execute_method("sleep")
    return results
