"""
Response length profiler for rollout phase.

Collects per-sample response lengths and computes:
- Distribution stats (mean, median, p50/p90/p99, min, max)
- Per-prompt variance (when n>1 samples per prompt)
- Padding waste ratio
- Histograms for wandb logging

Usage:
    profiler = ResponseLengthProfiler(max_response_len=7168)
    profiler.record(response_lengths, prompt_ids)
    metrics = profiler.compute_metrics()
    # Log metrics to wandb
    wandb.log(metrics)
    # Save raw data for offline analysis
    profiler.save_raw("lengths_step42.json")
"""

import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class ResponseLengthProfiler:
    """Collects response lengths and computes distribution + per-prompt stats."""

    def __init__(self, max_response_len: int, max_seq_len: Optional[int] = None):
        """
        Args:
            max_response_len: Maximum response length (R) — used to compute padding waste.
            max_seq_len: Maximum total sequence length (S) — if set, used for total padding waste.
        """
        self.max_response_len = max_response_len
        self.max_seq_len = max_seq_len
        self._lengths: List[int] = []
        self._prompt_ids: List[str] = []  # prompt identifier per sample
        self._prompt_lengths: List[int] = []

    def reset(self):
        """Clear all recorded data."""
        self._lengths.clear()
        self._prompt_ids.clear()
        self._prompt_lengths.clear()

    def record(
        self,
        response_lengths: Union[torch.Tensor, np.ndarray, List[int]],
        prompt_ids: Optional[Union[List[str], np.ndarray]] = None,
        prompt_lengths: Optional[Union[torch.Tensor, np.ndarray, List[int]]] = None,
    ):
        """Record response lengths for a batch.

        Args:
            response_lengths: [B] actual response lengths (number of non-pad response tokens).
            prompt_ids: [B] identifiers for each prompt (e.g., dataset index). Used to
                group samples from the same prompt when n>1.
            prompt_lengths: [B] actual prompt lengths (for total padding waste calculation).
        """
        if isinstance(response_lengths, torch.Tensor):
            response_lengths = response_lengths.cpu().tolist()
        elif isinstance(response_lengths, np.ndarray):
            response_lengths = response_lengths.tolist()
        self._lengths.extend(response_lengths)

        if prompt_ids is not None:
            if isinstance(prompt_ids, np.ndarray):
                prompt_ids = prompt_ids.tolist()
            self._prompt_ids.extend([str(pid) for pid in prompt_ids])
        else:
            # Generate unique IDs
            offset = len(self._prompt_ids)
            self._prompt_ids.extend([f"unk_{offset + i}" for i in range(len(response_lengths))])

        if prompt_lengths is not None:
            if isinstance(prompt_lengths, torch.Tensor):
                prompt_lengths = prompt_lengths.cpu().tolist()
            elif isinstance(prompt_lengths, np.ndarray):
                prompt_lengths = prompt_lengths.tolist()
            self._prompt_lengths.extend(prompt_lengths)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute distribution and per-prompt metrics.

        Returns dict suitable for wandb.log().
        """
        if not self._lengths:
            return {}

        lengths = np.array(self._lengths, dtype=np.float64)
        metrics = {}

        # --- Distribution stats ---
        metrics["response_length/mean"] = float(lengths.mean())
        metrics["response_length/median"] = float(np.median(lengths))
        metrics["response_length/std"] = float(lengths.std())
        metrics["response_length/min"] = float(lengths.min())
        metrics["response_length/max"] = float(lengths.max())
        metrics["response_length/p10"] = float(np.percentile(lengths, 10))
        metrics["response_length/p50"] = float(np.percentile(lengths, 50))
        metrics["response_length/p90"] = float(np.percentile(lengths, 90))
        metrics["response_length/p99"] = float(np.percentile(lengths, 99))

        # --- Padding waste ---
        total_padded_tokens = len(lengths) * self.max_response_len
        total_actual_tokens = float(lengths.sum())
        padding_waste = 1.0 - total_actual_tokens / total_padded_tokens if total_padded_tokens > 0 else 0.0
        metrics["response_length/padding_waste"] = padding_waste
        metrics["response_length/total_samples"] = len(lengths)
        metrics["response_length/total_actual_tokens"] = total_actual_tokens

        if self.max_seq_len and self._prompt_lengths:
            prompt_arr = np.array(self._prompt_lengths, dtype=np.float64)
            total_seq_padded = len(lengths) * self.max_seq_len
            total_seq_actual = float((prompt_arr + lengths).sum())
            metrics["response_length/total_padding_waste"] = 1.0 - total_seq_actual / total_seq_padded
            metrics["prompt_length/mean"] = float(prompt_arr.mean())

        # --- Per-prompt variance (for n>1 sampling) ---
        prompt_groups: Dict[str, List[float]] = defaultdict(list)
        for pid, length in zip(self._prompt_ids, self._lengths):
            prompt_groups[pid].append(float(length))

        # Only compute variance for prompts with multiple samples
        multi_sample_groups = {k: v for k, v in prompt_groups.items() if len(v) > 1}
        if multi_sample_groups:
            per_prompt_stds = [np.std(v) for v in multi_sample_groups.values()]
            per_prompt_ranges = [max(v) - min(v) for v in multi_sample_groups.values()]
            per_prompt_means = [np.mean(v) for v in multi_sample_groups.values()]
            per_prompt_cvs = [
                np.std(v) / np.mean(v) if np.mean(v) > 0 else 0.0
                for v in multi_sample_groups.values()
            ]

            metrics["response_length/per_prompt_std_mean"] = float(np.mean(per_prompt_stds))
            metrics["response_length/per_prompt_std_max"] = float(np.max(per_prompt_stds))
            metrics["response_length/per_prompt_range_mean"] = float(np.mean(per_prompt_ranges))
            metrics["response_length/per_prompt_range_max"] = float(np.max(per_prompt_ranges))
            metrics["response_length/per_prompt_cv_mean"] = float(np.mean(per_prompt_cvs))
            metrics["response_length/num_unique_prompts"] = len(prompt_groups)
            metrics["response_length/samples_per_prompt"] = float(
                np.mean([len(v) for v in multi_sample_groups.values()])
            )

            # Variance of per-prompt means — shows how much prompts differ
            metrics["response_length/between_prompt_std"] = float(np.std(per_prompt_means))

        return metrics

    def get_histogram_data(self, num_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Return histogram (counts, bin_edges) for plotting."""
        lengths = np.array(self._lengths)
        counts, bin_edges = np.histogram(lengths, bins=num_bins, range=(0, self.max_response_len))
        return counts, bin_edges

    def save_raw(self, path: str):
        """Save raw length data for offline analysis."""
        data = {
            "response_lengths": self._lengths,
            "prompt_ids": self._prompt_ids,
            "prompt_lengths": self._prompt_lengths if self._prompt_lengths else None,
            "max_response_len": self.max_response_len,
            "max_seq_len": self.max_seq_len,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def save_step(self, save_dir: str, step: int):
        """Append this step's data to the cumulative JSONL file.

        Each line is one step's data — easy to load incrementally in a notebook.
        Also saves the full per-sample data for this step.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Per-step raw data
        step_data = {
            "step": step,
            "response_lengths": self._lengths,
            "prompt_ids": self._prompt_ids,
            "prompt_lengths": self._prompt_lengths if self._prompt_lengths else None,
        }
        jsonl_path = os.path.join(save_dir, "response_lengths.jsonl")
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(step_data) + "\n")

    @staticmethod
    def load_all_steps(save_dir: str) -> List[Dict]:
        """Load all steps from the JSONL file for notebook visualization."""
        import os
        jsonl_path = os.path.join(save_dir, "response_lengths.jsonl")
        steps = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path) as f:
                for line in f:
                    if line.strip():
                        steps.append(json.loads(line))
        return steps

    @staticmethod
    def load_raw(path: str) -> "ResponseLengthProfiler":
        """Load raw data from a saved file."""
        with open(path, "r") as f:
            data = json.load(f)
        profiler = ResponseLengthProfiler(
            max_response_len=data["max_response_len"],
            max_seq_len=data.get("max_seq_len"),
        )
        profiler._lengths = data["response_lengths"]
        profiler._prompt_ids = data["prompt_ids"]
        profiler._prompt_lengths = data.get("prompt_lengths") or []
        return profiler
