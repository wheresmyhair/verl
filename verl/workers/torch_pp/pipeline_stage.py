"""
Pipeline stage — wraps a partitioned HuggingFace model chunk.

Each stage holds a contiguous subset of transformer layers. Stage 0 also
holds the embedding; the last stage holds norm + lm_head.

Key design: instead of extracting layers and calling them individually
(which requires manually replicating HF's attention mask, rotary embedding,
and SDPA logic), we **prune the model in-place** and delegate to its own
forward(). This guarantees numerical equivalence with single-GPU execution.

This version uses **partial loading** via safetensors:
- Instantiate model on meta device (zero memory)
- Prune on meta device (free — just removes module references)
- Materialize only the remaining parameters
- Selectively load weights from safetensors (reads only needed shards/keys)

Peak HBM: only this stage's params (not the full model).
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .partitioner import compute_layer_assignment


def _prune_model_inplace(
    model: nn.Module,
    start_layer: int,
    end_layer: int,
    is_first: bool,
    is_last: bool,
):
    inner = model.model if hasattr(model, "model") else model

    # Keep only this stage's layers
    all_layers = list(inner.layers)
    kept_layers = all_layers[start_layer:end_layer]
    inner.layers = nn.ModuleList(kept_layers)

    # Do NOT change model.config.num_hidden_layers — some models (e.g. Qwen3)
    # use it internally for per-layer attention patterns (max_window_layers).
    # Changing it causes NaN. The causal mask works correctly with the
    # original value since it depends on seq_len, not layer count.

    if not is_last:
        if hasattr(inner, "norm"):
            inner.norm = nn.Identity()
        if hasattr(model, "lm_head"):
            model.lm_head = nn.Identity()

    del all_layers


def _reinit_non_persistent_buffers(model: nn.Module, config):
    """
    Reinitialize non-persistent buffers (e.g. rotary embeddings) that are
    computed during __init__ but NOT stored in state_dict / safetensors.

    After from_config(meta) + to_empty(), these buffers are all-zeros,
    which causes NaN in attention.
    """
    for name, module in model.named_modules():
        # Handle rotary embedding (used by Qwen, Llama, Mistral, etc.)
        if hasattr(module, "inv_freq") and hasattr(module, "rope_init_fn"):
            # Use the current device of inv_freq; rope_init_fn creates float32 by default
            inv_freq, attention_scaling = module.rope_init_fn(module.config, device=module.inv_freq.device)
            module.inv_freq = inv_freq
            module.original_inv_freq = inv_freq
            module.attention_scaling = attention_scaling


def _load_partial_weights(
    model: nn.Module,
    model_path: str,
    start_layer: int,
    end_layer: int,
    pp_rank: int,
    pp_size: int,
):
    """
    Selectively load weights from safetensors — reads only the shards
    containing this stage's keys.

    After pruning, layers are renumbered from 0. This function loads the
    original global-indexed weights and remaps them to local indices.
    """
    from safetensors import safe_open

    # Resolve HuggingFace hub model IDs to local paths
    if not os.path.isdir(model_path):
        try:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(model_path)
        except Exception as e:
            raise FileNotFoundError(
                f"Model path '{model_path}' is not a local directory and "
                f"could not be downloaded from HuggingFace Hub: {e}"
            )

    is_first = pp_rank == 0
    is_last = pp_rank == pp_size - 1

    # Determine needed key prefixes
    needed_prefixes = []
    for layer_idx in range(start_layer, end_layer):
        needed_prefixes.append(f"model.layers.{layer_idx}.")
    if is_first:
        needed_prefixes.append("model.embed_tokens.")
    if is_last:
        needed_prefixes.append("model.norm.")
        needed_prefixes.append("lm_head.")

    # Read weight map (multi-shard) or find single safetensors file
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]
    else:
        # Single safetensors file
        single_file = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(single_file):
            raise FileNotFoundError(
                f"Neither {index_path} nor {single_file} found in {model_path}"
            )
        # Build weight_map from the single file
        weight_map = {}
        with safe_open(single_file, framework="pt") as f:
            for key in f.keys():
                weight_map[key] = "model.safetensors"

    # Filter to needed keys and determine which shard files to open
    needed_keys = {}
    for param_name, shard_file in weight_map.items():
        if any(param_name.startswith(prefix) for prefix in needed_prefixes):
            needed_keys[param_name] = shard_file

    # Group by shard file
    shard_to_keys: Dict[str, List[str]] = {}
    for param_name, shard_file in needed_keys.items():
        shard_to_keys.setdefault(shard_file, []).append(param_name)

    # Load from each shard, remap keys
    remapped_dict = {}
    for shard_file, keys in shard_to_keys.items():
        shard_path = os.path.join(model_path, shard_file)
        with safe_open(shard_path, framework="pt") as f:
            for key in keys:
                tensor = f.get_tensor(key)
                # Remap layer indices: model.layers.{global_idx}.* -> model.layers.{local_idx}.*
                new_key = _remap_global_to_local(key, start_layer)
                remapped_dict[new_key] = tensor

    model.load_state_dict(remapped_dict, strict=False, assign=True)


def _remap_global_to_local(key: str, start_layer: int) -> str:
    """Remap global layer index to local (pruned) index."""
    prefix = "model.layers."
    if key.startswith(prefix):
        rest = key[len(prefix):]
        dot_idx = rest.index(".")
        global_idx = int(rest[:dot_idx])
        local_idx = global_idx - start_layer
        return f"{prefix}{local_idx}.{rest[dot_idx+1:]}"
    return key


def restore_global_layer_keys(
    state_dict: Dict[str, torch.Tensor],
    start_layer: int,
) -> Dict[str, torch.Tensor]:
    """
    Remap pruned local layer indices back to global indices.

    Inverse of the load mapping — used when collecting weights from all
    stages for vLLM weight loading.
    """
    remapped = {}
    prefix = "model.layers."
    for key, tensor in state_dict.items():
        if key.startswith(prefix):
            rest = key[len(prefix):]
            dot_idx = rest.index(".")
            local_idx = int(rest[:dot_idx])
            global_idx = local_idx + start_layer
            new_key = f"{prefix}{global_idx}.{rest[dot_idx+1:]}"
            remapped[new_key] = tensor
        else:
            remapped[key] = tensor
    return remapped


class PipelineStage(nn.Module):
    """
    One stage of a naive pipeline-parallel model.

    Wraps a pruned HuggingFace model. The model's own forward() handles
    attention masks, rotary embeddings, and SDPA dispatch correctly.
    """

    def __init__(
        self,
        model: nn.Module,
        pp_rank: int,
        pp_size: int,
        start_layer: int,
        end_layer: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = device
        self.dtype = dtype

        self.is_first = pp_rank == 0
        self.is_last = pp_rank == pp_size - 1

        self.model = model

        # Per-batch metadata (set before running schedule)
        self._micro_input_ids: List[torch.Tensor] = []
        self._micro_attention_mask: List[torch.Tensor] = []
        self._micro_position_ids: List[torch.Tensor] = []

        # Activation stash for backward
        # micro_batch_id -> (input_hidden, output_hidden)
        self._stash: Dict[int, Tuple[torch.Tensor, ...]] = {}

    # ──────────────────────────────────────────────────────────────────
    # Batch setup
    # ──────────────────────────────────────────────────────────────────

    def set_batch_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_micro_batches: int,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Pre-chunk the full batch into micro-batches and store locally.
        """
        B = input_ids.size(0)
        if B % num_micro_batches != 0:
            raise ValueError(
                f"Batch size {B} not divisible by num_micro_batches {num_micro_batches}"
            )

        if position_ids is None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        self._micro_input_ids = list(input_ids.chunk(num_micro_batches, dim=0))
        self._micro_attention_mask = list(attention_mask.chunk(num_micro_batches, dim=0))
        self._micro_position_ids = list(position_ids.chunk(num_micro_batches, dim=0))
        self._stash.clear()

    def clear_batch_data(self):
        """Free micro-batch data and stash after a step."""
        self._micro_input_ids.clear()
        self._micro_attention_mask.clear()
        self._micro_position_ids.clear()
        self._stash.clear()

    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────

    def forward_step(
        self,
        micro_batch_id: int,
        input_hidden: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Forward one micro-batch through this stage.

        Args:
            return_hidden: If True and this is the last stage, return hidden
                states [B, S, H] from BEFORE lm_head instead of logits
                [B, S, V].  Used with FusedLinearForPPO to avoid
                materializing the full logits tensor.
        """
        ids = self._micro_input_ids[micro_batch_id]
        attn_mask = self._micro_attention_mask[micro_batch_id]
        pos_ids = self._micro_position_ids[micro_batch_id]

        if return_hidden and self.is_last:
            # Call the inner model (embed + layers + norm) to get hidden
            # states WITHOUT the lm_head projection.  This avoids
            # materializing the [B, S, V] logits tensor.
            inner = self.model.model  # e.g. Qwen3Model
            kwargs = dict(
                attention_mask=attn_mask,
                position_ids=pos_ids,
                use_cache=False,
            )
            if self.is_first:
                self._stash[micro_batch_id] = (None,)
                kwargs["input_ids"] = ids
            else:
                assert input_hidden is not None
                self._stash[micro_batch_id] = (input_hidden,)
                kwargs["inputs_embeds"] = input_hidden
            inner_out = inner(**kwargs)
            hidden = inner_out[0]  # last_hidden_state [B, S, H]
            self._stash[micro_batch_id] = (self._stash[micro_batch_id][0], hidden)
            return hidden

        # Standard path: full model forward (returns logits on last stage,
        # hidden states on non-last stages).
        if self.is_first:
            self._stash[micro_batch_id] = (None,)
            output = self.model(
                input_ids=ids,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                use_cache=False,
            )
        else:
            assert input_hidden is not None, "Non-first stage requires input_hidden"
            self._stash[micro_batch_id] = (input_hidden,)
            output = self.model(
                input_ids=None,
                inputs_embeds=input_hidden,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                use_cache=False,
            )

        out_tensor = output.logits if hasattr(output, "logits") else output[0]
        self._stash[micro_batch_id] = (self._stash[micro_batch_id][0], out_tensor)
        return out_tensor

    @property
    def lm_head_weight(self) -> torch.Tensor:
        """Access lm_head weight for fused linear cross-entropy."""
        return self.model.lm_head.weight

    # ──────────────────────────────────────────────────────────────────
    # Backward
    # ──────────────────────────────────────────────────────────────────

    def backward_step(
        self,
        micro_batch_id: int,
        grad_output: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Backward one micro-batch through this stage.
        """
        input_hidden, output_hidden = self._stash.pop(micro_batch_id)

        if self.is_last and grad_output is None:
            pass
        elif grad_output is not None:
            output_hidden.backward(grad_output)
        else:
            raise ValueError(
                "Non-last stage requires grad_output, or last stage needs "
                "loss.backward() to have been called."
            )

        if not self.is_first and input_hidden is not None and input_hidden.grad is not None:
            return input_hidden.grad.detach()
        return None

    # ──────────────────────────────────────────────────────────────────
    # Construction — partial loading via safetensors
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        pp_rank: int,
        pp_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        enable_gradient_checkpointing: bool = False,
    ) -> "PipelineStage":
        """
        Load a HuggingFace model with partial loading — no full model
        ever loaded into memory.

        1. Instantiate on meta device (zero memory)
        2. Prune on meta device (free — just removes module references)
        3. Materialize only remaining parameters
        4. Selectively load weights from safetensors
        """
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        num_layers = config.num_hidden_layers

        assignments = compute_layer_assignment(num_layers, pp_size)
        start, end = assignments[pp_rank]

        print(f"[PP rank {pp_rank}] Partial-loading layers {start}..{end-1}")
        t0 = time.time()

        is_first = pp_rank == 0
        is_last = pp_rank == pp_size - 1

        # 1. Instantiate model on meta device (zero memory)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)

        # 2. Prune on meta device (free — just removes module references)
        _prune_model_inplace(model, start, end, is_first, is_last)

        # 3. Materialize only the remaining parameters (allocates memory)
        model.to_empty(device="cpu")

        # 4. Selectively load weights from safetensors
        _load_partial_weights(model, model_path, start, end, pp_rank, pp_size)

        # 5. Move to device (before reinit — .to(dtype) would cast float32
        # buffers like inv_freq to bf16, losing precision)
        model.to(device=device, dtype=dtype)

        # 6. Reinitialize non-persistent buffers (e.g. rotary embeddings)
        # that are computed during __init__ but not stored in safetensors.
        # Must happen AFTER .to() so the buffers are created in float32
        # on the correct device (rope_init_fn creates inv_freq as float32).
        _reinit_non_persistent_buffers(model, config)

        # 7. Enable gradient checkpointing to reduce activation memory.
        # Critical for 1F1B: earlier PP stages stash multiple micro-batch
        # autograd graphs during warmup. Without checkpointing, all
        # intermediate activations are held in memory simultaneously.
        if enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        t_load = time.time() - t0

        stage = cls(
            model=model,
            pp_rank=pp_rank,
            pp_size=pp_size,
            start_layer=start,
            end_layer=end,
            device=device,
            dtype=dtype,
        )

        num_params = sum(p.numel() for p in stage.parameters())
        print(
            f"[PP rank {pp_rank}] Model loaded: "
            f"layers {start}..{end-1}, "
            f"{num_params:,} params | "
            f"partial_load={t_load:.1f}s"
        )

        return stage

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get this stage's parameters as a state dict.

        Returns standard HF key format (strips the ``model.`` prefix added
        by ``PipelineStage`` wrapping the HF model as ``self.model``).
        """
        prefix = "model."
        return {
            k[len(prefix):] if k.startswith(prefix) else k: v.cpu()
            for k, v in self.state_dict().items()
        }

    def get_global_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict with global layer indices restored.

        Used for weight collection across all PP stages.
        """
        local_sd = self.get_state_dict()
        return restore_global_layer_keys(local_sd, self.start_layer)

    def load_state_dict_from_full(self, full_state_dict: Dict[str, torch.Tensor]):
        """
        Load weights from a full model state dict (picking only this stage's
        parameters). Accepts standard HF key format and maps to internal keys.
        """
        prefix = "model."
        own_keys = set(self.state_dict().keys())
        filtered = {}
        for k, v in full_state_dict.items():
            internal_key = prefix + k
            if internal_key in own_keys:
                filtered[internal_key] = v
            elif k in own_keys:
                filtered[k] = v
        self.load_state_dict(filtered, strict=False)
