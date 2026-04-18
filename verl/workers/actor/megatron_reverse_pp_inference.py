"""
Megatron reverse-PP inference stage for fused forward.

Each rank holds:
- Training layers: [r·N/P, (r+1)·N/P)           (Megatron standard PP)
- Inference layers: [(P-1-r)·N/P, (P-r)·N/P)    (reverse direction, this module)

iF flows in reverse: rank P-1 → rank P-2 → ... → rank 0 (computes log_probs).
tF flows forward:    rank 0 → rank 1 → ... → rank P-1 (computes loss).

This replaces the 3.4 GB full HF replica with a 1/P sharded model (~850 MB
at 1.7B/PP=4). Combined with gloo P2P in reverse direction, iF becomes a
true distributed operation with no redundant compute.

Reuses torch_pp's `PipelineStage.from_pretrained` for partial loading
(meta device → prune → materialize → selective safetensors load).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from verl.workers.torch_pp.pipeline_stage import PipelineStage


class MegatronReversePPInferenceStage:
    """PP-sharded HF model for inference, loaded at the reversed PP position.

    Args:
        model_path: HF repo id or local path.
        train_pp_rank: this rank's training PP position (0..pp_size-1).
        pp_size: total PP size.
        device: CUDA device.
        dtype: param dtype (bf16 for bf16 training).
    """

    def __init__(
        self,
        model_path: str,
        train_pp_rank: int,
        pp_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
    ):
        self.train_pp_rank = train_pp_rank
        self.pp_size = pp_size
        self.infer_rank = pp_size - 1 - train_pp_rank
        self.device = device
        self.dtype = dtype
        self.model_path = model_path

        self.is_first_infer = self.infer_rank == 0            # embedding + first layers
        self.is_last_infer = self.infer_rank == pp_size - 1   # last layers + lm_head

        # Partial load via PipelineStage (meta → prune → materialize).
        self.stage = PipelineStage.from_pretrained(
            model_path=model_path,
            pp_rank=self.infer_rank,
            pp_size=pp_size,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            enable_gradient_checkpointing=False,
        )
        self.stage.eval()
        self.stage.requires_grad_(False)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.stage.parameters())

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_hidden: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """Run one forward pass through this PP shard.

        - First inference rank (infer_rank == 0): must pass ``input_ids``
          (+ attention_mask, position_ids). Output: hidden state [B, S, H].
        - Middle ranks: must pass ``input_hidden``. Output: hidden state.
        - Last inference rank (infer_rank == P-1):
          - If return_hidden: output hidden state (for fused log_probs
            computation outside).
          - Else: output logits [B, S, V].

        For HF-based model, we use `inputs_embeds` on non-first ranks to
        pass the hidden state from the previous rank.
        """
        if self.is_first_infer:
            assert input_ids is not None
            if return_hidden and self.is_last_infer:
                # Single-rank case: P=1. Run inner model only (no lm_head).
                inner = self.stage.model.model
                out = inner(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                return out[0]  # [B, S, H]
            out = self.stage.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
        else:
            assert input_hidden is not None, "non-first infer stage requires input_hidden"
            if return_hidden and self.is_last_infer:
                inner = self.stage.model.model
                out = inner(
                    input_ids=None,
                    inputs_embeds=input_hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                return out[0]  # [B, S, H]
            out = self.stage.model(
                input_ids=None,
                inputs_embeds=input_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
        # Non-last ranks return hidden_states (logits would fail since there's no lm_head on pruned middle stages)
        # But HF AutoModelForCausalLM always has lm_head on last stage. On non-last stages we pruned lm_head in
        # PipelineStage.from_pretrained, so the output is the hidden state from the final kept layer.
        out_tensor = out.logits if hasattr(out, "logits") else out[0]
        return out_tensor

    @torch.no_grad()
    def compute_log_probs_from_hidden(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        response_length: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Last inference rank: compute log_probs from hidden state.

        Mirrors MegatronFusedInferenceStage.forward_log_probs's slicing:
        Returns ``[B, response_length]`` log_probs for the response tokens.

        Uses chunked cross_entropy to avoid materializing [B*S, V] fp32.
        """
        assert self.is_last_infer
        # hidden_states: [B, S, H], input_ids: [B, S]
        B, S, H = hidden_states.shape
        lm_head = self.stage.model.lm_head  # (V, H) linear
        vocab_size = lm_head.out_features

        # Compute logits in chunks along (B*S) to cap fp32 peak memory.
        hs_flat = hidden_states.reshape(-1, H)  # [B*S, H]
        ids_flat = input_ids.reshape(-1)  # [B*S]
        labels = ids_flat.roll(shifts=-1)  # shift: labels[i] = ids[i+1]
        N = hs_flat.size(0)

        log_probs_flat = torch.empty(N, dtype=self.dtype, device=hidden_states.device)
        import os as _os_ce
        ce_chunk = int(_os_ce.environ.get("RLPIPE_CE_CHUNK", "1024"))
        for s in range(0, N, ce_chunk):
            e = min(s + ce_chunk, N)
            chunk_logits = lm_head(hs_flat[s:e]).float()  # [chunk, V]
            if temperature != 1.0:
                chunk_logits = chunk_logits / temperature
            nll = F.cross_entropy(chunk_logits, labels[s:e], reduction="none")
            log_probs_flat[s:e] = (-nll).to(self.dtype)

        # Reshape back: log_probs[i, j] = log_prob of token ids[i, j+1] given prefix.
        log_probs = log_probs_flat.reshape(B, S)  # positions 0..S-1, last is garbage (rolled)

        # Extract response slice: Megatron contract is log_probs[:, -response_length-1:-1].
        # Using packed layout (everything padded), response tokens are at positions
        # [S-response_length, S-1], log_probs at those positions come from positions
        # [S-response_length-1, S-2] of the log_probs tensor.
        resp_log_probs = log_probs[:, -response_length - 1 : -1].contiguous()
        return resp_log_probs.to(torch.float32)

    def to(self, device):
        """Move the inference stage (all params + buffers) to a device."""
        self.stage.to(device)
        self.device = device
        return self

    def __repr__(self) -> str:
        return (
            f"MegatronReversePPInferenceStage("
            f"infer_rank={self.infer_rank}/{self.pp_size}, "
            f"train_pp_rank={self.train_pp_rank}, "
            f"first={self.is_first_infer}, last={self.is_last_infer}, "
            f"num_params={self.num_parameters():,}, device={self.device})"
        )
