"""
Inference stage — lightweight wrapper for no-grad inference within fused forward.

Each GPU holds a training stage at its normal PP rank and an inference stage
at the **reversed** rank:
    infer_rank = pp_size - 1 - training_pp_rank

So:
    GPU0: training stage 0, inference stage P-1 (last  — has norm + lm_head)
    GPU3: training stage 3, inference stage 0   (first — has embedding)

The inference stage is used during pipeline bubbles to compute old_log_probs
in-band, eliminating the separate inference phase.
"""

from typing import Dict, List, Optional

import torch

from .loss import gather_response_log_probs, log_probs_from_logits
from .pipeline_stage import PipelineStage


class InferenceStage:
    """
    Wraps a PipelineStage for inference-only forward passes.

    - Loads model at the reversed rank (``infer_rank = pp_size - 1 - train_rank``)
    - All parameters have ``requires_grad=False``
    - Forward runs under ``torch.no_grad()``
    - No activation stashing (no backward needed)
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

        self.is_first_infer = self.infer_rank == 0
        self.is_last_infer = self.infer_rank == pp_size - 1

        # Load model at the reversed rank
        self.stage = PipelineStage.from_pretrained(
            model_path=model_path,
            pp_rank=self.infer_rank,
            pp_size=pp_size,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        self.stage.eval()
        self.stage.requires_grad_(False)

        # Per-batch metadata
        self._micro_input_ids: List[torch.Tensor] = []
        self._micro_attention_mask: List[torch.Tensor] = []
        self._micro_position_ids: List[torch.Tensor] = []

    def set_batch_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_micro_batches: int,
        position_ids: Optional[torch.Tensor] = None,
        use_remove_padding: bool = False,
    ):
        """Pre-chunk the full batch into micro-batches."""
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        # Delegate to underlying PipelineStage (handles rmpad internally)
        self.stage.set_batch_data(
            input_ids, attention_mask, num_micro_batches, position_ids,
            use_remove_padding=use_remove_padding,
        )
        # Mirror the stage's processed micro-batches
        self._micro_input_ids = self.stage._micro_input_ids
        self._micro_attention_mask = self.stage._micro_attention_mask
        self._micro_position_ids = self.stage._micro_position_ids
        self._micro_nnz = self.stage._micro_nnz

    def clear_batch_data(self):
        """Free micro-batch data after a step."""
        self._micro_input_ids.clear()
        self._micro_attention_mask.clear()
        self._micro_position_ids.clear()
        self.stage.clear_batch_data()

    @torch.no_grad()
    def forward_step(
        self,
        micro_batch_id: int,
        input_hidden: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Forward one micro-batch through this inference stage (no grad).

        Args:
            return_hidden: If True and this is the last infer stage, return
                hidden states [B, S, H] instead of logits [B, S, V].
        """
        ids = self._micro_input_ids[micro_batch_id]
        attn_mask = self._micro_attention_mask[micro_batch_id]
        pos_ids = self._micro_position_ids[micro_batch_id]

        if return_hidden and self.is_last_infer:
            # Call inner model only (no lm_head) to avoid materializing [B,S,V]
            inner = self.stage.model.model
            kwargs = dict(
                attention_mask=attn_mask,
                position_ids=pos_ids,
                use_cache=False,
            )
            if self.is_first_infer:
                kwargs["input_ids"] = ids
            else:
                assert input_hidden is not None
                kwargs["inputs_embeds"] = input_hidden
            inner_out = inner(**kwargs)
            return inner_out[0]  # [B, S, H]

        if self.is_first_infer:
            output = self.stage.model(
                input_ids=ids,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                use_cache=False,
            )
        else:
            assert input_hidden is not None, "Non-first inference stage requires input_hidden"
            output = self.stage.model(
                input_ids=None,
                inputs_embeds=input_hidden,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                use_cache=False,
            )

        out_tensor = output.logits if hasattr(output, "logits") else output[0]
        return out_tensor

    @torch.no_grad()
    def compute_log_probs(
        self,
        micro_batch_id: int,
        logits: torch.Tensor,
        response_start_positions: torch.Tensor,
        max_resp_len: int,
    ) -> torch.Tensor:
        """
        Compute old_log_probs from logits on the last inference stage.
        """
        assert self.is_last_infer, "compute_log_probs only on last inference stage"

        input_ids = self._micro_input_ids[micro_batch_id]
        full_log_probs = log_probs_from_logits(logits, input_ids)
        resp_log_probs = gather_response_log_probs(
            full_log_probs, response_start_positions, max_resp_len
        )
        return resp_log_probs

    @torch.no_grad()
    def compute_log_probs_fused(
        self,
        micro_batch_id: int,
        hidden_states: torch.Tensor,
        response_start_positions: torch.Tensor,
        max_resp_len: int,
    ) -> torch.Tensor:
        """
        Compute old_log_probs from hidden states using FusedLinearForPPO.
        Never materializes [B, S, V] logits.

        Handles both padded [micro_B, S] and packed [1, nnz] formats.
        """
        from verl.utils.experimental.torch_functional import FusedLinearForPPO

        assert self.is_last_infer, "compute_log_probs_fused only on last inference stage"

        input_ids = self._micro_input_ids[micro_batch_id]
        fused = FusedLinearForPPO(chunk_size=512)

        if (self.stage._micro_cu_seqlens
                and self.stage._micro_cu_seqlens[micro_batch_id] is not None):
            # === Packed format: cu_seqlens-based shifted labels ===
            from flash_attn.bert_padding import pad_input

            cu = self.stage._micro_cu_seqlens[micro_batch_id]
            ids_flat = input_ids.squeeze(0)  # [nnz]
            shifted = torch.empty_like(ids_flat)
            for j in range(len(cu) - 1):
                s, e = cu[j].item(), cu[j + 1].item()
                shifted[s:e - 1] = ids_flat[s + 1:e]
                shifted[e - 1] = 0

            lp, _ = fused.forward(
                hidden_states=hidden_states,
                vocab_weights=self.stage.lm_head_weight,
                input_ids=shifted.unsqueeze(0),
                temperature=1.0,
            )  # [1, nnz]

            # Extract valid positions (drop last per seq)
            valid_parts = []
            for j in range(len(cu) - 1):
                s, e = cu[j].item(), cu[j + 1].item()
                valid_parts.append(lp[0, s:e - 1])
            valid_lp = torch.cat(valid_parts)  # [nnz - nseqs]

            # Re-pad to [micro_B, S-1]
            _mb_B, _mb_S = self.stage._micro_batch_shape
            unpad_indices = self.stage._micro_unpad_indices[micro_batch_id]
            seq_lens_list = cu.diff().tolist()
            shifted_indices = []
            offset = 0
            for sl in seq_lens_list:
                shifted_indices.append(unpad_indices[offset:offset + sl - 1])
                offset += sl
            shifted_idx = (torch.cat(shifted_indices) if shifted_indices
                           else unpad_indices[:0])
            adjusted_idx = ((shifted_idx // _mb_S) * (_mb_S - 1)
                            + (shifted_idx % _mb_S))

            padded_lp = pad_input(
                valid_lp.unsqueeze(-1), adjusted_idx, _mb_B, _mb_S - 1
            ).squeeze(-1)  # [micro_B, S-1]

            resp_log_probs = gather_response_log_probs(
                padded_lp, response_start_positions, max_resp_len
            )
        else:
            # === Standard padded format ===
            rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
            full_log_probs, _ = fused.forward(
                hidden_states=hidden_states,
                vocab_weights=self.stage.lm_head_weight,
                input_ids=rolled_labels,
                temperature=1.0,
            )
            full_log_probs = full_log_probs[:, :-1]  # [B, S-1]

            resp_log_probs = gather_response_log_probs(
                full_log_probs, response_start_positions, max_resp_len
            )
        return resp_log_probs

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get this stage's parameters as a CPU state dict."""
        return {k: v.cpu() for k, v in self.stage.state_dict().items()}

    def load_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Load weights into this inference stage."""
        self.stage.load_state_dict(state_dict)
        self.stage.to(self.device)

    def to(self, device):
        """Move to device."""
        self.stage.to(device)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        return self

    def parameters(self):
        """Return stage parameters."""
        return self.stage.parameters()
