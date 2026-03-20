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
    ):
        """Pre-chunk the full batch into micro-batches."""
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        self._micro_input_ids = list(input_ids.chunk(num_micro_batches, dim=0))
        self._micro_attention_mask = list(attention_mask.chunk(num_micro_batches, dim=0))
        self._micro_position_ids = list(position_ids.chunk(num_micro_batches, dim=0))

        # Also set batch data on the underlying PipelineStage
        self.stage.set_batch_data(input_ids, attention_mask, num_micro_batches, position_ids)

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
    ) -> torch.Tensor:
        """
        Forward one micro-batch through this inference stage (no grad).
        """
        ids = self._micro_input_ids[micro_batch_id]
        attn_mask = self._micro_attention_mask[micro_batch_id]
        pos_ids = self._micro_position_ids[micro_batch_id]

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
