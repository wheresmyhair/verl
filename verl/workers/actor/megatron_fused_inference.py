"""
Megatron fused forward — HF-backed inference stage.

Provides a standalone inference-only wrapper that computes response log_probs
for a given batch via a full HF model, bypassing Megatron's pipeline-parallel
training path. Used by the fused schedule driver to run inference forward ops
inside Megatron training-forward / training-backward bubbles.

**Design choice — full replica, no PP sharding**
The torch_pp equivalent (`verl/workers/torch_pp/inference_stage.py`) shards the
inference model across PP ranks in reverse-direction order. For Megatron we
keep things simpler: each rank holds a full HF replica. At Qwen3-1.7B this is
~3.4 GB per rank, 13.6 GB aggregate on 4 ranks — affordable on 80 GB A100.
This avoids writing a second, Megatron-parallel set of P2P communicators and
shortens the MVP scope.

**Numerical contract**
`forward_log_probs(...)` returns log prob tensor with the same layout that
Megatron's `compute_log_prob` produces — specifically
``log_probs[:, -response_length - 1 : -1]`` — so it can be substituted for the
existing `old_log_probs` field in the GRPO loss without any caller changes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM
except ImportError as _import_err:  # pragma: no cover
    AutoModelForCausalLM = None
    _IMPORT_ERR = _import_err


class MegatronFusedInferenceStage:
    """HF-backed inference stage for the Megatron fused-forward port.

    Args:
        model_path: HF repo id or local path. Loaded once at construction.
        device: CUDA device for this rank's replica.
        dtype: weight/compute dtype. Match the Megatron training dtype.
        trust_remote_code: forwarded to `AutoModelForCausalLM.from_pretrained`.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
    ):
        if AutoModelForCausalLM is None:
            raise RuntimeError(
                "transformers is not installed — cannot build "
                "MegatronFusedInferenceStage"
            ) from _IMPORT_ERR
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.dtype = dtype
        self.model_path = model_path
        # flash_attention_2 is critical here: with 16K response lengths and
        # all-ones attention_mask, the sdpa/eager path materializes an
        # O(batch × heads × S²) matrix (~18 GB for batch-4/S-17K/fp16) and
        # blows runtime from seconds to minutes. flash_attention_2 handles
        # causal decoder masking in O(S) memory and matches Megatron's
        # flash backend for numerical parity.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2",
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def forward_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        response_length: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute response log probs for one micro-batch.

        Mirrors the slicing that Megatron's `compute_log_prob` does in
        ``megatron_actor.loss_func``: takes ``log_probs[:, -response_length-1:-1]``.

        Args:
            input_ids: [B, S] — concatenation of prompt + response tokens.
            attention_mask: [B, S] — 1 for valid, 0 for pad.
            position_ids: [B, S] — position ids (handles packed seqs if the
                caller built them correctly).
            response_length: number of response tokens at the end of the
                sequence. The returned log probs cover exactly this many
                positions (shifted by 1 so each entry is the log prob that
                the response token at position ``-response_length + i``
                would be emitted given the prefix up to ``-response_length + i - 1``).
            temperature: scales logits before softmax — Megatron applies
                this in its forward; we mirror it so the two log_prob
                distributions are numerically comparable.

        Returns:
            log_probs: [B, response_length] bfloat16 (or fp32 if dtype set).
        """
        # Pack to batch=1 + reset-style position_ids so HF's flash_attention_2
        # takes the `_prepare_from_posids` / `flash_varlen_fn` path and never
        # touches pad tokens. Running the padded forward directly burns ~10×
        # more compute (17K × 4 token micro-batch where ~90% is padding) and
        # takes the run from ~30 s to several minutes.
        B, S = input_ids.shape
        max_prompt = S - response_length
        valid_mask = attention_mask.bool()  # [B, S]
        lengths = attention_mask.sum(dim=-1).to(torch.long)  # [B]

        # packed_ids[0, j] = input_ids at valid positions, concatenated across B.
        packed_ids = input_ids[valid_mask].unsqueeze(0)  # [1, total_valid]
        # packed_pos[0, j] = within-seq offset. `cumsum(mask) - 1` at valid
        # positions gives 0..L_i-1 for each seq — vectorized, avoids a Python
        # loop over B.
        within_seq_pos = (torch.cumsum(attention_mask.to(torch.long), dim=-1) - 1)[
            valid_mask
        ]
        packed_pos = within_seq_pos.unsqueeze(0).to(torch.long)  # [1, total_valid]

        # HF's `_is_packed_sequence` check: batch=1 AND position_ids differs
        # from a pure monotone arange → treat as packed and take the
        # `flash_attn_varlen_func` path. Must pass attention_mask=None (the
        # 2D mask path would force `_upad_input` instead).
        outputs = self.model(
            input_ids=packed_ids,
            attention_mask=None,
            position_ids=packed_pos,
            use_cache=False,
        )
        logits = outputs.logits.squeeze(0)  # [total_valid, V]
        if temperature != 1.0:
            logits = logits / temperature

        # gathered[j] = log p(packed_ids[j+1] | attention restricted to seq_i).
        # The varlen mask from cu_seqlens ensures attention does not cross
        # seq boundaries, so this is equivalent to per-seq log_probs. The
        # `labels = packed_ids.roll(-1)` call produces wrong labels at each
        # seq_i → seq_{i+1} boundary (it feeds the next seq's first token),
        # but the scatter step below only reads the *last m_i - 1* entries
        # of each seq, all strictly before the boundary, so those garbage
        # entries are never touched.
        #
        # Cross-entropy is done in fp32 chunks to avoid materializing a
        # [total_valid × vocab=152064] fp32 logits tensor. Chunk size
        # controls peak memory: 4K tokens → 2.4 GB, 1K → 600 MB. For
        # fused forward path (where HF replica + Megatron training state
        # + stashed activations all reside on GPU simultaneously), use
        # a smaller chunk to stay within HBM budget. Env var allows
        # override for memory-constrained runs.
        packed_ids_1d = packed_ids.squeeze(0)  # [total_valid]
        labels = packed_ids_1d.roll(shifts=-1)
        total_valid = packed_ids_1d.size(0)
        packed_gathered = torch.empty(
            total_valid, dtype=self.dtype, device=input_ids.device
        )
        import os as _os_ce
        ce_chunk = int(_os_ce.environ.get("RLPIPE_CE_CHUNK", "1024"))
        for _s in range(0, total_valid, ce_chunk):
            _e = min(_s + ce_chunk, total_valid)
            nll = F.cross_entropy(
                logits[_s:_e].float(), labels[_s:_e], reduction="none"
            )
            packed_gathered[_s:_e] = (-nll).to(self.dtype)

        # Scatter per-seq response log_probs into [B, response_length].
        # Original Megatron slice is `gathered_padded[:, -response_length - 1 : -1]`.
        # In padded layout: slice position k → predicts token at padded index
        # (max_prompt + k). For k in [0, m_i - 1] this is R_1..R_m (real response).
        # For k in [m_i, response_length - 1] this predicts right-pad tokens
        # (garbage, but masked downstream, so we leave those as 0).
        #
        # In packed layout, R_{k+1}'s log_prob comes from the logit at packed
        # position `offset_i + n_i + k - 1`. Since L_i = n_i + m_i, that's
        # `offset_i + L_i - m_i + k - 1`, so out[i, :m_i] =
        # packed_gathered[offset_i + L_i - m_i - 1 : offset_i + L_i - 1].
        out = torch.zeros(
            B, response_length, dtype=self.dtype, device=input_ids.device
        )
        # `m_i = attention_mask[i, max_prompt:].sum()` counts real response tokens.
        # Single CPU transfer instead of B .item() calls.
        m_per_seq = attention_mask[:, max_prompt:].sum(dim=-1).to(torch.long).tolist()
        lengths_list = lengths.tolist()
        offset = 0
        for i in range(B):
            L_i = lengths_list[i]
            m_i = m_per_seq[i]
            if m_i > 0 and L_i >= m_i + 1:
                src_start = offset + L_i - m_i - 1
                src_end = offset + L_i - 1
                out[i, :m_i] = packed_gathered[src_start:src_end]
            offset += L_i
        return out

    @torch.no_grad()
    def forward_log_probs_packed(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        response_length: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Packed (remove-padding) variant — not yet implemented.

        Reserved for when the Megatron port switches from padded to packed
        layout. The Megatron baseline with `use_remove_padding=true` still
        pads on the boundary between Megatron's internal pipeline comm and
        HF's forward, so the padded path above is sufficient for Phase 1.
        """
        raise NotImplementedError(
            "packed forward_log_probs not implemented in Phase 1; use padded path"
        )

    def to(self, device) -> "MegatronFusedInferenceStage":
        """Move the HF replica across devices.

        Uses pinned CPU buffers and non-blocking copies for the GPU↔CPU
        direction — PyTorch's default `Module.to("cpu")` does per-parameter
        synchronous copies that benchmark at ~1.5 GB/s on this hardware,
        vs ~15-20 GB/s for pinned non-blocking DMA. At 16 GB HF-replica
        size (8B bf16) that's 10 s → ~1-2 s of transfer time.

        The pinned CPU buffer is lazily allocated on first offload and
        reused for subsequent offloads — it's a fixed 16 GB of pinned
        host memory at 8B, which is fine on cloud hosts with >256 GB RAM.
        """
        target = torch.device(device) if not isinstance(device, torch.device) else device

        if target.type == "cpu":
            # Lazy-init pinned CPU buffers, one per parameter, on first offload.
            if not hasattr(self, "_cpu_pinned_params") or not self._cpu_pinned_params:
                self._cpu_pinned_params = {}
                for name, p in self.model.named_parameters():
                    self._cpu_pinned_params[name] = torch.empty_like(
                        p.data, device="cpu", pin_memory=True
                    )
                for name, b in self.model.named_buffers():
                    self._cpu_pinned_params[f"__buf__{name}"] = torch.empty_like(
                        b.data, device="cpu", pin_memory=True
                    )
            # Non-blocking D2H copies into the pinned buffers, then reassign
            # .data pointers so PyTorch sees the params as on CPU.
            for name, p in self.model.named_parameters():
                if p.data.device.type == "cuda":
                    self._cpu_pinned_params[name].copy_(p.data, non_blocking=True)
                p.data = self._cpu_pinned_params[name]
            for name, b in self.model.named_buffers():
                k = f"__buf__{name}"
                if b.data.device.type == "cuda":
                    self._cpu_pinned_params[k].copy_(b.data, non_blocking=True)
                b.data = self._cpu_pinned_params[k]
            torch.cuda.synchronize()  # Wait for all non-blocking D2H copies
            torch.cuda.empty_cache()  # Actually free the GPU memory
            self.device = target
            return self

        # CPU → GPU (or GPU → GPU). Use non-blocking H2D from the pinned
        # CPU buffers if present, else fall back to Module.to() which does
        # the right thing for this direction (less hot path).
        if hasattr(self, "_cpu_pinned_params") and self._cpu_pinned_params:
            for name, p in self.model.named_parameters():
                if p.data.device.type == "cpu":
                    gpu_buf = torch.empty_like(p.data, device=target)
                    gpu_buf.copy_(p.data, non_blocking=True)
                    p.data = gpu_buf
            for name, b in self.model.named_buffers():
                if b.data.device.type == "cpu":
                    gpu_buf = torch.empty_like(b.data, device=target)
                    gpu_buf.copy_(b.data, non_blocking=True)
                    b.data = gpu_buf
            torch.cuda.synchronize()
        else:
            self.model.to(target)
        self.device = target
        return self

    def cpu_offload(self) -> "MegatronFusedInferenceStage":
        """Move the HF replica to CPU to free GPU memory during phases
        where inference is not active (e.g., Megatron's optimizer step)."""
        return self.to("cpu")

    def gpu_restore(self, device) -> "MegatronFusedInferenceStage":
        """Move back to GPU in preparation for the next inference pass."""
        return self.to(device)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def __repr__(self) -> str:
        return (
            f"MegatronFusedInferenceStage(model_path={self.model_path!r}, "
            f"device={self.device}, dtype={self.dtype}, "
            f"num_params={self.num_parameters()})"
        )
