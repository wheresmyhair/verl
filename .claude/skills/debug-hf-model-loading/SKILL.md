---
name: debug-hf-model-loading
description: >
  Guidance for debugging numerical issues when loading/partitioning HuggingFace
  models for pipeline parallelism or other distributed strategies. Covers
  NaN/precision issues from partial weight loading, non-persistent buffers,
  dtype casting, and NCCL deadlocks. Use when facing NaN, logit mismatches,
  precision divergence, or P2P communication hangs in PP setups.
argument-hint: [issue-description]
---

# Debugging HF Model Loading & Pipeline Parallelism

Lessons learned from building torch naive pipeline parallelism for verl,
loading Qwen3/Llama-style models via safetensors partial loading.

---

## 1. Non-Persistent Buffers (Root Cause of NaN)

**Problem**: When using `from_config(meta)` + `to_empty(device="cpu")` +
`load_state_dict(assign=True)` for memory-efficient partial loading,
**non-persistent buffers are lost**. They are:
- Computed during `__init__` but registered with `persistent=False`
- NOT saved in `state_dict()` or safetensors files
- After `to_empty()`, they become all-zeros

**Key example**: `model.rotary_emb.inv_freq` — a float32 tensor computed by
`rope_init_fn()` during model init. All-zero inv_freq produces NaN in
attention because cos/sin of the position embeddings become degenerate.

**Detection pattern**:
```python
# Compare buffers between from_pretrained and from_config
model_a = AutoModelForCausalLM.from_pretrained(path, ...)
with torch.device("meta"):
    model_b = AutoModelForCausalLM.from_config(config, ...)
model_b.to_empty(device="cpu")

bufs_a = dict(model_a.named_buffers())
bufs_b = dict(model_b.named_buffers())
for name in bufs_a:
    in_sd = name in dict(model_a.state_dict())
    if not in_sd:
        print(f"NON-PERSISTENT: {name}")  # These will be broken after to_empty
```

**Fix pattern**:
```python
def _reinit_non_persistent_buffers(model, config):
    for name, module in model.named_modules():
        if hasattr(module, "inv_freq") and hasattr(module, "rope_init_fn"):
            inv_freq, attention_scaling = module.rope_init_fn(
                module.config, device=module.inv_freq.device
            )
            module.inv_freq = inv_freq
            module.original_inv_freq = inv_freq
            module.attention_scaling = attention_scaling
```

---

## 2. `.to(dtype=)` Casts Non-Persistent Buffers (Precision Loss)

**Problem**: Even after correctly reinitializing `inv_freq` as float32,
calling `model.to(device=device, dtype=torch.bfloat16)` will cast ALL
tensors — including the float32 `inv_freq` — to bf16. This loses precision
in rotary embeddings, causing logit divergence (~0.3-0.6 max diff) that
compounds through layers.

HuggingFace's `from_pretrained` keeps `inv_freq` as float32 because it
registers it *after* the dtype cast. The rotary forward explicitly forces
float32 computation (`torch.autocast(enabled=False)`), but if the buffer
itself is bf16, the precision is already lost.

**Detection**: Compare inv_freq dtype after loading:
```python
model_good = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
print(model_good.model.rotary_emb.inv_freq.dtype)  # float32

model_bad = ...  # your partial loading
print(model_bad.model.rotary_emb.inv_freq.dtype)  # bf16 = BAD
```

**Fix**: Reinitialize non-persistent buffers AFTER `.to(dtype)`, not before:
```python
model.to(device=device, dtype=dtype)           # Step 5: cast everything
_reinit_non_persistent_buffers(model, config)   # Step 6: restore float32 buffers
```

---

## 3. model.config.num_hidden_layers Must Not Change

**Problem**: After pruning layers for PP, it's tempting to update
`model.config.num_hidden_layers` to match the pruned count. But some models
use this config value internally for per-layer behavior selection:

- **Qwen3**: `max_window_layers` controls which layers use sliding window
  attention. The condition is `layer_idx >= config.max_window_layers`. If you
  reduce `num_hidden_layers`, the layer-to-window mapping breaks, causing NaN.

- **Other models** may use `num_hidden_layers` for attention pattern selection,
  gradient checkpointing boundaries, etc.

**Fix**: Leave `config.num_hidden_layers` at the original value. The causal
mask depends on `seq_len`, not layer count, so it works correctly without
modification.

---

## 4. NCCL P2P Deadlock in Sequential Pipeline

**Problem**: NCCL creates pairwise sub-communicators lazily on first P2P
between any two ranks. This init requires rank 0 to distribute `ncclUniqueId`
via `TCPStore`. In a sequential PP pattern (rank 0->1, then 1->2, etc.),
rank 0 finishes its send and moves on, but ranks 2-3 are still waiting,
causing a deadlock in the lazy init.

**Symptoms**:
- Hangs on the first `dist.send()` or `dist.recv()` call
- Works with 2 GPUs but deadlocks with 4+
- Adding warmup allreduce doesn't help (allreduce uses collective comms, P2P uses separate sub-comms)

**Fix**: Use gloo backend for PP pair groups:
```python
pair_groups = {}
for i in range(pp_size - 1):
    # new_group is a world-wide collective — all ranks must call it
    pair_group = dist.new_group(ranks=[i, i + 1], backend="gloo")
    if rank == i or rank == i + 1:
        pair_groups[i] = pair_group
```

Gloo avoids NCCL's lazy init entirely. Tensors must be CPU-staged:
```python
# Send
out_cpu = tensor.detach().contiguous().cpu()
dist.send(out_cpu, dst=dst_rank, group=pair_groups[pair_idx])

# Recv
buf = torch.empty(shape, dtype=dtype, device="cpu")
dist.recv(buf, src=src_rank, group=pair_groups[pair_idx])
hidden = buf.to(device)
```

For 1F1B schedules, use `isend` (non-blocking) to avoid circular deadlocks
between adjacent stages. Keep references to CPU send buffers to prevent GC
before isend completes.

---

## 5. Debugging Methodology for Numerical Issues

When outputs don't match between two model loading approaches:

### Step 1: Verify weights match
```python
for (na, pa), (nb, pb) in zip(model_a.named_parameters(), model_b.named_parameters()):
    if not torch.equal(pa.data, pb.data):
        print(f"DIFF: {na}")
```

### Step 2: If weights match, check non-persistent buffers
```python
bufs_a = dict(model_a.named_buffers())
bufs_b = dict(model_b.named_buffers())
for name in bufs_a:
    if name in bufs_b and not torch.allclose(bufs_a[name].float(), bufs_b[name].float()):
        print(f"BUFFER DIFF: {name}, dtype A={bufs_a[name].dtype} B={bufs_b[name].dtype}")
```

### Step 3: If buffers match, check dtypes
A float32 buffer and a bf16 buffer can have the same *values* (when compared
after casting) but produce different *computation results* due to intermediate
precision. Check dtypes explicitly, not just values.

### Step 4: Trace layer-by-layer with hooks
```python
outputs = {}
def make_hook(name):
    def hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(out, torch.Tensor):
            outputs[name] = out.detach().clone()
    return hook

model.model.embed_tokens.register_forward_hook(make_hook("embed"))
for i, layer in enumerate(model.model.layers):
    layer.register_forward_hook(make_hook(f"layer_{i}"))
```

Run forward on both models, compare outputs per layer. The first divergence
pinpoints the issue.

### Step 5: Check module attributes (not just params/buffers)
```python
for (na, ma), (nb, mb) in zip(model_a.named_modules(), model_b.named_modules()):
    attrs_a = {k: v for k, v in vars(ma).items()
               if not k.startswith("_") and not isinstance(v, (nn.Module, nn.Parameter, torch.Tensor))}
    attrs_b = {k: v for k, v in vars(mb).items()
               if not k.startswith("_") and not isinstance(v, (nn.Module, nn.Parameter, torch.Tensor))}
    for k in attrs_a:
        if k in attrs_b and attrs_a[k] != attrs_b[k]:
            print(f"{na}.{k}: {attrs_a[k]} vs {attrs_b[k]}")
```

---

## 6. PP Stage Memory Imbalance

The last PP stage holds `lm_head` which produces `[B, S, vocab_size]` logits.
For large vocabularies (e.g., Qwen3's 151936), this dominates memory:

- Stages 0-2: ~1.8 GB each (transformer layers only)
- Stage 3: ~8-13 GB (layers + lm_head + logits tensor)

Mitigation techniques (from verl's megatron/FSDP strategies):
- **Vocab-parallel logits**: Split lm_head across TP ranks
- **Fused linear_cross_entropy**: Never materialize full logits tensor
- **log_probs per-sample logsumexp**: Compute log_probs without storing full softmax
- **Entropy chunking**: Process vocabulary in chunks for entropy computation
- **Flash-Attn cross_entropy**: Use flash-attn's memory-efficient CE kernel

---

## 7. Quick Checklist

When building partial model loading for PP:

- [ ] `from_config(meta)` + `to_empty()` leaves non-persistent buffers zeroed
- [ ] `load_state_dict(assign=True, strict=False)` — missing keys stay uninitialized
- [ ] `.to(dtype=bf16)` casts ALL tensors including float32 buffers that need precision
- [ ] `config.num_hidden_layers` may be used internally — don't modify it
- [ ] NCCL pair sub-communicators deadlock in sequential patterns — use gloo
- [ ] `isend` needs buffer references kept alive until send completes
- [ ] `new_group()` is a world-wide collective — all ranks must call it
