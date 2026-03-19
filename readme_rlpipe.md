# Modifications based on verl for RLPipe

## `verl/single_controller/base/decorator.py`
Add dispatcher based on index mapping, for routing based on rollout history length.

```python
def dispatch_dynamic_index_data_proto(worker_group, *args, **kwargs):
    """
    Dynamic dispatch function that reads index mapping from DataProto meta_info
    Supports uneven chunking and configurable mappings per call
    """
    from verl.single_controller.base.worker_group import WorkerGroup
    from verl.protocol import DataProto, DataProtoFuture
    assert isinstance(worker_group, WorkerGroup)
    
    # Extract index mapping from the first DataProto argument's meta_info
    index_mapping = None
    data_length = None
    for arg in args:
        if isinstance(arg, (DataProto, DataProtoFuture, BatchMeta)) and hasattr(arg, 'meta_info'):
            index_mapping = arg.meta_info.get('dp_index_mapping')
            data_length = len(arg)
            break
    
    # If no mapping provided, create even distribution
    if index_mapping is None:
        print(f"No index mapping provided, creating even distribution")
        index_mapping = {}
        num_samples_per_rank = data_length // worker_group.world_size
        for rank in range(worker_group.world_size):
            if rank == worker_group.world_size - 1:
                index_mapping[rank] = list(range(rank * num_samples_per_rank, data_length))
            else:
                index_mapping[rank] = list(range(rank * num_samples_per_rank, (rank + 1) * num_samples_per_rank))
    
    print(f"dispatch_dynamic_index_data_proto: {index_mapping=}")
    # Validate mapping covers all workers
    expected_ranks = set(range(worker_group.world_size))
    provided_ranks = set(index_mapping.keys())
    if expected_ranks != provided_ranks:
        raise ValueError(f"Index mapping must cover all DP ranks. Expected: {expected_ranks}, Got: {provided_ranks}")
    
    splitted_args = []
    for arg in args:
        assert isinstance(arg, (DataProto, DataProtoFuture, BatchMeta))
        chunks = []
        for dp_rank in range(worker_group.world_size):
            indices = index_mapping[dp_rank]
            if len(indices) == 0:
                # Create empty chunk with proper structure
                chunk = arg.select_idxs([])
            else:
                chunk = arg.select_idxs(indices)
            chunks.append(chunk)
        splitted_args.append(chunks)

    splitted_kwargs = {}
    for key, val in kwargs.items():
        assert isinstance(val, (DataProto, DataProtoFuture, BatchMeta))
        chunks = []
        for dp_rank in range(worker_group.world_size):
            indices = index_mapping[dp_rank]
            if len(indices) == 0:
                chunk = val.select_idxs([])
            else:
                chunk = val.select_idxs(indices)
            chunks.append(chunk)
        splitted_kwargs[key] = chunks

    return splitted_args, splitted_kwargs

def make_dynamic_index_dispatch_fn():
    """Factory function to create the dispatch mode"""
    return {
        "dispatch_fn": dispatch_dynamic_index_data_proto,
        "collect_fn": collect_dp_compute_data_proto,
    }

# Register the dispatch mode globally
DYNAMIC_INDEX_DISPATCH = make_dynamic_index_dispatch_fn()
```

## `verl/workers/megatron_workers.py`

`ActorRolloutRefWorker`'s `generate_sequences` is modified to use the dynamic index dispatcher.

```python
# @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
@register(dispatch_mode=DYNAMIC_INDEX_DISPATCH)
@GPUMemoryLogger(role="generate_sequences", logger=logger)
@DistProfiler.annotate(color="red")
def generate_sequences(self, prompts: DataProto):
    ...
```

## `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

For recording the rollout history length.

```python
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            # tensor_parallel_size=1,
            ...
```
```python
            ...
            response_lengths = [] # rlpipe modification
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    response_lengths.append(len(response_ids)) # rlpipe modification
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        non_tensor_batch["response_lengths"] = np.array(response_lengths, dtype=object) # rlpipe modification
        ...
```

## `verl/trainer/ppo/ray_trainer.py`

Support for response length balancing.

```python
class SampleLengthBalancer:
    ...
```

```python
    sample_lengths = {k: [[] for _ in range(self.config.trainer.total_epochs)] for k in range(len(self.train_dataset))} # rlpipe modification
    # {sample_idx: [[lengths_ep0_n0, lengths_ep0_n1, ...], [lengths_ep1_n0, lengths_ep1_n1, ...], ...]}
    response_length_balance = True # rlpipe modification
```

```python                            
    print(f"{gen_batch_output=}")
        
    if epoch != 0 and response_length_balance: # rlpipe modification
        # if epoch == 0, samples are being evenly distributed to dp ranks
        # if epoch != 0, samples are being balanced based on response lengths in the previous epoch
        # example: index_mapping = {0: [0, ], 1: [1,2,3]} # -> the 0th sample IN THIS GEN_BATCH_OUTPUT is mapped to dp rank 0, vise versa.
        # 1. get lengths in this batch
        sample_idx_this_batch = gen_batch_output.non_tensor_batch["index"]
        print(f"sample_idx_this_batch: {sample_idx_this_batch}")
        estimated_lengths_this_batch = []
        for sample_idx in sample_idx_this_batch:
            # since we drop last incomplete batch, there's chance that there's no length for this sample in the previous epoch
            # try to find the last length for this sample
            length_found = False
            for length in sample_lengths[sample_idx][:epoch][::-1]:
                if length:
                    estimated_lengths_this_batch.append(np.mean(length))
                    length_found = True
                    break
            if not length_found:
                estimated_lengths_this_batch.append(1024)
        print(f"estimated_lengths_this_batch: {estimated_lengths_this_batch}")
        # 2. rebalance
        balancer = SampleLengthBalancer(num_workers=self.actor_rollout_wg.world_size)
        index_mapping = balancer.balance(estimated_lengths_this_batch)
        print(f"index_mapping: {index_mapping}")
        balanced_num_samples_per_worker = len(estimated_lengths_this_batch) // self.actor_rollout_wg.world_size
        for k,v in index_mapping.items():
            print(
                f"index mapping rank: {k}, num_samples: "
                f"balanced: {len(v)}; "
                f"original: {sum(estimated_lengths_this_batch[k*balanced_num_samples_per_worker:(k+1)*balanced_num_samples_per_worker])}"
            )
        gen_batch_output.meta_info['dp_index_mapping'] = index_mapping
        
    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
    
    print(f"after generate_sequences: {gen_batch_output=}")
    for idx, sample_idx in enumerate(gen_batch_output.non_tensor_batch["index"]): # the global sample index across all datasets.
        sample_lengths[sample_idx][epoch].append(gen_batch_output.non_tensor_batch["response_lengths"][idx])
    print(f"sample_lengths: {sample_lengths}")
```

## `examples/data_preprocess/gsm8k.py`

For testing the response length balancing.

```python
    train_dataset = train_dataset.select(range(4)) # rlpipe modification for testing
    test_dataset = test_dataset.select(range(14)) # rlpipe modification for testing
```

## `examples/data_preprocess/gsm8k_all.py`

Backup script for the original `gsm8k.py` script.

## `exp_script/run_vllm_megatron_dp4.sh`

Script for running the training with 4 DP ranks.

