import os
from typing import List, Optional, Tuple, Any, Iterator

import torch
from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import PretrainedConfig

from AutoTuner.utils.memory import MemoryTracker, MemoryTrackerContext
from AutoTuner.utils.model_inputs import get_thd_model_input_from_bshd
from AutoTuner.utils.structs import InputTestCase
from AutoTuner.utils.timing import Timer, TimerContext
from AutoTuner.utils.nested_dict import NestedDict

from ..ops.embedding import LanguageModelEmbeddingForTest

os.environ["NVTE_NVTX_ENABLED"] = "1"


class TestLanguageModelEmbedding:
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        scatter_to_sequence_parallel: bool = True,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        profile_mode: bool = False,
        warmup_iters: int = 2,
    ):
        with MemoryTrackerContext("Embedding init") as memory_tracker_ctx:
            self.embedding = LanguageModelEmbeddingForTest(
                tf_config,
                hf_config,
                scatter_to_sequence_parallel=scatter_to_sequence_parallel,
                tp_group=tp_group,
                hook_activation=not profile_mode,
            )
        self.module_name = "Embedding"
        
        """
        timing_db structure:
        {
            "Embedding":
            {
                "forward":
                {
                    InputTestCase(batch_size=..., micro_batch_size=..., seqlen=..., max_token_len=..., shape='thd', system='megatron') : time_in_seconds,
                    ...
                },
                "backward":
                {
                    InputTestCase(batch_size=..., micro_batch_size=..., seqlen=..., max_token_len=..., shape='thd', system='megatron') : time_in_seconds,
                    ...
                }
            },
            ...
        }
        """
        """memory_db structure:
        {
            "weights": {
                "Embedding": memory_in_bytes,
                ...
            },
            "activations": {
                "Embedding":
                {
                    InputTestCase(batch_size=..., micro_batch_size=..., seqlen=..., max_token_len=..., shape='thd', system='megatron'): memory_in_bytes,
                    ...
                },
                ...
            }
        }
        """
        self.timing_db = NestedDict()
        self.memory_db = {"weights": {}, "activations": NestedDict()}
        self.memory_db["weights"][self.module_name] = memory_tracker_ctx.get_memory_usage()
        self.model_config = hf_config
        self.profile_mode = profile_mode
        self.warmup_iters = warmup_iters

    def run_test(self, test_case: InputTestCase, batch_data_generator: Iterator):
        batch = next(batch_data_generator)
        batch = batch.to(torch.cuda.current_device())
        batch = batch.contiguous()
        input_ids_rmpad, attention_mask, position_ids_rmpad, packed_seq_params = get_thd_model_input_from_bshd(batch)
        
        if self.profile_mode:
            if self.warmup_iters > 0:
                for _ in range(self.warmup_iters):
                    self.embedding(input_ids_rmpad, position_ids_rmpad)
            return self.embedding(input_ids_rmpad, position_ids_rmpad)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if self.warmup_iters > 0:
            for _ in range(self.warmup_iters):
                self.embedding(input_ids_rmpad, position_ids_rmpad)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        name = f"{self.module_name} forward {test_case}"
        with TimerContext(name) as timer_ctx:
            self.embedding(input_ids_rmpad, position_ids_rmpad)
        self.timing_db[self.module_name]["forward"][test_case] = timer_ctx.result
        self.memory_db["activations"][self.module_name][test_case] = self.embedding.get_activation_memory()

    def get_results(self) -> Tuple[dict, dict]:
        return self.timing_db, self.memory_db