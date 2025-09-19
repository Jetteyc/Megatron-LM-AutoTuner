import os
from typing import List, Optional, Tuple, Any, Iterator

import torch
from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import PretrainedConfig

from AutoTunner.utils.memory import MemoryTracker, MemoryTrackerContext
from AutoTunner.utils.models import get_thd_model_input_from_bshd
from AutoTunner.utils.structs import InputTestCase
from AutoTunner.utils.timing import Timer, TimerContext

from ..ops.embedding import LanguageModelEmbeddingForTest

os.environ["NVTE_NVTX_ENABLED"] = "1"


class TestLanguageModelEmbedding:
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        test_case: InputTestCase,
        batch_data_generator: List | Iterator,
        scatter_to_sequence_parallel: bool = True,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        profile_mode: bool = False,
        warmup: int = 2,
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
        self.timing_db = {}
        self.memory_db = {"weights": {}, "activations": {}}
        self.memory_db["weights"].update(
            {self.module_name: memory_tracker_ctx.get_memory_usage()}
        )
        self.test_case = test_case
        self.batch_data_generator = batch_data_generator
        self.model_config = hf_config
        self.profile_mode = profile_mode
        self.warmup = warmup

    def run_test(self):
        batch = next(self.batch_data_generator)
        batch = batch.to(torch.cuda.current_device())
        batch = batch.contiguous()
        input_ids_rmpad, attention_mask, position_ids_rmpad, packed_seq_params = get_thd_model_input_from_bshd(batch)
        
        if self.profile_mode:
            if self.warmup > 0:
                for _ in range(self.warmup):
                    self.embedding(input_ids_rmpad, position_ids_rmpad)
            return self.embedding(input_ids_rmpad, position_ids_rmpad)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if self.warmup > 0:
            for _ in range(self.warmup):
                self.embedding(input_ids_rmpad, position_ids_rmpad)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        name = f"{self.module_name} forward {self.test_case}"
        with TimerContext(name) as timer_ctx:
            self.embedding(input_ids_rmpad, position_ids_rmpad)
        self.timing_db[name] = timer_ctx.result
        self.memory_db["activations"].update(
            {name: self.embedding.get_activation_memory()}
        )

    def get_results(self) -> Tuple[dict, dict]:
        return self.timing_db, self.memory_db