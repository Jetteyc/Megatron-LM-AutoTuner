import os
from typing import List, Optional

import torch
from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import PretrainedConfig

from AutoTunner.utils.memory import MemoryTracker, MemoryTrackerContext
from AutoTunner.utils.models import get_model_input
from AutoTunner.utils.structs import InputTestCase
from AutoTunner.utils.timing import Timer, TimerContext

from ..ops.embedding import LanguageModelEmbeddingForTest

os.environ["NVTE_NVTX_ENABLED"] = "1"


class TestLanguageModelEmbedding:
    def __init__(
        self,
        tf_config: TransformerConfig,
        hf_config: PretrainedConfig,
        test_cases: List[InputTestCase],
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
        self.test_cases = test_cases
        self.model_config = hf_config
        self.profile_mode = profile_mode
        self.warmup = warmup

    def test_one_case(
        self, batch_size: int, seqlen: int, shape: Optional[List[int]], system: str
    ) -> None:
        input_ids, attention_mask, position_ids, packed_sequence_params = (
            get_model_input(self.model_config, batch_size, seqlen, shape, system)
        )

        if self.profile_mode:
            if self.warmup > 0:
                for _ in range(self.warmup):
                    self.embedding(input_ids, position_ids)
            return self.embedding(input_ids, position_ids)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if self.warmup > 0:
            for _ in range(self.warmup):
                self.embedding(input_ids, position_ids)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        name = f"{self.module_name} forward batch_size={batch_size} seqlen={seqlen} shape={shape} system={system}"
        with TimerContext(name) as timer_ctx:
            self.embedding(input_ids, position_ids)
        self.timing_db[name] = timer_ctx.result
        self.memory_db["activations"].update(
            {name: self.embedding.get_activation_memory()}
        )

    def run_all_tests(self):
        for case in self.test_cases:
            self.test_one_case(case.batch_size, case.seqlen, case.shape, case.system)
