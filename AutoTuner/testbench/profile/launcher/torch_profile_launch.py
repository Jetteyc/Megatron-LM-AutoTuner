# import os
import torch
# import torch.distributed as dist

from AutoTuner.utils.structs import InputTestCase
from verl.utils.memory_utils import MemorySnapshotSampler, enable_memory_visualize

from ..configs.config_struct import ProfileConfig, ProfileMode, TorchProfilerConfig
from ..op_mapping import OP_TEST_MAPPING
from .launcher import Launcher


class LaunchTorchProfileForOps(Launcher):
    def __init__(
        self,
        profile_config: ProfileConfig,
        test_cases: list[InputTestCase],
        model_name: str,
        override_model_kwargs: dict,
        override_tf_config_kwargs: dict,
        torch_profiler_config: TorchProfilerConfig,
    ):
        assert (
            profile_config.profile_mode == ProfileMode.torch_profiler
        ), "Nsys profile should enable torch profiler mode."
        super().__init__(
            profile_config=profile_config,
            test_cases=test_cases,
            model_name=model_name,
            override_model_kwargs=override_model_kwargs,
            override_tf_config_kwargs=override_tf_config_kwargs,
        )
        self.torch_profiler_config = torch_profiler_config
        self.torch_profiler_config.schedule = torch.profiler.schedule(
            wait=0,
            warmup=profile_config.warmup_iters,
            active=1,
            repeat=1,
        )
        self.torch_profiler_config.on_trace_ready = (
            torch.profiler.tensorboard_trace_handler(
                self.torch_profiler_config.output_dir
            )
        )
        self.prof = torch.profiler.profile(
            activities=self.torch_profiler_config.activities,
            schedule=self.torch_profiler_config.schedule,
            on_trace_ready=self.torch_profiler_config.on_trace_ready,
            record_shapes=self.torch_profiler_config.record_shapes,
            profile_memory=self.torch_profiler_config.profile_memory,
            with_stack=self.torch_profiler_config.with_stack,
            with_flops=self.torch_profiler_config.with_flops,
            with_modules=self.torch_profiler_config.with_modules,
        )
        self.snapshot_sampler = MemorySnapshotSampler(out_dir=self.torch_profiler_config.output_dir)
        enable_memory_visualize(
            trace_alloc_max_entries=200000, # 可以根据需要调整
            context='all',
            stacks='all'
        )

    def _run_op(self, op_name: str, test_case_idxs: list[int]):
        op_test_class = OP_TEST_MAPPING.get(op_name)
        if op_test_class is None:
            raise ValueError(f"Operator '{op_name}' is not supported.")
        op_class_instance = op_test_class(
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            tp_group=self.tp_group,
            profile_mode=self.profile_config.profile_mode,
            warmup_iters=self.profile_config.warmup_iters,
        )
        print(f"Running operator: {op_name}")
        if test_case_idxs is None:
            test_case_idxs = list(range(len(self.test_cases)))
        test_cases = [self.test_cases[i] for i in test_case_idxs]
        for test_case in test_cases:
            batch_data_generator = self.datasets.get_batch_generator(
                test_case
            )
            op_class_instance.run_test(test_case, batch_data_generator)
        return op_class_instance

    def run_op(self, op_name: str, test_case_idxs: list[int], inner_prof: bool = True):
        op_test_class = OP_TEST_MAPPING.get(op_name)
        output_dir = self.torch_profiler_config.output_dir
        # os.makedirs(output_dir, exist_ok=True)
        # current_rank = dist.get_rank() 
        # snapshot_file_path = os.path.join(output_dir, f"memory_snapshot_{op_name}_rank_{current_rank}.pickle")
        # torch.cuda.memory._record_memory_history(max_entries=100000)
        if op_test_class is None:
            raise ValueError(f"Operator '{op_name}' is not supported.")
        # torch.cuda.memory._record_memory_history(max_entries=100000)
        op_class_instance = op_test_class(
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            tp_group=self.tp_group,
            profile_mode=self.profile_config.profile_mode,
            warmup_iters=self.profile_config.warmup_iters,
            # snapshot_file_name=snapshot_file_path
        )
        print(f"Running operator: {op_name}")
        if test_case_idxs is None:
            test_case_idxs = list(range(len(self.test_cases)))
        test_cases = [self.test_cases[i] for i in test_case_idxs]
        if inner_prof:
            self.prof.start()
        for i in range(self.profile_config.warmup_iters + 1):
            for test_case in test_cases:
                batch_data_generator = self.datasets.get_batch_generator(
                    test_case
                )
                op_class_instance.run_test(test_case, batch_data_generator)
            if inner_prof:
                self.prof.step()
        if inner_prof:
            self.prof.stop()
        # torch.cuda.memory._dump_snapshot(snapshot_file_path)
        # torch.cuda.memory._record_memory_history(enabled=None)
        # print(f"Memory snapshot saved to {snapshot_file_path}")
        self.snapshot_sampler.dump_memory_snapshot(
            tag=f"{op_name}", # 使用 op_name 作为 tag
            out_dir=output_dir,
            # sub_dir 可以不指定，让它自动按时间戳创建
        )
        return op_class_instance

    def run_op_list(
        self,
        op_name_list: list[str],
        test_case_idxs: list[int],
        inner_prof: bool = True,
    ):
        if op_name_list is None:
            op_name_list = self.all_supported_ops
        if not inner_prof:
            self.prof.start()
        for i in range(self.profile_config.warmup_iters + 1):
            for op_name in op_name_list:
                print(f"Running operator: {op_name}")
                self.run_op(op_name, test_case_idxs, inner_prof=inner_prof)
        if not inner_prof:
            self.prof.stop()

    def run_all_supported_ops(self, test_case_idxs: list[int]):
        self.run_op_list(self.all_supported_ops, test_case_idxs)
