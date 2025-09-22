import logging
import os
import statistics
import subprocess as sp

from ..utils.logging import log_rank0


def get_gpu_memory() -> float:
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return statistics.mean(memory_free_values)


class MemoryTrackerContext:
    def __init__(self, name: str = "", log_level: int = logging.INFO):
        self.name = name
        self.log_level = log_level

    def __enter__(self) -> "MemoryTrackerContext":
        import torch

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        self.start_mem = torch.cuda.memory_allocated()
        self.start_peak_mem = torch.cuda.max_memory_allocated()
        self.start_real_mem = get_gpu_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        import torch

        torch.cuda.synchronize()
        self.end_mem = torch.cuda.memory_allocated()
        self.end_peak_mem = torch.cuda.max_memory_allocated()
        self.end_real_mem = get_gpu_memory()
        self.peak_mem_diff = self.end_peak_mem - self.start_peak_mem
        self.mem_diff = self.end_mem - self.start_mem
        self.real_mem_diff = self.end_real_mem - self.start_real_mem
        self.result = {
            "mem_diff": self.mem_diff,
            "peak_mem_diff": self.peak_mem_diff,
            "real_mem_diff": self.real_mem_diff,
        }
        reserved_mem = torch.cuda.memory_reserved()
        log_rank0(
            f"[MemoryTracker] {self.name} | "
            f"Memory Diff: {self.mem_diff / (1024 ** 2):.2f} MB | "
            f"Peak Memory Diff: {self.peak_mem_diff / (1024 ** 2):.2f} MB | "
            f"Real Memory Diff: {self.real_mem_diff / (1024 ** 2):.2f} MB | "
            f"Reserved Memory: {reserved_mem / (1024 ** 2):.2f} MB",
            level=self.log_level,
        )


class MemoryTracker:
    @staticmethod
    def track_function(func: callable, *args, **kwargs) -> tuple:
        with MemoryTrackerContext(name=func.__name__) as tracker:
            result = func(*args, **kwargs)
        return result, {
            "mem_diff": tracker.mem_diff,
            "peak_mem_diff": tracker.peak_mem_diff,
            "real_mem_diff": tracker.real_mem_diff,
        }

    @staticmethod
    def track_decorator(preffix: str = "") -> callable:
        def decorator(func: callable) -> callable:
            def wrapper(*args, **kwargs):
                with MemoryTrackerContext(name=f"{preffix} {func.__name__}") as tracker:
                    result = func(*args, **kwargs)
                return result, {
                    "mem_diff": tracker.mem_diff,
                    "peak_mem_diff": tracker.peak_mem_diff,
                    "real_mem_diff": tracker.real_mem_diff,
                }

            return wrapper

        return decorator


class ActivationHook:
    def __init__(
        self,
        enable: bool = True,
        module_name: str = "",
        logging_level: int = logging.INFO,
    ):
        self.activation_tensors = []
        self.enable = enable
        self.module_name = module_name
        self.logging_level = logging_level

    def save_hook(self, x) -> object:
        if self.enable:
            log_rank0(
                f"[{self.module_name} save] tensor shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}",
                level=self.logging_level,
            )
            self.activation_tensors.append(x)
        return x  # 必须返回 x，否则计算图会出错

    def load_hook(self, x) -> object:
        if self.enable:
            log_rank0(
                f"[{self.module_name} load] tensor shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}",
                level=self.logging_level,
            )
        return x

    def clear(self) -> None:
        self.activation_tensors = []

    def get_activation_memory(self) -> int:
        mem = 0
        for tensor in self.activation_tensors:
            mem += tensor.numel() * tensor.element_size()
        return mem
