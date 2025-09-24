import abc
import logging
from abc import ABC

from AutoTuner.utils.memory import ActivationHook, MemoryTracker, get_memory_str


class CommonOpsForTest(ABC):
    def __init__(
        self,
        hook_activation: bool = False,
        module_name: str = "common_ops",
        logging_level: int = logging.INFO,
    ):
        self.activation_hook = ActivationHook(
            enable=hook_activation, module_name=module_name, logging_level=logging_level
        )
        self.activation_tensors = []

    def get_activation_memory(self) -> int:
        return get_memory_str(self.activation_hook.get_activation_memory(), human_readable=True)
