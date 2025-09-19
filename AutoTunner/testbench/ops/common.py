import logging

from AutoTunner.utils.memory import ActivationHook, MemoryTracker


class CommonOpsForTest:
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
        return self.activation_hook.get_activation_memory()
