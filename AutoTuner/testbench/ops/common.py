import logging
import torch
from abc import ABC
from typing import Union

from AutoTuner.utils.memory import ActivationHook, MemoryTracker, get_memory_str


class CommonOpsForTest(ABC):
    def __init__(
        self,
        hook_activation: bool = False,
        module_name: str = "common_ops",
        hook_type: str = 'saved_tensors',
        logging_level: int = logging.INFO,
    ):
        self.hook_type = hook_type
        self._measured_activation_memory = 0
        
        if hook_activation and self.hook_type == 'forward_output':
            self.register_forward_hook(self._forward_output_hook)
            self.activation_hook = ActivationHook(
                enable=False, module_name=module_name, logging_level=logging_level
            )
        else:
            self.activation_hook = ActivationHook(
                enable=(hook_activation and self.hook_type == 'saved_tensors'), 
                module_name=module_name, 
                logging_level=logging_level
            )
        self.activation_tensors = []

    def _forward_output_hook(self, module, input, output: Union[torch.Tensor, tuple, list]):
        self._measured_activation_memory = 0
        if isinstance(output, torch.Tensor):
            self._measured_activation_memory = output.nelement() * output.element_size()
        elif isinstance(output, (list, tuple)):
            for t in output:
                if isinstance(t, torch.Tensor):
                    self._measured_activation_memory += t.nelement() * t.element_size()
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"[DEBUG 1/3 | Hook Execution] Hook calculated memory: {self._measured_activation_memory} bytes")
        
    def get_activation_memory(self) -> int:
        if self.hook_type == 'forward_output':
            return self._measured_activation_memory
        else: 
            return self.activation_hook.get_activation_memory()
