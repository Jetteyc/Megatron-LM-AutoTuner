import logging
from AutoTunner.utils.memory import MemoryTracker, ActivationHook

class CommonOpsForTest:
    def __init__(self, hook_activation=False, module_name="common_ops", logging_level=logging.INFO):
        self.activation_hook = ActivationHook(
            enable=hook_activation,
            module_name=module_name,
            logging_level=logging_level
        )
        self.activation_tensors = []
    
    def get_activation_memory(self):
        return self.activation_hook.get_activation_memory()