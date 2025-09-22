import subprocess

from megatron.core import parallel_state as mpu

from .op_mapping import OP_TEST_MAPPING
from .config import ProfileConfig
from AutoTuner.utils.config import get_hf_model_config, get_mcore_model_config_from_hf_config
from AutoTuner.utils.structs import InputTestCase

class LaunchDataCollectionForOps:
    def __init__(
        self,
        profile_config: ProfileConfig,
        test_cases: list[InputTestCase],
        model_name: str,
        override_model_kwargs: dict,
        override_tf_config_kwargs: dict,
    ):
        self.model_name = model_name
        self.profile_config = profile_config
        self.hf_config = get_hf_model_config(model_name, **override_model_kwargs)
        self.tf_config = get_mcore_model_config_from_hf_config(self.hf_config, **override_tf_config_kwargs)
        self.tp_group = mpu.get_tensor_model_parallel_group()
        self.test_cases = test_cases

    def run_op(self, op_name: str):
        op_test_class = OP_TEST_MAPPING.get(op_name)
        if op_test_class is None:
            raise ValueError(f"Operator '{op_name}' is not supported.")
        op_class_instance = op_test_class(
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            tp_group=self.tp_group,
            profile_mode=self.profile_config.profile_mode,
            warmup=self.profile_config.warmup,
        )