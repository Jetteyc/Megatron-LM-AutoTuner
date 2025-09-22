from .launcher import Launcher
from ..configs.config_struct import ProfileConfig
from AutoTuner.utils.structs import InputTestCase
from AutoTuner.utils.nested_dict import NestedDict

class LaunchDataCollectionForOps(Launcher):
    def __init__(
        self,
        profile_config: ProfileConfig,
        test_cases: list[InputTestCase],
        model_name: str,
        override_model_kwargs: dict,
        override_tf_config_kwargs: dict,
    ):
        assert profile_config.profile_mode == False, "Data collection should not be in profile mode."
        super().__init__(
            profile_config=profile_config,
            test_cases=test_cases,
            model_name=model_name,
            override_model_kwargs=override_model_kwargs,
            override_tf_config_kwargs=override_tf_config_kwargs,
        )
        
        self.total_timing_db = NestedDict()
        self.total_memory_db = {"weights": {}, "activations": NestedDict()}

    def run_op(self, op_name: str):
        op_class_instance = super().run_op()
        timing_db, memory_db = op_class_instance.get_results()
        self.total_timing_db.merge(timing_db)
        self.total_memory_db["weights"].update(memory_db["weights"])
        self.total_memory_db["activations"].merge(memory_db["activations"])
    
    def run_op_list(self, op_name_list: list[str]):
        for op_name in op_name_list:
            print(f"Running operator: {op_name}")
            self.run_op(op_name)
    
    def run_all_supported_ops(self):
        self.run_op_list(self.all_supported_ops)
    
    def return_results(self):
        return self.total_timing_db, self.total_memory_db