from ..configs.config_struct import ProfileConfig
from AutoTuner.utils.structs import InputTestCase

class LaunchProfileForOps:
    def __init__(
        self,
        profile_config: ProfileConfig,
        test_cases: list[InputTestCase],
        model_name: str,
        override_model_kwargs: dict,
        override_tf_config_kwargs: dict,
    ):
        assert profile_config.profile_mode == True, "Nsys profile should enable profile mode."
        super().__init__(
            profile_config=profile_config,
            test_cases=test_cases,
            model_name=model_name,
            override_model_kwargs=override_model_kwargs,
            override_tf_config_kwargs=override_tf_config_kwargs,
        )


