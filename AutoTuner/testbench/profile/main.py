import argparse
import json
import os
from typing import List

from .configs.config_struct import ProfileConfig
from .launcher.get_data_launch import LaunchDataCollectionForOps
from .launcher.profile_launch import LaunchProfileForOps
from AutoTuner.utils.structs import InputTestCase
from AutoTuner.utils.nvtx import enable_nvtx_profiling


def validate_args(args):
    args.real_test_cases_file = os.path.join(args.test_cases_dir, args.test_cases_file)
    assert os.path.exists(args.real_test_cases_file), f"{args.real_test_cases_file} not found"
    args.real_profile_config_file = os.path.join(args.config_dir, args.profile_config_file)
    assert os.path.exists(args.real_profile_config_file), f"{args.real_profile_config_file} not found"
    args.real_override_model_config_file = os.path.join(args.config_dir, args.override_model_config_file)
    assert os.path.exists(args.real_override_model_config_file), f"{args.real_override_model_config_file} not found"
    args.real_override_tf_config_file = os.path.join(args.config_dir, args.override_tf_config_file)
    assert os.path.exists(args.real_override_tf_config_file), f"{args.real_override_tf_config_file} not found"
    
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Profile Operators")
    parser.add_argument("--model-name", type=str, required=True, default="Qwen/Qwen3-0.6B", help="model name to test")
    parser.add_argument("--test_cases-dir", type=str, required=False, default="AutoTuner/testbench/profile/cases/", help="Base dir holds the test cases files, shall not modify this")
    parser.add_argument("--config-dir", type=str, required=False, default="AutoTuner/testbench/profile/configs/", help="Base dir holds the config files, shall not modify this")
    parser.add_argument("--test-cases-file", type=str, required=True, default="qwen3_0_6b.json", help="file in cases folder contains")
    parser.add_argument("--override-model-config-file", type=str, required=False, default="override_model_config.json", help="huggingface model configs to override")
    parser.add_argument("--override-tf-config-file", type=str, required=False, default="override_tf_config.json", help="TransformerConfig to override")
    parser.add_argument("--profile-config-file", type=str, required=False, default="profile_config.json", help="profile config file")
    
    parser.add_argument("--test-ops-list", type=int, nargs="+", default=None)
    parser.add_argument("--test-case-idxs", type=int, nargs="+", default=None)
    
    parser.add_argument("--output-dir", type=str, required=False, default="outputs", help="output directory to save the database results and nsys profile results, actual output_dir is args.output_dir/model_name/profile_mode")
    args = parser.parse_args()
    args = validate_args(args)
    return args


def handle_test_cases(args) -> List[InputTestCase]:
    with open(args.real_test_cases_file, "r") as fp:
        json_test_cases = json.load(fp)
    test_cases = []
    for json_test_case in json_test_cases["cases"]:
        test_case = InputTestCase(**json_test_case)
        test_cases.append(test_case)
    return test_cases


def handle_profile_configs(args) -> ProfileConfig:
    with open(args.real_profile_config_file, "r") as fp:
        profile_config = json.load(fp)
    return ProfileConfig(**profile_config)


def handle_model_config(args) -> dict:
    with open(args.real_override_model_config_file, "r") as fp:
        override_model_config = json.load(fp)
    return override_model_config

def handle_tf_config(args) -> dict:
    with open(args.real_tf_config_config, "r") as fp:
        override_tf_config = json.load(fp)
    return override_tf_config


def call_launcher(args, test_cases: List[InputTestCase], profile_config: ProfileConfig, override_model_config: dict, override_tf_config: dict):
    launcher_cls = LaunchProfileForOps if profile_config.profile_mode else LaunchDataCollectionForOps
    launcher = launcher_cls(profile_config, test_cases, model_name=args.model_name, override_model_kwargs=override_model_config, override_tf_config_kwargs=override_tf_config)
    
    if profile_config.profile_mode:
        enable_nvtx_profiling()
    
    if args.test_case_idxs is None and args.test_ops_list is None:
        launcher.run_all_supported_ops()
    else:
        launcher.run_op_list(args.test_ops_list, args.test_case_idxs)
    
    if not profile_config.profile_mode:
        total_timing_db, total_memory_db = launcher.return_results()
        output_dir = os.path.join(args.output_dir, args.model_name, "collect_data")
        with open(os.path.join(output_dir, "timing.json"), "a+") as fp:
            json.dump(total_timing_db, fp)
        with open(os.path.join(output_dir, "memory_activation.json"), "a+") as fp:
            json.dump(total_memory_db["activation"], fp)
        with open(os.path.join(output_dir, "memory_weights.json"), "a+") as fp:
            json.dump(total_memory_db["weights"], fp)


if __name__ == "main":
    args = parse_args()
    test_cases = handle_test_cases(args)
    profile_config = handle_profile_configs(args)
    override_model_config = handle_model_config(args)
    override_tf_config = handle_tf_config(args)
    call_launcher(args, test_cases, profile_config, override_model_config, override_tf_config)