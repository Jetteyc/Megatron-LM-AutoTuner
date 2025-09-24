import argparse
import json
import os
from typing import List

from AutoTuner.utils.nvtx import enable_nvtx_profiling
from AutoTuner.utils.structs import InputTestCase
from AutoTuner.utils.distributed import init_distributed_multi_nodes

from .configs.config_struct import ProfileConfig
from .launcher.get_data_launch import LaunchDataCollectionForOps
from .launcher.profile_launch import LaunchProfileForOps


def validate_args(args):
    args.real_test_cases_file = os.path.join(args.test_cases_dir, args.test_cases_file)
    assert os.path.exists(
        args.real_test_cases_file
    ), f"{args.real_test_cases_file} not found"
    args.real_override_model_config_file = os.path.join(
        args.config_dir, args.override_model_config_file
    )
    assert os.path.exists(
        args.real_override_model_config_file
    ), f"{args.real_override_model_config_file} not found"
    args.real_override_tf_config_file = os.path.join(
        args.config_dir, args.override_tf_config_file
    )
    assert os.path.exists(
        args.real_override_tf_config_file
    ), f"{args.real_override_tf_config_file} not found"

    # Validate distributed
    assert os.environ.get("WORLD_SIZE") is not None, "WORLD_SIZE is not set"
    assert os.environ.get("RANK") is not None, "RANK is not set"
    assert os.environ.get("LOCAL_RANK") is not None, "LOCAL_RANK is not set"

    return args


def parse_distributed_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        required=False,
        default=1,
        help="tp size of megatron",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        required=False,
        default=1,
        help="pp size of megatron",
    )
    parser.add_argument(
        "--virtual-pipeline-model-parallel-size",
        type=int,
        required=False,
        default=None,
        help="vpp size of megatron",
    )
    parser.add_argument(
        "--context-parallel-size",
        type=int,
        required=False,
        default=1,
        help="cp size of megatron",
    )
    parser.add_argument(
        "--expert-parallel-size",
        type=int,
        required=False,
        default=1,
        help="ep size of megatron",
    )
    parser.add_argument(
        "--expert-tensor-parallel-size",
        type=int,
        required=False,
        default=1,
        help="etp size of megatron",
    )
    return parser


def parse_args():
    parser = argparse.ArgumentParser(description="Profile Operators")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        default="Qwen/Qwen3-0.6B",
        help="model name to test",
    )
    # File and directories
    parser.add_argument(
        "--test_cases-dir",
        type=str,
        required=False,
        default="AutoTuner/testbench/profile/cases/",
        help="Base dir holds the test cases files, shall not modify this",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        required=False,
        default="AutoTuner/testbench/profile/configs/",
        help="Base dir holds the config files, shall not modify this",
    )
    parser.add_argument(
        "--test-cases-file",
        type=str,
        required=True,
        default="qwen3_0_6b.json",
        help="file in cases folder contains",
    )
    parser.add_argument(
        "--override-model-config-file",
        type=str,
        required=False,
        default="override_model_config.json",
        help="huggingface model configs to override",
    )
    parser.add_argument(
        "--override-tf-config-file",
        type=str,
        required=False,
        default="override_tf_config.json",
        help="TransformerConfig to override",
    )
    parser.add_argument(
        "--profile-config-file",
        type=str,
        required=False,
        default="profile_config.json",
        help="profile config file",
    )

    # Profile configs
    parser.add_argument(
        "--profile-mode",
        action="store_true",
        help="Enable it when using nsys profile, disable it in case of data collection",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=2,
        required=False,
        help="warmup op iterations",
    )

    # test choices for flexibility
    parser.add_argument("--test-ops-list", type=int, nargs="+", default=None)
    parser.add_argument("--test-case-idxs", type=int, nargs="+", default=None)

    # output
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="outputs",
        help="output directory to save the database results and nsys profile results, actual output_dir is args.output_dir/model_name/profile_mode",
    )
    
    # distributed
    parser = parse_distributed_args(parser)
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
    return ProfileConfig(args.profile_mode, args.warmup_iters)


def handle_model_config(args) -> dict:
    with open(args.real_override_model_config_file, "r") as fp:
        override_model_config = json.load(fp)
    return override_model_config


def handle_tf_config(args) -> dict:
    with open(args.real_tf_config_config, "r") as fp:
        override_tf_config = json.load(fp)
    return override_tf_config


def call_launcher(
    args,
    test_cases: List[InputTestCase],
    profile_config: ProfileConfig,
    override_model_config: dict,
    override_tf_config: dict,
):
    launcher_cls = (
        LaunchProfileForOps
        if profile_config.profile_mode
        else LaunchDataCollectionForOps
    )
    launcher = launcher_cls(
        profile_config,
        test_cases,
        model_name=args.model_name,
        override_model_kwargs=override_model_config,
        override_tf_config_kwargs=override_tf_config,
    )

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
    init_distributed_multi_nodes(
        tp=args.tensor_model_parallel_size,
        pp=args.pipeline_model_parallel_size,
        vpp=args.virtual_pipeline_model_parallel_size,
        cp=args.context_parallel_size,
        ep=args.expert_parallel_size,
        etp=args.expert_tensor_parallel_size,
    )
    test_cases = handle_test_cases(args)
    profile_config = handle_profile_configs(args)
    override_model_config = handle_model_config(args)
    override_tf_config = handle_tf_config(args)
    call_launcher(
        args, test_cases, profile_config, override_model_config, override_tf_config
    )
