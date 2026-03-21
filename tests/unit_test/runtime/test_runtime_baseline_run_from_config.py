import importlib.util
from pathlib import Path


def _load_runtime_baseline_run_from_config_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "functional_test"
        / "runtime"
        / "runtime_baseline_run_from_config.py"
    )
    spec = importlib.util.spec_from_file_location(
        "runtime_baseline_run_from_config", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_parallel_defaults_vpp_size_to_none_when_omitted() -> None:
    module = _load_runtime_baseline_run_from_config_module()

    parallel = module.build_parallel({}, {})

    assert parallel["vpp_size"] is None


def test_build_parallel_accepts_string_none_for_vpp_size() -> None:
    module = _load_runtime_baseline_run_from_config_module()

    parallel = module.build_parallel({"vpp_size": "None"}, {})

    assert parallel["vpp_size"] is None


def test_build_parallel_accepts_null_for_vpp_size() -> None:
    module = _load_runtime_baseline_run_from_config_module()

    parallel = module.build_parallel({"vpp_size": None}, {})

    assert parallel["vpp_size"] is None


def test_build_runtime_accepts_use_fused_kernels() -> None:
    module = _load_runtime_baseline_run_from_config_module()

    runtime = module.build_runtime({"use_fused_kernels": False}, {})

    assert runtime["use_fused_kernels"] is False


def test_build_command_emits_use_fused_kernels_flag() -> None:
    module = _load_runtime_baseline_run_from_config_module()

    spec = {
        "model_name": "Qwen/Qwen2.5-0.5B",
        "test_cases_dir": Path("tests/functional_test/runtime/generated_cases/qwen_longctx"),
        "test_cases_file": "dummy.json",
        "output_dir": Path("outputs"),
        "gpus_per_node": 1,
        "nproc_per_node": 1,
        "parallel": {
            "tp_size": 1,
            "cp_size": 1,
            "ep_size": 1,
            "etp_size": 1,
            "pp_size": 1,
            "vpp_size": None,
        },
        "runtime": {
            "num_test_cases": 1,
            "max_iterations": 10,
            "warmup_iterations": 3,
            "share_emb": None,
            "run_one_data": False,
            "no_ddp": False,
            "use_fused_kernels": False,
            "config_dir": None,
            "override_model_config_file": None,
            "override_tf_config_file": None,
            "tp_comm_overlap_cfg": None,
        },
        "distributed": {
            "master_addr": "localhost",
            "master_port": 6010,
            "num_nodes": 1,
            "node_rank": 0,
        },
        "env": {},
        "case": {
            "seqlen": 20480,
            "max_token_len": 40960,
            "batch_size": 1,
            "micro_batch_size": 1,
            "cases": [],
        },
    }

    cmd = module.build_command(spec)

    assert "--use-fused-kernels" in cmd
    assert cmd[cmd.index("--use-fused-kernels") + 1] == "false"


def test_build_command_honors_explicit_nproc_per_node() -> None:
    module = _load_runtime_baseline_run_from_config_module()

    spec = {
        "model_name": "Qwen/Qwen2.5-0.5B",
        "test_cases_dir": Path("tests/functional_test/runtime/generated_cases/qwen_longctx"),
        "test_cases_file": "dummy.json",
        "output_dir": Path("outputs"),
        "gpus_per_node": 1,
        "nproc_per_node": 8,
        "parallel": {
            "tp_size": 1,
            "cp_size": 1,
            "ep_size": 1,
            "etp_size": 1,
            "pp_size": 1,
            "vpp_size": None,
        },
        "runtime": {
            "num_test_cases": 1,
            "max_iterations": 10,
            "warmup_iterations": 3,
            "share_emb": None,
            "run_one_data": False,
            "no_ddp": False,
            "use_fused_kernels": None,
            "config_dir": None,
            "override_model_config_file": None,
            "override_tf_config_file": None,
            "tp_comm_overlap_cfg": None,
        },
        "distributed": {
            "master_addr": "localhost",
            "master_port": 6010,
            "num_nodes": 1,
            "node_rank": 0,
        },
        "env": {},
        "case": {
            "seqlen": 20480,
            "max_token_len": 40960,
            "batch_size": 1,
            "micro_batch_size": 1,
            "cases": [],
        },
    }

    cmd = module.build_command(spec)

    assert cmd[cmd.index("--nproc_per_node") + 1] == "8"


def test_build_run_spec_defaults_nproc_per_node_when_missing() -> None:
    module = _load_runtime_baseline_run_from_config_module()

    spec = module.build_run_spec(
        repo_root=Path("."),
        merged_cfg={
            "model_name": "Qwen/Qwen2.5-0.5B",
            "case": {
                "seqlen": 20480,
                "max_token_len": 40960,
                "batch_size": 1,
                "micro_batch_size": 1,
                "shapes": ["thd"],
                "system": "megatron",
            },
            "parallel": {
                "tp_size": 1,
                "cp_size": 1,
                "ep_size": 1,
                "etp_size": 1,
                "pp_size": 1,
            },
            "runtime": {
                "num_test_cases": 1,
                "max_iterations": 10,
                "warmup_iterations": 3,
            },
        },
        distributed_info={
            "master_addr": "localhost",
            "master_port": 6010,
            "num_nodes": 1,
            "node_rank": 0,
        },
    )

    assert spec["nproc_per_node"] == 1
