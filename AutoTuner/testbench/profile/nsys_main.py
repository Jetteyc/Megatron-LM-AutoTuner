import warnings
warnings.warn("Deprecared! Only support single node, use scripts instead!", DeprecationWarning, stacklevel=2)

import os
import subprocess
from collections import defaultdict

from megatron.core import parallel_state as mpu

from AutoTuner.utils.config import (
    get_hf_model_config,
    get_mcore_model_config_from_hf_config,
)
from AutoTuner.utils.model_inputs import DataSets
from AutoTuner.utils.nested_dict import NestedDict
from AutoTuner.utils.structs import InputTestCase
from AutoTuner.utils.distributed import init_distributed_single_node

from .configs.config_struct import ProfileConfig
from .main import parse_args
from .op_mapping import OP_TEST_MAPPING


class NsysLauncher:
    def __init__(
        self,
        args,
        timeout: int = 3000,
    ):
        self.environ = os.environ.copy()
        self.environ["NVTE_NVTX_ENABLED"] = "1"

        self.nsys_args = [
            # output
            "-w",
            "true",
            "-o",
            f"{args.output_dir}/{args.model_name}/nsys_profile",
            "-f",
            "true",
            "-x",
            "true",
            # cuda, nvtx, cudnn, cublas, osrt, syscall, python-gil
            "-t",
            "cuda,nvtx,cudnn,cublas,python-gil",
            # GPU/CUDA
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--cudabacktrace=all",
            "--cuda-memory-usage=true",
            "--python-backtrace=cuda",
            # "--gpuctxsw",
            # "--gpu-metrics-devices=all",
            # "--enable", "nvml_metrics",  # NVML Power and temperature
            # "--soc-metrics=true",
            # CPU
            # "--cpuctxsw",
            # Network
            "--enable",
            "network_interface",  # NIC/IB metrics
            # Python backtrace
            "--python-sampling=true",
            # "--python-function-trace=$NSYS_INSTALL_DIR/target-linux-x64/PythonFunctionsTrace/annotations.json",
        ]
        self.args = args

        self.command = (
            ["nsys", "profile"]
            + self.nsys_args
            + ["python3", "-m", "AutoTuner.profile.main"]
            + self.args
        )

        self.timeout = timeout
        init_distributed_single_node()

    def run_nsys_profile(self):
        process = subprocess.Popen(self.command, shell=True, env=self.environ)
        process.wait(timeout=self.timeout)
        print(f"stdout: {process.stdout}\nstderr: {process.stderr}")


if __name__ == "__main__":
    args = parse_args()
    launcher = NsysLauncher(args)
    launcher.run_nsys_profile()
