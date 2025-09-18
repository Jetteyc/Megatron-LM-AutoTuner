from megatron.core.utils import configure_nvtx_profiling, nvtx_range_push, nvtx_range_pop, nvtx_decorator
import os

def enable_nvtx_profiling():
    configure_nvtx_profiling(True)
    os.environ["NVTE_NVTX_ENABLED"] = "1"