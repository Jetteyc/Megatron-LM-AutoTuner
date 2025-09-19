import os

from megatron.core.utils import (
    configure_nvtx_profiling,
    nvtx_decorator,
    nvtx_range_pop,
    nvtx_range_push,
)


def enable_nvtx_profiling():
    configure_nvtx_profiling(True)
    os.environ["NVTE_NVTX_ENABLED"] = "1"
