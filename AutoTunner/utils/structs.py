from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class InputTestCase:
    batch_size: int
    seqlen: int
    shape: str = "thd"  # or "bshd"
    system: str = "megatron"  # or "fsdp"
