from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class InputTestCase:
    batch_size: int # mini batch size
    micro_batch_size: int  # micro batch size
    seqlen: int
    max_token_len: int | None = None
    shape: str = "thd"  # or "bshd"
    system: str = "megatron"  # or "fsdp"
