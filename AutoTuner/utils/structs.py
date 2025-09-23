from dataclasses import dataclass
from typing import Any, Tuple

from .nested_dict import nested_dict


@dataclass
class InputTestCase:
    batch_size: int  # mini batch size
    micro_batch_size: int  # micro batch size
    seqlen: int
    max_token_len: int | None = None
    shape: str = "thd"  # or "bshd"
    system: str = "megatron"  # or "fsdp"

    def __str__(self):
        return f"batch_size={self.batch_size}, micro_batch_size={self.micro_batch_size}, seqlen={self.seqlen}, max_token_len={self.max_token_len}, shape={self.shape}, system={self.system}"

    def __hash__(self):
        return hash(
            (
                self.batch_size,
                self.micro_batch_size,
                self.seqlen,
                self.max_token_len,
                self.shape,
                self.system,
            )
        )
