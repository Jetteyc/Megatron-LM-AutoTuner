import logging
import os

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_logging_level(level):
    logger.setLevel(level)


def log_with_rank(message, level=logging.INFO):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = int(os.getenv("RANK", "0"))
    logger.log(level, f"Rank {rank}: {message}")


def log_rank0(message, level=logging.INFO):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = int(os.getenv("RANK", "0"))
    if rank == 0:
        logger.log(level, message)
