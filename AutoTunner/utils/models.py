from typing import Any, Tuple

import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from transformers import PretrainedConfig

from verl.utils.model import compute_position_id_with_mask, create_random_mask

from .pack_sequence import generate_thd_input

"""
    Follow some convinient implementation in verl
    
    Return:
    - input_ids: (b, s) or (1, total_nnz) if thd
    - attention_mask: (b, s) or None if thd
    - position_ids: (b, s) or (1, total_nnz) if
    - packed_seq_params: Any, only for thd
"""


def _get_model_input_bshd(
    model_config: PretrainedConfig, batch_size: int, seqlen: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(
        low=0, high=model_config.vocab_size, size=(batch_size, seqlen), device="cuda"
    )
    attention_mask = create_random_mask(
        input_ids=input_ids,
        max_ratio_of_left_padding=0.1,
        max_ratio_of_valid_token=0.8,
        min_ratio_of_valid_token=0.5,
    )
    position_ids = compute_position_id_with_mask(attention_mask)
    return input_ids, attention_mask, position_ids, None


def _get_model_input_fsdp_thd(
    model_config: PretrainedConfig, batch_size: int, seqlen: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention_mask, position_ids = _get_model_input_bshd(
        model_config, batch_size, seqlen
    )
    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
    position_ids_rmpad = index_first_axis(
        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
    ).transpose(0, 1)
    return input_ids_rmpad, attention_mask, position_ids_rmpad, None


def _get_model_input_megatron_thd(
    model_config: PretrainedConfig, batch_size: int, seqlen: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention_mask, position_ids = _get_model_input_bshd(
        model_config, batch_size, seqlen
    )
    input_ids_rmpad, packed_seq_params = generate_thd_input(
        input_ids=input_ids, attention_mask=attention_mask
    )
    return input_ids_rmpad, attention_mask, position_ids, packed_seq_params


def get_model_input(
    model_config: PretrainedConfig,
    batch_size: int,
    seqlen: int,
    shape: str,
    system: str = "megatron",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    if shape == "thd":
        if system == "megatron":
            return _get_model_input_megatron_thd(model_config, batch_size, seqlen)
        elif system == "fsdp":
            return _get_model_input_fsdp_thd(model_config, batch_size, seqlen)
    else:
        return _get_model_input_bshd(model_config, batch_size, seqlen)
