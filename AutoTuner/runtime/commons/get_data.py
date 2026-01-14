import os
import json
from AutoTuner.utils.structs import InputTestCase
from torch.utils.data import Dataset
from omegaconf import DictConfig
import torch
import numpy as np
from typing import Optional
from verl.utils.model import create_random_mask, compute_position_id_with_mask


class RandomDataset(Dataset):
    def __init__(
        self,
        size: int = 10000,
        tokenizer: Optional[object] = None,
        config: Optional[DictConfig] = None,
        processor: Optional[object] = None,
        max_samples: int = -1,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config or {}
        
        self.max_prompt_length = self.config.get("max_prompt_length", 1024)
        self.min_prompt_length = self.config.get("min_prompt_length", 100)
        
        if tokenizer is not None:
            self.vocab_size = len(tokenizer) if hasattr(tokenizer, '__len__') else getattr(tokenizer, 'vocab_size', 32000)
        else:
            self.vocab_size = self.config.get("vocab_size", 32000)
        
        self.seed = self.config.get("dataset_seed", 42)
        self.max_ratio_of_valid_token = self.config.get("max_ratio_of_valid_token", 0.9)
        self.min_ratio_of_valid_token = self.config.get("min_ratio_of_valid_token", 0.5)
        self.max_ratio_of_left_padding = self.config.get("max_ratio_of_left_padding", 0.1)
        self.need_tools_kwargs = self.config.get("need_tools_kwargs", False)
        
        self.size = size
        if max_samples > 0:
            self.size = min(size, max_samples)
        
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self._prompt_lengths = np.random.randint(
            self.min_prompt_length, 
            self.max_prompt_length + 1, 
            size=self.size
        )
        
        print(f"RandomDataset initialized with {self.size} samples")
        print(f"Prompt length range: [{self.min_prompt_length}, {self.max_prompt_length}]")
        print(f"Vocab size: {self.vocab_size}")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx: int) -> dict:
        rng = np.random.RandomState(self.seed + idx)
        
        prompt_length = int(self._prompt_lengths[idx])
        
        input_ids = torch.randint(
            low=0, 
            high=self.vocab_size, 
            size=(prompt_length,),
            generator=torch.Generator().manual_seed(self.seed + idx)
        )

        attention_mask = create_random_mask(
            input_ids=input_ids.unsqueeze(0),
            max_ratio_of_left_padding=self.max_ratio_of_left_padding,
            max_ratio_of_valid_token=self.max_ratio_of_valid_token,
            min_ratio_of_valid_token=self.min_ratio_of_valid_token,
        ).squeeze(0)
        
        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0)).squeeze(0)
        
        raw_prompt = [
            {
                "role": "user",
                "content": f"Random prompt with {prompt_length} tokens (idx={idx})"
            }
        ]
        
        dummy_tensor = torch.tensor([0], dtype=torch.uint8)
        
        return {
            "raw_prompt": raw_prompt,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "dummy_tensor": dummy_tensor,
            "index": idx,
            "tools_kwargs": {},
            "interaction_kwargs": {},
            "data_source": "random_dataset",
            "extra_info": {
                "index": idx,
                "prompt_length": prompt_length,
                "tools_kwargs": {},
                "interaction_kwargs": {},
                "need_tools_kwargs": self.need_tools_kwargs,
            }
        }
    
def get_random_data(path, engine_args, model_args):
    with open(path, "r") as fp:
        json_test_cases = json.load(fp)        
    test_cases = []
    for json_test_case in json_test_cases["cases"]:
        test_case = InputTestCase(**json_test_case)
        test_case.tensor_model_parallel_size = engine_args.tensor_model_parallel_size
        test_case.pipeline_model_parallel_size = engine_args.pipeline_model_parallel_size
        test_case.virtual_pipeline_model_parallel_size = (
            engine_args.virtual_pipeline_model_parallel_size
        )
        test_case.context_parallel_size = engine_args.context_parallel_size
        test_case.expert_parallel_size = engine_args.expert_model_parallel_size
        test_case.expert_tensor_parallel_size = engine_args.expert_tensor_parallel_size
        test_cases.append(test_case)
    return test_cases 
        
def get_batch_data_generator(config):
    cases_args = config.cases
    engine_args = config.actor.megatron
    model_args = config.model
    if cases_args.randomize:
        real_test_cases_file = os.path.join(cases_args.test_cases_dir, cases_args.test_cases_file)
        assert os.path.exists(
            real_test_cases_file
        ), f"{real_test_cases_file} not found"
        return get_random_data(real_test_cases_file, engine_args, model_args)
    else:
        raise NotImplementedError("need to implement data reading")
    
# def construct_randomly(args):

from typing import Any, Optional
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import Sampler
from omegaconf import OmegaConf,open_dict

def create_train_dataloader(config, train_dataset, tokenizer, processor, collate_fn, train_sampler: Optional[Sampler]):
    """
    Creates the train and validation dataloaders.
    """
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

    if train_dataset is None:
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )

    if train_sampler is None:
        train_sampler = create_rl_sampler(config.data, train_dataset)
    if collate_fn is None:
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

        collate_fn = default_collate_fn

    num_workers = config.data["dataloader_num_workers"]

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=config.data.get("gen_batch_size", config.data.train_batch_size),
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=train_sampler,
    )

    assert len(train_dataloader) >= 1, "Train dataloader is empty!"

    print(
        f"Size of train dataloader: {len(train_dataloader)}"
    )

    total_training_steps = len(train_dataloader) * config.trainer.total_epochs

    if config.trainer.total_training_steps is not None:
        total_training_steps = config.trainer.total_training_steps

    total_training_steps = total_training_steps
    print(f"Total training steps: {total_training_steps}")

    try:
        OmegaConf.set_struct(config, True)
        with open_dict(config):
            if OmegaConf.select(config, "actor_rollout_ref.actor.optim"):
                config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
    except Exception as e:
        print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")
    return train_dataloader

from torch.utils.data import IterableDataset, Dataset
from typing import Optional

def create_rl_dataset(data_config, tokenizer, processor, data_paths=None, is_train=True, max_samples: int = -1):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    if data_paths is None:
        dummy_size = data_config.get("dummy_dataset_size", 10000)
        dataset = RandomDataset(
            size=dummy_size,
            tokenizer=tokenizer,
            processor=processor,
            config=data_config,
            max_samples=max_samples,
        )

    else:
        from verl.utils.dataset.rl_dataset import get_dataset_class

        # Get the dataset class
        dataset_cls = get_dataset_class(data_config)

        # Instantiate the dataset using the determined dataset class
        dataset = dataset_cls(
            data_files=data_paths,
            tokenizer=tokenizer,
            processor=processor,
            config=data_config,
            max_samples=max_samples,
        )

    return dataset