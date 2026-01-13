import os
import json
from AutoTuner.utils.structs import InputTestCase
from AutoTuner.utils.model_inputs import DataSets

import megatron.core.parallel_state as mpu
from AutoTuner.utils.config import (
    get_hf_model_config,
)

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
        dummy_size = data_config.get("dummy_dataset_size", 1000)
        dataset = DummyDataset(size=dummy_size)

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

class DummyDataset(Dataset):
    def __init__(self, size: int = 1000):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {"dummy_index": idx}

class DataSetsGeneratorDataset(IterableDataset):
    def __init__(self, datasets: DataSets, test_cases):
        self.datasets = datasets
        self.test_cases = test_cases

    def __iter__(self):
        for test_case in self.test_cases:
            batch_gen = self.datasets.get_batch_generator(test_case)
            for batch in batch_gen:
                yield {
                    "test_case": test_case,
                    "batch": batch,
                }

def create_test_cases(config, seqlen):
    json_test_cases = {
        "model": "deepseek-ai/DeepSeek-V3-Base",
        "cases": [
            {
                "batch_size": config.data.train_batch_size,
                "micro_batch_size": config.actor_rollout_ref.actor.ppo_micro_batch_size,
                "seqlen": seqlen,
                "max_token_len": config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu,
                "shape":  "thd" if config.actor_rollout_ref.actor.megatron.use_remove_padding else "bshd",
                "system": config.actor_rollout_ref.actor.strategy
            },
            {
                "batch_size": config.data.train_batch_size,
                "micro_batch_size": config.actor_rollout_ref.actor.ppo_micro_batch_size,
                "seqlen": seqlen,
                "max_token_len": config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu,
                "shape": "thd" if config.actor_rollout_ref.actor.megatron.use_remove_padding else "bshd",
                "system": config.actor_rollout_ref.actor.strategy
            }
        ]
    }
    test_cases = []
    for json_test_case in json_test_cases["cases"]:
        test_case = InputTestCase(**json_test_case)
        test_case.tensor_model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
        test_case.pipeline_model_parallel_size = config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
        test_case.virtual_pipeline_model_parallel_size = config.actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size
        test_case.context_parallel_size = config.actor_rollout_ref.actor.megatron.context_parallel_size
        test_case.expert_parallel_size = config.actor_rollout_ref.actor.megatron.expert_model_parallel_size
        test_case.expert_tensor_parallel_size = config.actor_rollout_ref.actor.megatron.expert_tensor_parallel_size
        test_cases.append(test_case)
    return test_cases
