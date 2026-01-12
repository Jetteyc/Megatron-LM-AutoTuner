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

# class BatchDataConstructor(ABC):
#     @abstractmethod
#     def gen_inputs(self):
#         pass
    
# class RandomBatchDataConstructor(BatchDataConstructor):
#     def __init__(self, test_cases, model_args, fix_compute_amount=True):
#         super().__init__()
#         self.hf_config = get_hf_model_config(model_args.hf_config_path)
#         self.fix_compute_amount = fix_compute_amount
#         self.test_cases = test_cases
#         self.datasets = None
    
#     def gen_inputs(self):
#         if self.datasets is None:
#             self.datasets = DataSets(
#                 self.hf_config,
#                 self.test_cases,
#                 fix_compute_amount=self.fix_compute_amount,
#                 use_dynamic_bsz_balance=True,
#                 vpp_size=mpu.get_virtual_pipeline_model_parallel_world_size()
#             )
            
        

# def handle_test_cases(args) -> List[InputTestCase]:
#     with open(args.real_test_cases_file, "r") as fp:
#         json_test_cases = json.load(fp)
#     test_cases = []
#     for json_test_case in json_test_cases["cases"]:
#         test_case = InputTestCase(**json_test_case)
#         test_case.tensor_model_parallel_size = args.tensor_model_parallel_size
#         test_case.pipeline_model_parallel_size = args.pipeline_model_parallel_size
#         test_case.virtual_pipeline_model_parallel_size = (
#             args.virtual_pipeline_model_parallel_size
#         )
#         test_case.context_parallel_size = args.context_parallel_size
#         test_case.expert_parallel_size = args.expert_parallel_size
#         test_case.expert_tensor_parallel_size = args.expert_tensor_parallel_size
#         test_cases.append(test_case)
#     return test_cases