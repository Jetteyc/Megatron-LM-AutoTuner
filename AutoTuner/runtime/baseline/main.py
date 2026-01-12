from .runtime_worker import ActorSimpleRuntimeWorker
from ..commons import get_batch_data_generator,create_train_dataloader
import hydra
from AutoTuner.utils.distributed import destroy_distributed
from tensordict import TensorDict
from verl.utils.config import validate_config
from AutoTuner.utils.model_inputs import DataSets
from AutoTuner.utils.structs import InputTestCase
from megatron.core import parallel_state as mpu
from transformers import PretrainedConfig
from transformers import AutoConfig
from verl.trainer.main_ppo import create_rl_sampler
from torchdata.stateful_dataloader import StatefulDataLoader



from torch.utils.data import IterableDataset

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

def run(config):
    validate_config(config=config, use_reference_policy=False, use_critic=False)
    actor = ActorSimpleRuntimeWorker(config.actor_rollout_ref)
    actor.init_model()
    test_cases = create_test_cases(config, seqlen=1024)
    dataset = DataSets(
        AutoConfig.from_pretrained(config.actor_rollout_ref.model.hf_config_path if config.actor_rollout_ref.model.hf_config_path else config.actor_rollout_ref.model.path),
        test_cases,
        fix_compute_amount=config.fix_compute_amount,
        use_dynamic_bsz_balance=True,
        vpp_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
    )
    print("---------------------created train dataset---------------------")
    # from verl.utils.fs import copy_to_local
    # local_path = copy_to_local(
    #     config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
    # )

    # # Instantiate the tokenizer and processor.
    # from verl.utils import hf_processor, hf_tokenizer

    # trust_remote_code = config.data.get("trust_remote_code", False)
    # tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    # # Used for multimodal LLM, could be None
    # processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
    # from verl.utils.dataset.rl_dataset import collate_fn
    # train_sampler = create_rl_sampler(config.data, train_dataset)
    # train_dataloader = create_train_dataloader(
    #     config, 
    #     train_dataset, 
    #     tokenizer, 
    #     processor, 
    #     collate_fn, 
    #     train_sampler
    # )
    train_dataset = DataSetsGeneratorDataset(
        datasets=dataset,
        test_cases=test_cases,
    )
    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=1,
        num_workers=0,
        drop_last=True,
    )

    print("---------------------created train dataloader---------------------")


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config):
    # print(config)
    run(config)
    destroy_distributed()

    
    
if __name__ == "__main__":
    main()