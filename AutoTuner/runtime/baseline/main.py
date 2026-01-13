from .runtime_worker import ActorSimpleRuntimeWorker
from ..commons import get_batch_data_generator,create_train_dataloader, create_rl_dataset
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
from verl.utils.import_utils import load_extern_object
from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def run(config):
    validate_config(config=config, use_reference_policy=False, use_critic=False)
    actor = ActorSimpleRuntimeWorker(config.actor_rollout_ref)
    actor.init_model()

    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.fs import copy_to_local
    trust_remote_code = config.data.get("trust_remote_code", False)
    local_path = copy_to_local(
        config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
    )
    # test_cases = create_test_cases(config, seqlen=1024)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    # Used for multimodal LLM, could be None
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
    train_dataset = create_rl_dataset(
        data_paths=config.data.train_files,
        data_config=config.data,
        tokenizer=tokenizer,
        processor=processor,
        is_train=True,
        max_samples=config.data.get("train_max_samples", -1),
    )
    # dataset = DataSets(
    #     AutoConfig.from_pretrained(config.actor_rollout_ref.model.hf_config_path if config.actor_rollout_ref.model.hf_config_path else config.actor_rollout_ref.model.path),
    #     test_cases,
    #     fix_compute_amount=config.fix_compute_amount,
    #     use_dynamic_bsz_balance=True,
    #     vpp_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
    # )
    print("---------------------created train dataset---------------------")
    train_sampler = create_rl_sampler(config.data, train_dataset)
    print("---------------------created data sampler---------------------")
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
    from verl.utils.dataset.rl_dataset import collate_fn
    # train_sampler = create_rl_sampler(config.data, train_dataset)
    train_dataloader = create_train_dataloader(
        config, 
        train_dataset, 
        tokenizer, 
        processor, 
        collate_fn, 
        train_sampler
    )
    # train_dataset = DataSetsGeneratorDataset(
    #     datasets=dataset,
    #     test_cases=test_cases,
    # )
    # train_dataloader = StatefulDataLoader(
    #     dataset=train_dataset,
    #     batch_size=1,
    #     num_workers=0,
    #     drop_last=True,
    # )
    
    print("---------------------created train dataloader---------------------")


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config):
    # print(config)
    run(config)
    destroy_distributed()

    
    
if __name__ == "__main__":
    main()