from .runtime_worker import ActorSimpleRuntimeWorker
from ..commons import get_batch_data_generator
import hydra
from AutoTuner.utils.distributed import destroy_distributed
from tensordict import TensorDict
from verl.utils.config import validate_config

def run(config):
    validate_config(config=config, use_reference_policy=False, use_critic=False)
    actor = ActorSimpleRuntimeWorker(config.actor_rollout_ref)
    # data = get_batch_data_generator(config.actor_rollout_ref)
    actor.init_model()
    test = TensorDict()
    actor.update_actor(test)

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config):
    # print(config)
    run(config)
    destroy_distributed()

    
    
if __name__ == "__main__":
    main()