from .runtime_worker import ActorSimpleRuntimeWorker
import hydra
from AutoTuner.utils.distributed import destroy_distributed

def run(config):
    actor = ActorSimpleRuntimeWorker(config.actor_rollout_ref)
    actor.init_model()

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config):
    # print(config)
    run(config)
    destroy_distributed()

    
    
if __name__ == "__main__":
    main()