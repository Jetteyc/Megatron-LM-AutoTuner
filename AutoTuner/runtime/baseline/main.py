from .runtime_worker import ActorSimpleRuntimeWorker
import hydra

def run(config):
    actor = ActorSimpleRuntimeWorker(config.actor_rollout_ref)
    actor.init_model()

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config):
    # print(config)
    run(config)

    
    
if __name__ == "__main__":
    main()