from verl.workers.engine_workers import TrainingWorker
from omegaconf import DictConfig
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import ActorConfig, HFModelConfig,TrainingWorkerConfig
from tensordict import TensorDict
from functools import partial
from verl.workers.utils.losses import ppo_loss
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
import torch
from verl.utils.profiler import DistProfiler, DistProfilerExtension


def _build_megatron_module(self):
    from verl.utils.megatron_utils import (
        McoreModuleWrapperConfig,
        make_megatron_module,
    )
    from verl.utils.model import print_model_size

    # TODO: add more cases
    is_value_model = (
        "ForTokenClassification" in self.model_config.architectures[0]
        or "ForSequenceClassification" in self.model_config.architectures[0]
    )

    self.is_value_model = is_value_model

    if self.engine_config.forward_only:
        wrap_with_ddp = False
    else:
        wrap_with_ddp = True

    wrap_config = McoreModuleWrapperConfig(
        is_value_model=is_value_model,  # actor is not value model
        share_embeddings_and_output_weights=self.model_config.share_embeddings_and_output_weights,
        wrap_with_ddp=wrap_with_ddp,
        use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
    )
    module, updated_tf_config = make_megatron_module(
        wrap_config=wrap_config,
        tf_config=self.tf_config,
        hf_config=self.model_config.hf_config,
        bridge=self.bridge,
        provider=self.provider,
        override_model_config=self.engine_config.override_mcore_model_config,
        override_ddp_config=self.engine_config.override_ddp_config,
        peft_cls=self.peft_cls,
        peft_config=self.model_config.get("lora", None),
    )
    self.tf_config = updated_tf_config
    print(f"module: {len(module)}")

    # if self.engine_config.use_dist_checkpointing:
    #     load_mcore_dist_weights(module, self.engine_config.dist_checkpointing_path, is_value_model=is_value_model)
    # else:
    #     if self.vanilla_bridge:
    #         self.bridge.load_weights(module, self.model_config.local_path)
    #     else:
    #         allowed_mismatched_params = []
    #         if self.is_value_model:
    #             allowed_mismatched_params = ["output_layer.weight"]
    #         self.bridge.load_hf_weights(
    #             module, self.model_config.local_path, allowed_mismatched_params=allowed_mismatched_params
    #         )

    if torch.distributed.get_rank() == 0:
        print_model_size(module[0])

    return module

from verl.workers.engine.megatron.transformer_impl import MegatronEngine
MegatronEngine._build_megatron_module = _build_megatron_module

class ActorSimpleRuntimeWorker:
    """
    A simple runtime worker copied and adjusted from ActorRolloutRefWorker

    NOTE: ActorRolloutRefWorker no longer support spmd mode and run native server mode.
    """

    def __init__(self, config: DictConfig, **kwargs):
        # super().__init__()
        self.config = config
        self.actor: TrainingWorker = None
        # assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
        # self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        # self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        # self._is_ref = self.role in ["ref", "actor_rollout_ref"]
        # for future support, we may have to add the profiler back, but now, we don't need it
        # if self._is_actor:
        #     omega_profiler_config = config.actor.get("profiler", {})
        # elif self._is_rollout:
        #     # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
        #     # This is for extendability in AsyncRL cases
        #     omega_profiler_config = config.rollout.get("profiler", {})
        # else:
        #     omega_profiler_config = config.ref.get("profiler", {})

        # profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        # if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
        #     tool_config = omega_conf_to_dataclass(
        #         omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
        #     )
        # else:
        #     tool_config = None

        # DistProfilerExtension.__init__(
        #     self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        # )

    # @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.actor.set_loss_fn(loss_fn=loss_fn)

    # @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        self.actor.to(device=device, model=model, optimizer=optimizer, grad=grad)

    # @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)

        # 2. build actor model
        actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
        actor_config.model_config = model_config

        actor_training_config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=actor_config.model_config,
            engine_config=actor_config.engine,
            optimizer_config=actor_config.optim,
            checkpoint_config=actor_config.checkpoint,
        )


        # assign engine configs
        actor_training_config.engine_config.use_dynamic_bsz = self.config.actor.use_dynamic_bsz
        # actor_training_config.engine_config.infer_max_token_len_per_gpu = (
        #     self.config.rollout.log_prob_max_token_len_per_gpu
        # )
        # actor_training_config.engine_config.infer_micro_batch_size_per_gpu = (
        #     self.config.rollout.log_prob_micro_batch_size_per_gpu
        # )
        actor_training_config.engine_config.max_token_len_per_gpu = self.config.actor.ppo_max_token_len_per_gpu
        actor_training_config.engine_config.micro_batch_size_per_gpu = (
            self.config.actor.ppo_micro_batch_size_per_gpu
        )
        actor_training_config.engine_config.use_remove_padding = model_config.use_remove_padding

        if self.config.actor.use_dynamic_bsz:
            assert self.config.actor.ppo_max_token_len_per_gpu is not None
        else:
            assert self.config.actor.ppo_micro_batch_size_per_gpu is not None

        self.loss_fn = partial(ppo_loss, config=actor_config)
        self.actor = TrainingWorker(config=actor_training_config)
        self.actor.reset()
        self.actor.set_loss_fn(self.loss_fn)
        # self.set_dispatch_collect(mesh_name="actor", **self.actor.get_dispatch_collect())

    # @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    # @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        output = self.actor.infer_batch(data)
        return output.cpu() if output is not None else None

    # @DistProfiler.annotate(color="red", role="actor_update")
    # @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: TensorDict) -> TensorDict:
        output = self.actor.train_mini_batch(data=data)
        return output.cpu() if output is not None else None

    # @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        # assert "actor" in self.role, "load_checkpoint only support actor role"
        self.actor.load_checkpoint(local_path, hdfs_path, del_local_after_load)

    # @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        # assert "actor" in self.role, "save_checkpoint only support actor role"
        self.actor.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)