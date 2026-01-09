from verl.workers.engine_workers import TrainingWorker
from omegaconf import DictConfig
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import ActorConfig, HFModelConfig,TrainingWorkerConfig
from tensordict import TensorDict
from functools import partial
from verl.workers.utils.losses import ppo_loss
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register


# def _init_dist_if_needed(timeout_second: Optional[int] = None) -> None:
#     """Initialize torch.distributed for torchrun (env://) if not already initialized."""
#     from datetime import timedelta

#     if dist.is_available() and not dist.is_initialized():
#         backend = "nccl" if torch.cuda.is_available() else "gloo"
#         dist.init_process_group(
#             backend=backend,
#             init_method="env://",
#             timeout=timedelta(seconds=timeout_second) if timeout_second is not None else None,
#         )


# class TrainingWorker:
#     """
#     A torchrun-only variant of verl TrainingWorker:
#     - No Ray
#     - No Worker / single_controller
#     - No DistProfilerExtension
#     - No @register / dispatch metadata

#     It directly wraps verl Engine and provides the same coarse-grained APIs.
#     """

#     def __init__(self, config):
#         """
#         Args:
#             config: TrainingWorkerConfig
#         """
#         from verl.workers.engine import BaseEngine, EngineRegistry

#         _init_dist_if_needed(timeout_second=None)

#         self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
#         self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
#         self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

#         self.config = config
#         self.model_config = self.config.model_config
#         self.engine_config = self.config.engine_config
#         self.optimizer_config = self.config.optimizer_config
#         self.checkpoint_config = self.config.checkpoint_config
#         self.device_name = get_device_name()

#         # use the one defined in model
#         self.engine_config.use_remove_padding = self.model_config.use_remove_padding

#         self.engine: BaseEngine = EngineRegistry.new(
#             model_type=self.config.model_type,
#             backend=self.engine_config.strategy,
#             model_config=self.model_config,
#             engine_config=self.engine_config,
#             optimizer_config=self.optimizer_config,
#             checkpoint_config=self.checkpoint_config,
#         )

#         self.flops_counter = FlopsCounter(self.model_config.hf_config)
#         self.loss_fn = None

#     # ---------------- basic controls ----------------

#     def to(self, device, model=True, optimizer=True, grad=True):
#         """Manual control of load/offload"""
#         assert device in ["cpu", "device"]
#         if device == "device":
#             device = get_device_name()
#         self.engine.to(device=device, model=model, optimizer=optimizer, grad=grad)

#     def set_loss_fn(self, loss_fn):
#         self.loss_fn = loss_fn

#     def reset(self):
#         """Reset the model engine to the initial state."""
#         self.engine.initialize()

#     # ---------------- helpers ----------------

#     def _postprocess_output(self, output, *, global_token_num, delta_time, forward_only):
#         metrics: dict = output.pop("metrics")

#         # reduce loss in DP group
#         loss = torch.sum(torch.tensor(output.pop("loss"), device=self.device_name))
#         dist.all_reduce(loss, op=dist.ReduceOp.AVG, group=self.engine.get_data_parallel_group())
#         loss = loss.item()

#         grad_norm = metrics.pop("grad_norm", None)
#         lr = metrics.pop("lr", None)

#         # other metrics allgather in DP group
#         final_metrics = allgather_dict_into_dict(data=metrics, group=self.engine.get_data_parallel_group())
#         final_metrics["loss"] = loss
#         if grad_norm is not None:
#             final_metrics["grad_norm"] = grad_norm
#         if lr is not None:
#             final_metrics["lr"] = lr

#         # MFU
#         if global_token_num is not None:
#             estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_token_num, delta_time)
#             final_metrics["mfu"] = estimated_flops / promised_flops / dist.get_world_size()
#             if forward_only:
#                 final_metrics["mfu"] /= 3.0

#         model_output = output.pop("model_output", {})
#         return tu.get_tensordict(tensor_dict=model_output, non_tensor_dict={"metrics": final_metrics})

#     # ---------------- training / inference ----------------

#     def train_mini_batch(self, data: TensorDict) -> Optional[TensorDict]:
#         """Split a batch into N mini-batches run for multiple epochs."""
#         batch_size_per_dp = data.shape[0]
#         disable_auto_offload = tu.pop(data, key="disable_auto_offload", default=False)
#         mini_batch_size = tu.pop(data, key="mini_batch_size", default=None)
#         num_mini_batch = tu.pop(data, key="num_mini_batch", default=None)
#         epochs = tu.pop(data, key="epochs", default=1)
#         seed = tu.pop(data, key="seed", default=42)
#         dataloader_kwargs = tu.pop(data, key="dataloader_kwargs", default={})

#         assert mini_batch_size is not None or num_mini_batch is not None

#         if mini_batch_size is None:
#             assert batch_size_per_dp % num_mini_batch == 0, f"Got {batch_size_per_dp=} and {num_mini_batch=}"
#             mini_batch_size_per_gpu = batch_size_per_dp // num_mini_batch
#         else:
#             assert mini_batch_size % self.engine.get_data_parallel_size() == 0, (
#                 f"Got {mini_batch_size=} and {self.engine.get_data_parallel_size()=}"
#             )
#             mini_batch_size_per_gpu = mini_batch_size // self.engine.get_data_parallel_size()

#         dataloader = tu.make_iterator(
#             data,
#             mini_batch_size=mini_batch_size_per_gpu,
#             epochs=epochs,
#             seed=seed + self.engine.get_data_parallel_rank(),
#             dataloader_kwargs=dataloader_kwargs,
#         )

#         with (
#             self.engine.train_mode(disable_auto_offload=disable_auto_offload),
#             Timer(name="train_batch", logger=None),
#         ):
#             output_lst = []
#             total_num_iterations = data.shape[0] // mini_batch_size_per_gpu * epochs

#             for batch_idx, mini_batch_td in enumerate(dataloader):
#                 global_token_num = mini_batch_td["input_ids"].offsets().diff().tolist()

#                 # allgather token nums from DP ranks
#                 global_token_num_output = [None] * self.engine.get_data_parallel_size()
#                 dist.all_gather_object(global_token_num_output, global_token_num, self.engine.get_data_parallel_group())
#                 global_token_num = [x for xs in global_token_num_output for x in xs]

#                 tu.assign_non_tensor(
#                     mini_batch_td,
#                     global_token_num=NonTensorData(global_token_num),
#                     update_lr_scheduler=batch_idx == total_num_iterations - 1,
#                     disable_auto_offload=True,
#                 )
#                 out = self.train_batch(mini_batch_td)
#                 output_lst.append(out)

#             if self.engine.is_mp_src_rank_with_outputs():
#                 metrics_list = [tu.get(o, "metrics") for o in output_lst]
#                 metrics = {}
#                 for m in metrics_list:
#                     for key, val in m.items():
#                         if isinstance(val, list):
#                             m[key] = list(chain.from_iterable(val))
#                     append_to_dict(metrics, m)

#                 output = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()
#             else:
#                 output = None

#         return output

#     def train_batch(self, data: TensorDict) -> Optional[TensorDict]:
#         assert self.loss_fn is not None, "loss function can't be None when calling train_batch"
#         assert not self.engine_config.forward_only, "Can't run `train_batch` when forward_only is in the engine config."

#         global_token_num = tu.get(data, key="global_token_num")
#         disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)

#         default_keys = dict(
#             use_remove_padding=self.model_config.use_remove_padding,
#             use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
#             max_token_len_per_gpu=self.engine_config.max_token_len_per_gpu,
#             micro_batch_size_per_gpu=self.engine_config.micro_batch_size_per_gpu,
#             use_fused_kernels=self.engine_config.use_fused_kernels,
#         )
#         for key, val in default_keys.items():
#             if key not in data.keys():
#                 tu.assign_non_tensor(data, **{key: val})

#         with (
#             self.engine.train_mode(disable_auto_offload=disable_auto_offload),
#             Timer(name="train_batch", logger=None) as timer,
#         ):
#             output = self.engine.train_batch(data, loss_function=self.loss_fn)

#         delta_time = timer.last

#         update_lr_scheduler = tu.get(data, key="update_lr_scheduler", default=False)
#         lr = self.engine.lr_scheduler_step() if update_lr_scheduler else None

#         if self.engine.is_mp_src_rank_with_outputs():
#             output.pop("model_output", None)
#             if lr is not None:
#                 output["metrics"]["lr"] = lr
#             final_output = self._postprocess_output(
#                 output, global_token_num=global_token_num, delta_time=delta_time, forward_only=False
#             ).cpu()
#         else:
#             final_output = None

#         return final_output

#     def infer_batch(self, data: TensorDict) -> Optional[TensorDict]:
#         global_token_num = tu.get(data, key="global_token_num")
#         compute_loss = tu.get(data, key="compute_loss", default=True)
#         disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)

#         default_keys = dict(
#             use_remove_padding=self.model_config.use_remove_padding,
#             use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
#             max_token_len_per_gpu=self.engine_config.infer_max_token_len_per_gpu,
#             micro_batch_size_per_gpu=self.engine_config.infer_micro_batch_size_per_gpu,
#             use_fused_kernels=self.engine_config.use_fused_kernels,
#         )
#         for key, val in default_keys.items():
#             if key not in data.keys():
#                 tu.assign_non_tensor(data, **{key: val})

#         loss_function = self.loss_fn if compute_loss else None

#         with (
#             self.engine.eval_mode(disable_auto_offload=disable_auto_offload),
#             Timer(name="eval_batch", logger=None) as timer,
#         ):
#             output = self.engine.infer_batch(data, loss_function=loss_function)

#         delta_time = timer.last

#         if self.engine.is_mp_src_rank_with_outputs():
#             final_output = self._postprocess_output(
#                 output, global_token_num=global_token_num, delta_time=delta_time, forward_only=True
#             ).cpu()
#         else:
#             final_output = None

#         return final_output

#     # ---------------- checkpoint ----------------

#     def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
#         return self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

#     def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
#         return self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)

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

    # @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.actor.set_loss_fn(loss_fn=loss_fn)

    # @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        self.actor.to(device=device, model=model, optimizer=optimizer, grad=grad)

    # @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        print(self.config)
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