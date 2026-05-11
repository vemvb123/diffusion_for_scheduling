import sys
import torch

from rl4co.envs import FJSPEnv
from rl4co.models.zoo.l2d import L2DModel
from rl4co.models.zoo.l2d.policy import L2DPolicy
from rl4co.utils.trainer import RL4COTrainer


import code.dataset_code.benchmark_utils as benchmark_utils
import code.dataset_code.benchmark_values as benchmark_values




# ============================================================
# Custom Environment
# ============================================================

class LimitedMachineFJSPEnv(FJSPEnv):

    def __init__(
        self,
        *args,
        num_jobs,
        num_machines,
        limited_machine_id=2,   # machine 3
        max_machine_usage=20,   # max allowed usages
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.num_jobs_custom = num_jobs
        self.num_machines_custom = num_machines

        self.limited_machine_id = limited_machine_id
        self.max_machine_usage = max_machine_usage

    # ========================================================
    # RESET
    # ========================================================

    def _reset(self, td=None, batch_size=None):

        td = super()._reset(td, batch_size)

        batch = td.batch_size

        # Track machine usage
        td["machine_usage"] = torch.zeros(
            (*batch, self.num_machines_custom),
            device=td.device,
            dtype=torch.long
        )

        return td

    # ========================================================
    # STEP
    # ========================================================

    def _step(self, td):

        next_td = super()._step(td)

        action = td["action"]

        # ====================================================
        # RL4CO FJSP decoding
        #
        # action = machine_id * num_jobs + job_id + 1
        #
        # action == 0 -> wait action
        # ====================================================

        valid_action = action > 0

        machine_id = (action - 1) // self.num_jobs_custom

        # ====================================================
        # Update machine usage
        # ====================================================

        next_td["machine_usage"] = td["machine_usage"].clone()

        batch_idx = torch.arange(
            action.shape[0],
            device=td.device
        )

        next_td["machine_usage"][
            batch_idx[valid_action],
            machine_id[valid_action]
        ] += 1

        # ====================================================
        # Check limit
        # ====================================================

        machine_over_limit = (
            next_td["machine_usage"][:, self.limited_machine_id]
            >= self.max_machine_usage
        )

        # ====================================================
        # Disable actions for machine 3
        # ====================================================

        if machine_over_limit.any():

            action_mask = next_td["action_mask"].clone()

            # --------------------------------------------
            # Action indices for machine 3
            # --------------------------------------------

            start_idx = (
                self.limited_machine_id
                * self.num_jobs_custom
                + 1
            )

            end_idx = (
                start_idx + self.num_jobs_custom
            )

            # Disable machine actions
            action_mask[
                machine_over_limit,
                start_idx:end_idx
            ] = False

            next_td["action_mask"] = action_mask

        return next_td


# ============================================================
# TRAINING
# ============================================================

print("Beginning training of target model")


_, _, _, _, filepath_benchmark_instance, _, _, _, _, name = \
    benchmark_values.get_benchmark_values(sys.argv[1])

print(f'making model {name}')
print(f"benchmark instance: {filepath_benchmark_instance}")
print('parameters:')


parameters = benchmark_utils.get_rl4co_parameters_from_brandimarte_instance(
    filepath_benchmark_instance
)

print(parameters)


jobs = parameters['n_jobs']
ma = parameters['n_machines']

max_proc = parameters['max_processing_time']
min_proc = parameters['min_processing_time']

max_op_per_job = parameters['most_operations']
min_op_per_job = parameters['fewest_operations']

max_eligable_ma_per_op = parameters['max_machine_options']
min_eligable_ma_per_op = parameters['min_machine_options']


# ============================================================
# Generator params
# ============================================================

generator_params = {
    "num_jobs": jobs,
    "num_machines": ma,
    "min_ops_per_job": min_op_per_job,
    "max_ops_per_job": max_op_per_job,
    "min_processing_time": min_proc,
    "max_processing_time": max_proc,
    "min_eligible_ma_per_op": min_eligable_ma_per_op,
    "max_eligible_ma_per_op": max_eligable_ma_per_op,
}


# ============================================================
# ENV
# ============================================================

env = LimitedMachineFJSPEnv(
    generator_params=generator_params,

    num_jobs=jobs,
    num_machines=ma,

    _torchrl_mode=True,
    stepwise_reward=True,

    # machine 3
    limited_machine_id=2,

    # max number of usages
    max_machine_usage=20
)


# ============================================================
# Hardware
# ============================================================

if torch.cuda.is_available():

    accelerator = "gpu"

    batch_size = 24

    train_data_size = 2_000

    embed_dim = 128

    num_encoder_layers = 4

else:

    accelerator = "cpu"

    batch_size = 32

    train_data_size = 1_000

    embed_dim = 64

    num_encoder_layers = 2


# ============================================================
# Policy
# ============================================================

policy = L2DPolicy(
    embed_dim=embed_dim,
    num_encoder_layers=num_encoder_layers,
    env_name="fjsp"
)


# ============================================================
# Train
# ============================================================

lrs = [1e-4]

for lr in lrs:

    model = L2DModel(
        env,
        policy=policy,
        baseline="rollout",
        batch_size=batch_size,
        train_data_size=train_data_size,
        val_data_size=1_000,
        optimizer_kwargs={"lr": lr}
    )

    trainer = RL4COTrainer(
        max_epochs=100,
        accelerator=accelerator,
        devices=1,
        logger=None,
    )

    trainer.fit(model)

    model_name = f'/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk01/rl4co_model_{lr}_{name}_less2util.ckpt'

    trainer.save_checkpoint(model_name)

    print(f"saved model for {model_name}")