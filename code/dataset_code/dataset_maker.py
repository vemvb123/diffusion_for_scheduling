from code.scheduling.schedule import schedule_actions_batch


import torch
from rl4co.envs import FJSPEnv
from rl4co.models.zoo.l2d import L2DModel
from tensordict import TensorDict


from typing import Dict, List, Tuple


def make_target(env: FJSPEnv, td: TensorDict, checkpoint_path: str, order: bool = False) -> Tuple[TensorDict, List]:
    model = L2DModel.load_from_checkpoint(checkpoint_path)
    model = model.to("cpu")

    with torch.inference_mode():
        out = model(td,
                    decode_type="multistart_sampling",
                    num_starts=5,
                    select_best=True,
                    return_actions=True)
    actions = out["actions"]
    td_scheduled, ordered_assignments = schedule_actions_batch(env, actions, td.copy(), order)


    return td_scheduled, actions, ordered_assignments


def make_instance(
    n_ma, n_jobs, max_op_per_job, min_op_per_job, max_proc_time, min_proc_time, max_eligable_ma_per_op, min_eligable_ma_per_op, batch_size
) -> Tuple[FJSPEnv, TensorDict, Dict]:



    generator_params = {
        "num_jobs": n_jobs,
        "num_machines": n_ma,
        "min_ops_per_job": min_op_per_job,
        "max_ops_per_job": max_op_per_job,
        "min_processing_time": min_proc_time,
        "max_processing_time": max_proc_time,
        "min_eligible_ma_per_op": min_eligable_ma_per_op,
        "max_eligible_ma_per_op": max_eligable_ma_per_op,
    }

    env = FJSPEnv(
        generator_params=generator_params,
        _torchrl_mode=True,
        stepwise_reward=True
    )
    td = env.reset(batch_size=[batch_size])
    return env, td, generator_params