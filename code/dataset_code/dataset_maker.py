from code.scheduling.schedule import schedule_actions_batch, modify_actions_batch_mautil, batched_schedule_rollout_keep_order, schedule_batch_instances_mautil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



from code.dataset_code.benchmark_utils import make_td_from_benchmark_working
import code.dataset_code.benchmark_values as benchmark_values


import torch
from rl4co.envs import FJSPEnv
from rl4co.models.zoo.l2d import L2DModel
from tensordict import TensorDict


from typing import Dict, List, Tuple


def make_actions_for_instance(td: TensorDict, checkpoint_path: str) -> Tuple[TensorDict, List]:
    print('make actions')
    model = L2DModel.load_from_checkpoint(checkpoint_path)
    model = model.to("cpu")

    with torch.inference_mode():
        out = model(td,
                    decode_type="multistart_sampling",
                    num_starts=5,
                    select_best=True,
                    return_actions=True)


    actions = out["actions"]

    return actions



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


def make_target_ma_util(env: FJSPEnv, td: TensorDict, checkpoint_path: str, order: bool = False) -> Tuple[TensorDict, List]:
    model = L2DModel.load_from_checkpoint(checkpoint_path)
    model = model.to("cpu")

    with torch.inference_mode():
        out = model(td,
                    decode_type="multistart_sampling",
                    num_starts=5,
                    select_best=True,
                    return_actions=True)
    actions = out["actions"]

    actions_modified = modify_actions_batch_mautil(actions, td.copy())
    td_scheduled, ordered_assignments = schedule_batch_instances_mautil(env, td.copy(), actions_modified)

    return td_scheduled, actions_modified, ordered_assignments



def make_target_mauti_mautill():

    checkpoint_path, _, _, _, filepath_benchmark_instance, _, _, _, _, name = \
    benchmark_values.get_benchmark_values('mk01')
    checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk01/rl4co_model_0.01_10j_6ma_6op_mk01_limited_machine2_lr2.ckpt'

    td, env = make_td_from_benchmark_working(
        filepath_benchmark_instance,
        return_env=True
    )

    td = TensorDict(
    {k: v.unsqueeze(0) for k, v in td.items()},
    batch_size=[1]
    )

    model = L2DModel.load_from_checkpoint(checkpoint_path)
    model = model.to("cpu")

    with torch.inference_mode():
        out = model(td,
                    decode_type="multistart_sampling",
                    num_starts=5,
                    select_best=True,
                    return_actions=True)
    actions = out["actions"]
    order = True
    # td_scheduled, ordered_assignments = schedule_actions_batch_mautil(env, actions, td.copy(), order)
    td_scheduled, _ = schedule_batch_instances_mautil(env, actions, td, order)

    path_save_image = f'/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/code/inference/scheduled_by_model_lessmachilessmachine22_hmm2.png'
    print(td_scheduled[0]['time'])
    env.render(td_scheduled, 0)
    if path_save_image:
        plt.savefig(
            path_save_image,
            dpi=150,
            bbox_inches='tight'
        )


    ##  return td_scheduled, actions, ordered_assignments

# make_target_mauti_mautill()


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
        # mask_no_ops=False
    )
    td = env.reset(batch_size=[batch_size])
    return env, td, generator_params