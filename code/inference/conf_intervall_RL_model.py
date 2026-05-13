from code.dataset_code.benchmark_utils import make_td_from_benchmark_working
from code.dataset_code.dataset_maker import make_actions_for_instance, make_instance
from code.scheduling.schedule import schedule_actions_batch
from code.dataset_code.benchmark_values import get_benchmark_values
from tensordict import from_dict
import torch
import csv
import os
from tensordict import TensorDict

import code.dataset_code.target_maker_mautil_schedule as schedule_mautil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from code.dataset_code import benchmark_values
from code.dataset_code import benchmark_utils

def schedule_instance_with_model(checkpoint_path, benchmark_path, path_save_image):

    _, model_path, dataset_folder, _, filepath_brandimarte_instance, valid_h, valid_w, mask_w, mask_h, _ = benchmark_values.get_benchmark_values('mk01')

    parameters = benchmark_utils.get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance)
    env, td, generator_params = make_instance(
        n_ma=parameters['n_machines'], 
        n_jobs=parameters['n_jobs'], 
        max_op_per_job=parameters['most_operations'], 
        min_op_per_job=parameters['fewest_operations'], 
        max_proc_time=parameters['max_processing_time'], 
        min_proc_time=parameters['min_processing_time'], 
        max_eligable_ma_per_op=parameters['max_machine_options'], 
        min_eligable_ma_per_op=parameters['min_machine_options'], 
        batch_size=1
    )
    '''
    td, env = make_td_from_benchmark_working(
        benchmark_path,
        return_env=True
    )

    td = TensorDict(
    {k: v.unsqueeze(0) for k, v in td.items()},
    batch_size=[1]
    )
    '''
    print(type(td))
    print(td.shape)

    # make actions from td
    actions = schedule_mautil.make_actions_for_instance(td, checkpoint_path)
    # actions = make_actions_for_instance(td, checkpoint_path)
    print(actions)

    # schedule actions
    td_scheduled, _ = schedule_actions_batch(env, actions, td, False)


    print(td_scheduled['time'])

    env.render(td_scheduled, 0)

    if path_save_image:

        plt.savefig(
            path_save_image,
            dpi=150,
            bbox_inches='tight'
        )

        print(f"Saved scheduled image at path {path_save_image}")



def make_conf_RL_model(
    checkpoint_path,
    benchmark_path,
    batch_size,
    n_in_intervall,
    report_path
    ):
    # get td for benchmark, with batched
    td, env = make_td_from_benchmark_working(
        benchmark_path,
        return_env=True
    )
    td = TensorDict(
    {
        k: v.unsqueeze(0).expand(n_in_intervall, *v.shape).clone()
        for k, v in td.items()
    },
    batch_size=[n_in_intervall],
    device=next(iter(td.values())).device
    )
    # td = td.unsqueeze(0).expand(n_in_intervall).clone()

    # make actions from td
    actions = make_actions_for_instance(td, checkpoint_path)

    # schedule actions
    td, _ = schedule_actions_batch(env, actions, td, False)

    makespans = td["time"]

    file_exists = os.path.isfile(report_path)

    with open(report_path, "a", newline="") as f:
        writer = csv.writer(f)

        # CASE 1:
        # No batching -> write every makespan individually
        if batch_size is None:

            if not file_exists:
                writer.writerow(["Makespan"])

            for ms in makespans:
                writer.writerow([ms.item()])

        # CASE 2:
        # Original LB / AVG / UP batching
        else:

            if not file_exists:
                writer.writerow(["LB", "AVG", "UP"])

            batch_chunks = torch.split(makespans, batch_size)

            for c in batch_chunks:
                lb = torch.min(c).item()
                avg = torch.mean(c).item()
                up = torch.max(c).item()

                writer.writerow([lb, avg, up])




benchmark = "mk01"
checkpoint_path, _, _, _, filepath_benchmark_instance, _, _, _, _, _ = get_benchmark_values(benchmark)
# checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk01/rl4co_model_0.0001_10j_6ma_6op_mk01_less_machine2.ckpt'
checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/mk01/rl4co_model_0.01_10j_6ma_6op_mk01_limited_machine2_lr2.ckpt'
path_save_image = f'/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/code/inference/scheduled_by_model_lessmachilessmachine22_for_{benchmark}.png'
schedule_instance_with_model(checkpoint_path, filepath_benchmark_instance, path_save_image)
# bruker ikke LB for dette trur jeg, skal bare vise til at er basert på RL modell
# uansett trener jo ikke modell på beste utvalg fra RL modellen
'''
benchmark = "mk01"
report_path = f'/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/confidence_intervals/RL_conf/RL_conf_{benchmark}.csv'
checkpoint_path, _, _, _, filepath_benchmark_instance, _, _, _, _, _ = get_benchmark_values(benchmark)
make_conf_RL_model(checkpoint_path, filepath_benchmark_instance, None, 500, report_path)

benchmark = "mk10"
report_path = f'/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/confidence_intervals/RL_conf/RL_conf_{benchmark}.csv'
checkpoint_path, _, _, _, filepath_benchmark_instance, _, _, _, _, _ = get_benchmark_values(benchmark)
make_conf_RL_model(checkpoint_path, filepath_benchmark_instance, None, 500, report_path)

'''




