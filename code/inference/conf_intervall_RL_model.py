from code.dataset_code.benchmark_utils import make_td_from_benchmark_working
from code.dataset_code.dataset_maker import make_actions_for_instance
from code.scheduling.schedule import schedule_actions_batch
from code.dataset_code.benchmark_values import get_benchmark_values
from tensordict import from_dict
import torch
import csv
import os

def make_conf_RL_model(checkpoint_path, benchmark_path, batch_size, n_in_intervall, report_path):
    # get td for benchmark, with batched
    td, env= make_td_from_benchmark_working(benchmark_path, return_env=True)
    td = from_dict(td)      # dict → TensorDict
    td = td.unsqueeze(0).expand(n_in_intervall * batch_size).clone()

    # make actions from td
    actions = make_actions_for_instance(td, checkpoint_path)

    # schedule actions
    td, _ = schedule_actions_batch(env, actions, td, False)

    # format results
    batch_chunks = torch.split(td['time'], batch_size)

    file_exists = os.path.isfile(report_path)
    with open(report_path, "a", newline="") as f:
        writer = csv.writer(f)

        # write header ONLY if file does not exist
        if not file_exists:
            writer.writerow(["LB", "AVG", "UP"])

        for c in batch_chunks:
            lb = torch.min(c).item()
            avg = torch.mean(c).item()
            up = torch.max(c).item()

            writer.writerow([lb, avg, up])

benchmark = "mk02"

report_path = f'/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/confidence_intervals/RL_conf/RL_conf_{benchmark}.csv'
checkpoint_path, _, _, _, filepath_benchmark_instance, _, _, _, _, _ = get_benchmark_values(benchmark)
make_conf_RL_model(checkpoint_path, filepath_benchmark_instance, 32, 10, report_path)
