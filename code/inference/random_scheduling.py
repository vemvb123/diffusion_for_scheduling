from code.inference.utils import compute_job_lengths
from code.scheduling import schedule
import torch
import code.dataset_code.benchmark_utils as benchmark_utils

from code.scheduling import utils
from joblib import Parallel, delayed
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tensordict import from_dict

def save_matrix_as_png(matrix, filename="matrix.png"):
    data = matrix.numpy()
    rows, cols = data.shape

    fig, ax = plt.subplots(figsize=(cols * 0.35, rows * 0.6))
    ax.axis('off')

    cell_text = []
    cell_colors = []

    for i in range(rows):
        text_row = []
        color_row = []

        for j in range(cols):
            val = data[i, j]

            if val == 0:
                text_row.append("")
                color_row.append("#f0f0f0")  # spacer / empty
            else:
                text_row.append(f"{val:.1f}")
                color_row.append("#90EE90")  # green

        cell_text.append(text_row)
        cell_colors.append(color_row)

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        cellLoc='center',
        loc='center'
    )

    # grid styling
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')

    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 1.5)

    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()

# shape = (h, w)
def generate_matrices(n, shape, groups, n_ops):
    value_range = (1, n_ops)
    groups = [6, 5, 5, 5, 6, 6, 5, 5, 6, 6, 1, 1, 1, 1, 1]

    matrices = []

    for _ in range(n):
        mat = torch.zeros(shape)

        col_idx = 0

        for g in groups:
            if g == 1:
                # skip or leave zero column
                col_idx += 1
                continue

            # generate strictly increasing random values
            vals = torch.sort(
                torch.rand(g) * (value_range[1] - value_range[0]) + value_range[0]
            ).values

            for i in range(g):
                row = torch.randint(0, shape[0], (1,)).item()
                mat[row, col_idx] = vals[i]
                col_idx += 1

        matrices.append(mat)

    return torch.stack(matrices)


# ==== RANDOM
def schedule_randomly(td, env, n_assignments, report_file):
    # Make random schedules
    ops_seq_order = td["ops_sequence_order"]
    job_lengts = compute_job_lengths(ops_seq_order)
    n_ops = sum(i for i in job_lengts if i > 1)
    h, w = td['ops_ma_adj'].shape

    assignments = generate_matrices(n_assignments, (h, w), job_lengts, n_ops)
    # report_file_mat = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/code/inference/random_schedule_report.png'
    # save_matrix_as_png(assignments[0], filename=report_file_mat)



    # convert schedules into actions
    print(f"making actions for assignments with {os.cpu_count()} processors")

    n_jobs = schedule.infer_n_jobs(td['ops_sequence_order']) 
    jobs_to_make = int(os.cpu_count() / 6)
    B = assignments.size(0)
    all_actions = Parallel(n_jobs=jobs_to_make)(
        delayed(utils.map_assignments_to_actions_text)( assignments[b], True, n_jobs )
        for b in range(B)
    )
    all_actions = torch.stack(all_actions)
    all_actions = all_actions.to(torch.int64) 

    # TODO må hente ut info for hver 32.. kan kanskje fortsatt skedulere alt samtidig.. kan eks bare kappe opp seinere i 32ere
    # schedule all actions 
    td = from_dict(td)
    td = td.unsqueeze(0).repeat(n_assignments, *([1] * td.ndimension()))
    td, ordered_assignments = schedule.schedule_actions_batch(env, all_actions, td, True)

    n_machines = h
    tds = Parallel(n_jobs=jobs_to_make)(
        delayed(schedule.do_actions)( all_actions, n_machines, td.copy(), env )
    )
    # format results and save to file



benchmark = "mk01"
ins = f'/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/{benchmark}.txt'
td, env = benchmark_utils.make_td_from_benchmark_working(ins, return_env=True)
schedule_randomly(td, env, 10, "random_schedule_report.txt")
