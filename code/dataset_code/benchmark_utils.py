import json
from typing import Dict

import torch
from rl4co.envs import FJSPEnv
from rl4co.models.zoo.l2d import L2DModel
from tensordict import TensorDict

from rl4co.envs import FJSPEnv
import torch

import numpy as np

from tensordict import from_dict

'''
use make_td_from_benchmark(file: str) to make a benchmark instance into a td
use get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance: str) -> Dict to get parameters from a benchmark to generate a similair instance (not same)

ignore the other functions
'''


def parse_mk01_json(json_data):
    """Parse the mk01 JSON description into job → ops → machine-processing lists."""
    jobs = json_data["jobs"]
    num_jobs = len(jobs)
    num_machines = json_data["machines"]

    # Count total operations
    ops = []
    job_to_ops = []
    for j, job in enumerate(jobs):
        for op in job:
            ops.append(op)
            job_to_ops.append(j)
    total_ops = len(ops)

    return jobs, job_to_ops, num_jobs, num_machines, total_ops

def parse_mk01_txt(txt_str):
    """Parse the Brandimarte .txt representation."""
    lines = txt_str.strip().splitlines()
    nj, nm = map(int, lines[0].split())
    jobs = []
    ptr = 1
    for _ in range(nj):
        parts = list(map(int, lines[ptr].split()))
        ptr += 1
        n_ops = parts[0]
        idx = 1
        job_ops = []
        for _ in range(n_ops):
            n_mach = parts[idx]
            idx += 1
            op_list = []
            for _m in range(n_mach):
                m_idx = parts[idx]
                p_time = parts[idx + 1]
                idx += 2
                op_list.append({"machine": m_idx, "processing": p_time})
            job_ops.append(op_list)
        jobs.append(job_ops)
    return {"machines": nm, "jobs": jobs}


def make_rl4co_instance(jobs, num_machines):
    """
    jobs: list of jobs, each job is list of ops,
          each op is list of {"machine", "processing"}
    """

    # Flatten all operations in sequence
    job_op_tuples = []
    for job_idx, job in enumerate(jobs):
        for op_idx, op in enumerate(job):
            job_op_tuples.append((job_idx, op_idx))
    total_ops = len(job_op_tuples)

    # proc_times: (num_machines, total_ops)
    proc_times = torch.zeros((num_machines, total_ops), dtype=torch.int32)
    for op_idx, (_job, _op) in enumerate(job_op_tuples):
        for ma in jobs[_job][_op]:
            proc_times[ma["machine"], op_idx] = ma["processing"]

    # pad_mask: zeros are valid ops; ones indicate padding if needed
    pad_mask = torch.zeros(total_ops, dtype=torch.bool)

    # job_ops_adj: (num_jobs, total_ops) adjacency (1 if op belongs to job)
    num_jobs = len(jobs)
    job_ops_adj = torch.zeros((num_jobs, total_ops), dtype=torch.int32)
    for op_idx, (job_idx, _) in enumerate(job_op_tuples):
        job_ops_adj[job_idx, op_idx] = 1

    # ops_sequence_order: sequence index for ops in each job
    ops_sequence_order = torch.zeros(total_ops, dtype=torch.int32)
    for op_idx, (j_idx, o_idx) in enumerate(job_op_tuples):
        ops_sequence_order[op_idx] = o_idx

    # start_op_per_job: first op index of each job
    start_op_per_job = torch.full((num_jobs,), -1, dtype=torch.int32)
    end_op_per_job = torch.full((num_jobs,), -1, dtype=torch.int32)
    for j_idx in range(num_jobs):
        ops_for_j = [i for i, (jj, _) in enumerate(job_op_tuples) if jj == j_idx]
        start_op_per_job[j_idx] = min(ops_for_j)
        end_op_per_job[j_idx] = max(ops_for_j)

    td = {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "start_op_per_job": start_op_per_job,
        "end_op_per_job": end_op_per_job,
        "proc_times": proc_times,
        "pad_mask": pad_mask,
        "job_ops_adj": job_ops_adj,
        "ops_sequence_order": ops_sequence_order,
    }
    return td

def make_td_from_benchmark(file: str):
    # load either JSON or TXT
    if file.endswith(".json"):
        data = json.load(open(file))
        parsed = parse_mk01_json(data)
        jobs = parsed["jobs"]
        num_machines = parsed["machines"]
    else:
        txt_content = open(file).read()
        parsed = parse_mk01_txt(txt_content)
        jobs = parsed["jobs"]
        num_machines = parsed["machines"]

    # build instance dictionary
    td = make_rl4co_instance(jobs, num_machines)

    # Convert everything to tensors with explicit batch dim
    batched_td = {}

    for key, value in td.items():
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        # ensure there is a batch dimension of 1
        # if the tensor is scalar or 1-D, we unsqueeze
        if value.dim() == 0:
            value = value.unsqueeze(0)  # shape [1]
        elif value.size(0) != 1:
            value = value.unsqueeze(0)  # shape [1, ...]

        batched_td[key] = value

    # Convert to TensorDict
    batch_td = TensorDict(batched_td, batch_size=[1])

    # reset the RL4CO environment
    env = FJSPEnv(generator=None)
    env_state = env.reset(batch_td)
    #print(env_state)

    return env_state

# Lager eksempler fra benchmark... lager ikke benchmark instanse, men problemer av samme storrelse
def get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance: str) -> Dict:

    with open(filepath_brandimarte_instance, "r") as f:
        lines = [line.strip() for line in f if line.strip()]


    # First line: number of jobs, number of machines
    first = lines[0].split()
    n_jobs = int(first[0])
    n_machines = int(first[1])

    # Stats
    global_min_pt = float("inf")
    global_max_pt = float("-inf")
    min_ops = float("inf")
    max_ops = float("-inf")

    # New stats for machine options per operation
    min_machine_options = float("inf")
    max_machine_options = float("-inf")

    # Loop through job lines
    for i in range(1, 1 + n_jobs):
        parts = list(map(int, lines[i].split()))
        idx = 0

        # Number of operations in this job
        n_ops = parts[idx]
        idx += 1

        # Update min/max number of operations
        min_ops = min(min_ops, n_ops)
        max_ops = max(max_ops, n_ops)

        # Loop through each operation
        for _ in range(n_ops):
            m_count = parts[idx]
            idx += 1

            # Track machine options stats
            min_machine_options = min(min_machine_options, m_count)
            max_machine_options = max(max_machine_options, m_count)

            # m_count pairs of (machine, processing time)
            for _ in range(m_count):
                machine_id = parts[idx]        # machine index (not needed for stats)
                proc_time = parts[idx + 1]     # processing time
                idx += 2

                # Track processing time
                global_min_pt = min(global_min_pt, proc_time)
                global_max_pt = max(global_max_pt, proc_time)


    return {
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "min_processing_time": global_min_pt,
        "max_processing_time": global_max_pt,
        "fewest_operations": min_ops,
        "most_operations": max_ops,
        "min_machine_options": min_machine_options,
        "max_machine_options": max_machine_options
    }


def parse_mk01_json(json_data):
    """Parse the mk01 JSON description into job → ops → machine-processing lists."""
    jobs = json_data["jobs"]
    num_jobs = len(jobs)
    num_machines = json_data["machines"]

    # Count total operations
    ops = []
    job_to_ops = []
    for j, job in enumerate(jobs):
        for op in job:
            ops.append(op)
            job_to_ops.append(j)
    total_ops = len(ops)

    return jobs, job_to_ops, num_jobs, num_machines, total_ops




def parse_mk01_txt(txt_str):
    lines = txt_str.strip().splitlines()
    nj, nm = map(int, lines[0].split())
    jobs = []
    ptr = 1

    for _ in range(nj):
        parts = list(map(int, lines[ptr].split()))
        ptr += 1

        n_ops = parts[0]
        idx = 1
        job_ops = []

        for _ in range(n_ops):
            n_mach = parts[idx]
            idx += 1

            op_list = []
            for _m in range(n_mach):
                m_idx = parts[idx]
                p_time = parts[idx + 1]
                idx += 2
                op_list.append({"machine": m_idx, "processing": p_time})

            job_ops.append(op_list)

        jobs.append(job_ops)

    return {"machines": nm, "jobs": jobs}


def make_rl4co_instance(jobs, num_machines):
    num_jobs = len(jobs)

    # 🔥 KEY FIX: compute padded size
    max_ops_per_job = max(len(job) for job in jobs)
    max_total_ops = num_jobs * max_ops_per_job

    # Flatten real operations
    job_op_tuples = []
    for job_idx, job in enumerate(jobs):
        for op_idx, _ in enumerate(job):
            job_op_tuples.append((job_idx, op_idx))

    total_real_ops = len(job_op_tuples)

    # -------------------------
    # proc_times (M, max_total_ops)
    # -------------------------
    proc_times = torch.zeros((num_machines, max_total_ops), dtype=torch.int32)

    for op_idx, (j, o) in enumerate(job_op_tuples):
        for ma in jobs[j][o]:
            proc_times[ma["machine"], op_idx] = ma["processing"]

    # -------------------------
    # pad_mask (True = padded)
    # -------------------------
    pad_mask = torch.zeros(max_total_ops, dtype=torch.bool)
    pad_mask[total_real_ops:] = True

    # -------------------------
    # job_ops_adj (num_jobs, max_total_ops)
    # -------------------------
    job_ops_adj = torch.zeros((num_jobs, max_total_ops), dtype=torch.int32)
    for op_idx, (job_idx, _) in enumerate(job_op_tuples):
        job_ops_adj[job_idx, op_idx] = 1

    # -------------------------
    # ops_sequence_order (max_total_ops)
    # -------------------------
    ops_sequence_order = torch.zeros(max_total_ops, dtype=torch.int32)
    for op_idx, (_, op_in_job) in enumerate(job_op_tuples):
        ops_sequence_order[op_idx] = op_in_job
    # padded tail stays 0 (correct)

    # -------------------------
    # start/end per job
    # -------------------------
    start_op_per_job = torch.zeros(num_jobs, dtype=torch.int32)
    end_op_per_job = torch.zeros(num_jobs, dtype=torch.int32)

    cursor = 0
    for j, job in enumerate(jobs):
        start_op_per_job[j] = cursor
        end_op_per_job[j] = cursor + len(job) - 1
        cursor += len(job)

    td = {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "start_op_per_job": start_op_per_job,
        "end_op_per_job": end_op_per_job,
        "proc_times": proc_times,
        "pad_mask": pad_mask,
        "job_ops_adj": job_ops_adj,
        "ops_sequence_order": ops_sequence_order,
    }

    return td


def make_td_from_benchmark_working(file: str, batch_size: int = 1,return_env: bool = False):
    # -------------------------
    # load data
    # -------------------------
    if file.endswith(".json"):
        data = json.load(open(file))
        jobs = data["jobs"]
        num_machines = data["machines"]
    else:
        txt_content = open(file).read()
        parsed = parse_mk01_txt(txt_content)
        jobs = parsed["jobs"]
        num_machines = parsed["machines"]

    td = make_rl4co_instance(jobs, num_machines)

    # -------------------------
    # add batch dim (ONLY ONCE)
    # -------------------------
    batched_td = {}
    for key, value in td.items():
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        batched_td[key] = value.unsqueeze(0)

    batch_td = TensorDict(batched_td, batch_size=[1])


    # -------------------------
    # reset env
    # -------------------------
    env = FJSPEnv(generator=None)
    env_state = env.reset(batch_td)

    # removing first dimension [batch, height, width] into [height, width]
    env_state = {k: v.squeeze(0) for k, v in env_state.items()}

    # TODO la til dette, for å konvertere til tensordict. hvis ikke funker, så fjern
    td = from_dict(td) 

    if return_env:
        return env_state, env
    return env_state












































# Lager eksempler fra benchmark... lager ikke benchmark instanse, men problemer av samme storrelse
def get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance: str) -> Dict:

    with open(filepath_brandimarte_instance, "r") as f:
        lines = [line.strip() for line in f if line.strip()]


    # First line: number of jobs, number of machines
    first = lines[0].split()
    n_jobs = int(first[0])
    n_machines = int(first[1])

    # Stats
    global_min_pt = float("inf")
    global_max_pt = float("-inf")
    min_ops = float("inf")
    max_ops = float("-inf")

    # New stats for machine options per operation
    min_machine_options = float("inf")
    max_machine_options = float("-inf")

    # Loop through job lines
    for i in range(1, 1 + n_jobs):
        parts = list(map(int, lines[i].split()))
        idx = 0

        # Number of operations in this job
        n_ops = parts[idx]
        idx += 1

        # Update min/max number of operations
        min_ops = min(min_ops, n_ops)
        max_ops = max(max_ops, n_ops)

        # Loop through each operation
        for _ in range(n_ops):
            m_count = parts[idx]
            idx += 1

            # Track machine options stats
            min_machine_options = min(min_machine_options, m_count)
            max_machine_options = max(max_machine_options, m_count)

            # m_count pairs of (machine, processing time)
            for _ in range(m_count):
                machine_id = parts[idx]        # machine index (not needed for stats)
                proc_time = parts[idx + 1]     # processing time
                idx += 2

                # Track processing time
                global_min_pt = min(global_min_pt, proc_time)
                global_max_pt = max(global_max_pt, proc_time)


    return {
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "min_processing_time": global_min_pt,
        "max_processing_time": global_max_pt,
        "fewest_operations": min_ops,
        "most_operations": max_ops,
        "min_machine_options": min_machine_options,
        "max_machine_options": max_machine_options
    }