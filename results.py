"""
results.py contains code for gathering results.
Such as gathering mean makespan of scheduled instances, graphs, training results, etc
"""

from scheduling_utils import get_td_from_path, make_instance, get_feature_adj_from_instance, make_adj_with_order
from inference import adj_inference, guide_adj_inference

from torchtyping import TensorType
from typing import Callable, Dict, List, Tuple
from torch import Tensor
import torch

import os
import logging
import bisect

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




def get_td_from_path(path: str, instance_idx: int) -> Tensor:
    # Get sorted list of data files
    files = sorted([f for f in os.listdir(path) if f.endswith(".pt")])
    # Build file ranges
    cum_sizes = []
    ranges = []
    total = 0

    for fname in files:
        start, end = map(int, fname.replace(".pt", "").split("_"))
        size = end - start
        cum_sizes.append(total)
        ranges.append((start, end))
        total += size

    # Find which file contains the instance
    file_idx = bisect.bisect_right(cum_sizes, instance_idx) - 1
    if file_idx < 0:
        raise ValueError(f"Instance {instance_idx} not found in {path}")

    file_path = os.path.join(path, files[file_idx])

    # Load with weights_only=False so that TensorDict objects (or other custom objects)
    # can be unpickled properly. Only do this if the file is from a trusted source.
    batch = torch.load(
        file_path,
        map_location="cpu",
        weights_only=False,  # use full pickle, not restricted weights_only loader
    )

    # Compute local index within this batch
    local_idx = instance_idx - cum_sizes[file_idx]
    return batch[local_idx]




def map_assignemnts_to_actions(assignments, order: bool):

    # remove batch/channel dims if present
    if assignments.dim() == 4:
        assignments = assignments.squeeze(0).squeeze(0)  # (4,16)

    H, W = assignments.shape  # H=4, W=16
    section_width = 4
    num_sections = W // section_width
    """
    actions = []
    if order:
        x = assignments

        x = x.squeeze(0).squeeze(0)

        # get all nonzero positions
        rows, cols = torch.nonzero(x, as_tuple=True)

        # get the values at those positions
        vals = x[rows, cols]

        # sort by value (1 → 16)
        order = torch.argsort(vals)
        rows = rows[order]
        cols = cols[order]

        # compute mapped values
        sections = cols // 4
        mapped = sections * 4 + (rows + 1)

        return mapped.tolist()


    # TODO inkluder dette igjen i funksjonen seinere
    """
    if order:
        # order by largest value first
        _, indices = torch.topk(assignments.flatten(), H * W)

        for idx in indices:
            row = idx // W
            col = idx % W
            section = col // section_width
            action = section * H + row + 1
            actions.append(action.item())

    else:
        cols_per_section = 4
        actions = []

        for col in range(assignments.shape[1]):
            section_idx = col // cols_per_section
            for row in range(assignments.shape[0]):
                if assignments[row, col] == 1: 
                    value = section_idx * cols_per_section + (row + 1)
                    actions.append(value)

        return actions


def make_step(env, td, action):
    td['action'] = torch.tensor([action])
    td = env.step(td)['next']


    return td




def inferenced_schedule(assignments, order: bool, env, td, path_save_image: str):
    actions = map_assignemnts_to_actions(assignments, order)
    # print(assignments)
    # print(td["opt_assignment"])
    # print(actions)
    # print(td["opt_actions"])
    # exit()
    td.del_("opt_assignment")
    td.del_("opt_actions")

    #print(td.shape)
    #td = TensorDict.from_dict(td, auto_batch_size=True)
    #print(td.shape)
    #td = TensorDict(td, batch_size=[1])
    #print(td.shape)

    td = td.unsqueeze(0)
    env.render(td, 0)


    #---
    while not td["done"].all():
        # loop true indexer
        # for en true index, se at maskinen den oppgaven skal skeduleres til ikke er opptatt
        # det ses ved at: mapped = [((v - 1) % 4) + 1 for v in actions]  ... fra ctions
        # hvis opptatt, gå til neste gå til neste true.
        # så den oppgaven endelig kn skeduleres, skeduleres den, så starter du å loope true fra starten av
        # time.sleep(10)
        #print(td["time"])
        #print(td["busy_until"])
        #print(td["is_ready"])
        #print(assignments)

        if order:
            pass

        else:
            ready_ops = torch.nonzero(td["is_ready"], as_tuple=True)[1]

            ma_indices_for_actions = [((v - 1) % 4) for v in actions]  # 0‑based
            # print("machine indices:", ma_indices_for_actions)

            for op in ready_ops:
                op = op.item()
                machine_idx = ma_indices_for_actions[op]
                busy_val = td["busy_until"][0, machine_idx]
                current_time = td["time"][0]

                if busy_val <= current_time:
                    td['action'] = torch.tensor([actions[op]])
                    td = env.step(td)['next']
                    env.render(td, 0)
                else:
                    td["time"] = busy_val.unsqueeze(0)

    if path_save_image:
        plt.savefig(path_save_image, dpi=150, bbox_inches='tight')
    return td







# første skedulerte har minst verdi, sist skedulerte har størst verdi
def show_order_clear(x, n_values):

    # flatten all values
    flat = x.flatten()

    # find the top 16 values and their indices
    topk_vals, topk_idx = torch.topk(flat, n_values)

    # sort those top 16 in descending order so largest -> rank 1
    sorted_vals, sorted_order = torch.sort(topk_vals, descending=False)
    top16_idx_sorted = topk_idx[sorted_order]

    # create an output tensor of zeros
    out = torch.zeros_like(flat)

    # assign ranks 1..16 to those positions
    for rank, idx in enumerate(top16_idx_sorted, start=1):
        out[idx] = rank

    # reshape back to original
    return out.view_as(x)





def round_to_values(x: torch.Tensor, n_values: int) -> torch.Tensor:
    orig_shape = x.shape

    # assume shape [1,1,4,16] or similar, so flatten leading dims
    flat = x.view(-1, x.shape[-2], x.shape[-1])  # [B, 4, 16]

    # find max in each column along row dim (dim=1)
    max_vals, _ = flat.max(dim=1, keepdim=True)  # [B, 1, 16]

    # compare with max and binarize
    mask = torch.isclose(flat, max_vals)  # True where value == max

    # convert to float (1.0/0.0)
    result = mask.float()

    # restore original leading dims
    result = result.view(orig_shape)

    return result
    """
    # flatten and get top k indices
    topk_vals, topk_idx = torch.topk(x.flatten(), n_values)

    # start with all zeros
    y = torch.zeros_like(x).flatten()

    # set top entries to 1
    y[topk_idx] = 1.0

    # reshape back to original shape
    return y.view(x.shape)

    """






def check_when_inference_makes_final_schedule(assignments_over_time: List[Tensor], final_assignment: Tensor, order: bool):
    for i, assignment_at_time in enumerate(assignments_over_time):
        assignment_at_time = assignment_at_time[:, :, :4, :16]
        if order:
            assignment_at_time = round_to_values(assignment_at_time, 16)
        else:
            assignment_at_time = show_order_clear(assignment_at_time, 16)

        if torch.equal(assignment_at_time, assignments_over_time):
            print(f"assignments are exactly the same at point {i}")
            print(assignment_at_time)
            print(final_assignment)
            break




def get_inference_result(model_type: str, order: bool, adj_model_path, enc_model_path, dataset_folder, instance_idx):

    # GETTING DATA OF TEST INSTANCE TO CHECK
    td = get_td_from_path(dataset_folder, instance_idx)

    env, td_ignore, generator_params = make_instance(4,4,4,5,50, batch_size=1)

    target_assignments, proc_times, job_id, pos_job = get_feature_adj_from_instance(td, env, order)

    target_assignments = target_assignments.unsqueeze(0)
    proc_times = proc_times.unsqueeze(0)
    job_id = job_id.unsqueeze(0)
    pos_job = pos_job.unsqueeze(0)

    # GETTING THE INFERENCED RESULT
    embed_size = 80
    n_samples = 1

    print("Running inference")
    elapsed = None
    inference_assignments = None
    if model_type == "adj":
        inference_assignments, elapsed, assignments_over_time = adj_inference(proc_times, job_id, pos_job, adj_model_path, n_samples)
    elif model_type == "f":
        pass
    else:
        raise ValueError("Model type must be either adj or f")

    print(f"Inference done. Took {elapsed} time")

    inference_assignments = inference_assignments[:, :, :4, :16]


    
    # CHANGING THE INFERENCED REPRESENTATION, FOR SCHEDULING AND VIZULISATION
    if order:
        inference_assignments = show_order_clear(inference_assignments, 16)
    else:
        inference_assignments = round_to_values(inference_assignments, 16)

   # CHECKING WHEN IN INFERENCE THE RESULT BECAME SIMILAIR TO THE END RESULT
    check_when_inference_makes_final_schedule(assignments_over_time, inference_assignments, order)

    # SCHEDULING THE INFERENCED SCHEDULE
    graph_folder  = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/graphs"
    graph_name = f"scheduled_model_type_{model_type} order_{order}.png"
    graph_save_path = f"{graph_folder}/{graph_name}"

    td_scheduled = inferenced_schedule(inference_assignments, order, env, td.copy(), graph_save_path)

    # GETTING THE MKESPAN OF THE SCHEDULED INFERENCED
    makespan = td_scheduled['time']
    print(makespan)


# HER
def compare_inference_guiding(model_type: str, order: bool, adj_model_path, enc_model_path, dataset_folder, instance_idx):

    # GETTING DATA OF TEST INSTANCE TO CHECK
    td = get_td_from_path(dataset_folder, instance_idx)

    env, td_ignore, generator_params = make_instance(4,4,4,5,50, batch_size=1)

    target_assignments, proc_times, job_id, pos_job = get_feature_adj_from_instance(td, env, order)

    target_assignments = target_assignments.unsqueeze(0)
    proc_times = proc_times.unsqueeze(0)
    job_id = job_id.unsqueeze(0)
    pos_job = pos_job.unsqueeze(0)

    # GETTING THE INFERENCED RESULT
    embed_size = 80
    n_samples = 1

    print("Running inference")


    conditions = torch.cat([            
        proc_times,
        job_id,
        pos_job,
    ], dim=1)

    inference_assignments = None
    if model_type == "adj":
        print("running inference")
        # NORMAL INFERENCE
        inference_assignments, elapsed, assignments_over_time = adj_inference(proc_times, job_id, pos_job, adj_model_path, n_samples)
        print("running guided inference")
        # INFERENCE GUIDE
        guided_inference_assignments, guided_elapsed, guided_assignments_over_time, errors = guide_adj_inference(conditions,3+1, adj_model_path, n_samples, 1)

    elif model_type == "f":
        pass
    else:
        raise ValueError("Model type must be either adj or f")


    guided_inference_assignments = guided_inference_assignments[:, :, :4, :16]
    inference_assignments = inference_assignments[:, :, :4, :16]


    
    # CHANGING THE INFERENCED REPRESENTATION, FOR SCHEDULING AND VIZULISATION
    #if order:
    inference_assignments = round_to_values(inference_assignments, 16)
    guided_inference_assignments = round_to_values(guided_inference_assignments, 16)
    #else:
    #    inference_assignments = show_order_clear(inference_assignments, 16)
    #    guided_inference_assignments = show_order_clear(guided_inference_assignments, 16)

    print("RESULT")
    print(inference_assignments)
    print(guided_inference_assignments)

    for value in [x.item() for x in errors]:
        print(value)

    check_when_inference_makes_final_schedule(guided_assignments_over_time, guided_inference_assignments, order)
    check_when_inference_makes_final_schedule(assignments_over_time, inference_assignments, order)

    # SCHEDULING THE INFERENCED SCHEDULE
    graph_folder  = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/graphs"
    graph_name = f"scheduled_model_type_{model_type} order_{order}.png"
    graph_save_path = f"{graph_folder}/{graph_name}"

    # TODO fjern
    td_scheduled = inferenced_schedule(target_assignments, order, env, td.copy(), graph_save_path)
    # TODO gjør seinere så ikke kommenter ut
    # td_scheduled = inferenced_schedule(inference_assignments, order, env, td.copy(), graph_save_path)

    # GETTING THE MKESPAN OF THE SCHEDULED INFERENCED
    makespan = td_scheduled['makespan']
    print(makespan)






model_type = "adj"
order = False
adj_model_path = None
enc_model_path = None
if order:
    adj_model_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/feature_v_adj/adj_type_2.pth'
else:
    adj_model_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/feature_v_adj/adj_type_1.pth'

dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/with_targets/test_batched_444'
instance_idx = 10

print("inference result")
get_inference_result(model_type, order, adj_model_path, enc_model_path, dataset_folder, instance_idx)
exit()
   







