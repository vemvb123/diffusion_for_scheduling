import code.dataset_code
import code.dataset_code.dataset_maker
import code.dataset_code.dataset_utils as dataset_utils
import code.inference
import code.inference.experimental
import code.inference.inference as inference
import code.inference.inferenced_to_schedule
import code.inference.utils as utils
import code.scheduling.schedule as schedule


import torch


import code


def compare_inference(model_type: str, order: bool, adj_model_path, enc_model_path, dataset_folder, instance_idx):

    w = 60
    h = 10
    n_jobs = 10

    # GETTING DATA OF TEST INSTANCE TO CHECK
    td = dataset_utils.get_dataset_instance(dataset_folder, instance_idx)

    env, td_ignore, generator_params = code.dataset_code.dataset_maker.make_instance(
        n_ma=4,
        n_jobs=4,
        max_op_per_job=4,
        min_op_per_job=4,
        max_proc_time=50,
        min_proc_time=5,
        max_eligable_ma_per_op=4,
        min_eligable_ma_per_op=4,
        batch_size=1
    )


    target_assignments, proc_times, job_ops_adj, ops_ma_adj = dataset_utils.get_dataset_features(td, env, order)

    target_assignments = target_assignments.unsqueeze(0)
    proc_times = proc_times.unsqueeze(0)
    job_ops_adj = job_ops_adj.unsqueeze(0)
    ops_ma_adj = ops_ma_adj.unsqueeze(0)

    # GETTING THE INFERENCED RESULT
    embed_size = 80
    n_samples = 1

    print("Running inference")


    conditions = torch.cat([
        proc_times,
        job_ops_adj,
        ops_ma_adj,
    ], dim=1)

    inference_assignments = None
    if model_type == "adj":
        print("running inference")
        # NORMAL INFERENCE
        inference_assignments, elapsed, assignments_over_time = inference.adj_inference_ddpm_batch_influence(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples)
        print("running comparison inference")
        # INFERENCE DDIM
        comparison_inference_assignments, comparison_elapsed, comparison_assignments_over_time = code.inference.experimental.adj_inference_ddim(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples)
        # INFERENCE GUIDE
        # guided_inference_assignments, guided_elapsed, guidence.guided_assignments_over_time, errors = guide_adj_inference(conditions,3+1, adj_model_path, n_samples, 1)

    elif model_type == "f":
        pass
    else:
        raise ValueError("Model type must be either adj or f")


    print("skjekker når final result er...")
    print(f"comparison of assignlents.. {len(comparison_assignments_over_time)}")
    utils.check_when_inference_makes_final_schedule(comparison_assignments_over_time, comparison_inference_assignments, order)
    utils.check_when_inference_makes_final_schedule(assignments_over_time, inference_assignments, order)

    print(f"Inference took {elapsed} time")
    print(f"ddim inference took {comparison_elapsed} time")

    comparison_inference_assignments = comparison_inference_assignments[:, :, :h, :w]
    inference_assignments = inference_assignments[:, :, :h, :w]

     #for value in [x.item() for x in errors]:
    #    print(value)

    # CHANGING THE INFERENCED REPRESENTATION, FOR SCHEDULING AND VIZULISATION
    #if order:
    inference_assignments = code.inference.inferenced_to_schedule.round_to_values(inference_assignments, 16, ops_ma_adj)
    comparison_inference_assignments = code.inference.inferenced_to_schedule.round_to_values(comparison_inference_assignments, 16, ops_ma_adj)
    #else:
    #    inference_assignments = show_order_clear(inference_assignments, 16)
    #    guided_inference_assignments = show_order_clear(guided_inference_assignments, 16)

    print("RESULT")




    print(inference_assignments)
    print(comparison_inference_assignments)

    # SCHEDULING THE INFERENCED SCHEDULE
    graph_folder  = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results"
    graph_name = f"scheduled_model_type_{model_type} order_{order}.png"
    graph_name_comp = f"scheduled_model_type_{model_type} order_{order}_comp.png"
    graph_save_path = f"{graph_folder}/{graph_name}"
    graph_save_path_comp = f"{graph_folder}/{graph_name_comp}"

    # TODO fjern
    # td_scheduled = inferenced_schedule(target_assignments, order, env, td.copy(), graph_save_path)
    # TODO gjør seinere så ikke kommenter ut
    td_scheduled = schedule.schedule_from_inference(inference_assignments, order, env, td.copy(), graph_save_path, n_jobs)
    td_scheduled_comp = schedule.schedule_from_inference(comparison_inference_assignments, order, env, td.copy(), graph_save_path_comp, n_jobs)

    # GETTING THE MKESPAN OF THE SCHEDULED INFERENCED
    makespan = td_scheduled['time']
    makespan_comp = td_scheduled_comp['time']
    print(makespan)
    print(makespan_comp)