import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)


import code.inference.report_infeasibilities
from code.inference.utils import compute_job_lengths
import code.inference.inference as inference
import code.scheduling.schedule as schedule
import code.dataset_code.dataset_utils as dataset_utils
import code.inference.confidence_interval as confidence_interval_utils
import code.dataset_code.benchmark_utils as benchmark_utils
import code.dataset_code.benchmark_values as benchmark_values
import sys
import torch
import matplotlib

matplotlib.use('Agg')

import torch



def get_problem_type(problem_type : str):
    (td, env, mask_h, mask_w, target_assignments, proc_times, job_ops_adj, ops_ma_adj, valid_h, valid_w) = (None,) * 10

    _, model_path, dataset_folder, _, filepath_brandimarte_instance, valid_h, valid_w, mask_w, mask_h, _ = benchmark_values.get_benchmark_values(problem_type)

    parameters = code.dataset_code.benchmark_utils.get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance)
    env, td_ignore, generator_params = code.dataset_code.dataset_maker.make_instance(
        n_ma=parameters['n_machines'], 
        n_jobs=parameters['n_jobs'], 
        max_op_per_job=parameters['most_operations'], 
        min_op_per_job=parameters['fewest_operations'], 
        max_proc_time=parameters['max_processing_time'], 
        min_proc_time=parameters['min_processing_time'], 
        max_eligable_ma_per_op=parameters['max_machine_options'], 
        min_eligable_ma_per_op=parameters['min_machine_options'], 
        batch_size=1, 
        schedule_manually=True
    )

    return dataset_folder, env, mask_h, mask_w, valid_h, valid_w, model_path





# inference types: ddpm, guide, random# inference types: ddpm, guide, random
def run_inference(problem_type, instance_idx, benchmark_instance = None, inference_type = "ddpm", gamma_input = 0.0):
    (td, env, mask_h, mask_w, target_assignments, proc_times, job_ops_adj, ops_ma_adj, valid_h, valid_w) = (None,) * 10

    dataset_folder, env, mask_h, mask_w, valid_h, valid_w, adj_model_path = get_problem_type(problem_type)

    if benchmark_instance is None:
        td = dataset_utils.get_dataset_instance(dataset_folder, instance_idx)
    else:
        td = benchmark_utils.make_td_from_benchmark_working(benchmark_instance)
    target_assignments, proc_times, job_ops_adj, ops_ma_adj, ops_sequence_order, opt_actions = dataset_utils.get_dataset_features(td, env, True, mask_h, mask_w, 
                                                                                                                                  include_ops_sequence=True, 
                                                                                                                                  from_benchmark_instance=benchmark_instance is not None,
                                                                                                                                  include_ops_actions=False)
    target_assignments = target_assignments.unsqueeze(0)
    proc_times = proc_times.unsqueeze(0)
    job_ops_adj = job_ops_adj.unsqueeze(0)
    ops_ma_adj = ops_ma_adj.unsqueeze(0)


    elapsed = None
    inference_assignments = None
    columns_done = None
    n_ops = None

    n_samples =32
    n_ops = int(torch.count_nonzero(target_assignments))
    cos = True

    # timesteps used for different benchmarks
    TIMESTEPS = {
        "mk01": 100,
        "mk01_mautil": 100,
        "mk02": 100,
        "mk03": 120,
        "mk04": 150,
        "mk05": 150,
        "mk06": 150,
        "mk07": 150,
        "mk08": 170,
        "mk09": 170,
        "mk10": 170,
    }
    try:
        timesteps = TIMESTEPS[problem_type]
    except KeyError:
        raise ValueError(f"Timesteps not set for benchmark {problem_type}")

    graph_folder  = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/{problem_type}"
    graph_name = f"result_schedule_{problem_type}_bench_{inference_type}.png"
    graph_save_path = f"{graph_folder}/{graph_name}"

    report_file_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/inference_reports/inference_report_file_{timesteps}t_{n_samples}b_zero_{problem_type}_{inference_type}.txt"
    report_file_path_fix = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/inference_reports/inference_report_file_{timesteps}t_{n_samples}b_zero_{problem_type}_{inference_type}.txt"

    # === DDIM
    #inference_assignments, elapsed, assignments_over_time = experimental.adj_inference_ddim(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w, sampling_steps = sampling_steps, ddim_eta = ddim_eta, timesteps = timesteps)
    #elapsed = None 
    #assignments_over_time = None
    ## == CACHE
    # inference_assignments, elapsed, assignments_over_time, columns_done, done_at_t = cache_inference.adj_inference_ddpm(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, order, mask_h, mask_w, after_ts_check, valid_h, valid_w, n_ops, threshold )

    # === DDPM
    inference_assignments, elapsed, assignments_over_time, variation_over_time = None, None, None, None
    if inference_type == "ddpm":
        logging.info("Running inference with DDPM")

        inference_assignments, elapsed, assignments_over_time, variation_over_time = inference.adj_inference_ddpm(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w, timesteps,
            jump=None, 
            cos=cos,
            smart_init=False,
            n_ops=n_ops, ops_sequence_order=ops_sequence_order, td=td, valid_h=valid_h, valid_w=valid_w)

    # GUIDING
    if inference_type == "guide":
        logging.info(f"Running inference with sampling modification and gamma {gamma_input}")

        inference_assignments, elapsed, assignments_over_time, variation_over_time = inference.inference_guide(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w, timesteps,
            jump=None, 
            cos=cos,
            smart_init=False,
            job_lengths=compute_job_lengths(td["ops_sequence_order"]),
            td=td,
            valid_h=valid_h, valid_w=valid_w,
            gamma_input=gamma_input)
        
    # BATCH REPLACEMENT
    if inference_type == "batchrep":
        t_replace = 2
        logging.info(f"Running inference with batch replacement, replacing at timestep {t_replace}")

        inference_assignments, elapsed, assignments_over_time, variation_over_time = inference.adj_inference_ddpm_batch_replacement(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w, timesteps,
            jump=None, 
            cos=cos,
            smart_init=False,
            valid_h=valid_h, valid_w=valid_w, t_replace=t_replace, n_ops=n_ops, ops_sequence_order=ops_sequence_order
        )

    # === LOOK AHEAD
    """
    inference_assignments, elapsed, assignments_over_time, variation_over_time = experimental.inference_lookahead(proc_times, job_ops_adj, ops_ma_adj, adj_model_path, n_samples, mask_h, mask_w, timesteps,
        jump=None, 
        cos=cos,
        smart_init=False
        )
    """
    # ==== RANDOM SCHEDULING
    if inference_type == "random":
        logging.info("Randomly scheduling")
        assignments_to_make = 32
        ops_seq_order = td["ops_sequence_order"]
        job_lengts = compute_job_lengths(ops_seq_order)
        # valid_w = 280
        inference_assignments = schedule.schedule_randomly(ops_ma_adj, valid_h, valid_w, job_lengts, N=n_ops)
        for i in range(assignments_to_make-1):
            instance_batch= schedule.schedule_randomly(ops_ma_adj, valid_h, valid_w, job_lengts)
            inference_assignments = torch.cat([inference_assignments, instance_batch], dim=0)

        elapsed = 0
        assignments_over_time = None

        ops_seq_order = td["ops_sequence_order"]
        job_lengts = compute_job_lengths(ops_seq_order)
        n_ops = sum(x for x in job_lengts if x != 1)

    logging.info(f"Inference done. Took {elapsed} time")

    inference_assignments = inference_assignments[:, :, :valid_h, :valid_w]
    ops_ma_adj = ops_ma_adj[:, :, :valid_h, :valid_w]
    job_ops_adj = job_ops_adj[:, :, :valid_h, :valid_w]
    proc_times = proc_times[:, :, :valid_h, :valid_w]

    return td, env, mask_h, mask_w, target_assignments, proc_times, job_ops_adj, ops_ma_adj, inference_assignments, elapsed, assignments_over_time, valid_h, valid_w, report_file_path, adj_model_path, graph_save_path, report_file_path_fix, n_ops



def get_inference_result(instance_idx, w, h, problem_type, benchmark_instance = None, inference_type = "ddpm", result_path_name=None, gamma_input = 0.0):
    td, env, mask_h, mask_w, target_assignments, proc_times, job_ops_adj, ops_ma_adj, inference_assignments, elapsed, assignments_over_time, valid_h, valid_w, report_file_path, adj_model_path, graph_save_path, report_file_path_fix, n_ops = run_inference(problem_type, instance_idx, benchmark_instance, inference_type, gamma_input=gamma_input)

    logging.info(f"Shape of result: {inference_assignments.shape}")

    # counting infeasibilities
    inference_assignments_order = code.inference.inferenced_to_schedule.show_order_clear(inference_assignments, n_ops, ops_ma_adj, r_global=False) # tidligere order visning ... DENNE ER KLART BEDRE, far langt mindre feil for mk01
    report, total_errors, error_list,   total_error, total_error_p, multi_p, seq_p, infeas_rate, multi_rate, seq_rate, amf_infeas, amt_infeas_p = code.inference.report_infeasibilities.count_infeasibilities(inference_assignments_order, td["ops_sequence_order"][:valid_w], report_file_path=report_file_path, n_ops=n_ops)
    code.inference.report_infeasibilities.add_elapsed_time_report(elapsed, report_file_path)

    # uncomment to check model outputs
    '''
    save_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/analysis"
    matrix_graph_schedule.show_single_sched(inference_assignments_order, 2, td["ops_sequence_order"], ops_ma_adj, valid_h, valid_w, save_path, n_ops, cell_size=0.8, show_values=True)
    '''
    #save_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/analysis"
    #assignments_over_time.append(inference_assignments_order)
    #matrix_graph_schedule.show_sched(2, ops_ma_adj, inference_assignments_order, assignments_over_time, td["ops_sequence_order"], n_ops, valid_h, valid_w, save_path)


    logging.info("fixing infeasibilities")
    inference_assignments_fixed = code.inference.infeasibilities.fix_infeas_mk10(inference_assignments_order, ops_ma_adj, td["ops_sequence_order"], 
                                                                                 valid_w=valid_w, n_ops=n_ops, td=td, analyse_infeas=False) # Bruker denne .. inkluder bare de nedre argumenta hvis skal analysere
    inference_assignments_order = code.inference.inferenced_to_schedule.show_order_clear(inference_assignments_fixed, n_ops, ops_ma_adj)
    

    report_fix, total_errors_fix, error_list, total_error_fix, total_error_p_fix, multi_p_fix, seq_p_fix, infeas_rate_fix, multi_rate_fix, seq_rate_fix, amf_infeas_fix, amt_infeas_p_fix = code.inference.report_infeasibilities.count_infeasibilities(
    inference_assignments_order,
    td["ops_sequence_order"][:valid_w],
    report_file_path=report_file_path_fix,
    n_ops=n_ops
    )


    # uncomment to check model outputs after fixing infeasibilities
    # save_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/analysis_fix"
    # matrix_graph_schedule.show_sched(2, ops_ma_adj, inference_assignments, assignments_over_time, td["ops_sequence_order"], n_ops, valid_h, valid_w, save_path)
    '''
    save_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/analysis_fix"
    matrix_graph_schedule.show_single_sched(inference_assignments_fixed, 2, td["ops_sequence_order"], ops_ma_adj, valid_h, valid_w, save_path, n_ops, cell_size=0.8, show_values=True)
    '''


    n_machines = valid_h
    td_scheduled, min_makespan, max_makespan, avg_makespan, busy_count, ma_counts = schedule.schedule_from_inference(inference_assignments_order, env, td.copy(), ops_sequence_order=td["ops_sequence_order"])

    code.inference.report_infeasibilities.add_makespans_report(min_makespan, max_makespan, avg_makespan, report_file_path)


    benchmark = problem_type
    if result_path_name is not None:
        logging.info(f'writing results to {csv_path}')

        path_before_csv = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/results/confidence_intervals'
        csv_path = f"{path_before_csv}/{result_path_name}"
        confidence_interval_utils.append_results(
            csv_path,
            min_makespan, avg_makespan, max_makespan, elapsed,
            total_errors, total_error, total_error_p, multi_p, seq_p, infeas_rate, multi_rate, seq_rate, amf_infeas, amt_infeas_p,
            total_errors_fix, total_error_fix, total_error_p_fix, multi_p_fix, seq_p_fix, infeas_rate_fix, multi_rate_fix, seq_rate_fix, amf_infeas_fix, amt_infeas_p_fix,
            benchmark, busy_count
        )

        csv_path = f"{path_before_csv}/{result_path_name}"
        csv_path = csv_path.replace(".csv", "")
        csv_path += '_ma_usage.csv'
        confidence_interval_utils.append_ma_usage_result(csv_path, ma_counts)


    else: 
        logging.info('Not recording results, as result_path_name is set to None')
    



# run with argument 'inf' to gather 500 samples (run inference 500 times)
times = 1
if sys.argv[1] == 'inf':
    times = 500

# if run inference on dataset instance, use the 10th instance.
instance_idx = 10

# values to use: ddpm, random, guide
inference_type = 'ddpm'
benchmark = 'mk01'
# gamma = 0.5

ins = f'/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk01.txt'
result_path_name = f"runs_batchrep/batch_runs_metrics_{benchmark}_{inference_type}.csv"

get_inference_result(instance_idx, benchmark, 
                            # gamma_input = gamma, # used for when running with sampling modification

                            # benchmark_instance=None, # set to None to run on dataset instance, otherwise run on benchmark instance
                            benchmark_instance = f'/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/{benchmark}.txt',

                            inference_type = inference_type, 

                            result_path_name=result_path_name # set to None to not save result for statistical analysis
                            # result_path_name=None)
                            )

