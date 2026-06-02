import code.dataset_code.benchmark_utils
import code.dataset_code.dataset_utils as dataset_utils
import code.scheduling.schedule as schedule
from code.dataset_code.dataset import Dataset_RL4CO
from code.dataset_code import benchmark_utils
import sys
from code.dataset_code.benchmark_values import get_benchmark_values
from code.inference.utils import compute_job_lengths

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)



def main(instance_type = None):
    ### Lag dataset
    logging.info("Beginning dataset creation...")

    train_size = 100000
    test_size = int(train_size * 0.2)
    n = train_size + test_size

    if instance_type == "444":

        dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_444'
        dataset_utils.make_dataset(
            n, dataset_folder,
            n_jobs=4,
            n_ma=4,
            max_op_per_job=4,
            min_op_per_job=4,
            max_proc_time=50,
            min_proc_time=5,
            max_eligable_ma_per_op=4,
            min_eligable_ma_per_op=4,
            target_model='/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_0.0001.ckpt',
            order=True,
        )
        logging.info("done making 444")
        exit()

    target_model, _, dataset_folder, _, filepath_benchmark_instance, _, _, _, _, _ = get_benchmark_values(sys.argv[1])


    parameters = code.dataset_code.benchmark_utils.get_rl4co_parameters_from_brandimarte_instance(filepath_benchmark_instance)
    logging.info(parameters)

    logging.info(f"dataset folder: {dataset_folder}")
    logging.info(f"benchmark instance: {filepath_benchmark_instance}")
    logging.info(f"target model: {target_model}")

    dataset_utils.make_dataset(
        n, dataset_folder,
        n_jobs=parameters['n_jobs'],
        n_ma=parameters['n_machines'],
        max_op_per_job=parameters['most_operations'],
        min_op_per_job=parameters['fewest_operations'],
        max_proc_time=parameters['max_processing_time'],
        min_proc_time=parameters['min_processing_time'],
        max_eligable_ma_per_op=parameters['max_machine_options'],
        min_eligable_ma_per_op=parameters['min_machine_options'],
        target_model=target_model,
        order=True
    )

    logging.info("Done making dataset")


def dataset_ma_util(problem_type):
    ### Lag dataset
    logging.info("Beginning dataset creation...")

    # train_size = 100000
    train_size = 50000
    test_size = int(train_size * 0.2)
    n = train_size + test_size

    target_model, _, dataset_folder, _, filepath_benchmark_instance, _, _, _, _, _ = get_benchmark_values(problem_type)
    dataset_folder = dataset_folder + '_mautil'

    parameters = code.dataset_code.benchmark_utils.get_rl4co_parameters_from_brandimarte_instance(filepath_benchmark_instance)
    logging.info(parameters)

    logging.info("stated making dataset")
    logging.info(f"dataset folder: {dataset_folder}")
    logging.info(f"benchmark instance: {filepath_benchmark_instance}")
    logging.info(f"target model: {target_model}")

    dataset_utils.make_dataset_ma_util(
        n, dataset_folder,
        n_jobs=parameters['n_jobs'],
        n_ma=parameters['n_machines'],
        max_op_per_job=parameters['most_operations'],
        min_op_per_job=parameters['fewest_operations'],
        max_proc_time=parameters['max_processing_time'],
        min_proc_time=parameters['min_processing_time'],
        max_eligable_ma_per_op=parameters['max_machine_options'],
        min_eligable_ma_per_op=parameters['min_machine_options'],
        target_model=target_model,
        order=True
    )
    logging.info("Done making dataset")







def check_benchmark_parameters(benchmark: str):
    print(f"checking benchmark parameters... {benchmark}")
    #dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/'
    filepath_brandimarte_instance = f'/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/{benchmark}.txt'
    #filepath_brandimarte_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/dauzere/18a.txt'
    parameters = code.dataset_code.benchmark_utils.get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance)
    print(parameters)
    benchmark = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/{benchmark}.txt"
    td = code.dataset_code.benchmark_utils.make_td_from_benchmark_working(benchmark)
    ops_seq_order = td["ops_sequence_order"]
    job_lengts = compute_job_lengths(ops_seq_order)
    valid_assignments = td['ops_ma_adj']

    n_ops = sum(x for x in job_lengts if x != 1)
    n_ma = valid_assignments.shape[0]

    print(f"n ops: {n_ops}")
    print(f"n ma: {n_ma}")
    print(f"n valid assignments: {valid_assignments.sum().item()}")
    print(f"Size: {n_ma * n_ops}")




# benchmark instance = "mk10" - for example
def check_dataset(benchmark_instance: str):
    print(f'checking: {benchmark_instance}')
    # --- process test image
    instance_idx = 10
    _, _, dataset_folder, _, _, _, _, _, _, _ = get_benchmark_values(benchmark_instance)
    td = dataset_utils.get_dataset_instance(dataset_folder, instance_idx)


    print(td['job_ops_adj'])
    print(td.keys())
    print(td['opt_assignment'])
    print(td['opt_actions'])
    print(td['opt_assignment_order'])
    print(td['opt_assignment_order'].shape)




