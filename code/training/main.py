import os
from code.dataset_code import benchmark_values
import torch
import code.dataset_code.benchmark_utils as dataset_utils
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)

from typing import Callable, Dict, List, Tuple
import sys
from datetime import datetime

import code.training.diffusion_improv as train_improv
import code.training.diffusion as training
from code.dataset_code.dataset import Dataset_RL4CO



model_type = "adj"
order = True

# continue training an existing model
# takes an extra parameter: lr (learning rate)
def continue_training(
    model_type, order,
    generator_params,
    embed_size,
    h, w,
    testing_epochs, run_epochs,
    graph_name, graph_save_folder,
    train_dataset_path, test_dataset_path,
    model_path_enc, model_path_adj,
    valid_h, valid_w,
    lr=None

    ):

    train_dataset = Dataset_RL4CO(train_dataset_path, generator_params, order, h, w)
    test_dataset = Dataset_RL4CO(test_dataset_path, generator_params, order, h, w)

    scheduler_timesteps = 1000
    timesteps = 1000
    batch_size = 32

    use_cos = True
    penalty = False

    # TODO ukommenter det over for a finne beste lr, og ikke sett manuelt beste lr
    logging.info(f"continue training model {model_path_adj}  with lr {lr}")
    some, model_path, last_epoch_loss = training.diffusion(
        model_type, train_dataset, test_dataset,
        embed_size, model_path_adj, model_path_enc,
        graph_name, graph_save_folder, valid_h, valid_w,
        timesteps, scheduler_timesteps,
        run_epochs, lr, batch_size=batch_size,
        use_cos=use_cos, penalty=penalty
    )

    logging.info(f"Ended training model {model_path} at time {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")




# used for training a model
def train_models(
    model_type, order,
    generator_params,
    embed_size,
    h, w,
    testing_epochs, run_epochs,
    graph_name, graph_save_folder,
    train_dataset_path, test_dataset_path,
    model_path_enc, model_path_adj,
    valid_h, valid_w
    ):


    train_dataset = Dataset_RL4CO(train_dataset_path, generator_params, order, h, w)
    test_dataset = Dataset_RL4CO(test_dataset_path, generator_params, order, h, w)

    scheduler_timesteps = 1000
    timesteps = 1000
    batch_size = 32

    # cosine scheduler
    use_cos = True
    # penalty .. experimental, not fully implemented
    penalty = False

    best_loss = 1
    best_lr = None

    # finds the optimal lr value. Only runs for testing_epochs amount of epochs
    lrs = [1e-4, 1e-5, 1e-6]
    best_lr = lrs[0]
    for lr in lrs:
        logging.info('training')
        some, model_path, last_epoch_loss = training.diffusion(
            model_type, train_dataset, test_dataset,
            embed_size, model_path_adj, model_path_enc,
            graph_name, graph_save_folder, valid_h, valid_w,
            timesteps, scheduler_timesteps,
            testing_epochs, lr, batch_size=batch_size,
            use_cos=use_cos, penalty=penalty
        )
        if best_loss > last_epoch_loss: 
            best_lr = lr
            best_loss = last_epoch_loss
    logging.info(f"Best lr found: {best_lr}, for model {model_path_adj}")

    some, model_path, last_epoch_loss = training.diffusion(
        model_type, train_dataset, test_dataset,
        embed_size, model_path_adj, model_path_enc,
        graph_name, graph_save_folder, valid_h, valid_w,
        timesteps, scheduler_timesteps,
        run_epochs, best_lr, batch_size=batch_size,
        use_cos=use_cos, penalty=penalty
    )

    logging.info(f"Ended training model {model_path} at time {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")







# benchmark_instance = "mk01" .. for example
def benchmark_parameters(benchmark_instance: str):
    _, model_path_adj, train_dataset_path, test_dataset_path, filepath_brandimarte_instance, valid_h, valid_w, w, h, _ = benchmark_values.get_benchmark_values(benchmark_instance)

    parameters = dataset_utils.get_rl4co_parameters_from_brandimarte_instance(filepath_brandimarte_instance)
    logging.info(parameters)
    generator_params = {
        "num_jobs": parameters['n_jobs'],
        "num_machines": parameters['n_machines'],
        "min_ops_per_job": parameters['fewest_operations'],
        "max_ops_per_job": parameters['most_operations'],
        "min_processing_time": parameters['min_processing_time'],
        "max_processing_time": parameters['max_processing_time'],
        "min_eligible_ma_per_op": parameters['min_machine_options'],
        "max_eligible_ma_per_op": parameters['max_machine_options'],
    }

    # Parameter not used
    embed_size = 80

    testing_epochs = 2
    run_epochs = 50

    graph_name = benchmark_instance
    # TODO User may have to change full_path
    full_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt'
    graph_save_folder = f"{full_path}/results/{benchmark_instance}/train" 
    model_path_enc = f'{full_path}/models/{benchmark_instance}/{benchmark_instance}.pth'
 
    return generator_params, embed_size,h, w, testing_epochs, run_epochs, graph_name, graph_save_folder, train_dataset_path, test_dataset_path, model_path_enc, model_path_adj, valid_h, valid_w




# TODO Set benchmark_instance .. for example benchmark_instance = "mk01"
benchmark_instance = None
generator_params, embed_size,h, w, testing_epochs, run_epochs, graph_name, graph_save_folder, train_dataset_path, test_dataset_path, model_path_enc, model_path_adj, valid_h, valid_w = benchmark_parameters(benchmark_instance)


logging.info(f"Began training model {model_path_adj} at time {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
logging.info("CUDA available:", torch.cuda.is_available())
logging.info("Number of GPUs:", torch.cuda.device_count())
logging.info("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
logging.info(f"Saving model to {model_path_adj}")



common_args = (
    model_type,
    order,
    generator_params,
    embed_size,
    h,
    w,
    testing_epochs,
    run_epochs,
    graph_name,
    graph_save_folder,
    train_dataset_path,
    test_dataset_path,
    model_path_enc,
    model_path_adj,
    valid_h,
    valid_w,
)

train_models(*common_args)

lr = None
# continue_training(*common_args, lr)

