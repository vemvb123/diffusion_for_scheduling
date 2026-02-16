
import code.dataset_code.utils as dataset_utils
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)

from typing import Callable, Dict, List, Tuple
import sys
from datetime import datetime


import code.training.diffusion as training
from code.dataset_code.dataset import Dataset_RL4CO


# 1 adj ordered
# 2 adj
# 3 f ordered
# 4 f

# maps file execution parameter (1-4) to some model to train
model_to_train = None
use_cos = True
if len(sys.argv) > 1:
    model_to_train = int(sys.argv[1])
    if int(sys.argv[2]) == 1:
        use_cos = True
    elif int(sys.argv[2]) == 2:
        use_cos = False

    logging.info(f"Training models nr {model_to_train}")
else:
    logging.info("Please provide a training number!")

timesteps = None
if model_to_train == 1:
    timesteps = 1000
elif model_to_train == 2:
    timesteps = 800

print(f"training model on timestep {timesteps} with scheduler cos: {use_cos}")




model_type = "adj"
order = True

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

    lrs = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

       
    best_loss = 1
    best_lr = None

    logging.info(f"Began training timestep model {model_path_adj} at time {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

    for lr in lrs:
        logging.info(f"Training with lr {lr}")
        path_enc, path_adj, last_epoch_loss = training.diffusion(
            model_type, train_dataset, test_dataset,
            embed_size, model_path_adj, model_path_enc,
            graph_name, graph_save_folder, valid_h, valid_w, timesteps, testing_epochs, lr,
            use_cos=use_cos
        ) 
        if best_loss > last_epoch_loss: 
            best_lr = lr
            best_loss = last_epoch_loss

    logging.info(f"Best lr found: {best_lr}, for model {path_adj}")
    path_enc, path_adj, last_epoch_loss = training.diffusion(
        model_type, train_dataset, test_dataset,
        embed_size, model_path_adj, model_path_enc,
        graph_name, graph_save_folder, valid_h, valid_w, timesteps, run_epochs, best_lr,
        use_cos=use_cos
    )

    logging.info(f"Ended training model {path_adj} at time {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")



def four():
    
    generator_params = {
        "num_jobs": 4,
        "num_machines": 4,
        "min_ops_per_job": 4,
        "max_ops_per_job": 4,
        "min_processing_time": 5,
        "max_processing_time": 50,
        "min_eligible_ma_per_op": 4,
        "max_eligible_ma_per_op": 4,
    }

    embed_size = 80

    h = 24
    w = 24
    valid_h = 4
    valid_w = 16
    testing_epochs = 2
    run_epochs = 100


    graph_name = f"model {model_type}, with order: {order}"

    full_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt'

    graph_save_folder = f"{full_path}/results/444" 
    train_dataset_path = f'{full_path}/data/batched_444'
    test_dataset_path = f'{full_path}/data/batched_444_TEST'

    
    model_path_enc = f'{full_path}/models/444/enc_type_{model_type}_order_{order}.pth'
    model_path_adj = f'{full_path}/models/444/adj_type_{model_type}_order_{order}.pth'
    
    return generator_params, embed_size,h, w, testing_epochs, run_epochs, graph_name, graph_save_folder, train_dataset_path, test_dataset_path, model_path_enc, model_path_adj, valid_h, valid_w
    


def mk01():
    filepath_brandimarte_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/brandimarte/mk01.txt'
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

    embed_size = 80

    h = 24
    w = 64
    valid_h = 6
    valid_w = 60
    testing_epochs = 1
    run_epochs = 50


    graph_name = f"model {timesteps}, with order: {order}"

    full_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt'

    graph_save_folder = f"{full_path}/results/timestep_mk01" 

    train_dataset_path = f'{full_path}/data/batched_mk01_10j_6ma_6op_mk01'
    test_dataset_path = f'{full_path}/data/batched_mk01_10j_6ma_6op_mk01_TEST'

    model_path_enc = f'{full_path}/models/beta/enc_timestep_{timesteps}.pth'

    if use_cos:
        model_path_adj = f'{full_path}/models/cos_beta/timestep_{timesteps}_cos.pth'
    else:
        model_path_adj = f'{full_path}/models/cos_beta/timestep_{timesteps}_beta.pth'
    
    return generator_params, embed_size,h, w, testing_epochs, run_epochs, graph_name, graph_save_folder, train_dataset_path, test_dataset_path, model_path_enc, model_path_adj, valid_h, valid_w
    


# generator_params, embed_size,h, w, testing_epochs, run_epochs, graph_name, graph_save_folder, train_dataset_path, test_dataset_path, model_path_enc, model_path_adj, valid_h, valid_w = four()
generator_params, embed_size,h, w, testing_epochs, run_epochs, graph_name, graph_save_folder, train_dataset_path, test_dataset_path, model_path_enc, model_path_adj, valid_h, valid_w = mk01()

train_models(model_type, order,
    generator_params,
    embed_size,
    h, w,
    testing_epochs, run_epochs,
    graph_name, graph_save_folder,
    train_dataset_path, test_dataset_path,
    model_path_enc, model_path_adj, 
    valid_h, valid_w)
 
