
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)

from typing import Callable, Dict, List, Tuple
import sys
from datetime import datetime

from diffusion import diffusion
from dataset import Dataset_RL4CO


# 1 adj
# 2 adj ordered
# 3 f
# 4 f ordered

# maps file execution parameter (1-4) to some model to train
model_to_train = None
if len(sys.argv) > 1:
    model_to_train = int(sys.argv[1])
    logging.info(f"Training models nr {model_to_train}")
else:
    logging.info("Please provide a training number!")


def map_file_parameter_to_model_type(model_to_train: int) -> Tuple[str, bool]:
    model_type = None
    order = None
    if model_to_train < 3:
        model_type = "adj"
    elif model_to_train > 2:
        model_type = "f"
    if model_to_train == 2 or model_to_train == 4:
        order = True
    elif model_to_train == 1 or model_to_train == 3:
        order = False

    print("Using model: type: {type}, order: {order}")
    if model_type == None or order == None:
        raise ValueError("That model type dosent exist. pecify one between 1 and 2")

    return model_type, order

map_file_parameter_to_model_type(model_to_train)






def train_models(model_type: str, order: bool):

    jobs = 4
    ma = 4
    ops_per_job = 4
    min_proc = 5
    max_proc = 50

    base_embed = 3
    embed_size = 80

    generator_params = {
        "num_jobs": jobs,
        "num_machines": ma,
        "min_ops_per_job": ops_per_job,
        "max_ops_per_job": ops_per_job,
        "min_processing_time": min_proc,
        "max_processing_time": max_proc,
        "min_eligible_ma_per_op": ma,
        "max_eligible_ma_per_op": ma,
    }
    # TODO full path

    lrs = [1e-3, 1e-4, 1e-5, 1e-6]
    testing_epochs = 2
    run_epochs = 100

    training_func = None
    graph_name = None
    if  model_type == "f": 
        training_func = diffusion
    elif model_type == "adj": 
        training_func = diffusion
    graph_name = f"model {model_type}, with order: {order}"

    full_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt'

    loss_image_path = f'{full_path}/models/feature_v_adj'
    graph_save_folder = "/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/graphs" 

    train_dataset_path = f'{full_path}/data/with_targets/batched_444'
    test_dataset_path = f'{full_path}/data/with_targets/test_batched_444'

    train_dataset = Dataset_RL4CO(train_dataset_path, generator_params, order)
    test_dataset = Dataset_RL4CO(test_dataset_path, generator_params, order)

    model_path_enc = f'{full_path}/models/feature_v_adj/enc_type_{model_type}_order_{order}.pth'
    model_path_adj = f'{full_path}/models/feature_v_adj/adj_type_{model_type}_order_{order}.pth'
    
    best_loss = 1
    best_lr = None

    print(f"Began training model {model_type} order_{order} at time {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

    for lr in lrs:
        logging.info(f"Training with lr {lr}")
        path_enc, path_adj, last_epoch_loss = training_func(
            model_type, loss_image_path, train_dataset, test_dataset,
            base_embed, embed_size, model_path_adj, model_path_enc,
            graph_name, graph_save_folder, testing_epochs, lr
        ) 
        if best_loss > last_epoch_loss: 
            best_lr = lr
            best_loss = last_epoch_loss

    print(f"Best lr found: {best_lr}, for model {model_type} order_{order} training full model now")
    path_enc, path_adj, last_epoch_loss = training_func(
        model_type, loss_image_path, train_dataset, test_dataset,
        base_embed, embed_size, model_path_adj, model_path_enc,
        graph_name, graph_save_folder, run_epochs, best_lr
    )

    print(f"Ended training model {model_type} order_{order} at time {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

