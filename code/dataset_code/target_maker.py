"""
target_maker.py contains code for training a rl4co reinforcment model on the FJSP problem,
to make targets.

The targets can later be used in a dataset
"""


import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
import time
import networkx as nx
import matplotlib.pyplot as plt
from rl4co.envs import FJSPEnv
from rl4co.models.zoo.l2d import L2DModel
from rl4co.models.zoo.l2d.policy import L2DPolicy
from rl4co.models.zoo.l2d.decoder import L2DDecoder
from rl4co.models.nn.graph.hgnn import HetGNNEncoder
from rl4co.utils.trainer import RL4COTrainer

import code.dataset_code.benchmark_utils as utils


# Trains RL model to make scheduling targets
def train_model():
    print("Beginning training of target model")


    filepath_benchmark_instance = None
    name = None
    if int(sys.argv[1]) == 1:
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk15.txt'
        name = "30j_15ma_11op_mk15"
    elif int(sys.argv[1]) == 2:
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk10.txt'
        name = "20j_15ma_14op_mk10"
    elif int(sys.argv[1]) == 3:
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/dauzere/18a.txt'
        name = "20j_10ma_25op_18a"
    elif int(sys.argv[1]) == 4:
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk02.txt'
        name = "10j_6ma_6op_mk02"
    elif int(sys.argv[1]) == 5:
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk03.txt'
        name = "15j_8ma_10op_mk03"
    elif int(sys.argv[1]) == 6:
        filepath_benchmark_instance = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/mk04.txt'
        name = "15j_8ma_9op_mk04"






    print(f'making model {name}')
    print(f"benchmark instance: {filepath_benchmark_instance}")
    print('parameters:')

    parameters = utils.get_rl4co_parameters_from_brandimarte_instance(filepath_benchmark_instance)
    print(parameters)

    jobs = parameters['n_jobs']
    ma = parameters['n_machines']
    max_proc = parameters['max_processing_time']
    min_proc = parameters['min_processing_time']
    max_op_per_job = parameters['most_operations']
    min_op_per_job = parameters['fewest_operations']
    max_eligable_ma_per_op = parameters['max_machine_options']
    min_eligable_ma_per_op = parameters['min_machine_options']

# Lets generate a more complex instance

    generator_params = {
      "num_jobs": jobs,  # the total number of jobs
      "num_machines": ma,  # the total number of machines that can process operations
      "min_ops_per_job": min_op_per_job,  # minimum number of operatios per job
      "max_ops_per_job": max_op_per_job,  # maximum number of operations per job
      "min_processing_time": min_proc,  # the minimum time required for a machine to process an operation
      "max_processing_time": max_proc,  # the maximum time required for a machine to process an operation
      "min_eligible_ma_per_op": min_eligable_ma_per_op,  # the minimum number of machines capable to process an operation
      "max_eligible_ma_per_op": max_eligable_ma_per_op,  # the maximum number of machines capable to process an operation
    }

    env = FJSPEnv(
        generator_params=generator_params, 
        _torchrl_mode=True, 
        stepwise_reward=True
    )


    if torch.cuda.is_available():
        accelerator = "gpu"
        batch_size = 24
        train_data_size = 2_000
        embed_dim = 128
        num_encoder_layers = 4
    else:
        accelerator = "cpu"
        batch_size = 32
        train_data_size = 1_000
        embed_dim = 64
        num_encoder_layers = 2


    # Policy: neural network, in this case with encoder-decoder architecture
    policy = L2DPolicy(embed_dim=embed_dim, num_encoder_layers=num_encoder_layers, env_name="fjsp")

    # Model: default is AM with REINFORCE and greedy rollout baseline
    lrs = [1e-4]
    for lr in lrs:
        model = L2DModel(env,
                         policy=policy, 
                         baseline="rollout",
                         batch_size=batch_size,
                         train_data_size=train_data_size,
                         val_data_size=1_000,
                         optimizer_kwargs={"lr": lr})
        
        trainer = RL4COTrainer(
            max_epochs=100,
            accelerator=accelerator,
            devices=1,
            logger=None,
        )
        
        trainer.fit(model)

        model_name = f'models/rl4co_model_{lr}_{name}.ckpt'
        trainer.save_checkpoint(model_name)
        print(f"saved model for {model_name}")


train_model()
