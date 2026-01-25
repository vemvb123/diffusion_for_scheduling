"""
target_maker.py contains code for training a rl4co reinforcment model on the FJSP problem,
to make targets.

The targets can later be used in a dataset
"""


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




def train_model():
    print("Beginning training of target model")

    jobs = 10
    ma = 6
    max_proc = 6
    min_proc = 1
    max_op_per_job = 6
    min_op_per_job = 5
    max_eligable_ma_per_op = 6
    min_eligable_ma_per_op = 1

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
        batch_size = 256
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
            max_epochs=10,
            accelerator=accelerator,
            devices=1,
            logger=None,
        )
        
        trainer.fit(model)

        model_name = f'models/rl4co_model_{lr}.ckpt'
        trainer.save_checkpoint(model_name)
        print(f"saved model for {model_name}")


train_model()
