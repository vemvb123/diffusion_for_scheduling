import code.dataset_code.benchmark_utils as benchmark_utils
import code.dataset_code.benchmark_values as benchmark_values

import csv
import torch
from rl4co.envs import FJSPEnv
from rl4co.models.zoo.l2d import L2DModel
from rl4co.models.zoo.l2d.decoder import TensorDict
from rl4co.models.zoo.l2d.policy import L2DPolicy


def use_trained_model_on_benchmark_instance(instance_value: str):
    checkpoint_path = None
    if instance_value == "mk02":
        checkpoint_path = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_0.001_10j_6ma_6op_mk02.ckpt'
    else:
        raise ValueError(f"instance value {instance_value} does not exist. Either not made code for, or invalid")

    path_brandimarte = f'/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/{instance_value}.txt'
    td_dict = benchmark_utils.make_td_from_benchmark_working(path_brandimarte)
    td = TensorDict(
        {k: torch.tensor(v) for k, v in td_dict.items()},
        batch_size=[]
    )



    parameters = benchmark_utils.get_rl4co_parameters_from_brandimarte_instance(path_brandimarte)
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

    # TODO lurer på om ikke dette lager en ny instanse
    td = td.unsqueeze(0)
    td = env.reset(td)


    if torch.cuda.is_available():
        accelerator = "gpu"
        batch_size = 24
        train_data_size = 2_000
        embed_dim = 128
        num_encoder_layers = 4

    policy = L2DPolicy(embed_dim=embed_dim, num_encoder_layers=num_encoder_layers, env_name="fjsp")



    model = L2DModel.load_from_checkpoint(
        checkpoint_path,
        env=env,          # IMPORTANT
        policy=policy     # IMPORTANT
    )

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    td = td.to(device)

    with torch.no_grad():
        out = model(
            td.clone(),
            phase="test",
            decode_type="greedy",
            return_actions=True
        )

    actions = out["actions"]
    makespan = -out["reward"]

    print(f"makespan: {makespan}")
    print(f"actions: {actions}")


def solve_trained_model_on_benchmark_instance_batch(instance_value: str, batch_size: int = 32, csv_path: str = "results.csv"):

    # ---------------------------
    # 1. LOAD CHECKPOINT
    # ---------------------------

    checkpoint_path, _, _, _, _, _, _, _, _ = benchmark_values.get_benchmark_values(instance_value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # 2. LOAD SINGLE INSTANCE
    # ---------------------------
    path = f'/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/benchmarks/brandimarte/{instance_value}.txt'

    td_dict = benchmark_utils.make_td_from_benchmark_working(path)

    td = TensorDict(
        {k: torch.tensor(v, device=device) for k, v in td_dict.items()},
        batch_size=[]
    )

    # ---------------------------
    # 3. ENV PARAMETERS
    # ---------------------------
    parameters = benchmark_utils.get_rl4co_parameters_from_brandimarte_instance(path)

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

    env = FJSPEnv(
        generator_params=generator_params,
        _torchrl_mode=True,
        stepwise_reward=True
    )

    env = env.to(device)

    # ---------------------------
    # 4. CREATE BATCH
    # ---------------------------
    td = td.unsqueeze(0)  # [1, ...]

    # Expand to batch (no memory copy, efficient)
    td = td.expand(batch_size, *td.shape[1:]).clone()
    # expand creates a view, clone makes it safe :contentReference[oaicite:0]{index=0}

    td = env.reset(td)
    td = td.to(device)

    # ---------------------------
    # 5. LOAD MODEL
    # ---------------------------
    policy = L2DPolicy(
        embed_dim=128,
        num_encoder_layers=4,
        env_name="fjsp"
    )

    model = L2DModel.load_from_checkpoint(
        checkpoint_path,
        env=env,
        policy=policy
    )

    model = model.to(device)
    model.eval()

    # ---------------------------
    # 6. RUN INFERENCE
    # ---------------------------
    with torch.no_grad():
        out = model(
            td,
            phase="test",
            decode_type="greedy",
            return_actions=True
        )

    makespans = -out["reward"]  # shape: [batch_size]

    # ---------------------------
    # 7. SAVE TO CSV
    # ---------------------------
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["instance_id", "makespan"])

        for i, m in enumerate(makespans):
            writer.writerow([i, m.item()])

    print(f"Saved {batch_size} results to {csv_path}")

