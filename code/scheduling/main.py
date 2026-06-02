from code.dataset_code.dataset_maker import make_instance, make_target


def main():

    env, td, generator_params = make_instance(
                    n_jobs=10,
                    n_ma=6,
                    min_proc_time=1,
                    max_proc_time=6,
                    min_op_per_job=5,
                    max_op_per_job=6,
                    min_eligable_ma_per_op=1,
                    max_eligable_ma_per_op=3,
                    batch_size=1
                    )
    # ma .. 6max op, n jobs 10 .... så for hver ma, er alle mulig op der...
    # for en op, hvordan vise at en op ikke kan skeduleres til den ma-en?
    checkpoint_path = f"/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/models/rl4co_model_0.0001_10j_6ma_6op_mk01.ckpt"
    td_scheduled, actions = make_target(env, td.copy(), checkpoint_path)

    for key in td.keys():
        print(key)

    print("seq order")
    print(td["ops_sequence_order"])  # gir rekkefølgen av operasjonene
    print(td["ops_sequence_order"].shape)
    print("job ops adj")
    print(td["job_ops_adj"])  # gir om en op tilhører jobben... conditionerer på denne også
    print(td["job_ops_adj"].shape)
    print("ops ma adj")
    print(td["ops_ma_adj"]) # denne gir om ma kan benytte op, kan lett conditione på den.
    print(td["ops_ma_adj"].shape)
    print("proc_times")
    print(td["proc_times"])
    print(f"Actions: {actions}")
    print("scheduled:")
    print(td_scheduled["ma_assignment"])
    print(td_scheduled["ma_assignment"].shape)



print(2)

from rl4co.envs import FJSPEnv
from rl4co.models.zoo.l2d import L2DModel


import code.dataset_code.benchmark_utils as dataset_utils

print(3)



def make_benchmark_instance_into_td(file_path: str):

    parameters = dataset_utils.get_rl4co_parameters_from_brandimarte_instance(file_path)
    print(parameters)

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
    td = env.reset(batch_size=[1])

    for key, value in td.items():
        print(f"{key}: {value.shape}")
        print(value)
        print("---")


