import torch
import code.scheduling.schedule as schedule
# Your schedule tensor

from rl4co.envs import FJSPEnv
# ------------------------------------------------------------
# Load instance
# ------------------------------------------------------------

dataset_folder = '/cluster/datastore/vemundvb/diffusion/diff_project/mindre_prosjekt/data/batched_444_TEST'
instance_idx = 10

td = schedule.get_td_from_path(dataset_folder, instance_idx)
td.del_("opt_assignment")
td.del_("opt_assignment_order")
td.del_("opt_actions")
td = td.unsqueeze(0)

env = FJSPEnv()
td = env.reset(td)

