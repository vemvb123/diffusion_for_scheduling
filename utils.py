import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d - %(message)s"
)



import random
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import Tuple, Optional
from IPython.display import display, clear_output

# logging.info("imports")

"""
Contrains the following:
Method to vizulise a schedule
A class representing a dataset, which combined op and ma embeddings, and can also return a combined data back into a ma and op embedding
"""
# logging.info(1)
import torch
from torch.utils.data import Dataset
# logging.info("skjekk")
from rl4co.envs import FJSPEnv
from rl4co.models.zoo.l2d import L2DModel
from rl4co.models.zoo.l2d.policy import L2DPolicy
from rl4co.models.zoo.l2d.decoder import L2DDecoder
from rl4co.models.nn.graph.hgnn import HetGNNEncoder
from rl4co.utils.trainer import RL4COTrainer
from IPython.display import display, clear_output

# logging.info(2)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import time
import random
from torchvision import transforms

# logging.info(3)
from torch.utils.data import Dataset
from PIL import Image
import glob
import os

import torch

import numpy as np
import networkx as nx  # Make sure you install networkx if you don’t have it


import json

# logging.info(4)

from tensordict import TensorDict, from_dict
from torch import Tensor
from torchtyping import TensorType
from typing import Callable
# import inference



# logging.info("start of file")






# TODO
# Når holder på videre, gjør følgende:
# fjern alt under, og hent fra prev kun det du trenger




def tensordict_to_dict(td):
    simple_dict = {}
    for key, value in td.items():
        # If value is a tensor or Tensordict style, convert
        if isinstance(value, torch.Tensor):
            simple_dict[key] = value.cpu().tolist()   # list of numbers
        else:
            # you might need further checks if value is a TensorDict itself or custom type
            try:
                simple_dict[key] = value  # assume already serializable
            except TypeError:
                simple_dict[key] = str(value)

    return simple_dict
     


def visulize_schedule(td, env, embed_dim, ma_emb, op_emb, batch_size):
    td = env.reset(batch_size=[batch_size])
    ma_emb = ma_emb.cpu()
    op_emb = op_emb.cpu()
    # Visulizing a sample
    decoder = L2DDecoder(env_name=env.name, embed_dim=embed_dim)
    logits, mask = decoder(td, (ma_emb, op_emb), num_starts=0)
    # (1 + num_jobs * num_machines)

    def make_step(td):
        logits, mask = decoder(td, (ma_emb, op_emb), num_starts=0)
        action = logits.masked_fill(~mask, -torch.inf).argmax(1)
        td["action"] = action
        td = env.step(td)["next"]
        return td


    while not td["done"].all():
        td = make_step(td)

    makespan = get_makespan(td)
    return makespan


def get_makespan(td):
    """
    length = len(td["finish_times"][0]) - 1
    makespan = td["finish_times"][0][length]
    return makespan
    """
    return td["time"]




class EmbeddingDataset(Dataset):
    def __init__(self, op_emb, ma_emb, td, env):
        # store channel sizes from the initial embeddings
        self.td = td
        self.env = env

        self.C_op = op_emb.size(1)
        self.C_ma = ma_emb.size(1)

        # store original heights (before padding) assuming width unchanged
        self.H = op_emb.size(2)

        self.op_emb = self.reshape_data(op_emb)
        self.ma_emb = self.reshape_data(ma_emb)


    def reshape_data(self, x_emb):
        x_emb = x_emb.unsqueeze(1)
        x = x_emb.detach().clone()
        x = x.squeeze(0)
        return x


    def decomposition_instance(self, sample_data):
        """
        data_instance : Tensor of shape [B, 1, C_op + C_ma, H] or [B, C_op + C_ma, H] (depending on whether
        the middle dimension 1 is kept).
        This returns:
          op_rec : shape [B, C_op, H]
          ma_rec : shape [B, C_ma, H]
        """
        # If there is a singleton dimension at dim=1 (the “1”), remove it.
        if sample_data.dim() == 4 and sample_data.size(1) == 1:
            # from [B,1,channels,H] → [B,channels,H]
            sample_data = sample_data.squeeze(1)

        # split channels
        op_rec, ma_rec = torch.split(sample_data, [self.C_op, self.C_ma], dim=1)

        # Crop spatial dimension if needed (in your case height =16 for both so likely no crop)
        op_rec = op_rec[:, :, :self.H]
        ma_rec = ma_rec[:, :, :self.H]

        return op_rec, ma_rec



    def __len__(self):
        return self.op_emb.size(0)


    def __getitem__(self, idx):
        op = self.op_emb[idx]
        ma = self.ma_emb[idx]
        
        # Pad the smaller embedding to match the larger one
        if op.shape[2] < ma.shape[2]:
            padding = (0, 0, 0, ma.shape[2] - op.shape[2])  # Pad height
            op = torch.nn.functional.pad(op, padding)
        elif ma.shape[2] < op.shape[2]:
            padding = (0, 0, 0, op.shape[2] - ma.shape[2])  # Pad height
            ma = torch.nn.functional.pad(ma, padding)
        
        # Concatenate along the channel dimension
        combined = torch.cat([op, ma], dim=1)
        
        return combined


def min_dist_between_points(coords):
    # compute all pairwise distances (excluding self‑distances)
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    # set the diagonal to a large number so we ignore zero distances to self
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min()
    return min_dist




def map_index_to_action(r_i, op_i, generator_params):
    job_size = generator_params["num_jobs"]
    ma_size = generator_params["num_machines"]

    return (op_i + 1) + (r_i // job_size) * ma_size



# HERSAN
def apply_fcfs(env, td, generator_params):
    


    actions = []
    ma_range = generator_params["num_machines"]

    # inneholder en liste for hver maskin, en liste inneholder sekvensen for hver op assigned til maskinen, i format [x,x]
    assignments = []
    while not td['done'].all():
        min_proc_time = 100
        action_i = 0

        rma_i = 0
        opma_i = 0
        
        ready_indexes = [i for i, val in enumerate(td['is_ready'][0]) if val]
        for r_i in ready_indexes:
            for op_i in range(ma_range):
                v = td['proc_times'][0][op_i][r_i]
                is_not_busy = td['busy_until'][0][op_i] <= td['time'][0]
                if v < min_proc_time and is_not_busy:
                    min_proc_time = v
                    action_i = map_index_to_action(r_i, op_i, generator_params)
                    # adding action taken, to assignment
                    rma_i = r_i # operasjon
                    opma_i = op_i # maskin
        

        assignments.append((opma_i, rma_i))


        actions.append(action_i)
        td['action'] = torch.tensor([action_i])
        td = env.step(td)['next']
    # print("actions during FCFS")
    # print(actions)

    return td, assignments



def make_data_input(td):
    mask = td["ma_assignment"]
    values = td["start_times"]

    arr = values.cpu().numpy().flatten()

    # compute the scaled ranks
    v_min, v_max = arr.min(), arr.max()
    ranks = 0.1 + (arr - v_min) / (v_max - v_min) * (0.99 - 0.1)

    # --- Step 2: build a mapped matrix by broadcasting ranks onto mask
    # We want to replace columns in mask where mask == 1 with the corresponding rank
    mask_np = mask.cpu().numpy()

    # Create an output array initially zeros (or some default)
    out = np.zeros_like(mask_np)

    # For each column index j, where mask == 1, set out[:, :, j] = ranks[j]
    for j, r in enumerate(ranks):
        out[:, :, j][mask_np[:, :, j] == 1] = r

    out_tensor = torch.tensor(out)
    return out_tensor




def make_instance_json(env, generator_params, filename: str, sch_rule):
    td = None
    td_old = None
    while True:
        try:
            # Initialiserer enviroment
            td = env.reset(batch_size=[1])

            # Lagrer enviroment til fil
            simple_dict = tensordict_to_dict(td)
            with open(filename, "w") as f:
                json.dump(simple_dict, f, indent=4)

            # Lager optimal solution
            td_old = td.copy()
            td = apply_fcfs(env, td, generator_params)
            input_data = make_data_input(td) # mix of ma assignments and starttimes, to get the assignments, and the order of assignment
            

            simple_dict = tensordict_to_dict(td)

            data = None
            with open(filename, "r", encoding="utf‑8") as f:
                data = json.load(f)
            data["ma_assignment"] = simple_dict["ma_assignment"]
            data["input_data"] = input_data.tolist()
            
            # Lagrer enviroment med optimal solution til fil
            with open(filename, "w", encoding="utf‑8") as f:
                json.dump(data, f, indent=4)


            break
        except ValueError as err:
            continue
    # print(td["time"])
    return td, td_old



def make_assignment(td, instance_type):
    if instance_type == "422":
        u = torch.tensor(td["ma_assignment"])

        # Remove the batch dimension (size 1)
        adj_small = u.squeeze(0)  # now shape is (4, 4)
        num_machines, num_ops = adj_small.shape  # should be 4, 4
        N = num_machines * num_ops  # 16
        # Create the big adjacency matrix
        adj_expanded = torch.zeros((N, N))

        for i in range(num_machines):
            for j in range(num_ops):
                for k in range(num_ops):
                    idx1 = i * num_ops + j
                    idx2 = i * num_ops + k
                    adj_expanded[idx1, idx2] = adj_small[i, k]
                    # or adj_small[i, j] depending on how you interpret adjacency

        return adj_expanded


def give_proc_times(td, instance_type):
    if instance_type == "422":
        x = torch.tensor(td['proc_times'], dtype=torch.float32)

        v = x.flatten()

        # min & max bounds
        min_val = 5.0
        max_val = 20.0
        v_norm = (v - min_val) / (max_val - min_val)
        # optionally: clip to [0,1] if values might go outside bounds
        # v_norm = torch.clamp(v_norm, min=0.0, max=1.0)

        res_mat = torch.diag(v_norm)
        return res_mat
    if instance_type == "444":
        x = td["proc_times"]
        x = torch.tensor(x, dtype=torch.float32)

        # normelize between for values between 5 and 20, smallest value being 0.1, and largest 0.99
        # in_min, in_max = 5.0, 20.0
        # out_min, out_max = 0.1, 0.99
        # x = out_min + (x - in_min) * (out_max - out_min) / (in_max - in_min)

        mat = x.squeeze(0)
        n_rows, n_cols = mat.squeeze(0).shape
        to_add = 16 - n_rows
        pad_block = torch.full((to_add, n_cols), -1.0, dtype=mat.dtype)
        return torch.cat([mat, pad_block], dim=0)



def arrangement_valid(x):
    mat = x.squeeze(0)  # shape (4,4)

    # Sum down columns: this gives a 1‑D tensor of length 4
    col_sums = mat.sum(dim=0)

    # Check if each column sum == 1
    check = (col_sums == 1).all().item()
    return check






def arrange_actions_valid(actions, td, env):
    while True:
        td_try = td.copy()
        try:
            for action in actions:
                td_try['action'] = torch.tensor([action])
                td_try = env.step(td_try)['next']
            if not arrangement_valid(td_try["ma_assignment"]):
                raise ValueError("")
            break
        except:
            random.shuffle(actions)
            continue

    return actions




# oversetter preprossesert rep til en felles rep som scheduler bruker
def make_action_list_from_assignments(assignments, instance_kind: str, td, env):
    if instance_kind == "new_small":
        if (assignments.size(0) > 4):
            x_blocks = assignments.view(4, 4, -1)  # shape (4 blocks, 4 rows per block, 16 columns)
            x = x_blocks[:, 0, :]  # shape (4, 16)
        else:
            x = assignments

                # Suppose we assume 4 ops, each op has 4 features/columns:
        x = x.view(x.shape[0], 4, 4)    # shape = (4 rows, 4 ops, 4 features per op)
        combined = x.max(dim=0)[0]    # or sum(dim=0), depending on how you want to combine
        x = combined.unsqueeze(0)   # shape (1,4,4)
        
        

        actions = []
        x = x.squeeze(0)  # shape (4,4)
        n_rows, n_ops = x.shape
        ops_per_group = 2
        for col in range(n_ops):
            for row in range(n_rows):
                if x[row, col] == 1:
                    group = col // ops_per_group
                    base = group * 4
                    action = base + (row + 1)
                    actions.append(action)

        # new_order = [0, 2, 1, 3]
        # actions = [actions[i] for i in new_order]
        arrange_actions_valid(actions, td, env)
        return actions

    if instance_kind == "444":
        pass


# print("start")



def schedule_actions(actions, env, td, path_to_instance, filename):
    
    # reading instance file
    data = None
    # with open(path_to_instance, "r", encoding="utf‑8") as f:
    #     data = json.load(f)
    # td = TensorDict.from_dict(data)
    # td["ma_assignment"] = torch.zeros_like( td["ma_assignment"] )
    # td["ma_assignment"] = 0
    for action in actions:
        td['action'] = torch.tensor([action])
        td = env.step(td)['next']
        env.render(td, 0)
        # Display updated plot
        # display(plt.gcf())
        plt.savefig(filename, dpi=150, bbox_inches='tight')


def make_schedule_from_assignment_444_fasan(problem_path: str, model_path: str):
    # lager eksempel instase som er 4ma2j2op, med target løsning fra fcfs
    logging.info("fasan")
    ma = 4
    ops_per_job = 4
    jobs = 4
    min_proc = 5
    max_proc = 20
    name = 'instances/test_444/test_4ma_4op_4j'
    filetype = 'json'
    sch_rule = apply_fcfs

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

    equaled = 0
    elapsed_sum = 0
    num = 1
    for i in range(num):
        logging.info(i)


        env = FJSPEnv(
            generator_params=generator_params,
            _torchrl_mode=True,
            stepwise_reward=True
        )


        td, td_old = make_instance_json(env, generator_params, name + f"_{i}" + "." + filetype, sch_rule)
        # gjør om til samme måte som modellen for 4ma2j2op ser den
        proc_times_model = give_proc_times(td_old, "444")
        adj_model = td["ma_assignment"]


        # lager metode som kan oversette adj matrise modell lager til liste med aksjoner
        # TODO kommenter ut
        # actions = make_action_list_from_assignments(adj_model, "444", td_old, env)
        # schedule_actions(actions, env, td_old, name + "." + filetype)


        # gi testinstanse med adj og proc til modellen
        # få det genererte outputtet
        model_path = "models/new_4x4_diffusion_0.0001.pth"
        n_samples = 1
        x, elapsed = inference.apply_inference_444(proc_times_model, model_path, n_samples, dim_x=16, dim_y=16)
        # print("x gotten")
        x = x.squeeze()

        x4 = x[:4, :]
        max_indices = x4.argmax(dim=0)   # shape [16]
        result = torch.zeros_like(x4)
        result[max_indices, torch.arange(x4.size(1))] = 1

        # print(x.shape)
        # print("x")
        # print(x)
        # print("proc")
        # print(td["proc_times"])
        # print("result")
        # print(result)
        # print("adj")
        print(adj_model)



        """
        # 1. Group rows into 4 blocks of 4 and sum them  → (4, 16)
        x_grouped = x.view(4, 4, 16).sum(dim=1)
        flat = x_grouped.flatten()
        top4 = torch.topk(flat, k=4)
        binary = torch.zeros_like(flat)
        binary[top4.indices] = 1
        binary = binary.view_as(x_grouped)


        adj_model = adj_model.view(4, 4, adj_model.size(1))  # here x.size(1)=16 columns
        adj_model, _ = adj_model.max(dim=1)
        
        binary = binary.to("cuda")
        adj_model = adj_model.to("cuda")
            

        if torch.equal(binary, adj_model):
            equaled += 1
        elapsed_sum += elapsed

    print(equaled / num)
    print(elapsed_sum / num)
    """
    logging.info("end")


def map_assignments_to_adj_mat(assignments, td, generator_params):
    ma_n = generator_params["num_machines"]
    op_n = generator_params["max_ops_per_job"] * generator_params["num_jobs"]
    sum_n = ma_n + op_n
    
    adj = torch.zeros(sum_n, sum_n)
    
    last_op_for_ma = [-1 for _ in range(ma_n)]
    for assignment in assignments:
        ma, op = assignment
        if last_op_for_ma[ma] == -1:
            adj[ma, ma_n + op] = 1
        else:
            adj[last_op_for_ma[ma] + ma_n, ma_n + op] = 1
        last_op_for_ma[ma] = op
    return adj



class AdjTransform:
    def __call__(self, x, x2):
        # Normalize x
        real_x_mask = x != 1.0
        x_min = x[real_x_mask].min()
        x_max = x[real_x_mask].max()
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        x = torch.where(real_x_mask, x_norm, x)  # keep padding as 1

        # Normalize x2
        real_x2_mask = x2 != 1.0
        x2_min = x2[real_x2_mask].min()
        x2_max = x2[real_x2_mask].max()
        x2_norm = (x2 - x2_min) / (x2_max - x2_min + 1e-8)
        x2 = torch.where(real_x2_mask, x2_norm, x2)  # keep padding as 1

        return x, x2



class AdjDataset(Dataset):
    def __init__(self, 
            root_dir: str, 
            instance_size: Tuple[int, int, int], 
            pad_size: Tuple[int, int],
            test=False
            ):

        self.root_dir = root_dir
        self.pad_size = pad_size
        self.pad_value = 1
        ma, j, op = instance_size
        size_str = f"{ma}ma_{j}j_{op}op"
        if test:
            size_str = f"test_{ma}ma_{j}j_{op}op"

        transform = AdjTransform() 

        self.file_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith('.json') and size_str in fname
        ]
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # read one json file
        path = self.file_paths[idx]
        with open(path, 'r') as f:
            data = json.load(f)
        # get the part you need
        input_data = data['input']
        proc_times = data["proc_times"]

        # Convert input_data to a tensor (or whatever your model expects)
        x = torch.tensor(input_data, dtype=torch.float)
        x2 = torch.tensor(proc_times, dtype=torch.float).squeeze()
        x_unpad_shape = x.shape
        x2_unpad_shape = x2.shape

        # normelize
        x_min = x.min()
        x_max = x.max()
        x = (x - x_min) / (x_max - x_min)

        # Min-max normalization for x2
        x2_min = x2.min()
        x2_max = x2.max()
        x2 = (x2 - x2_min) / (x2_max - x2_min)
        

        # padding
        xs = [x, x2]
        for i in range( len(xs) ):
            target_h, target_w = self.pad_size
            h, w = xs[i].shape
            pad_h = target_h - h
            pad_w = target_w - w
            xs[i] = F.pad(xs[i], (0, pad_w, 0, pad_h), mode='constant', value=self.pad_value)
        x, x2 = xs
        # if self.transform:
        #     x, x2 = self.transform(x, x2)
        return x, x2, x_unpad_shape, x2_unpad_shape

"""
print("check dataset value")
dataset = AdjDataset('instances/adj', (4,4,4), (20, 20))
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

x, x2 = dataset[10]
print(x)
print(x2)
"""





def write_to_json(td, td_assignment, input_data, filename):
    td_dict = tensordict_to_dict(td)
    td_a_dict = tensordict_to_dict(td_assignment)

    td_dict["ma_assignment"] = td_a_dict["ma_assignment"]
    td_dict["input"] = input_data.tolist()

    with open(filename, "w") as f:
        json.dump(td_dict, f, indent=4)

    return filename







# HERSAN
def make_dataset_adjacency():

    ma = 4
    ops_per_job = 4
    jobs = 4
    min_proc = 5
    max_proc = 20
    sch_rule = apply_fcfs

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

    n = 10
    # instances/adj/{ma}ma_{jobs}j_{ops_per_job}op_{i}.json
    for i in range(n):
        filename = f"test_{ma}ma_{jobs}j_{ops_per_job}op_{i}.json"

        env = FJSPEnv(
            generator_params=generator_params,
            _torchrl_mode=True,
            stepwise_reward=True
        )

        td = env.reset(batch_size=[1])
        td_scheduled, assignments = sch_rule(env, td.copy(), generator_params)
        adj = map_assignments_to_adj_mat(assignments, td_scheduled, generator_params)

        write_to_json(td, td_scheduled, adj, filename)
        logging.info(f"made file {filename}")
        


def make_schedule_from_assignment(problem_path: str, model_path: str):
    # lager eksempel instase som er 4ma2j2op, med target løsning fra fcfs
    logging.info("i funka")

    ma = 4
    ops_per_job = 2
    jobs = 2
    min_proc = 5
    max_proc = 20
    name = 'instances/test_444/test_4ma_2op_2j'
    filetype = 'json'
    sch_rule = apply_fcfs

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

    env = FJSPEnv(
        generator_params=generator_params,
        _torchrl_mode=True,
        stepwise_reward=True
    )


    equaled = 0
    elapsed_sum = 0
    num = 1
    for i in range(num):


        env = FJSPEnv(
            generator_params=generator_params,
            _torchrl_mode=True,
            stepwise_reward=True
        )


        td, td_old = make_instance_json(env, generator_params, name + f"_{i}" + "." + filetype, sch_rule)
        # gjør om til samme måte som modellen for 4ma2j2op ser den
        proc_times_model = give_proc_times(td_old, "422")
        adj_model = make_assignment(td, "422")

        # lager metode som kan oversette adj matrise modell lager til liste med aksjoner
        actions = make_action_list_from_assignments(adj_model, "new_small", td_old, env)
        # .... dette skal brukes senere, men nå må jeg lage så løst instanse skeduleres
        schedule_actions(actions, env, td_old, name + "." + filetype, "fcfs.png")


        # gi testinstanse med adj og proc til modellen
        # få det genererte outputtet
        model_path = "models/new_4x2_diffusion_mnist_1e-05.pth"
        # model_path = "models/new_4x4_diffusion_0.0001.pth"
        n_samples = 1
        
        logging.info("starting inference")
        x, elapsed = inference.apply_inference_4x2(proc_times_model, model_path, n_samples, dim_x=16, dim_y=16)
        # print("x gotten")
        x = x.squeeze()
        # print(x.shape)

        # print(x)

        # 1. Group rows into 4 blocks of 4 and sum them  → (4, 16)
        x_grouped = x.view(4, 4, 16).sum(dim=1)
        flat = x_grouped.flatten()
        top4 = torch.topk(flat, k=4)
        binary = torch.zeros_like(flat)
        binary[top4.indices] = 1
        binary = binary.view_as(x_grouped)


        adj_model = adj_model.view(4, 4, adj_model.size(1))  # here x.size(1)=16 columns
        adj_model, _ = adj_model.max(dim=1)
        
        binary = binary.to("cuda")
        adj_model = adj_model.to("cuda")
            
        actions = make_action_list_from_assignments(binary, "new_small", td_old, env)
        # .... dette skal brukes senere, men nå må jeg lage så løst instanse skeduleres
        schedule_actions(actions, env, td_old, name + "." + filetype, "model_sch.png")
        

        if torch.equal(binary, adj_model):
            equaled += 1
        elapsed_sum += elapsed

    logging.info(equaled / num)
    logging.info(elapsed_sum / num)
    
    logging.info("end")












"""
print("start")

ma = 4
ops_per_job = 4
jobs = 4
min_proc = 5
max_proc = 20

generator_params = {
"num_jobs": jobs,  # the total number of jobs
"num_machines": ma,  # the total number of machines that can process operations
"min_ops_per_job": ops_per_job,  # minimum number of operatios per job
"max_ops_per_job": ops_per_job,  # maximum number of operations per job
"min_processing_time": min_proc,  # the minimum time required for a machine to process an operation
"max_processing_time": max_proc,  # the maximum time required for a machine to process an operation
"min_eligible_ma_per_op": ma,  # the minimum number of machines capable to process an operation
"max_eligible_ma_per_op": ma,  # the maximum number of machines capable to process an operation
}




env = FJSPEnv(
    generator_params=generator_params, 
    _torchrl_mode=True, 
    stepwise_reward=True
)
batch_size = 1
td = env.reset(batch_size=[batch_size])

td = apply_fcfs_free_size(env, td, generator_params)

print(td["ma_assignment"])
"""




def indexes():
    ma = 4
    ops_per_job = 4
    jobs = 4
    min_proc = 5
    max_proc = 20

    generator_params = {
    "num_jobs": jobs,  # the total number of jobs
    "num_machines": ma,  # the total number of machines that can process operations
    "min_ops_per_job": ops_per_job,  # minimum number of operatios per job
    "max_ops_per_job": ops_per_job,  # maximum number of operations per job
    "min_processing_time": min_proc,  # the minimum time required for a machine to process an operation
    "max_processing_time": max_proc,  # the maximum time required for a machine to process an operation
    "min_eligible_ma_per_op": ma,  # the minimum number of machines capable to process an operation
    "max_eligible_ma_per_op": ma,  # the maximum number of machines capable to process an operation
    }

    env = FJSPEnv(generator_params=generator_params)
    td = env.reset(batch_size=[1])

    decoder = L2DDecoder(env_name=env.name, embed_dim=32)
    encoder = HetGNNEncoder(embed_dim=32, num_layers=2)
    (ma_emb, op_emb), init = encoder(td)



    def make_step(td):
        logits, mask = decoder(td, (ma_emb, op_emb), num_starts=0)
        action = logits.masked_fill(~mask, -torch.inf).argmax(1)
        td["action"] = action
        td = env.step(td)["next"]

        return td




    env.render(td, 0)
    # Update plot within a for loop
    while not td["done"].all():
        # Clear the previous output for the next iteration
        clear_output(wait=True)

        td = make_step(td)
        env.render(td, 0)
        # Display updated plot
        display(plt.gcf())
        plt.show()

        # Pause for a moment to see the changes
        time.sleep(.4)






def encode_to_image(generator_params, env, td, filename, threshold_point_closeness, sch_rule, image_size):
    w = td['proc_times'][0]           # shape (num_machines, n_points)
    jobs = generator_params['num_jobs']
    num_machines = generator_params['num_machines']
    n_points = w.shape[1]
    
    max_w = 20.0

    desired_px = image_size
    dpi = 100  # choose a dpi
    fig_size_in = desired_px / dpi  # 32/100 = 0.32 inches

    fig = plt.figure(figsize=(fig_size_in, fig_size_in), dpi=dpi)

    # corner positions for machines
    corners = np.array([[0, 0],
                        [image_size-1, 0],
                        [0, image_size-1],
                        [image_size-1, image_size-1]])

    coords = []
    for i in range(n_points):
        norm = w[:, i] / max_w
        scaled = norm ** 0.5
        inv = 1.0 - scaled
        inv = np.clip(inv, 0.0, None)
        weighted_pos = (inv[:, None] * corners).sum(axis=0)
        total = inv.sum()
        pos = weighted_pos / total if total > 0 else corners.mean(axis=0)
        pos += np.random.uniform(-0.3, 0.3, size=2)
        coords.append(pos)
    coords = np.array(coords)

    # distansen burde vere mer en 0.3
    closeness = min_dist_between_points(coords)
    if closeness < threshold_point_closeness:
        message = f'Points are too close.. : {str(closeness)}'
        # print(message)
        raise ValueError(message)

    # Setup graph for edges
    G = nx.DiGraph()
    # Add machine nodes
    for m in range(num_machines):
        G.add_node(f"M{m}", pos=tuple(corners[m]), type='machine')
    # Add operation nodes
    for op in range(n_points):
        G.add_node(f"O{op}", pos=tuple(coords[op]), type='operation')

    # TODO her er sch rule
    # td = sch_rule(env, td, 2)
    relations = td['ma_assignment'][0]  # assume shape (num_machines, n_points)

    # Add edges: machine → first op; then chain ops for each machine
    edge_colors_per_machine = ['red', 'blue', 'green', 'purple']  # one colour per machine

    # Instead of drawing all edges at once, build edgelists per machine
    edgelists = []
    edge_colors = []
    for m in range(num_machines):
        ops_for_machine = [op for op in range(n_points) if relations[m, op] == 1]
        if not ops_for_machine:
            continue
        sorted_ops = sorted(ops_for_machine)
        # First edge: machine → first op
        edgelists.append((f"M{m}", f"O{sorted_ops[0]}"))
        edge_colors.append(edge_colors_per_machine[m])
        # Then chain op edges
        for i in range(len(sorted_ops)-1):
            edgelists.append((f"O{sorted_ops[i]}", f"O{sorted_ops[i+1]}"))
            edge_colors.append(edge_colors_per_machine[m])

    # Plot
    # Scatter: machines
    plt.scatter(corners[:,0], corners[:,1],
                s=100, color='red', marker='X', label='Machines')

    markers = ['o', 's', '^', 'D']
    colors  = ['red', 'blue', 'green', 'purple']

    jobs = generator_params['num_jobs']
    n_ops_per_job = generator_params['max_ops_per_job']

    # Ensure we have enough markers for each job
    assert jobs <= len(markers), "Not enough marker types for jobs"
    # Ensure we have enough colors for each operation
    assert n_ops_per_job <= len(colors), "Not enough colors for operations"

    for group_idx in range(jobs):
        marker = markers[group_idx]
        start = group_idx * n_ops_per_job
        end   = start + n_ops_per_job
        for op_idx, idx in enumerate(range(start, end)):
            x, y = coords[idx]
            color = colors[op_idx]
            plt.scatter(x, y,
                        s=10,
                        marker=marker,
                        color=color,
                        alpha=0.8)


    # Draw edges using NetworkX
    pos_dict = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_edges(
        G,
        pos_dict,
        edgelist=edgelists,
        edge_color=edge_colors,
        arrows=False,
        width=1
    )

    plt.xlim(-1, image_size+1)
    plt.ylim(-1, image_size+1)
    plt.gca().set_aspect('equal', 'box')
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    fig.savefig(filename, dpi=dpi, bbox_inches=None, pad_inches=0, facecolor='white')
    plt.close(fig)
    img = Image.open(filename).convert('L')  # luminance mode
    img.save(filename)
    plt.close('all')
    # plt.show()





# td kan være dictionary, eller dict hentet fra instanse
def encode_instance_to_image(n_ops_per_job, num_machines, num_jobs, td, filename, image_size, draw_edges, draw_all_edges):
    if draw_edges and draw_all_edges:
        raise ValueError("Both draw_edges and draw_all_edges cannot both be True")

    td = TensorDict(td, batch_size=[1])

    w = td['proc_times'][0]           # shape (num_machines, n_points)
    jobs = num_jobs
    n_points = w.shape[1]
    
    max_w = 20.0

    desired_px = image_size
    dpi = 100  # choose a dpi
    fig_size_in = desired_px / dpi  # 32/100 = 0.32 inches

    fig = plt.figure(figsize=(fig_size_in, fig_size_in), dpi=dpi)

    # corner positions for machines
    corners = np.array([[0, 0],
                        [image_size-1, 0],
                        [0, image_size-1],
                        [image_size-1, image_size-1]])

    coords = []
    for i in range(n_points):
        norm = w[:, i] / max_w
        scaled = norm ** 0.5
        inv = 1.0 - scaled
        inv = np.clip(inv, 0.0, None)
        weighted_pos = (inv[:, None] * corners).sum(axis=0)
        total = inv.sum()
        pos = weighted_pos / total if total > 0 else corners.mean(axis=0)
        pos += np.random.uniform(-0.3, 0.3, size=2)
        coords.append(pos)
    coords = np.array(coords)

    # distansen burde vere mer en 0.3
    # closeness = min_dist_between_points(coords)
    # if closeness < threshold_point_closeness:
    #    message = f'Points are too close.. : {str(closeness)}'
        # print(message)
    #    raise ValueError(message)

    # Setup graph for edges
    if draw_edges or draw_all_edges:
        G = nx.DiGraph()
        # Add machine nodes
        for m in range(num_machines):
            G.add_node(f"M{m}", pos=tuple(corners[m]), type='machine')
        # Add operation nodes
        for op in range(n_points):
            G.add_node(f"O{op}", pos=tuple(coords[op]), type='operation')

    # TODO her er sch rule
    # td = sch_rule(env, td, 2)
    if draw_edges:
        relations = td['ma_assignment'][0]  # assume shape (num_machines, n_points)

    # Add edges: machine → first op; then chain ops for each machine
    edge_colors_per_machine = ['red', 'red', 'red', 'red']  # one colour per machine

    # Instead of drawing all edges at once, build edgelists per machine
    if draw_edges:
        edgelists = []
        edge_colors = []
        for m in range(num_machines):
            ops_for_machine = [op for op in range(n_points) if relations[m, op] == 1]
            if not ops_for_machine:
                continue
            sorted_ops = sorted(ops_for_machine)
            # First edge: machine → first op
            edgelists.append((f"M{m}", f"O{sorted_ops[0]}"))
            edge_colors.append(edge_colors_per_machine[m])
            # Then chain op edges
            for i in range(len(sorted_ops)-1):
                edgelists.append((f"O{sorted_ops[i]}", f"O{sorted_ops[i+1]}"))
                edge_colors.append(edge_colors_per_machine[m])
    
    if draw_all_edges:
        edgelists = []
        edge_colors = []
        # Create a full connection among all nodes (machines + operations)
        all_nodes = [f"M{m}" for m in range(num_machines)] + [f"O{op}" for op in range(n_points)]
        # You could choose directed edges in one direction; here assume machine → operation and operation → operation
        # For example, connect every machine to every operation:
        for m in range(num_machines):
            for op in range(n_points):
                edgelists.append((f"M{m}", f"O{op}"))
                edge_colors.append(edge_colors_per_machine[m % len(edge_colors_per_machine)])
        # And maybe also connect every operation to every other operation (or just chain them all)
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    edgelists.append((f"O{i}", f"O{j}"))
                    # choose a color, e.g. use one for operations
                    edge_colors.append('red')


    # Plot
    # Scatter: machines
    plt.scatter(corners[:,0], corners[:,1],
                s=2, color='red', marker='X', label='Machines')

    markers = ['o', 's', '^', 'D']
    colors  = ['red', 'blue', 'green', 'purple']
    colors  = ['grey', 'grey', 'grey', 'grey']

    # Ensure we have enough markers for each job
    assert jobs <= len(markers), "Not enough marker types for jobs"
    # Ensure we have enough colors for each operation
    assert n_ops_per_job <= len(colors), "Not enough colors for operations"





    # Draw edges using NetworkX
    if draw_edges or draw_all_edges:
        pos_dict = nx.get_node_attributes(G, 'pos')
        nx.draw_networkx_edges(
            G,
            pos_dict,
            edgelist=edgelists,
            edge_color=edge_colors,
            arrows=False,
            width=1
        )


    for group_idx in range(jobs):
        marker = markers[group_idx]
        start = group_idx * n_ops_per_job
        end   = start + n_ops_per_job
        for op_idx, idx in enumerate(range(start, end)):
            x, y = coords[idx]
            color = colors[op_idx]
            plt.scatter(x, y,
                        s=2,
                        marker=marker,
                        color=color,
                        alpha=0.8)




    plt.xlim(-1, image_size+1)
    plt.ylim(-1, image_size+1)
    plt.gca().set_aspect('equal', 'box')
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    fig.savefig(filename, dpi=dpi, bbox_inches=None, pad_inches=0, facecolor='white')
    plt.close(fig)
    if draw_all_edges == False:
        img = Image.open(filename).convert('L')  # luminance mode
        img.save(filename)
    plt.close('all')
    # plt.show()





def make_images():
    num_machines = 4
    num_jobs = 2
    n_ops_per_job = 2
    image_size = 64
    td = None

    for i in range(100000):
        instance_file = f"instances/4ma_2op_2j_{i}"

        logging.info(f"making image for instance {i}")

        with open(instance_file + ".json", "r", encoding="utf‑8") as f:
            td = json.load(f)
        
        draw_all_edges = False
        filename = instance_file + "_nodes.png"
        draw_edges = False
        encode_instance_to_image(n_ops_per_job, num_machines, num_jobs, td, filename, image_size, draw_edges, draw_all_edges)
        
        filename = instance_file + "_edges.png"
        draw_edges = True
        encode_instance_to_image(n_ops_per_job, num_machines, num_jobs, td, filename, image_size, draw_edges, draw_all_edges)

        filename = instance_file + "_all_edges.png"
        draw_edges = False
        draw_all_edges = True
        encode_instance_to_image(n_ops_per_job, num_machines, num_jobs, td, filename, image_size, draw_edges, draw_all_edges)




# ex of instance: 4ma_2op_2j_3
# output: Tensor(1, 64, 64) .. (channels, height, width)
def make_mask(instance: str, img_height: int, img_width: int, device: str) \
        -> TensorType["batch_size, output channels, height, width"]:

    img = Image.open(instance).convert("RGB") 
    img_tensor = transforms.ToTensor()(img) # converts to [C, H, W], floats in [0,1]

    # Detect red pixels: you might choose a threshold
    # For example, red channel high, green & blue low:
    red = img_tensor[0, :, :]   # red channel
    green = img_tensor[1, :, :]
    blue = img_tensor[2, :, :]
    red_thresh = 0.9
    non_red_thresh = 0.9

    # is_red is a boolean tensor of shape [H, W]
    is_red = (red > red_thresh) & (green < non_red_thresh) & (blue < non_red_thresh)
    # Now mask = 1 where NOT red (i.e., fixed region), mask = 0 where red (i.e., free region)
    mask_tensor = (~is_red).to(torch.float32)  # float mask: 1.0 = fixed, 0.0 = free
    mask_tensor = mask_tensor.unsqueeze(0)

    # 3) Expand/unsqueeze to batch of n_samples
    # mask_batch = mask_tensor.unsqueeze(0).repeat(1, 1, 1) 
    # mask_batch = mask_batch.to(device)
    return mask_tensor



# input: Tensor(1, 64, 64) .. (channels, height, width)
def visulize_mask(mask: TensorType["channels, height, width"], save_path: str) \
        -> TensorType["channels, height, width"]:

    mask_np = mask.cpu().numpy()              # [batch_size, 1, H, W]
    mask_np = mask_np[0, 0, :, :]                    # pick first batch, first channel → [H, W]
    vis_mask = (1.0 - mask_np) * 255
    vis_mask = vis_mask.astype(np.uint8)

    vis_im = Image.fromarray(vis_mask, mode='L')

    if save_path:
        vis_im.save(save_path)

    # print("Mask shape:", mask.shape)



# mask = make_mask("4ma_2op_2j_0", 1, 64, 64, "cuda")
# visulize_mask(mask, "mask_0.png")
# print("done running")





def make_dataset_json():

    ma = 4
    ops_per_job = 4
    jobs = 4
    min_proc = 5
    max_proc = 20

    folder = 'instances'
    name = '4ma_4op_4j'
    filetype = 'json'

    samples_to_make = 50000

    sch_rule = apply_fcfs


    generator_params = {
    "num_jobs": jobs,  # the total number of jobs
    "num_machines": ma,  # the total number of machines that can process operations
    "min_ops_per_job": ops_per_job,  # minimum number of operatios per job
    "max_ops_per_job": ops_per_job,  # maximum number of operations per job
    "min_processing_time": min_proc,  # the minimum time required for a machine to process an operation
    "max_processing_time": max_proc,  # the maximum time required for a machine to process an operation
    "min_eligible_ma_per_op": ma,  # the minimum number of machines capable to process an operation
    "max_eligible_ma_per_op": ma,  # the maximum number of machines capable to process an operation
    }


    env = FJSPEnv(
        generator_params=generator_params,
        _torchrl_mode=True,
        stepwise_reward=True
    )
    
    for n in range(samples_to_make):
        filename = folder + '/' + name + '_' + str(n) + '.' + filetype
        make_instance_json(env, generator_params, filename, sch_rule)




# print("making dataset")
# make_dataset_json()
# print("done making dataset")



# optimizer function must take the following parameters: env, td, 2 or 4
def make_instance_and_solution(threshold, optimizer_function, samples_to_make):
    generator_params = {
        "num_jobs": 2,
        "num_machines": 4,
        "min_ops_per_job": 2,
        "max_ops_per_job": 2,
        "min_processing_time": 5,
        "max_processing_time": 20,
        "min_eligible_ma_per_op": 4,
        "max_eligible_ma_per_op": 4,
    }

    env = FJSPEnv(
        generator_params=generator_params,
        _torchrl_mode=True,
        stepwise_reward=True
    )

    threshold_point_closeness = threshold


    for n in range(samples_to_make):
        folder = 'instances'
        imagename = '4ma_2op_2j'
        filetype = 'json'
        filename = folder + '/' + imagename + '_' + str(n) + '.' + filetype

        sch_rule = apply_fcfs
        while True:
            try:
                # Initialiserer enviroment
                td = env.reset(batch_size=[1])
 
                image_size = 64
                max_w = 20.0
                w = td['proc_times'][0]  
                n_points = w.shape[1]

               
                # Finner distanse mellom noder
                corners = np.array([[0, 0],
                                    [image_size-1, 0],
                                    [0, image_size-1],
                                    [image_size-1, image_size-1]])

                coords = []
                for i in range(n_points):
                    norm = w[:, i] / max_w
                    scaled = norm ** 0.5
                    inv = 1.0 - scaled
                    inv = np.clip(inv, 0.0, None)
                    weighted_pos = (inv[:, None] * corners).sum(axis=0)
                    total = inv.sum()
                    pos = weighted_pos / total if total > 0 else corners.mean(axis=0)
                    pos += np.random.uniform(-0.3, 0.3, size=2)
                    coords.append(pos)
                coords = np.array(coords)

                # distansen burde vere mer en 0.3
                # Lagrer ikke instansen hvis ikke nok distanse mellom noder
                closeness = min_dist_between_points(coords)
                if closeness < threshold_point_closeness:
                    message = f'Points are too close.. : {str(closeness)}'
                    raise ValueError(message)

                # Lagrer enviroment til fil

                simple_dict = tensordict_to_dict(td)
                with open(filename, "w") as f:
                    json.dump(simple_dict, f, indent=4)

                # Lager optimal solution
                td = sch_rule(env, td, 2)
                simple_dict = tensordict_to_dict(td)

                data = None
                with open(filename, "r", encoding="utf‑8") as f:
                    data = json.load(f)
                data["ma_assignment"] = simple_dict["ma_assignment"]
                
                # Lagrer enviroment med optimal solution til fil
                with open(filename, "w", encoding="utf‑8") as f:
                    json.dump(data, f, indent=4)


                logging.info('Made instance ' + str(n))
                break
            except ValueError as err:
                continue

"""
print("running")

sch_rule = apply_fcfs
image_size = 64
threshold = 5.1
n_samples = 100000
optimizer_function = apply_fcfs

make_instance_and_solution(threshold, optimizer_function, n_samples)

print("done running")
"""
# Den lysere noden er den furste i sekvensen
# Hver jobb har et eget symbol
def make_dataset():
    generator_params = {
        "num_jobs": 2,
        "num_machines": 4,
        "min_ops_per_job": 2,
        "max_ops_per_job": 2,
        "min_processing_time": 5,
        "max_processing_time": 20,
        "min_eligible_ma_per_op": 4,
        "max_eligible_ma_per_op": 4,
    }

    env = FJSPEnv(
        generator_params=generator_params,
        _torchrl_mode=True,
        stepwise_reward=True
    )
    
    image_size = 64
    start_from = 0
    end_at = 25000
    for i in range(start_from, end_at):

        folder = 'images'
        imagename = '4ma_2op_2j'
        filetype = 'png'
        filename = folder + '/' + imagename + '_' + str(i) + '.' + filetype

        sch_rule = apply_fcfs
        while True:
            try:
                td = env.reset(batch_size=[1])
                encode_to_image(generator_params, env, td, filename, 8.4, sch_rule, image_size)
                logging.info('Made image ' + str(i - start_from))
                break
            except ValueError as err:
                continue



def apply_random(td):
    while not td['done'].all():
        min_proc_time = 100
        min_proc_op_i = 0
        for ma_i in range(  len(td['proc_times'][0])  ):
            ready_indexes = [i for i, val in enumerate(td['is_ready'][0]) if val]
            min_proc_op_i = random.choice(ready_indexes)


            job_id = map_index_to_jobid(min_proc_op_i)

            td['action'] = torch.tensor([job_id])
            td = env.step(td)['next']
    return td






def make_assignment(td):
    u = torch.tensor(td['ma_assignment'])

    # Remove the batch dimension (size 1)
    adj_small = u.squeeze(0)  # now shape is (4, 4)
    num_machines, num_ops = adj_small.shape  # should be 4, 4
    N = num_machines * num_ops  # 16
    # Create the big adjacency matrix
    adj_expanded = torch.zeros((N, N))

    for i in range(num_machines):
        for j in range(num_ops):
            for k in range(num_ops):
                idx1 = i * num_ops + j
                idx2 = i * num_ops + k
                adj_expanded[idx1, idx2] = adj_small[i, k]
                # or adj_small[i, j] depending on how you interpret adjacency

    return adj_expanded

def give_proc_times(td):
    x = torch.tensor(td['proc_times'], dtype=torch.float32)

    v = x.flatten()

    # min & max bounds
    min_val = 5.0
    max_val = 20.0
    v_norm = (v - min_val) / (max_val - min_val)
    # optionally: clip to [0,1] if values might go outside bounds
    # v_norm = torch.clamp(v_norm, min=0.0, max=1.0)

    res_mat = torch.diag(v_norm)
    return res_mat


def make_dataset():

    for i in range(120, 25000):
        instance_file = f"instances/4ma_2op_2j_{i}"

        logging.info(f"making data for instance {i}")
        
        with open(instance_file + ".json", "r", encoding="utf‑8") as f:
            td = json.load(f)

            data = {}
            adj = make_assignment(td)











def pad_to_16x16(mat):
    mat = mat.squeeze(0)
    n_rows, n_cols = mat.squeeze(0).shape
    assert n_cols == 16, "expected 16 columns"
    if n_rows > 16:
        raise ValueError("Already have more than 16 rows")
    # Number of rows to add:
    to_add = 16 - n_rows
    # Make a block of shape (to_add, 16) filled with -1:
    pad_block = torch.full((to_add, n_cols), -1.0, dtype=mat.dtype)
    # Concatenate along the row dimension (dim=0):
    return torch.cat([mat, pad_block], dim=0)


class MyJSONMatrixDataset(Dataset):
    def __init__(self, root_dir="instances", prefix="4ma_4op_4j", start=1, end=49999):
        """
        root_dir: directory where JSON files are stored
        prefix: beginning part of filename before number
        start, end: inclusive range of file numbers
        """
        self.root_dir = root_dir
        self.prefix = prefix
        # build list of filepaths
        self.file_paths = [
            os.path.join(root_dir, f"{prefix}_{i}.json")
            for i in range(start, end + 1)
        ]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        adj = data["ma_assignment"]
        proc_times = data["proc_times"]

        # convert to tensors
        adj = torch.tensor(adj, dtype=torch.float32)
        proc_times = torch.tensor(proc_times, dtype=torch.float32)

        adj = pad_to_16x16(adj)
        proc_times = pad_to_16x16(proc_times)

        return adj, proc_times







def make_dataset():

    for i in range(120, 25000):
        instance_file = f"instances/4ma_2op_2j_{i}"

        logging.info(f"making data for instance {i}")

        with open(instance_file + ".json", "r", encoding="utf‑8") as f:
            td = json.load(f)

            data = {}
            adj = make_assignment(td)
            proc = give_proc_times(td)
            data["adj"] = adj.tolist()
            data["proc_times"] = proc.tolist()

            with open(f"instances/new_4ma_2op_2j_{i}.json", "w") as w:
                json.dump(data, w, indent=4)


def see_instance():
    instance_file = f"instances/new_4ma_2op_2j_130"

    with open(instance_file + ".json", "r", encoding="utf‑8") as f:
        td = json.load(f)
        logging.info(td)

    with open(f"instances/4ma_2op_2j_130.json", "r") as w:
        hm = json.load(w)
        logging.info(hm["proc_times"])



def check_data():

    path = "instances/4ma_4op_4j_50.json"
    with open(path, "r", encoding="utf‑8") as f:
        td = json.load(f)
        logging.info(td["ma_assignment"])
        logging.info("..")
        logging.info(td["proc_times"])
        logging.info(td["start_times"])





def batch_resize(input_folder, output_folder, size=(16,16), 
                 file_exts=('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through files in input folder
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(file_exts):
            continue  # skip non-image files

        input_path  = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        try:
            with Image.open(input_path) as img:
                # Resize image to given size (no aspect ratio preservation)
                img_resized = img.resize(size, Image.LANCZOS)
                img_resized.save(output_path)
                logging.info(f"Resized {filename} → {size}")
        except Exception as e:
            logging.info(f"Skipping {filename}: {e}")

# in_folder  = "images"
# out_folder = "images"
# batch_resize(in_folder, out_folder, size=(16,16))

# logging.info("called hm.py")


