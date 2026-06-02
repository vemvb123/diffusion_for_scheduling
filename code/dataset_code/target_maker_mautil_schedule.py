import torch
import __main__

from rl4co.envs import FJSPEnv
from rl4co.models.zoo.l2d import L2DModel
from rl4co.envs.common.base import RL4COEnvBase



original_setstate = RL4COEnvBase.__setstate__

def patched_setstate(self, state):
    if "rng" in state:
        rng = state["rng"]

        if not isinstance(rng, torch.ByteTensor):
            rng = torch.tensor(
                rng,
                dtype=torch.uint8,
                device="cpu"
            )
        else:
            rng = rng.cpu().byte()

        state["rng"] = rng

    original_setstate(self, state)

RL4COEnvBase.__setstate__ = patched_setstate

class LimitedMachineFJSPEnv(FJSPEnv):
    pass
__main__.LimitedMachineFJSPEnv = LimitedMachineFJSPEnv





def make_actions_for_instance(
    td,
    checkpoint_path
):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = L2DModel.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        load_baseline=False
    )

    model = model.to(device)

    model.eval()

    td = td.to(device)

    with torch.no_grad():

        out = model(
            td,
            decode_type="greedy",
            return_actions=True
        )

    return out["actions"].cpu()

