import ml_utils.save_io as io
import locgame.models as models
import locgame.environments as environments
import time
from ml_utils.utils import load_json, try_key
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0")

"""
To use this script, argue a model folder or checkpt to be examined

$ python3 watch_model.py <path_to_model>
"""

env_name = "~/loc_games/LocationGameLinux_1/LocationGameLinux.x86_64"

checkpt = io.load_checkpoint(sys.argv[1])
hyps = checkpt['hyps']
if "absoluteCoords" not in hyps['float_params']:
    params = hyps['float_params']
    params['absoluteCoords'] = float(not params["egoCentered"])
if env_name is not None:
    hyps['env_name'] = env_name

print("Making Env")
hyps['seed'] = int(time.time())
env = environments.UnityGymEnv(**hyps)

print("Making model")
model = getattr(models,hyps['model_class'])(**hyps)
model.to(DEVICE)
model.load_state_dict(checkpt["state_dict"])

done = True
model.eval()
sum_rew = 0
n_loops = 0
with torch.no_grad():
    while True:
        if done:
            obs,_ = env.reset()
            model.reset_h()
            plt.imshow(obs.squeeze().permute(1,2,0).data.numpy()/3)
            plt.show()
            if n_loops > 0:
                print("Running Mean Rew:", sum_rew/n_loops)
        pred,rew_pred,color,shape = model(obs[None].to(DEVICE))
        obs,targ,rew,done,_ = env.step(pred)
        sum_rew += rew
        if color is not None:
            color = torch.argmax(color[0]).item()
            shape = torch.argmax(shape[0]).item()
        pred = pred.squeeze().cpu().data.tolist() + [color,shape]
        print("targ:", targ.squeeze())
        print("pred:", pred)
        print()
        plt.imshow(obs.squeeze().permute(1,2,0).data.numpy()/3)
        plt.show()
        n_loops += 1
