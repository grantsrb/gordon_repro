import ml_utils.save_io as io
import locgame.models as models
import locgame.environments as environments
import time
from ml_utils.utils import load_json, try_key
import sys
import torch
import numpy as np
import cv2
from tqdm import tqdm

DEVICE = torch.device("cuda:0")

repeat = 15 # Set this to longer to linger on images longer
n_unique_frames = 100 # Set this to longer to get more unique game xp

"""
To use this script, argue a model folder or checkpt to be examined

$ python3 watch_model.py <path_to_model>
"""

env_name = None

checkpt = io.load_checkpoint(sys.argv[1])
hyps = checkpt['hyps']
if "absoluteCoords" not in hyps['float_params']:
    params = hyps['float_params']
    params['absoluteCoords'] = float(not params["egoCentered"])
if env_name is not None:
    hyps['env_name'] = env_name

print("Making Env")
env = environments.UnityGymEnv(**hyps)

print("Making model")
model = getattr(models,hyps['model_class'])(**hyps)
model.to(DEVICE)
model.load_state_dict(checkpt["state_dict"])

done = True
model.eval()
sum_rew = 0
n_loops = 0
frames = []
with torch.no_grad():
    while len(frames) < n_unique_frames:
        if done:
            obs,_ = env.reset()
            model.reset_h()
            img = obs.squeeze().permute(1,2,0).data.numpy()/3
            frames.append(np.tile(img[None],(repeat,1,1,1)))
        pred,rew_pred,color,shape = model(obs[None].to(DEVICE))
        obs,targ,rew,done,_ = env.step(pred)
        sum_rew += rew
        if color is not None:
            color = torch.argmax(color[0]).item()
            shape = torch.argmax(shape[0]).item()
        pred = pred.squeeze().cpu().data.tolist() + [color,shape]
        img = obs.squeeze().permute(1,2,0).data.numpy()/3
        frames.append(np.tile(img[None],(repeat,1,1,1)))
        n_loops += 1
        print("{:.2f}%".format(len(frames)/n_unique_frames), end="      \r")

frames = np.vstack(frames)
frames = np.uint8(frames*255/2+255/2)
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, img.shape[:2])
for frame in frames:
    out.write(frame) # frame is a numpy.ndarray with shape (1280,720,3)
out.release()
