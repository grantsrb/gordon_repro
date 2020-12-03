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
import torch.nn.functional as F

DEVICE = torch.device("cuda:0")

repeat = 15 # Set this to longer to linger on images longer
n_unique_frames = 50 # Set this to longer to get more unique game xp
env_name = None # If none, defaults to argued model's env
seed = None # If none, defaults to argued model's seed
validate = False # Determine if environment should be heldout set
output_name = "output.mp4"
if len(sys.argv) > 2:
    for arg in sys.argv[2:]:
        if arg == "validate": validate = True
        else: output_name = arg

print("Saving to", output_name)
"""
To use this script, argue a model folder or checkpt to be examined

$ python3 record_seed.py <path_to_model>
"""


checkpt = io.load_checkpoint(sys.argv[1])
hyps = checkpt['hyps']
if "absoluteCoords" not in hyps['float_params']:
    params = hyps['float_params']
    params['absoluteCoords'] = float(not params["egoCentered"])
if validate: print("Validation environment")
hyps['float_params']["validation"] = validate
if env_name is not None:
    hyps['env_name'] = env_name
if seed is not None:
    hyps['seed'] = seed

print("Making Env")
env = environments.UnityGymEnv(**hyps)

print("Making model")
model = getattr(models,hyps['model_class'])(**hyps)
model.to(DEVICE)
try:
    model.load_state_dict(checkpt["state_dict"])
except:
    keys = list(checkpt['state_dict'].keys())
    for k in keys:
        if "pavlov" == k.split(".")[0]:
            del checkpt['state_dict'][k]
    model.load_state_dict(checkpt['state_dict'])
    print("state dict success")
model.eval()

done = True
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
        tup = model(obs[None].to(DEVICE))
        if len(tup)==3: pred,color,shape = tup
        else: pred,color,shape,rew_p = tup
        obs,targ,rew,done,_ = env.step(pred)
        sum_rew += rew
        if color is not None and len(color) > 0:
            color = torch.argmax(color[0]).item()
            shape = torch.argmax(shape[0]).item()
        loc = pred.squeeze().cpu().data.tolist()
        if len(tup) > 3 and len(rew_p) > 0: rew_p = rew_p.cpu().data.tolist()
        else: rew_p = []
        disp_pred = loc + [color,shape]
        #print("pred:", disp_pred)
        #print("targ:", targ)
        loc_loss = F.mse_loss(pred,torch.FloatTensor(targ[:2])[None].cuda())
        #print("locL:", loc_loss.item())
        #print()
        img = obs.squeeze().permute(1,2,0).data.numpy()/3
        frames.append(np.tile(img[None],(repeat,1,1,1)))
        n_loops += 1
        print("{:.2f}%".format(len(frames)/n_unique_frames*100), end="      \r")

frames = np.vstack(frames)
frames = np.uint8(frames*255/2+255/2)
out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, img.shape[:2])
for frame in frames:
    out.write(frame) # frame is a numpy.ndarray with shape (1280,720,3)
out.release()
env.close()
