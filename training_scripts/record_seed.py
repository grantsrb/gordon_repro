import ml_utils.save_io as io
import locgame.models as models
import locgame.environments as environments
from locgame.training import DummyFwdModel
import time
from ml_utils.utils import load_json, try_key
import sys
import torch
import numpy as np
import cv2
from tqdm import tqdm
import torch.nn.functional as F

DEVICE = torch.device("cuda:0")

use_fwd_preds = True
repeat = 15 # Set this to longer to linger on images longer
n_unique_frames = 100 # Set this to longer to get more unique game xp
env_name = None # If none, defaults to argued model's env
seed = None # If none, defaults to argued model's seed
validate = False # Determine if environment should be heldout set
output_name = "output.mp4"
if len(sys.argv) > 2:
    for arg in sys.argv[2:]:
        if arg == "validate": validate = True
        elif arg=="vae": use_fwd_preds = False
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
#hyps['float_params']['objectColorIdx'] = 0
#hyps['float_params']['objectShapeIdx'] = -1
#hyps['env_name'] = "~/loc_games/LocationGame2dLinux_11/LocationGame2dLinux.x86_64"

print("Float Params")
print("\n".join([k + ": " + str(v) for k,v in hyps['float_params'].items()]))

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

fwd_dynamics = try_key(hyps,'use_fwd_dynamics',False) and\
               try_key(hyps,"countOut",False)
if fwd_dynamics:
    fwd_model = getattr(models,hyps['fwd_class'])(**hyps)
    fwd_model.cuda()
    fwd_model.load_state_dict(checkpt['fwd_state_dict'])
    fwd_model.eval()
else:
    fwd_model = DummyFwdModel()

done = True
sum_rew = 0
n_loops = 0
frames = []
obscatpreds = []
with torch.no_grad():
    while len(frames) < n_unique_frames:
        if done:
            obs,targ = env.reset()
            model.reset_h()
            fwd_model.reset_h()
            img = obs.squeeze().permute(1,2,0).data.numpy()/3
            frames.append(np.tile(img[None],(repeat,1,1,1)))

            obscatpred = torch.cat([obs/6+.5,torch.zeros_like(obs)],dim=2)
            obscatpred = obscatpred.permute(1,2,0).data.numpy()
            obscatpreds.append(np.tile(obscatpred[None],(repeat,1,1,1)))
        if len(targ) > 4: num_idx = targ[4:5].long()[None].cuda()
        else: num_idx = None
        color_idx = targ[2:3].long()[None].cuda()
        shape_idx = targ[3:4].long()[None].cuda()

        # Actor
        tup = model(obs[None].to(DEVICE), h=None,
                            color_idx=color_idx,
                            shape_idx=shape_idx,
                            count_idx=num_idx)
        if len(tup)==3: pred,color,shape = tup
        else: pred,color,shape,rew_p = tup

        # Fwd Dynamics
        if fwd_dynamics:
            h,mu,sigma,mu_pred,sigma_pred = fwd_model(obs[None].cuda(),
                                                   h=None,
                                                   color_idx=color_idx,
                                                   shape_idx=shape_idx,
                                                   count_idx=num_idx)
            if not use_fwd_preds:
                s = mu + sigma*torch.randn_like(sigma)
                obs_pred = fwd_model.decode(s,h).cpu()
                if not try_key(hyps,'end_sigmoid',False):
                    obs_pred = torch.clamp(obs_pred/6+.5,0,1)
                temp = obs/6+.5
                obscatpred = torch.cat([temp,obs_pred[0]],dim=2)
                obscatpred = obscatpred.permute(1,2,0).data.numpy()
                obscatpreds.append(np.tile(obscatpred[None],(repeat,1,1,1)))

        # Step
        obs,targ,rew,done,_ = env.step(pred)

        # Fwd Dynamics Cont
        if fwd_dynamics and use_fwd_preds:
            s = mu_pred + sigma_pred*torch.randn_like(sigma_pred)
            obs_pred = fwd_model.decode(s,h).cpu()
            if not try_key(hyps,'end_sigmoid',False):
                obs_pred = torch.clamp(obs_pred/6+.5,0,1)
            temp = obs/6+.5
            obscatpred = torch.cat([temp,obs_pred[0]],dim=2)
            obscatpred = obscatpred.permute(1,2,0).data.numpy()
            obscatpreds.append(np.tile(obscatpred[None],(repeat,1,1,1)))

        sum_rew += rew
        if color is not None and len(color) > 0:
            color = torch.argmax(color[0]).item()
            shape = torch.argmax(shape[0]).item()
        loc = pred.squeeze().cpu().data.tolist()
        if len(tup) > 3 and len(rew_p) > 0:
            rew_p = rew_p.cpu().data.tolist()
        else:
            rew_p = []
        disp_pred = loc + [color,shape]
        print("pred:", disp_pred)
        print("targ:", targ)
        print()
        loc_loss = F.mse_loss(pred,torch.FloatTensor(targ[:2])[None].cuda())
        #print("locL:", loc_loss.item())
        #print()
        img = obs.squeeze().permute(1,2,0).data.numpy()/3
        frames.append(np.tile(img[None],(repeat,1,1,1)))
        n_loops += 1
        #print("{:.2f}%".format(len(frames)/n_unique_frames*100), end="      \r")

frames = np.vstack(frames)
frames = np.uint8(frames*255/2+255/2)
out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, img.shape[:2])
for frame in frames:
    out.write(frame)
out.release()

if fwd_dynamics:
    output_name = output_name[:-4] + "_fwd.mp4"
    obscatpreds = np.vstack(obscatpreds)
    obscatpreds = np.uint8(obscatpreds*255)
    shape = (obscatpred.shape[1],obscatpred.shape[0])
    fwdout = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'),
                                              30, shape)
    for obscatpred in obscatpreds:
        fwdout.write(obscatpred)
    fwdout.release()
env.close()
