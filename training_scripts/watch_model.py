import ml_utils.save_io as io
import locgame.models as models
import locgame.environments as environments
from locgame.training import DummyFwdModel
import time
from ml_utils.utils import load_json, try_key
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn.functional as F

DEVICE = torch.device("cuda:0")

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
hyps['seed'] = int(time.time())
env = environments.UnityGymEnv(**hyps)

print("Making model")
model = getattr(models,hyps['model_class'])(**hyps)
model.to(DEVICE)
model.load_state_dict(checkpt["state_dict"])

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
obs = None
model.eval()
print(model.aud_targs)
sum_rew = 0
n_loops = 0
with torch.no_grad():
    while True:
        if done:
            if obs is not None:
                pred,color,shape,rew_pred = model(obs[None].to(DEVICE),
                                        None,
                                        targ[2:3].long()[None].cuda(),
                                        targ[3:4].long()[None].cuda(),
                                        targ[4:5].long()[None].cuda())

                if len(color) > 0:
                    color = torch.argmax(color[0]).item()
                    shape = torch.argmax(shape[0]).item()
                loc = pred.squeeze().cpu().data.tolist()
                disp_pred = loc + [color,shape]
                print("ending pred:", disp_pred)
                print("ending targ:", targ)
                loc_loss = 10*F.mse_loss(pred,
                             torch.FloatTensor(targ[:2])[None].cuda())
                print("ending locL:", loc_loss.item())
                print()
            obs,targ = env.reset()
            model.reset_h()
            fwd_model.reset_h()
            plt.imshow(obs.squeeze().permute(1,2,0).data.numpy()/6+0.5)
            plt.show()
            if n_loops > 0:
                print("Running Mean Rew:", sum_rew/n_loops)
        pred,color,shape,rew_pred = model(obs[None].to(DEVICE),
                                        None,
                                        targ[2:3].long()[None].cuda(),
                                        targ[3:4].long()[None].cuda(),
                                        targ[4:5].long()[None].cuda())
        # Fwd Dynamics
        if fwd_dynamics:
            _,mu,sigma,mu_pred,sigma_pred = fwd_model(obs[None].cuda(),
                                        None,
                                        targ[2:3].long()[None].cuda(),
                                        targ[3:4].long()[None].cuda(),
                                        targ[4:5].long()[None].cuda())
            if not use_fwd_preds:
                s = mu + sigma*torch.randn_like(sigma)
                obs_pred = fwd_model.decode(s).cpu()
                if not try_key(hyps,'end_sigmoid',False):
                    obs_pred = torch.clamp(obs_pred/6+.5,0,1)
                temp = obs/6+.5
                obscatpred = torch.cat([temp,obs_pred[0]],dim=2)
                obscatpred = obscatpred.permute(1,2,0).data.numpy()
                obscatpreds.append(np.tile(obscatpred[None],(repeat,1,1,1)))

        obs,targ,rew,done,_ = env.step(pred)
        sum_rew += rew
        if len(color) > 0:
            color = torch.argmax(color[0]).item()
            shape = torch.argmax(shape[0]).item()
        loc = pred.squeeze().cpu().data.tolist()
        disp_pred = loc + [color,shape]
        print("pred:", disp_pred)
        print("targ:", targ)
        loc_loss = 10*F.mse_loss(pred,torch.FloatTensor(targ[:2])[None].cuda())
        print("locL:", loc_loss.item())
        print()
        plt.imshow(obs.squeeze().permute(1,2,0).data.numpy()/6+0.5)
        plt.show()
        n_loops += 1

