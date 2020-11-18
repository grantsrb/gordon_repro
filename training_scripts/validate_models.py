import locgame.models as models
import locgame.environments as environments
import time
from ml_utils.utils import load_json, try_key
import ml_utils.save_io as io
import sys
import torch
import numpy as np
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

"""
To use this script, argue an experiment folder to be examined

$ python3 watch_model.py <path_to_folder>
"""
if __name__=="__main__":
    torch.cuda.set_device(0)
    manifest_name = "systematicity_manifest.csv"
    for folder in sys.argv[1:]:
        print("Exp Folder:", folder)
        manifest_path = os.path.join(folder,manifest_name)
        if os.path.exists(manifest_path):
            manifest = pd.read_csv(manifest_path,sep="!")
        else:
            manifest = None
        model_folders = io.get_model_folders(folder,True)
        print("Model Folders:", model_folders)
        env_name = "~/loc_games/LocationGameLinux_1/LocationGameLinux.x86_64"
        lossfxn = nn.MSELoss()
        for model_folder in model_folders:
            table = {"model_name":[],"validation_rew":[],
                    "validation_loss":[],
                    "validation_obj_acc":[],
                    "validation_color_acc":[],
                    "validation_shape_acc":[]}
            checkpt = io.load_checkpoint(model_folder)
            hyps = checkpt['hyps']
            if "absoluteCoords" not in hyps['float_params']:
                params = hyps['float_params']
                params['absoluteCoords'] = float(not params["egoCentered"])
            hyps['float_params']['validation'] = 1
            hyps['env_name'] = env_name
            
            print("Making Env")
            table['seed'] = [hyps['seed']]
            hyps['seed'] = 121314
            env = environments.UnityGymEnv(**hyps)
        
            print("Making model")
            model = getattr(models,hyps['model_class'])(**hyps)
            model.cuda()
            model.load_state_dict(checkpt["state_dict"])
        
            done = True
            model.eval()
            sum_rew = 0
            n_loops = 0
            preds = []
            targs = []
            color_preds = []
            shape_preds = []
            n_eps = -1
            with torch.no_grad():
                while n_eps < 100:
                    if done:
                        obs,_ = env.reset()
                        model.reset_h()
                        n_eps += 1
                    tup = model(obs[None].cuda())
                    pred,rew_pred,color_pred,shape_pred = tup
                    preds.append(pred)
                    color_preds.append(color_pred)
                    shape_preds.append(shape_pred)
                    obs,targ,rew,done,_ = env.step(pred)
                    targs.append(targ)
                    sum_rew += rew
                    n_loops += 1
            val_rew = sum_rew/n_loops
            preds = torch.stack(preds).squeeze()
            targs = torch.stack(targs)
            targs,obj_targs = targs[:,:2],targs[:,2:]
            loss = lossfxn(preds,targs.cuda())
            if color_preds[0] is not None:
                obj_targs = obj_targs.long().cuda()
                color_preds = torch.stack(color_preds).squeeze()
                color_loss = F.cross_entropy(color_preds,
                                             obj_targs[:,0])
                shape_preds = torch.stack(shape_preds).squeeze()
                shape_loss = F.cross_entropy(shape_preds,
                                             obj_targs[:,1])
                obj_loss = color_loss + shape_loss
                with torch.no_grad():
                    maxes = torch.argmax(color_preds,dim=-1)
                    color_acc = (maxes==obj_targs[:,0]).float().mean()
                    maxes = torch.argmax(shape_preds,dim=-1)
                    shape_acc = (maxes==obj_targs[:,1]).float().mean()
                    obj_acc = ((color_acc + shape_acc)/2).item()
                table["validation_obj_acc"].append(obj_acc)
                table["validation_color_acc"].append(color_acc.item())
                table["validation_shape_acc"].append(shape_acc.item())
            else:
                table["validation_obj_acc"].append(0)
                table["validation_color_acc"].append(0)
                table["validation_shape_acc"].append(0)
            table['model_name'].append(model_folder)
            table['validation_rew'].append(val_rew)
            table['validation_loss'].append(loss.item())
            df = pd.DataFrame(table)
            params = checkpt['hyps']['float_params']
            df['egoCentered'] =    try_key(params,'egoCentered',False)
            df['absoluteCoords'] = try_key(params,'absoluteCoords',False)
            df['smoothMovement'] = try_key(params,'smoothMovement',False)
            df['restrictCamera'] = try_key(params,'restrictCamera',False)
            df['randomizeObjs'] =  try_key(params,'randomizeObjs',False)
            df['specGoalObjs'] =   try_key(params,'specGoalObjs',False)
            df['obj_recog'] = try_key(checkpt['hyps'],'obj_recog',False)
            df['model_class'] = checkpt['hyps']['model_class']
            df['model_type'] = "Unk"
            idx = (df["egoCentered"]>=1)&(df["absoluteCoords"]>=1)
            df.loc[idx,"model_type"] = "EgoAbsolute"
            idx = (df["egoCentered"]>=1)&(df["absoluteCoords"]<=0)
            df.loc[idx,"model_type"] = "EgoRelative"
            idx = (df["egoCentered"]<=0)&(df["absoluteCoords"]>=1)
            df.loc[idx,"model_type"] = "AlloAbsolute"
            idx = (df["egoCentered"]<=0)&(df["absoluteCoords"]<=0)
            df.loc[idx,"model_type"] = "AlloRelative"
            if manifest is None:
                manifest = df
            else:
                manifest = manifest.loc[manifest['model_name']!=model_folder]
                manifest = manifest.append(df,sort=True)
            manifest.to_csv(manifest_path, header=True,index=False,sep="!")
            for k,v in table.items():
                print(k,":",v)
            print()
            env.close()
