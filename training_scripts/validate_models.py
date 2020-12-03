import locgame.models as models
import locgame.environments as environments
from locgame.training import Runner
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
env_name = None
seed = None

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
        lossfxn = nn.MSELoss()
        for model_folder in model_folders:
            table = {"model_name":[],
                    "gen_rew":[],
                    "gen_loss":[],
                    "gen_loc_loss":[],
                    "gen_rew_loss":[],
                    "gen_obj_loss":[],
                    "gen_color_loss":[],
                    "gen_shape_loss":[],
                    "gen_obj_acc":[],
                    "gen_color_acc":[],
                    "gen_shape_acc":[]}
            checkpt = io.load_checkpoint(model_folder)
            hyps = checkpt['hyps']
            if "absoluteCoords" not in hyps['float_params']:
                params = hyps['float_params']
                params['absoluteCoords'] = float(not params["egoCentered"])
            hyps['float_params']['validation'] = 1
            if env_name is not None:
                hyps['env_name'] = env_name
            if seed is not None:
                hyps['seed'] = seed
            
            print("Making Env")
            table['seed'] = [hyps['seed']]
            env = environments.UnityGymEnv(**hyps)
        
            print("Making model")
            model = getattr(models,hyps['model_class'])(**hyps)
            model.cuda()
            model.load_state_dict(checkpt["state_dict"])
            model.eval()
            val_runner = Runner(rank=0,hyps=hyps, shared_data=None,
                                                  gate_q=None,
                                                  stop_q=None,
                                                  end_q=None)
            val_runner.env = env
            val_runner.model = model
            rew_alpha = checkpt['hyps']['rew_alpha']
            alpha = checkpt['hyps']['alpha']
            print("Rolling out model")
            with torch.no_grad():
                loss_tup = val_runner.rollout(0,validation=True,n_tsteps=500)
                loss_tup = [x.item() for x in loss_tup]
                val_loc_loss,val_color_loss,val_shape_loss=loss_tup[:3]
                val_rew_loss,val_color_acc,val_shape_acc,val_rew=loss_tup[3:]
                val_obj_loss = ((val_color_loss + val_shape_loss)/2)
                val_obj_acc = ((val_color_acc + val_shape_acc)/2)
                temp = rew_alpha*val_loc_loss + (1-rew_alpha)*val_rew_loss
                val_loss = alpha*temp + (1-alpha)*val_obj_loss

            table["gen_loc_loss"].append(val_loc_loss)
            table["gen_rew_loss"].append(val_rew_loss)
            table["gen_obj_loss"].append(val_obj_loss)
            table["gen_color_loss"].append(val_color_loss)
            table["gen_shape_loss"].append(val_shape_loss)
            table["gen_obj_acc"].append(val_obj_acc)
            table["gen_color_acc"].append(val_color_acc)
            table["gen_shape_acc"].append(val_shape_acc)
            table['gen_rew'].append(val_rew)
            table['gen_loss'].append(val_loss)

            table['model_name'].append(model_folder)
            df = pd.DataFrame(table)

            # These keys are ignored
            ignores = {"del_prev_sd","key_descriptions",
                       "search_keys","float_params"}
            for k,v in checkpt['hyps'].items():
                # Note that any key ending with an underscore is ignored
                if k not in ignores and k[-1] != "_":
                    try:
                        df[k] = v
                    except:
                        df[k] = str(v)
            params = checkpt['hyps']['float_params']
            for k,v in params.items():
                df[k] = v

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
