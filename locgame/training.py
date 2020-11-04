import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import PoissonNLLLoss, MSELoss
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,MultiStepLR
import time
from tqdm import tqdm
import math
from queue import Queue
from collections import deque
import psutil
import json
import torch.multiprocessing as mp
import ml_utils.save_io as io
from ml_utils.training import get_exp_num, record_session, get_save_folder,get_resume_checkpt
from ml_utils.utils import try_key
import locgame.models as models
import locgame.environments as environments

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

def train(hyps, verbose=True):
    """
    hyps: dict
        contains all relavent hyperparameters
    """
    hyps['main_path'] = try_key(hyps,'main_path',"./")
    checkpt,hyps = get_resume_checkpt(hyps)
    if checkpt is None:
        hyps['exp_num']=get_exp_num(hyps['main_path'], hyps['exp_name'])
        hyps['save_folder'] = get_save_folder(hyps)
    if not os.path.exists(hyps['save_folder']):
        os.mkdir(hyps['save_folder'])
    # Set manual seed
    hyps['seed'] = try_key(hyps,'seed', int(time.time()))
    torch.manual_seed(hyps['seed'])
    np.random.seed(hyps['seed'])

    if "float_params" not in hyps:
        keys = ["validation", "egoCentered", "absoluteCoords",
                "smoothMovement", "restrictCamera"]
        hyps['float_params'] = {k:try_key(hyps,k,0) for k in keys}

    model_class = hyps['model_class']
    hyps['n_loss_loops'] = try_key(hyps,'n_loss_loops',1)

    if verbose:
        print("Making Env")
    env = environments.UnityGymEnv(**hyps)

    hyps["img_shape"] = env.shape
    hyps["targ_shape"] = env.targ_shape

    if verbose:
        print("Making model")
    model = getattr(models,model_class)(**hyps)

    if try_key(hyps,'multi_gpu',False):
        ids = [i for i in range(torch.cuda.device_count())]
        model = torch.nn.DataParallel(model, device_ids=ids)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'],
                                           weight_decay=hyps['l2'])
    if checkpt is not None:
        if verbose:
            print("Loading state dicts from", checkpt['save_folder'])
        model.load_state_dict(checkpt["state_dict"])
        optimizer.load_state_dict(checkpt["optim_dict"])
    lossfxn = getattr(nn,try_key(hyps,'lossfxn',"MSELoss"))()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                                                    patience=6,
                                                    verbose=True)
    if verbose:
        print("Beginning training for {}".format(hyps['save_folder']))
        print("Img Shape:", hyps['img_shape'])
    record_session(hyps,model)

    if hyps['exp_name'] == "test":
        hyps['n_epochs'] = 2
    epoch = -1 if checkpt is None else checkpt['epoch']

    alpha = try_key(hyps,'rew_alpha',.7)
    print()
    while epoch < hyps['n_epochs']:
        epoch += 1
        print("Epoch:{} | Model:{}".format(epoch, hyps['save_folder']))
        starttime = time.time()
        avg_loss = 0
        avg_rew = 0
        avg_pred_loss = 0
        avg_rew_loss = 0
        model.train()
        print("Training...")
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        # Collect new rollouts
        done = False
        for rollout in range(hyps['n_rollouts']):
            obs,targ = env.reset()
            model.reset_h()
            obsrs = [obs]
            targs = []
            preds = []
            rew_preds = []
            done_preds = []
            rews  = []
            dones = []
            while len(rews) < hyps['batch_size']:
                pred,rew_pred = model(obs[None].to(DEVICE))
                preds.append(pred)
                rew_preds.append(rew_pred)
                obs,targ,rew,done,_ = env.step(pred)
                rews.append(rew)
                dones.append(done)
                targs.append(targ)
                if done:
                    obs,_ = env.reset()
                    model.reset_h()
                obsrs.append(obs)
            # Calc Loss
            preds = torch.stack(preds).squeeze()
            targs = torch.stack(targs)[:,:2]
            pred_loss = lossfxn(preds,targs.to(DEVICE))
            rew_preds = torch.stack(rew_preds)
            rews = torch.FloatTensor(rews)
            rew_loss = lossfxn(rew_preds.squeeze(),rews.to(DEVICE))

            loss = alpha*pred_loss + (1-alpha)*rew_loss
            loss = loss / hyps['n_loss_loops']
            loss.backward()

            if rollout % hyps['n_loss_loops'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            avg_loss += loss.item()
            avg_pred_loss += pred_loss.item()
            avg_rew_loss += rew_loss.item()

            rew_mean = rews.mean().item()
            avg_rew += rew_mean
            s = "Loc:{:.5f} | RewLoss:{:.5f} | Rew:{:.5f} | {:.0f}%"
            s = s.format(pred_loss.item(), rew_loss.item(), rew_mean,
                                      rollout/hyps['n_rollouts']*100)
            print(s, end=len(s)*" " + "\r")
            if hyps['exp_name'] == "test" and rollout>3: break
        print()
        train_avg_loss = avg_loss / hyps['n_rollouts']
        train_pred_loss = avg_pred_loss / hyps['n_rollouts']
        train_rew_loss = avg_rew_loss / hyps['n_rollouts']
        train_avg_rew = avg_rew / hyps['n_rollouts']

        s = "Train - Loss:{:.5f} | Loc:{:.5f} | "
        s += "RewLoss:{:.5f} | Rew:{:.5f}\n"
        stats_string = s.format(train_avg_loss, train_pred_loss,
                                                train_rew_loss,
                                                train_avg_rew)
        scheduler.step(train_avg_loss)

        print("Evaluating")
        done = True
        model.eval()
        sum_rew = 0
        n_eps = -1
        n_loops = 0
        sleep_time = 1
        with torch.no_grad():
            while n_eps < 5:
                if done:
                    obs,_ = env.reset()
                    model.reset_h()
                    time.sleep(sleep_time)
                    n_eps += 1
                pred,rew_pred = model(obs[None].to(DEVICE))
                obs,_,rew,done,_ = env.step(pred)
                sum_rew += rew
                time.sleep(sleep_time)
                n_loops += 1
        val_rew = sum_rew/n_loops
        stats_string += "Evaluation Avg Rew: {:.5f}\n".format(val_rew)

        optimizer.zero_grad()
        save_dict = {
            "epoch":epoch,
            "hyps":hyps,
            "train_loss":train_avg_loss,
            "train_pred_loss":train_pred_loss,
            "train_rew_loss":train_rew_loss,
            "train_rew":train_avg_rew,
            "val_rew":val_rew,
            "state_dict":model.state_dict(),
            "optim_dict":optimizer.state_dict(),
        }
        save_name = "checkpt"
        save_name = os.path.join(hyps['save_folder'],save_name)
        io.save_checkpt(save_dict, save_name, epoch, ext=".pt",
                                del_prev_sd=hyps['del_prev_sd'])
        stats_string += "Exec time: {}\n".format(time.time()-starttime)
        print(stats_string)
        s = "Epoch:{} | Model:{}\n".format(epoch, hyps['save_folder'])
        stats_string = s + stats_string
        log_file = os.path.join(hyps['save_folder'],"training_log.txt")
        with open(log_file,'a') as f:
            f.write(str(stats_string)+'\n')
    del save_dict['state_dict']
    del save_dict['optim_dict']
    del save_dict['hyps']
    save_dict['save_folder'] = hyps['save_folder']
    env.close()
    del env
    return save_dict

