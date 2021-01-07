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
from ml_utils.utils import try_key, load_json
import locgame.models as models
import locgame.environments as environments
import matplotlib.pyplot as plt
from datetime import datetime

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

def get_game_variables(hyps):
    """
    Uses the env_name to find the n_number, n_color, and n_shape counts
    for the argued version of the game.
    """
    env_path = "/".join(hyps['env_name'].split("/")[:-1])
    variables = load_json(env_path+"/variables.json")
    hyps['endAtOrigin'] = try_key(hyps,'endAtOrigin',0)
    for k in variables.keys():
        hyps[k] = variables[k] + hyps['endAtOrigin']
    hyps['n_numbers'] = try_key(hyps,'maxObjCount',5)
    return hyps

def train(rank, hyps, verbose=True):
    """
    hyps: dict
        contains all relavent hyperparameters
    """
    # Initialize settings
    if "randomizeObjs" in hyps:
        assert False, "you mean randomizeObs, not randomizeObjs"
    if "audibleTargs" in hyps and hyps['audibleTargs'] > 0:
        hyps['aud_targs'] = True
        if verbose: print("Using audible targs!")
    countOut = try_key(hyps, 'countOut', 0)
    if countOut and not hyps['endAtOrigin']:
        assert False, "endAtOrigin must be true for countOut setting"
    hyps['main_path'] = try_key(hyps,'main_path',"./")
    checkpt,hyps = get_resume_checkpt(hyps,verbose=verbose) #incrs seed
    if checkpt is None:
        hyps['exp_num']=get_exp_num(hyps['main_path'], hyps['exp_name'])
        hyps['save_folder'] = get_save_folder(hyps)
    if not os.path.exists(hyps['save_folder']):
        os.mkdir(hyps['save_folder'])
    hyps = get_game_variables(hyps)
    # Set manual seed
    hyps['seed'] = try_key(hyps,'seed', int(time.time()))
    if "torch_seed" in hyps: torch.manual_seed(hyps['torch_seed'])
    else: torch.manual_seed(hyps['seed'])
    if "numpy_seed" in hyps: np.random.seed(hyps['numpy_seed'])
    else: np.random.seed(hyps['seed'])

    if "float_params" not in hyps:
        keys = hyps['game_keys']
        hyps['float_params'] = {k:try_key(hyps,k,0) for k in keys}
        if "minObjLoc" not in hyps:
            hyps['float_params']["minObjLoc"] = 0.27
            hyps['float_params']["maxObjLoc"] = 0.73

    print("Float Params:", hyps['float_params'])
    hyps['n_loss_loops'] = try_key(hyps,'n_loss_loops',1)

    if verbose:
        print("Making Env(s)")
    hyps['n_runners'] = try_key(hyps,'n_runners',1)
    validation = hyps['float_params']['validation']
    if hyps['n_runs'] > 1: hyps['float_params']['validation']=1
    env = environments.get_env(hyps)
    hyps['float_params']['validation'] = validation

    hyps["img_shape"] = env.shape
    hyps["targ_shape"] = env.targ_shape

    if verbose:
        print("Making model")
    model = getattr(models,hyps['model_class'])(**hyps)
    model.to(DEVICE)
    fwd_dynamics = try_key(hyps,'use_fwd_dynamics',True) and countOut
    fwd_model = None
    if fwd_dynamics:
        fwd_model = getattr(models, hyps['fwd_class'])(**hyps)
        fwd_model.cuda()
        fwd_optim = torch.optim.Adam(fwd_model.parameters(),
                                     lr=hyps['fwd_lr'],
                                     weight_decay=hyps['l2'])
        fwd_scheduler = ReduceLROnPlateau(fwd_optim, 'min', factor=0.5,
                                                     patience=6,
                                                     verbose=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'],
                                           weight_decay=hyps['l2'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                                                    patience=6,
                                                    verbose=True)
    if checkpt is not None:
        if verbose:
            print("Loading state dicts from", hyps['save_folder'])
        model.load_state_dict(checkpt["state_dict"])
        optimizer.load_state_dict(checkpt["optim_dict"])
        if fwd_dynamics:
            fwd_model.load_state_dict(checkpt['fwd_state_dict'])
            fwd_optim.load_state_dict(checkpt['fwd_optim_dict'])
    batch_size = hyps['batch_size']

    ## Multi Processing
    # The number of actual runs performed
    hyps['n_runs'] = try_key(hyps,'n_runs',1)
    # The number of steps for each run
    hyps['n_tsteps'] = hyps['batch_size']//hyps['n_runs']
    # The total number of steps included in the update
    hyps['batch_size'] = hyps['n_tsteps']*hyps['n_runs']
    shared_data = {
            'rews': torch.zeros(hyps['batch_size']),
            "hs":torch.zeros(hyps['batch_size'],model.h_shape[-1]),
            "starts":torch.zeros(hyps['batch_size']).long(),
            "dones":torch.zeros(hyps['batch_size']).long(),
            "loc_targs":torch.zeros(hyps['batch_size'],2),
            "obj_targs":torch.zeros(hyps['batch_size'],2).long(),
            "reset_fwd_hs":torch.zeros(hyps['batch_size']).long(),
            "counts":torch.zeros(hyps['batch_size'],1).long()
            }
    shared_data = {k:v.share_memory_().cuda() for k,v in\
                                     shared_data.items()}
    shared_data['obsrs'] = torch.zeros(hyps['batch_size'],*env.shape)

    gate_q = mp.Queue(hyps['n_runs'])
    stop_q = mp.Queue(hyps['n_runs'])
    end_q = mp.Queue(1)

    # Make Runners
    hyps['n_runners'] = try_key(hyps,'n_runners',None)
    if hyps['n_runners'] is None: hyps['n_runners'] = hyps['n_runs']
    runners = []
    for i in range(hyps['n_runners']):
        runner = Runner(rank=i, hyps=hyps, shared_data=shared_data,
                                           gate_q=gate_q,
                                           stop_q=stop_q,
                                           end_q=end_q)
        runners.append(runner)
    val_runner = Runner(rank=0,hyps=hyps, shared_data=None,
                                          gate_q=None,
                                          stop_q=None,
                                          end_q=None)
    val_runner.env = env
    if len(runners) > 1:
        procs = []
        for i in range(len(runners)):
            proc = mp.Process(target=runners[i].run, args=(model,True))
            procs.append(proc)
            proc.start()
        if verbose:
            print("Waiting for environments to load")
        for i in range(len(runners)):
            stop_q.get()
    else:
        runner.env = env

    if verbose:
        print("Beginning training for {}".format(hyps['save_folder']))
        print("Img Shape:", hyps['img_shape'])
        print("Num Samples Per Update:", batch_size)
    record_session(hyps,model)

    if hyps['exp_name'] == "test":
        hyps['n_epochs'] = 2
    epoch = -1 if checkpt is None else checkpt['epoch']

    alpha = try_key(hyps,'alpha',.5)
    rew_alpha = try_key(hyps,'rew_alpha',.9)
    obj_recog = try_key(hyps,'obj_recog',False)
    best_val_rew = -np.inf
    fwd_hs = None
    print()
    while epoch < hyps['n_epochs']:
        epoch += 1
        print("Epoch:{} | Model:{}".format(epoch, hyps['save_folder']))
        starttime = time.time()
        avg_loss = 0
        avg_rew = 0
        avg_loc_loss = 0
        avg_obj_loss = 0
        avg_color_loss = 0
        avg_color_acc = 0
        avg_shape_loss = 0
        avg_shape_acc = 0
        avg_rew_loss = 0
        avg_obj_acc = 0
        avg_fwd_loss = 0

        first_avg_loc_loss = 0
        first_avg_obj_loss = 0
        first_avg_color_loss = 0
        first_avg_color_acc = 0
        first_avg_shape_loss = 0
        first_avg_shape_acc = 0
        first_avg_rew_loss = 0
        first_avg_obj_acc = 0

        model.train()
        print("Training...")
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        # Collect new rollouts
        done = False
        for rollout in range(hyps['n_rollouts']):
            iter_start = time.time()
            if len(runners) > 1:
                # Start the runners
                for i in range(hyps['n_runs']):
                    gate_q.put(i)
                # Wait for all runners to stop
                for i in range(hyps['n_runs']):
                    stop_q.get()
            else:
                runner.run(model, multi_proc=False)

            # Collect data from runners
            rews = shared_data['rews']
            hs = shared_data['hs']
            dones = shared_data['dones']
            starts = shared_data['starts']
            obsrs = shared_data['obsrs']
            loc_targs = shared_data['loc_targs']
            obj_targs = shared_data['obj_targs']
            counts = shared_data['counts']
            reset_fwd_hs = shared_data['reset_fwd_hs']
            color_targs,shape_targs = obj_targs[:,0],obj_targs[:,1]

            # Make predictions
            if try_key(hyps,"use_bptt",False):
                pred_tup = bptt(hyps,model,obsrs,hs,dones,obj_targs,
                                                          counts)
            else:
                color_idx = obj_targs[:,:1]
                shape_idx = obj_targs[:,1:2]
                pred_tup = model(obsrs.cuda(), h=hs.cuda(),
                                               color_idx=color_idx,
                                               shape_idx=shape_idx,
                                               number_idx=counts)
            loc_preds,color_preds,shape_preds,rew_preds = pred_tup

            # Calc Losses
            post_obj_preds = try_key(hyps,'post_obj_preds',False)
            post_rew_preds = try_key(hyps,'post_rew_preds',False)
            loss_tup = calc_losses(
                         loc_preds,color_preds,shape_preds, rew_preds,
                         loc_targs,color_targs,shape_targs, rews,
                         starts,dones, post_obj_preds=post_obj_preds,
                         post_rew_preds=post_rew_preds,
                         hyps=hyps, firsts=reset_fwd_hs)
            loc_loss,color_loss,shape_loss,rew_loss = loss_tup[:4]
            color_acc,shape_acc = loss_tup[4:6]
            first_loc_loss,first_color_loss=loss_tup[6:8]
            first_shape_loss, first_rew_loss = loss_tup[8:10]
            first_color_acc,first_shape_acc = loss_tup[10:12]
            loss = rew_alpha*loc_loss + (1-rew_alpha)*rew_loss
            obj_loss = (color_loss + shape_loss)/2
            obj_acc = ((color_acc + shape_acc)/2)
            first_obj_loss = (first_color_loss + first_shape_loss)/2
            first_obj_acc = ((first_color_acc +  first_shape_acc)/2)
            loss = alpha*loss + (1-alpha)*obj_loss

            back_loss = loss / hyps['n_loss_loops']
            back_loss.backward()

            avg_loss      += loss.item()
            avg_rew       += rews.mean()
            avg_loc_loss  += loc_loss.item()
            avg_obj_loss  += obj_loss.item()
            avg_color_loss+= color_loss.item()
            avg_shape_loss+= shape_loss.item()
            avg_rew_loss  += rew_loss.item()
            avg_color_acc += color_acc.item()
            avg_shape_acc += shape_acc.item()
            avg_obj_acc   += obj_acc.item()
            first_avg_loc_loss  += first_loc_loss.item()
            first_avg_obj_loss  += first_obj_loss.item()
            first_avg_color_loss+= first_color_loss.item()
            first_avg_shape_loss+= first_shape_loss.item()
            first_avg_rew_loss  += first_rew_loss.item()
            first_avg_color_acc += first_color_acc.item()
            first_avg_shape_acc += first_shape_acc.item()
            first_avg_obj_acc   += first_obj_acc.item()

            if rollout % hyps['n_loss_loops'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Fwd dynamics loss
            s = ""
            if fwd_dynamics:
                preds,fwd_hs = fwd_preds(hyps, fwd_model,
                                               obsrs=obsrs,
                                               hs=fwd_hs,
                                               resets=reset_fwd_hs,
                                               obj_targs=obj_targs,
                                               counts=counts)
                fwd_loss = calc_fwd_loss(preds,obsrs,starts,dones)
                fwd_loss = fwd_loss / hyps['n_loss_loops']
                fwd_loss.backward()
                if rollout % hyps['n_loss_loops'] == 0:
                    fwd_optim.step()
                    fwd_optim.zero_grad()
                avg_fwd_loss += fwd_loss.item()
                s = "Fwd: {:.5f} | ".format(fwd_loss.item())

            s += "LocL:{:.5f} | Obj:{:.5f} | {:.0f}% | t:{:.2f}"
            s = s.format(loc_loss.item(), obj_loss.item(),
                                   rollout/hyps['n_rollouts']*100,
                                   time.time()-iter_start)
            print(s, end=len(s)//4*" " + "\r")
            if hyps['exp_name'] == "test" and rollout>=3: break
        print()
        train_avg_loss = avg_loss / hyps['n_rollouts']
        train_loc_loss = avg_loc_loss / hyps['n_rollouts']
        train_color_loss = avg_color_loss / hyps['n_rollouts']
        train_shape_loss = avg_shape_loss / hyps['n_rollouts']
        train_rew_loss = avg_rew_loss / hyps['n_rollouts']
        train_obj_loss = avg_obj_loss / hyps['n_rollouts']
        train_color_acc = avg_color_acc / hyps['n_rollouts']
        train_shape_acc = avg_shape_acc / hyps['n_rollouts']
        train_obj_acc = avg_obj_acc / hyps['n_rollouts']
        train_avg_rew = avg_rew / hyps['n_rollouts']
        train_fwd_loss = avg_fwd_loss / hyps['n_rollouts']

        first_train_loc_loss = first_avg_loc_loss / hyps['n_rollouts']
        first_train_color_loss=first_avg_color_loss / hyps['n_rollouts']
        first_train_shape_loss=first_avg_shape_loss / hyps['n_rollouts']
        first_train_rew_loss = first_avg_rew_loss / hyps['n_rollouts']
        first_train_obj_loss = first_avg_obj_loss / hyps['n_rollouts']
        first_train_color_acc =first_avg_color_acc / hyps['n_rollouts']
        first_train_shape_acc =first_avg_shape_acc / hyps['n_rollouts']
        first_train_obj_acc =  first_avg_obj_acc / hyps['n_rollouts']

        s = "Train- Loss:{:.5f} | Loc:{:.5f} | Rew:{:.5f}\n"
        s +="Train- Obj Loss:{:.5f} | Obj Acc:{:.5f} | Fwd:{:.5f}\n"
        stats_string = s.format(train_avg_loss, train_loc_loss,
                                                train_avg_rew,
                                                train_obj_loss,
                                                train_obj_acc,
                                                train_fwd_loss)
        # Sample images
        rand = int(np.random.randint(0,len(obsrs)))
        obs = obsrs[rand].permute(1,2,0).cpu().data.numpy()/6+0.5
        plt.imsave("imgs/sample"+str(epoch)+".png", obs)
        if fwd_dynamics:
            rand = int(np.random.randint(0,len(preds)))
            #obs = preds[rand].permute(1,2,0).cpu().data.numpy()/6+0.5
            obs = preds[rand].permute(1,2,0).cpu().data.numpy()
            obs = np.clip(obs, 0, 1)
            path = os.path.join(hyps['save_folder'],
                               "pred_sample"+str(epoch)+".png")
            plt.imsave(path, obs)
        print("Evaluating")
        done = True
        model.eval()
        val_runner.model = model
        with torch.no_grad():
            loss_tup = val_runner.rollout(0,validation=True,n_tsteps=200,
                                            fwd_model=fwd_model)
            loss_tup = [x.item() for x in loss_tup]
            val_loc_loss,val_color_loss,val_shape_loss = loss_tup[:3]
            val_rew_loss,val_color_acc,val_shape_acc = loss_tup[3:6]
            val_rew,val_fwd_loss = loss_tup[6:8]

            first_val_loc_loss,first_val_color_loss = loss_tup[8:10]
            first_val_shape_loss,first_val_rew_loss = loss_tup[10:12]
            first_val_color_acc,first_val_shape_acc = loss_tup[12:14]

            val_obj_loss = ((val_color_loss + val_shape_loss)/2)
            val_obj_acc = ((val_color_acc + val_shape_acc)/2)
            first_val_obj_loss = ((first_val_color_loss+\
                                   first_val_shape_loss)/2)
            first_val_obj_acc = ( (first_val_color_acc +\
                                   first_val_shape_acc)/2)
            temp = rew_alpha*val_loc_loss + (1-rew_alpha)*val_rew_loss
            val_loss = alpha*temp + (1-alpha)*val_obj_loss

        s = "Val- Loss:{:.5f} | Loc:{:.5f} | Rew:{:.5f}\n"
        s +="Val- Obj Loss:{:.5f} | Obj Acc:{:.5f} | Fwd:{:.5f}\n"
        stats_string += s.format(val_loss, val_loc_loss, val_rew,
                                                        val_obj_loss,
                                                        val_obj_acc,
                                                        val_fwd_loss)

        scheduler.step(train_avg_loss)
        optimizer.zero_grad()
        if fwd_dynamics:
            fwd_scheduler.step(train_fwd_loss)
            fwd_optim.zero_grad()
        save_dict = {
            "epoch":epoch,
            "hyps":hyps,

            "train_loss":train_avg_loss,
            "train_loc_loss":train_loc_loss,
            "train_color_loss": train_color_loss,
            "train_shape_loss": train_shape_loss,
            "train_rew_loss": train_rew_loss,
            "train_obj_loss":train_obj_loss,
            "train_color_acc": train_color_acc,
            "train_shape_acc": train_shape_acc,
            "train_obj_acc":train_obj_acc,
            "train_fwd_loss":train_fwd_loss,

            "first_train_loc_loss":   first_train_loc_loss,
            "first_train_color_loss": first_train_color_loss,
            "first_train_shape_loss": first_train_shape_loss,
            "first_train_rew_loss":   first_train_rew_loss,
            "first_train_obj_loss":   first_train_obj_loss,
            "first_train_color_acc":  first_train_color_acc,
            "first_train_shape_acc":  first_train_shape_acc,
            "first_train_obj_acc":    first_train_obj_acc,

            "val_loss":val_loss,
            "val_loc_loss":val_loc_loss,
            "val_color_loss": val_color_loss,
            "val_shape_loss": val_shape_loss,
            "val_rew_loss": val_rew_loss,
            "val_obj_loss":val_obj_loss,
            "val_color_acc": val_color_acc,
            "val_shape_acc": val_shape_acc,
            "val_obj_acc":val_obj_acc,
            "val_fwd_loss":val_fwd_loss,

            "first_val_loc_loss":   first_val_loc_loss,
            "first_val_color_loss": first_val_color_loss,
            "first_val_shape_loss": first_val_shape_loss,
            "first_val_rew_loss":   first_val_rew_loss,
            "first_val_obj_loss":   first_val_obj_loss,
            "first_val_color_acc":  first_val_color_acc,
            "first_val_shape_acc":  first_val_shape_acc,
            "first_val_obj_acc":    first_val_obj_acc,

            "train_rew":train_avg_rew,
            "val_rew":val_rew,
            "state_dict":model.state_dict(),
            "optim_dict":optimizer.state_dict(),
        }
        if fwd_dynamics:
            save_dict['fwd_state_dict'] = fwd_model.state_dict()
            save_dict['fwd_optim_dict'] = fwd_optim.state_dict()
        save_name = "checkpt"
        save_name = os.path.join(hyps['save_folder'],save_name)
        io.save_checkpt(save_dict, save_name, epoch, ext=".pt",
                                   del_prev_sd=hyps['del_prev_sd'],
                                   best=(val_rew>best_val_rew))
        best_val_rew = max(val_rew, best_val_rew)
        stats_string += "Exec time: {}\n".format(time.time()-starttime)
        print(stats_string)
        s = "Epoch:{} | Model:{}\n".format(epoch, hyps['save_folder'])
        stats_string = s + stats_string
        log_file = os.path.join(hyps['save_folder'],"training_log.txt")
        with open(log_file,'a') as f:
            if epoch==0:
                dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                f.write(dt_string+"\n\n")
            f.write(str(stats_string)+'\n')
    del save_dict['state_dict']
    del save_dict['optim_dict']
    del save_dict['hyps']
    if 'fwd_state_dict' in save_dict:
        del save_dict['fwd_state_dict']
        del save_dict['fwd_optim_dict']
    save_dict['save_folder'] = hyps['save_folder']
    env.close()
    del env
    end_q.put(1)
    if len(runners) > 1:
        # Stop the runners
        for i in range(hyps['n_runs']):
            gate_q.put(i)
        for proc in procs:
            proc.join()
    time.sleep(5) # Sleeping performed to let envs power down
    return save_dict

class Runner:
    def __init__(self, rank, hyps, shared_data, gate_q, stop_q, end_q):
        """
        rank: int
            the id of the runner
        hyps: dict
            dict of hyperparams
        shared_data: dict of shared tensors
            keys: str
                obsrs: shared_tensors
                targs: shared_tensors
        gate_q: multi processing Queue
            a signalling q for collection
        stop_q: multi processing Queue
            a signalling q to indicate the data has been collected
        end_q: multi processing Queue
            a signalling q to indicate that the training is complete
        """
        self.rank = rank
        self.hyps = hyps
        self.shared_data = shared_data
        self.gate_q = gate_q
        self.stop_q = stop_q
        self.end_q = end_q
        self.env = None
        self.prev_h = None
        self.fwd_hs = None

    def run(self, model, multi_proc=True):
        """
        Call this function for starting the process
        """
        self.model = model
        if self.env is None:
            self.hyps['seed'] = self.hyps['seed'] + self.rank
            self.env = environments.get_env(self.hyps)
            print("env made rank:", self.rank)
            self.stop_q.put(self.rank)
        if multi_proc:
            while self.end_q.empty():
                idx = self.gate_q.get() # Opened from main process
                _ = self.rollout(idx)
                # Signals to main process that data has been collected
                self.stop_q.put(idx)
            self.env.close()
        else:
            self.rollout(0)

    def rollout(self, idx, validation=False, n_tsteps=None,
                                             fwd_model=None):
        """
        rollout handles the actual rollout of the environment for
        n steps in time. It is called from run and performs a single
        rollout, placing the collected data into the shared lists
        found in the datas dict.

        idx: int
            identification number distinguishing the portion of the
            shared array designated for this runner
        validation: bool
            if true, this runner is used as a validation runner
        n_tsteps: int
            number of steps to take in this rollout
        fwd_model: (optional) torch Module
            the fwd dynamics model. Only used in validation mode
        """
        hyps = self.hyps
        obj_recog = try_key(hyps,'obj_recog',False)
        rew_recog = try_key(hyps,'rew_recog',False)
        post_obj_preds = try_key(hyps,'post_obj_preds',False)
        post_rew_preds = try_key(hyps,'post_rew_preds',False)
        n_tsteps = hyps['n_tsteps'] if n_tsteps is None else n_tsteps

        self.model.reset_h(batch_size=1)
        # Prev h will only be None if this is the first rollout of the
        # training. If we ended on a done in the last session, the env
        # hasn't been restarted yet. So, we can reset here.
        if self.prev_h is None or self.prev_done:
            self.prev_obs,self.prev_targ = self.env.reset()
            self.prev_rew = 0
            self.prev_done = 0
            self.prev_start = 1
            reset_fwd_hs = [1]
        else:
            self.model.h = self.prev_h
            reset_fwd_hs = [0]
        obs = self.prev_obs.cuda()

        obsrs = [obs]
        hs = [self.model.h]
        targs = [self.prev_targ]
        rews  = [self.prev_rew]
        dones = [0] # Will never be 1 due to reset a few lines above
        # Easiest to mark this as a start for indexing. But means we
        # need to be careful and use done signals in the bptt.
        # Otherwise we'll restart the h vector when we don't want to
        starts = [1]

        loc_preds = []
        color_preds = []
        shape_preds = []
        rew_preds = []

        with torch.no_grad():
            while len(rews) < n_tsteps:
                temp = targs[-1].squeeze()[None].long()
                color_idx=torch.LongTensor(temp[:,2:3])
                shape_idx=torch.LongTensor(temp[:,3:4])
                if temp.shape[1]>=5:
                    number_idx = torch.LongTensor(temp[:,4:5]).cuda()
                else:
                    number_idx = None
                tup = self.model(obs[None], None, color_idx.cuda(),
                                                  shape_idx.cuda(),
                                                  number_idx)
                pred,color_pred,shape_pred,rew_pred = tup

                obs,targ,rew,done,_ = self.env.step(pred)
                start = 0
                done = int(done)
                obs = obs.cuda()

                loc_preds.append(pred)
                if len(color_pred)>0:
                    color_preds.append(color_pred)
                    shape_preds.append(shape_pred)
                if rew_recog:
                    rew_preds.append(rew_pred)

                obsrs.append(obs)
                hs.append(self.model.h)
                reset_fwd_hs.append(0)
                targs.append(targ)
                rews.append(rew)
                starts.append(start)
                dones.append(done)

                if done>0 and len(rews)<n_tsteps:
                    # Finish out last step
                    temp = targs[-1].squeeze()[None].long()
                    color_idx=torch.LongTensor(temp[:,2:3])
                    shape_idx=torch.LongTensor(temp[:,3:4])
                    if temp.shape[1]>=5:
                        number_idx = torch.LongTensor(temp[:,4:5]).cuda()
                    else:
                        number_idx = None
                    tup = self.model(obs[None], None, color_idx.cuda(),
                                                      shape_idx.cuda(),
                                                      number_idx)
                    pred,color_pred,shape_pred,rew_pred = tup
                    loc_preds.append(pred)
                    if len(color_pred)>0:
                        color_preds.append(color_pred)
                        shape_preds.append(shape_pred)
                    if rew_recog:
                        rew_preds.append(rew_pred)

                    obs,targ = self.env.reset()
                    rew = 0
                    done = 0
                    start = 1
                    self.model.reset_h()
                    obs = obs.cuda()

                    obsrs.append(obs)
                    hs.append(self.model.h)
                    reset_fwd_hs.append(1)
                    targs.append(targ)
                    rews.append(rew)
                    starts.append(start)
                    dones.append(done)
        dones[-1] = 1

        self.prev_h = self.model.h
        self.prev_obs = obs
        self.prev_targ = targ
        self.prev_rew = rew
        self.prev_done = int(done)
        self.prev_start = start

        rews = torch.FloatTensor(rews).cuda()
        hs = torch.vstack(hs).cuda()
        dones = torch.LongTensor(dones).cuda()
        starts = torch.LongTensor(starts).cuda()
        obsrs = torch.stack(obsrs)
        targs = torch.stack(targs).cuda()
        reset_fwd_hs = torch.LongTensor(reset_fwd_hs).cuda()
        loc_targs,obj_targs = targs[:,:2],targs[:,2:4].long()
        counts = None
        if targs.shape[1] > 4:
            counts = targs[:,4:5].long()

        if not validation:
            # Send data to main proc
            startx = idx*n_tsteps
            endx = (idx+1)*n_tsteps
            self.shared_data['rews'][startx:endx] = rews
            self.shared_data['hs'][startx:endx] = hs
            self.shared_data['dones'][startx:endx] = dones
            self.shared_data['starts'][startx:endx] = starts
            self.shared_data['obsrs'][startx:endx] = obsrs
            self.shared_data['loc_targs'][startx:endx] = loc_targs
            self.shared_data['obj_targs'][startx:endx] = obj_targs
            self.shared_data['reset_fwd_hs'][startx:endx] = reset_fwd_hs
            if counts is not None:
                self.shared_data['counts'][startx:endx] = counts

        if validation:
            color_idx = obj_targs[-1:,:1]
            shape_idx = obj_targs[-1:,1:2]
            number_idx = None
            if counts is not None:
                number_idx = counts[-1:].cuda()
            tup = self.model(obs[None], None, color_idx.cuda(),
                                              shape_idx.cuda(),
                                              number_idx)
            pred,color_pred,shape_pred,rew_pred = tup
            loc_preds.append(pred)
            loc_preds = torch.vstack(loc_preds)
            if len(color_pred)>0:
                color_preds.append(color_pred)
                shape_preds.append(shape_pred)
                color_preds = torch.vstack(color_preds)
                shape_preds = torch.vstack(shape_preds)
            if rew_recog:
                rew_preds.append(rew_pred)
                rew_preds = torch.vstack(rew_preds)

            color_targs,shape_targs = obj_targs[:,0],obj_targs[:,1]

            loss_tup = calc_losses(
                            loc_preds,color_preds,shape_preds,rew_preds,
                            loc_targs,color_targs,shape_targs,rews,
                            starts,dones,
                            post_obj_preds=post_obj_preds,
                            post_rew_preds=post_rew_preds)
            loc_loss,color_loss,shape_loss,rew_loss = loss_tup[:4]
            color_acc,shape_acc = loss_tup[4:6]
            first_loc_loss,first_color_loss = loss_tup[6:8]
            first_shape_loss, first_rew_loss = loss_tup[8:10]
            first_color_acc,first_shape_acc = loss_tup[10:12]

            fwd_loss = torch.zeros(1)
            if fwd_model is not None:
                preds,self.fwd_hs = fwd_preds(hyps,fwd_model,
                                          obsrs=obsrs,
                                          hs=self.fwd_hs,
                                          resets=reset_fwd_hs,
                                          obj_targs=obj_targs,
                                          counts=counts)
                fwd_loss = calc_fwd_loss(preds,obsrs,starts,dones)

            return loc_loss,color_loss,shape_loss,rew_loss,\
                    color_acc,shape_acc,rews.mean(),fwd_loss,\
                    first_loc_loss,first_color_loss,first_shape_loss,\
                    first_rew_loss,first_color_acc,first_shape_acc

def calc_losses(loc_preds,color_preds,shape_preds,rew_preds,
                loc_targs,color_targs,shape_targs,rew_targs,
                starts,dones,
                post_obj_preds=False, post_rew_preds=False,
                hyps=None, firsts=None):
    """
    loc_preds: FloatTensor (N,2)
        the location predictions
    color_preds: FloatTensor (N, N_COLORS)
        the color predictions
    shape_preds: FloatTensor (N, N_SHAPES)
        the shape predictions
    loc_targs: FloatTensor (N,2)
        the location targets
    color_targs: LongTensor (N,)
        the color targets
    shape_targs: LongTensor (N,)
        the shape targets
    dones: FloatTensor (N,)
        the done signals
    starts: FloatTensor (N,)
        the start signals
    post_obj_preds: bool
        if the object predictions come after the movement
    post_rew_preds: bool
        if the rew predictions come after the movement
    firsts: torch LongTensor (N,)
        Binary array denoting the indices in which a fresh h should be
        used. aka the real start of an episode. starts can be misleading
        because they can simply indicate where a model started in again
        in a partially completed episode.
    """
    d_idxs = (1-dones).bool()
    s_idxs = (1-starts).bool()

    # Loc Loss
    l_preds = loc_preds[d_idxs]
    l_targs = loc_targs[s_idxs]
    # HEADS UP: Added a multiplcation factor of 10
    loc_loss = 10*F.mse_loss(l_preds.cuda(), l_targs.cuda())
    if firsts is not None:
        assert hyps is not None, "if using firsts, must argue hyps"
        n_runs = hyps['n_runs']
        n_tsteps = hyps['n_tsteps']
        b_size = n_runs*n_tsteps
        firsts = firsts.reshape(n_runs,n_tsteps).clone()
        firsts[:,-1] = 0 # don't care about ending firsts
        firsts = firsts.reshape(-1)
        roll = torch.roll(firsts,shifts=1,dims=0)
        with torch.no_grad():
            l_preds = loc_preds[firsts]
            l_targs = loc_targs[roll]
            first_loc_loss = 10*F.mse_loss(l_preds,l_targs)
    else:
        first_loc_loss = torch.zeros(1).cuda()

    if len(color_preds) > 0:
        idxs = d_idxs
        if post_obj_preds:
            idxs = s_idxs
        c_preds = color_preds[idxs]
        s_preds = shape_preds[idxs]
        c_targs = color_targs[s_idxs]
        s_targs = shape_targs[s_idxs]

        color_loss = F.cross_entropy(c_preds, c_targs)
        shape_loss = F.cross_entropy(s_preds, s_targs)
        with torch.no_grad():
            maxes = torch.argmax(c_preds,dim=-1)
            color_acc = (maxes==c_targs).float().mean()
            maxes = torch.argmax(s_preds,dim=-1)
            shape_acc = (maxes==s_targs).float().mean()
        if firsts is not None:
            with torch.no_grad():
                c_preds = color_preds[firsts]
                s_preds = shape_preds[firsts]
                if post_obj_preds:
                    c_targs = color_targs[roll]
                    s_targs = shape_targs[roll]
                else:
                    c_targs = color_targs[firsts]
                    s_targs = shape_targs[firsts]
                first_color_loss = F.cross_entropy(c_preds,c_targs)
                first_shape_loss = F.cross_entropy(s_preds,s_targs)
                maxes = torch.argmax(c_preds,dim=-1)
                first_color_acc = (maxes==c_targs).float().mean()
                maxes = torch.argmax(s_preds,dim=-1)
                first_shape_acc = (maxes==s_targs).float().mean()
    else:
        color_loss = torch.zeros(1).cuda()
        shape_loss = torch.zeros(1).cuda()
        color_acc = torch.zeros(1)
        shape_acc = torch.zeros(1)
        first_color_loss = torch.zeros(1).cuda()
        first_shape_loss = torch.zeros(1).cuda()
        first_color_acc = torch.zeros(1)
        first_shape_acc = torch.zeros(1)
    if len(rew_preds) > 0:
        idxs = d_idxs
        if post_rew_preds:
            idxs = s_idxs
        r_preds = rew_preds[d_idxs]
        r_targs = rew_targs[s_idxs]
        rew_loss = F.mse_loss(r_preds.squeeze().cuda(),
                              r_targs.squeeze().cuda())
        if firsts is not None:
            with torch.no_grad():
                r_preds = rew_preds[firsts]
                if post_obj_preds:
                    r_targs = rew_targs[roll]
                else:
                    r_targs = rew_targs[firsts]
                first_rew_loss = F.mse_loss(r_preds.squeeze().cuda(),
                                      r_targs.squeeze().cuda())
    else:
        rew_loss = torch.zeros(1).cuda()
        first_rew_loss = torch.zeros(1).cuda()

    return loc_loss,color_loss,shape_loss,rew_loss,color_acc,shape_acc,\
            first_loc_loss,first_color_loss,first_shape_loss,\
            first_rew_loss,first_color_acc,first_shape_acc

def bptt(hyps, model, obsrs, hs, dones, obj_targs, counts):
    """
    Used to include dependencies over time. It is assumed each rollout
    is of fixed length.

    R = Number of runs
    N = Number of steps per run

    obsrs: torch FloatTensor (R*N,C,H,W)
        MDP states at each timestep t
    hs: FloatTensor (R*N,H)
        Recurrent states at timestep t
    dones: torch LongTensor (R*N,)
        Binary array denoting the indices at the end of an episode
    obj_targs: long tensor (R*N,2)
        the color and shape indexes
    counts: long tensor (R*N,1)
        the indices of the number of objects to touch
    """
    n_runs = hyps['n_runs']
    n_tsteps = hyps['n_tsteps']
    b_size = n_runs*n_tsteps
    assert len(obsrs) == b_size

    obsrs = obsrs.reshape(n_runs,n_tsteps,*obsrs.shape[1:])
    dones = dones.reshape(n_runs,n_tsteps,1)
    resets = 1-dones
    h_inits = model.reset_h(batch_size=n_runs)
    model.h = hs.reshape(n_runs,n_tsteps,-1)[:,0]
    color_idxs = obj_targs[:,:1].reshape(n_runs,n_tsteps,1)
    shape_idxs = obj_targs[:,1:].reshape(n_runs,n_tsteps,1)
    number_idxs = counts.reshape(n_runs,n_tsteps,1)
    loc_preds = []
    color_preds = []
    shape_preds = []
    rew_preds = []
    h = model.h
    for i in range(n_tsteps):
        obs = obsrs[:,i]
        color_idx = color_idxs[:,i]
        shape_idx = shape_idxs[:,i]
        number_idx = number_idxs[:,i]
        loc_pred,color_pred,shape_pred,rew_pred = model(obs.cuda(), h,
                                                        color_idx,
                                                        shape_idx,
                                                        number_idx)
        loc_preds.append(loc_pred)
        color_preds.append(color_pred)
        shape_preds.append(shape_pred)
        rew_preds.append(rew_pred)
        # Might be tempting to use starts here, BUT DON"T DO IT
        # The start indices are unreliable due to being used as markers
        # for the start of recording an episode, not necessarily the
        # real start of a new episode
        h = model.h
        if dones[:,i].sum() > 0:
            #h = h*resets[:,i].data+h_inits.data*dones[:,i].data
            new_hs = []
            for j in range(len(dones)):
                new_h = h[j] if dones[j,i] < 1 else h_inits[j].data
                new_hs.append(new_h)
            h = torch.stack(new_hs)
    shape = (b_size, loc_pred.shape[-1])
    loc_preds = torch.stack(loc_preds,dim=1).reshape(shape)
    if model.obj_recog and not model.aud_targs:
        shape = (b_size,color_pred.shape[-1])
        color_preds = torch.stack(color_preds,dim=1).reshape(shape)
        shape = (b_size,shape_pred.shape[-1])
        shape_preds = torch.stack(shape_preds,dim=1).reshape(shape)
    else:
        color_preds,shape_preds = [],[]
    if model.rew_recog:
        rew_preds = torch.stack(rew_preds,dim=1).reshape(b_size)
    else:
        rew_preds = []
    return loc_preds, color_preds, shape_preds, rew_preds

def fwd_preds(hyps, fwd_model, obsrs, hs, resets, obj_targs, counts):
    """
    Used to include dependencies over time. It is assumed each rollout
    is of fixed length.

    R = Number of runs
    N = Number of steps per run

    obsrs: torch FloatTensor (R*N,C,H,W)
        MDP states at each timestep t
    hs: FloatTensor (R,H)
        Recurrent states at timestep t
    resets: torch LongTensor (R*N,)
        Binary array denoting the indices in which a fresh h should be
        used
    obj_targs: long tensor (R*N,2)
        the color and shape indexes
    counts: long tensor (R*N,1)
        the indices of the number of objects to touch
    """
    n_runs = hyps['n_runs']
    n_tsteps = hyps['n_tsteps']
    b_size = n_runs*n_tsteps
    # Case of validation
    if len(obsrs) != b_size:
        n_runs = 1
        n_tsteps = len(obsrs)
        b_size = n_runs*n_tsteps

    obsrs = obsrs.reshape(n_runs,n_tsteps,*obsrs.shape[1:])
    resets = resets.reshape(n_runs,n_tsteps,1)
    h_inits = fwd_model.reset_h(batch_size=n_runs)
    if hs is None: hs = fwd_model.reset_h(batch_size=n_runs)
    fwd_model.h = hs
    color_idxs = obj_targs[:,:1].reshape(n_runs,n_tsteps,1)
    shape_idxs = obj_targs[:,1:].reshape(n_runs,n_tsteps,1)
    number_idxs = counts.reshape(n_runs,n_tsteps,1)
    preds = []
    h = fwd_model.h
    for i in range(n_tsteps):
        # Might be tempting to use starts or dones here, BUT DON"T DO IT
        # The start indices are unreliable due to being used as markers
        # for the start of recording an episode, not necessarily the
        # real start of a new episode. The dones are unreliable because
        # of the way the runners reset the environment only if the done
        # does not occur on the final step of the rollout. Use the
        # reset_fwd_hs vector
        if resets[:,i].sum() > 0:
            new_hs = []
            for j in range(len(resets)):
                new_h = h[j] if resets[j,i] < 1 else h_inits[j].data
                new_hs.append(new_h)
            h = torch.stack(new_hs)
        obs = obsrs[:,i]
        color_idx = color_idxs[:,i]
        shape_idx = shape_idxs[:,i]
        number_idx = number_idxs[:,i]
        pred = fwd_model(obs.cuda(), h, color_idx=color_idx,
                                 shape_idx=shape_idx,
                                 number_idx=number_idx)
        preds.append(pred)
        h = fwd_model.h
    shape = (b_size, *pred.shape[1:])
    preds = torch.stack(preds,dim=1).reshape(shape)
    return preds, h.data

def calc_fwd_loss(preds, obsrs, starts, dones):
    """
    A function to calculate the fwd dynamics loss
    
    preds: torch Float Variable (B,C,H,W)
    obsrs: torch FloatTensor (B,C,H,W)
    starts: torch LongTensor (B,)
    dones: torch LongTensor (B,)
    """
    idxs = (1-dones.squeeze()).bool()
    preds = preds[idxs]
    idxs = (1-starts.squeeze()).bool()
    targs = obsrs[idxs]
    targs = targs/6+0.5 # forced between 0 and 1
    return 10*F.mse_loss(preds.cuda(), targs.cuda())

