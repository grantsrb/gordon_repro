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
    checkpt,hyps = get_resume_checkpt(hyps,verbose=verbose) #incrs seed
    if checkpt is None:
        hyps['exp_num']=get_exp_num(hyps['main_path'], hyps['exp_name'])
        hyps['save_folder'] = get_save_folder(hyps)
    if not os.path.exists(hyps['save_folder']):
        os.mkdir(hyps['save_folder'])
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
    model_class = hyps['model_class']
    hyps['n_loss_loops'] = try_key(hyps,'n_loss_loops',1)

    if verbose:
        print("Making Env(s)")
    hyps['n_runners'] = try_key(hyps,'n_runners',1)
    env = environments.UnityGymEnv(**hyps)

    hyps["img_shape"] = env.shape
    hyps["targ_shape"] = env.targ_shape

    if verbose:
        print("Making model")
    model = getattr(models,model_class)(**hyps)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'],
                                           weight_decay=hyps['l2'])
    if checkpt is not None:
        if verbose:
            print("Loading state dicts from", hyps['save_folder'])
        model.load_state_dict(checkpt["state_dict"])
        optimizer.load_state_dict(checkpt["optim_dict"])
    lossfxn = getattr(nn,try_key(hyps,'lossfxn',"MSELoss"))()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                                                    patience=6,
                                                    verbose=True)
    batch_size = hyps['batch_size']

    ## Multi Processing
    # The number of actual runs performed
    hyps['n_runs'] = try_key(hyps,'n_runs',1)
    # The number of steps for each run
    hyps['n_tsteps'] = hyps['batch_size']//hyps['n_runs']
    # The total number of steps included in the update
    hyps['batch_size'] = hyps['n_tsteps']*hyps['n_runs']
    shared_data = {
            'rews': torch.zeros(hyps['n_runs']).share_memory_(),
            'losses': torch.zeros(hyps['n_runs']).share_memory_(),
            'loc_losses':torch.zeros(hyps['n_runs']).share_memory_(),
            'rew_losses':torch.zeros(hyps['n_runs']).share_memory_(),
            'obj_losses':torch.zeros(hyps['n_runs']).share_memory_(),
            'color_losses':torch.zeros(hyps['n_runs']).share_memory_(),
            'shape_losses':torch.zeros(hyps['n_runs']).share_memory_(),
            'obj_accs':torch.zeros(hyps['n_runs']).share_memory_(),
            'color_accs':torch.zeros(hyps['n_runs']).share_memory_(),
            'shape_accs':torch.zeros(hyps['n_runs']).share_memory_(),
            }

    gate_q = mp.Queue(hyps['n_runs'])
    stop_q = mp.Queue(hyps['n_runs'])
    reward_q = mp.Queue(1)

    # Make Runners
    hyps['n_runners'] = try_key(hyps,'n_runners',None)
    if hyps['n_runners'] is None:
        hyps['n_runners'] = hyps['n_runs']
    runners = []
    for i in range(hyps['n_runners']):
        runner = Runner(rank=i, hyps=hyps, shared_data=shared_data,
                                           gate_q=gate_q,
                                           stop_q=stop_q)
        runners.append(runner)
    if len(runners) > 1:
        procs = []
        for i in range(len(runners)):
            proc = mp.Process(target=runners[i].run, args=(model,True))
            procs.append(proc)
            proc.start()
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

    alpha = try_key(hyps,'rew_alpha',.7)
    obj_recog = try_key(hyps,'obj_recog',False)
    best_val_rew = -np.inf
    print()
    while epoch < hyps['n_epochs']:
        epoch += 1
        print("Epoch:{} | Model:{}".format(epoch, hyps['save_folder']))
        starttime = time.time()
        avg_loss = 0
        avg_rew = 0
        avg_loc_loss = 0
        avg_rew_loss = 0
        avg_obj_loss = 0
        avg_color_loss = 0
        avg_color_acc = 0
        avg_shape_loss = 0
        avg_shape_acc = 0
        avg_obj_acc = 0
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
                for i in range(hyps['n_runners']):
                    gate_q.put(i)
                # Wait for all runners to stop
                for i in range(hyps['n_runners']):
                    stop_q.get()
            else:
                runner.run(model,multi_proc=False)

            avg_loss +=     shared_data['losses'].mean()
            avg_rew +=      shared_data['rews'].mean()
            avg_loc_loss += shared_data['loc_losses'].mean().item()
            avg_rew_loss += shared_data['rew_losses'].mean().item()

            avg_obj_loss += shared_data['obj_losses'].mean().item()
            avg_color_loss+= shared_data['color_losses'].mean().item()
            avg_shape_loss+= shared_data['shape_losses'].mean().item()
            avg_color_acc += shared_data['color_accs'].mean().item()
            avg_shape_acc += shared_data['shape_accs'].mean().item()
            avg_obj_acc   += shared_data['obj_accs'].mean().item()

            if rollout % hyps['n_loss_loops'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            s = "LocL:{:.5f} | RewL:{:.5f} |" 
            s +=" Obj:{:.5f} | {:.0f}% | t:{:.2f}"
            div = (rollout+1)
            s= s.format(avg_loc_loss/div,avg_rew_loss/div,
                                         avg_obj_loss/div,
                                         rollout/hyps['n_rollouts']*100,
                                         time.time()-iter_start)
            print(s, end=len(s)*" " + "\r")
            if hyps['exp_name'] == "test" and rollout>=3: break
        print()
        train_avg_loss = avg_loss / hyps['n_rollouts']
        train_loc_loss = avg_loc_loss / hyps['n_rollouts']
        train_rew_loss = avg_rew_loss / hyps['n_rollouts']
        train_color_loss = avg_color_loss / hyps['n_rollouts']
        train_shape_loss = avg_shape_loss / hyps['n_rollouts']
        train_obj_loss = avg_obj_loss / hyps['n_rollouts']
        train_color_acc = avg_color_acc / hyps['n_rollouts']
        train_shape_acc = avg_shape_acc / hyps['n_rollouts']
        train_obj_acc = avg_obj_acc / hyps['n_rollouts']
        train_avg_rew = avg_rew / hyps['n_rollouts']

        s = "Train - Loss:{:.5f} | Loc:{:.5f} | "
        s += "RewLoss:{:.5f} | Rew:{:.5f}\n"
        s += "Obj Loss:{:.5f} | Obj Acc:{:.5f}\n"
        stats_string = s.format(train_avg_loss, train_loc_loss,
                                                train_rew_loss,
                                                train_avg_rew,
                                                train_obj_loss,
                                                train_obj_acc)
        scheduler.step(train_avg_loss)

        print("Evaluating")
        done = True
        model.eval()
        n_eps = -1
        n_loops = 0
        targs = []
        preds = []
        rews = []
        rew_preds = []
        color_preds = [] if try_key(hyps,'obj_recog',False) else None
        shape_preds = [] if try_key(hyps,'obj_recog',False) else None
        sum_loss = 0
        sum_rew = 0
        sum_loc_loss = 0
        sum_rew_loss = 0
        sum_obj_loss = 0
        sum_obj_acc = 0
        with torch.no_grad():
            while n_eps < 20:
                if done:
                    obs,_ = env.reset()
                    model.reset_h()
                    n_eps += 1
                tup = model(obs[None].to(DEVICE))
                pred,rew_pred,color_pred,shape_pred = tup
                obs,targ,rew,done,_ = env.step(pred)
                sum_rew += rew
                n_loops += 1

                preds.append(pred)
                rew_preds.append(rew_pred)
                rews.append(rew)
                if obj_recog:
                    color_preds.append(color_pred)
                    shape_preds.append(shape_pred)
                targs.append(targ)

            preds = torch.stack(preds).squeeze()
            targs = torch.stack(targs)
            targs,obj_targs = targs[:,:2],targs[:,2:]
            val_loc_loss = lossfxn(preds,targs.to(DEVICE)).item()

            rew_preds = torch.stack(rew_preds)
            rews = torch.FloatTensor(rews)
            val_rew_loss = lossfxn(rew_preds.squeeze(),rews.to(DEVICE))
            val_rew_loss = val_rew_loss.item()

            if obj_recog:
                obj_targs = obj_targs.long().to(DEVICE)
                color_preds = torch.stack(color_preds).squeeze()
                val_color_loss = F.cross_entropy(color_preds,
                                             obj_targs[:,0])
                shape_preds = torch.stack(shape_preds).squeeze()
                val_shape_loss = F.cross_entropy(shape_preds,
                                             obj_targs[:,1])
                val_obj_loss = color_loss + shape_loss

                maxes = torch.argmax(color_preds,dim=-1)
                val_color_acc = (maxes==obj_targs[:,0]).float().mean()
                maxes = torch.argmax(shape_preds,dim=-1)
                val_shape_acc = (maxes==obj_targs[:,1]).float().mean()
                val_obj_acc = ((color_acc + shape_acc)/2).item()

            else:
                val_obj_loss = torch.zeros(1).to(DEVICE)
                val_color_loss = torch.zeros(1).to(DEVICE)
                val_shape_loss = torch.zeros(1).to(DEVICE)
                val_color_acc = 0
                val_shape_acc = 0
                val_obj_acc = 0

            val_loss = alpha*(val_loc_loss+val_obj_loss)
            val_loss = val_loss+(1-alpha)*val_rew_loss
            val_loss = val_loss / hyps['n_loss_loops']
        val_rew = sum_rew/n_loops
        stats_string += "Evaluation Avg Rew: {:.5f}\n".format(val_rew)

        optimizer.zero_grad()
        save_dict = {
            "epoch":epoch,
            "hyps":hyps,

            "train_loss":train_avg_loss,
            "train_loc_loss":train_loc_loss,
            "train_rew_loss":train_rew_loss,
            "train_color_loss": train_color_loss,
            "train_shape_loss": train_shape_loss,
            "train_obj_loss":train_obj_loss,
            "train_color_acc": train_color_acc,
            "train_shape_acc": train_shape_acc,
            "train_obj_acc":train_obj_acc,

            "val_loss":val_loss,
            "val_loc_loss":val_loc_loss,
            "val_rew_loss":val_rew_loss,
            "val_color_loss": val_color_loss,
            "val_shape_loss": val_shape_loss,
            "val_obj_loss":val_obj_loss,
            "val_color_acc": val_color_acc,
            "val_shape_acc": val_shape_acc,
            "val_obj_acc":val_obj_acc,

            "train_rew":train_avg_rew,
            "val_rew":val_rew,
            "state_dict":model.state_dict(),
            "optim_dict":optimizer.state_dict(),
        }
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
            f.write(str(stats_string)+'\n')
    del save_dict['state_dict']
    del save_dict['optim_dict']
    del save_dict['hyps']
    save_dict['save_folder'] = hyps['save_folder']
    env.close()
    del env
    return save_dict

class Runner:
    def __init__(self, rank, hyps, shared_data, gate_q, stop_q):
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
        """
        self.rank = rank
        self.hyps = hyps
        self.shared_data = shared_data
        self.gate_q = gate_q
        self.stop_q = stop_q
        self.env = None

    def run(self, model, multi_proc=True):
        """
        Call this function for starting the process
        """
        self.model = model
        lossfxn_name = try_key(self.hyps,'lossfxn',"MSELoss")
        self.lossfxn = getattr(nn,lossfxn_name)()
        if self.env is None:
            self.hyps['seed'] = self.hyps['seed'] + self.rank
            self.env = environments.UnityGymEnv(**self.hyps)
            print("env made rank:", self.rank)
        if multi_proc:
            while True:
                idx = self.gate_q.get() # Opened from main process
                self.rollout(idx)
                # Signals to main process that data has been collected
                self.stop_q.put(idx)
        else:
            self.rollout(0)

    def rollout(self, idx):
        """
        rollout handles the actual rollout of the environment for
        n steps in time. It is called from run and performs a single
        rollout, placing the collected data into the shared lists
        found in the datas dict.

        idx: int
            identification number distinguishing the portion of the
            shared array designated for this runner
        """
        hyps = self.hyps
        n_tsteps = hyps['n_tsteps']
        self.model.reset_h()
        obs,targ = self.env.reset()
        obsrs = [obs]
        targs = []
        preds = []
        rew_preds = []
        obj_recog = try_key(hyps,'obj_recog',False)
        color_preds = [] if obj_recog else None
        shape_preds = [] if obj_recog else None
        rews  = []
        dones = []
        alpha = try_key(hyps,'rew_alpha',.7)
        while len(rews) < n_tsteps:
            tup = self.model(obs[None].to(DEVICE))
            pred,rew_pred,color_pred,shape_pred = tup
            preds.append(pred)
            rew_preds.append(rew_pred)
            if obj_recog:
                color_preds.append(color_pred)
                shape_preds.append(shape_pred)
            obs,targ,rew,done,_ = self.env.step(pred)
            rews.append(rew)
            dones.append(done)
            targs.append(targ)
            if done:
                obs,_ = self.env.reset()
                self.model.reset_h()
            obsrs.append(obs)
        # Calc Loss
        preds = torch.stack(preds).squeeze()
        targs = torch.stack(targs)
        targs,obj_targs = targs[:,:2],targs[:,2:]
        loc_loss = self.lossfxn(preds,targs.to(DEVICE))
        rew_preds = torch.stack(rew_preds)
        rews = torch.FloatTensor(rews)
        rew_loss = self.lossfxn(rew_preds.squeeze(),rews.to(DEVICE))

        if obj_recog:
            obj_targs = obj_targs.long().to(DEVICE)
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
        else:
            obj_loss = torch.zeros(1).to(DEVICE)
            color_loss = torch.zeros(1).to(DEVICE)
            shape_loss = torch.zeros(1).to(DEVICE)
            color_acc = 0
            shape_acc = 0
            obj_acc = 0

        loss = alpha*(loc_loss+obj_loss) + (1-alpha)*rew_loss
        loss = loss / hyps['n_loss_loops'] / hyps['n_runs']
        loss.backward()

        startx = idx*n_tsteps
        endx = (idx+1)*n_tsteps
        self.shared_data['rews'][idx] = rews.mean()
        self.shared_data['losses'][idx] = loss
        self.shared_data['loc_losses'][idx] = loc_loss
        self.shared_data['rew_losses'][idx] = rew_loss

        self.shared_data['obj_losses'][idx] = obj_loss
        self.shared_data['color_losses'][idx] = color_loss
        self.shared_data['color_accs'][idx] = color_acc
        self.shared_data['shape_losses'][idx] = shape_loss
        self.shared_data['shape_accs'][idx] = shape_acc
        self.shared_data['obj_accs'][idx] = obj_acc
