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

def train(rank, hyps, verbose=True):
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
    env = environments.get_env(hyps)

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
            'rews': torch.zeros(hyps['batch_size']),
            "hs":torch.zeros(hyps['batch_size'],model.h_shape[-1]),
            "starts":torch.zeros(hyps['batch_size']).long(),
            "dones":torch.zeros(hyps['batch_size']).long(),
            "obsrs":torch.zeros(hyps['batch_size'],*env.shape),
            "loc_targs":torch.zeros(hyps['batch_size'],2),
            "obj_targs":torch.zeros(hyps['batch_size'],2).long()
            }
    shared_data = {k:v.share_memory_().cuda() for k,v in\
                                     shared_data.items()}

    gate_q = mp.Queue(hyps['n_runs'])
    stop_q = mp.Queue(hyps['n_runs'])
    end_q = mp.Queue(1)

    # Make Runners
    hyps['n_runners'] = try_key(hyps,'n_runners',None)
    if hyps['n_runners'] is None:
        hyps['n_runners'] = hyps['n_runs']
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
                runner.run(model, multi_proc=False)

            # Collect data from runners
            rews = shared_data['rews']
            hs = shared_data['hs']
            dones = shared_data['dones']
            starts = shared_data['starts']
            obsrs = shared_data['obsrs']
            loc_targs = shared_data['loc_targs']
            obj_targs = shared_data['obj_targs']
            color_targs,shape_targs = obj_targs[:,0],obj_targs[:,1]

            # Make predictions
            if try_key(hyps,"use_bptt",False):
                pred_tup = bptt(hyps,model,obsrs,hs,dones)
                #pred_tup = bptt(hyps,model,obsrs,starts)
            else:
                pred_tup = model(obsrs.cuda(), hs.cuda())
            loc_preds,color_preds,shape_preds = pred_tup

            # Calc Losses
            post_obj_preds = try_key(hyps,'post_obj_preds',False)
            loss_tup = calc_losses(loc_preds,color_preds,shape_preds,
                                   loc_targs,color_targs,shape_targs,
                                   starts,dones,
                                   post_obj_preds=post_obj_preds)
            loc_loss,color_loss,shape_loss,color_acc,shape_acc=loss_tup
            obj_loss = (color_loss + shape_loss)/2
            obj_acc = ((color_acc + shape_acc)/2)
            if obj_recog:
                loss = alpha*loc_loss + (1-alpha)*obj_loss
            else:
                loss = loc_loss

            back_loss = loss / hyps['n_loss_loops']
            back_loss.backward()

            avg_loss +=      loss.item()
            avg_rew +=       rews.mean()
            avg_loc_loss +=  loc_loss.item()
            avg_obj_loss +=  obj_loss.item()
            avg_color_loss+= color_loss.item()
            avg_shape_loss+= shape_loss.item()
            avg_color_acc += color_acc.item()
            avg_shape_acc += shape_acc.item()
            avg_obj_acc   += obj_acc.item()

            if rollout % hyps['n_loss_loops'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            s = "LocL:{:.5f} | Obj:{:.5f} | {:.0f}% | t:{:.2f}"
            s = s.format(loc_loss.item(), obj_loss.item(),
                                   rollout/hyps['n_rollouts']*100,
                                   time.time()-iter_start)
            print(s, end=len(s)*" " + "\r")
            if hyps['exp_name'] == "test" and rollout>=3: break
        print()
        train_avg_loss = avg_loss / hyps['n_rollouts']
        train_loc_loss = avg_loc_loss / hyps['n_rollouts']
        train_color_loss = avg_color_loss / hyps['n_rollouts']
        train_shape_loss = avg_shape_loss / hyps['n_rollouts']
        train_obj_loss = avg_obj_loss / hyps['n_rollouts']
        train_color_acc = avg_color_acc / hyps['n_rollouts']
        train_shape_acc = avg_shape_acc / hyps['n_rollouts']
        train_obj_acc = avg_obj_acc / hyps['n_rollouts']
        train_avg_rew = avg_rew / hyps['n_rollouts']

        s = "Train - Loss:{:.5f} | Loc:{:.5f} | Rew:{:.5f}\n"
        s += "Obj Loss:{:.5f} | Obj Acc:{:.5f}\n"
        stats_string = s.format(train_avg_loss, train_loc_loss,
                                                train_avg_rew,
                                                train_obj_loss,
                                                train_obj_acc)
        scheduler.step(train_avg_loss)

        print("Evaluating")
        done = True
        model.eval()
        val_runner.model = model
        with torch.no_grad():
            loss_tup = val_runner.rollout(0,validation=True,n_tsteps=200)
            loss_tup = [x.item() for x in loss_tup]
            val_loc_loss,val_color_loss,val_shape_loss=loss_tup[:3]
            val_color_acc,val_shape_acc,val_rew=loss_tup[3:]
            val_obj_loss = ((val_color_loss + val_shape_loss)/2)
            val_obj_acc = ((val_color_acc + val_shape_acc)/2)
            val_loss = alpha*val_loc_loss + (1-alpha)*val_obj_loss

        stats_string += "Evaluation Avg Rew: {:.5f}\n".format(val_rew)

        optimizer.zero_grad()
        save_dict = {
            "epoch":epoch,
            "hyps":hyps,

            "train_loss":train_avg_loss,
            "train_loc_loss":train_loc_loss,
            "train_color_loss": train_color_loss,
            "train_shape_loss": train_shape_loss,
            "train_obj_loss":train_obj_loss,
            "train_color_acc": train_color_acc,
            "train_shape_acc": train_shape_acc,
            "train_obj_acc":train_obj_acc,

            "val_loss":val_loss,
            "val_loc_loss":val_loc_loss,
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
    end_q.put(1)
    if len(runners) > 1:
        # Start the runners
        for i in range(hyps['n_runners']):
            gate_q.put(i)
        for proc in procs:
            proc.join()
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

    #def rollout(self, idx, validation=False, n_tsteps=None):
    #    """
    #    rollout handles the actual rollout of the environment for
    #    n steps in time. It is called from run and performs a single
    #    rollout, placing the collected data into the shared lists
    #    found in the datas dict.

    #    idx: int
    #        identification number distinguishing the portion of the
    #        shared array designated for this runner
    #    validation: bool
    #        if true, this runner is used as a validation runner
    #    n_tsteps: int
    #        number of steps to take in this rollout
    #    """
    #    hyps = self.hyps
    #    obj_recog = try_key(hyps,'obj_recog',False)
    #    n_tsteps = hyps['n_tsteps'] if n_tsteps is None else n_tsteps

    #    self.model.reset_h(batch_size=1)
    #    if self.prev_h is None:
    #        obs,targ = self.env.reset()
    #    else:
    #        self.model.h = self.prev_h
    #        obs = self.prev_obs
    #        targ = self.prev_targ
    #    obs = obs.cuda()

    #    obsrs = []
    #    hs = []
    #    targs = []
    #    rews  = []
    #    dones = []
    #    starts = []
    #    loc_preds = []
    #    color_preds = []
    #    shape_preds = []

    #    with torch.no_grad():
    #        while len(rews) < n_tsteps:
    #            hs.append(self.model.h)
    #            obsrs.append(obs)

    #            tup = self.model(obs[None])
    #            pred,color_pred,shape_pred = tup

    #            obs,targ,rew,done,_ = self.env.step(pred)
    #            obs = obs.cuda()

    #            loc_preds.append(pred)
    #            if obj_recog:
    #                color_preds.append(color_pred)
    #                shape_preds.append(shape_pred)

    #            targs.append(targ)
    #            rews.append(rew)
    #            dones.append(done)
    #            starts.append(0)

    #            if done:
    #                obs,targ = self.env.reset()
    #                self.model.reset_h()
    #                obs = obs.cuda()
    #                starts[-1] = 1

    #    self.prev_h = self.model.h
    #    self.prev_obs = obs
    #    self.prev_targ = targ

    #    rews = torch.FloatTensor(rews).cuda()
    #    hs = torch.vstack(hs).cuda()
    #    dones = torch.LongTensor(dones).cuda()
    #    starts = torch.LongTensor(starts).cuda()
    #    obsrs = torch.stack(obsrs)
    #    targs = torch.stack(targs).cuda()
    #    loc_targs,obj_targs = targs[:,:2],targs[:,2:].long()

    #    if not validation:
    #        # Send data to main proc
    #        startx = idx*n_tsteps
    #        endx = (idx+1)*n_tsteps
    #        self.shared_data['rews'][startx:endx] = rews
    #        self.shared_data['hs'][startx:endx] = hs
    #        self.shared_data['dones'][startx:endx] = dones
    #        self.shared_data['starts'][startx:endx] = starts
    #        self.shared_data['obsrs'][startx:endx] = obsrs
    #        self.shared_data['loc_targs'][startx:endx] = loc_targs
    #        self.shared_data['obj_targs'][startx:endx] = obj_targs

    #    if validation:
    #        loc_preds = torch.vstack(loc_preds)
    #        if len(color_preds) > 0:
    #            color_preds = torch.vstack(color_preds)
    #            shape_preds = torch.vstack(shape_preds)
    #        color_targs,shape_targs = obj_targs[:,0],obj_targs[:,1]

    #        post_obj_preds = try_key(hyps,'post_obj_preds',False)
    #        loss_tup = calc_losses(loc_preds,color_preds,shape_preds,
    #                               loc_targs,color_targs,shape_targs,
    #                               starts,dones,
    #                               post_obj_preds=post_obj_preds)
    #        loc_loss,color_loss,shape_loss,color_acc,shape_acc=loss_tup

    #        return loc_loss,color_loss,shape_loss,\
    #                color_acc,shape_acc,rews.mean()

    def rollout(self, idx, validation=False, n_tsteps=None):
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
        """
        hyps = self.hyps
        obj_recog = try_key(hyps,'obj_recog',False)
        post_obj_preds = try_key(hyps,'post_obj_preds',False)
        n_tsteps = hyps['n_tsteps'] if n_tsteps is None else n_tsteps

        self.model.reset_h(batch_size=1)
        if self.prev_h is None or self.prev_done:
            self.prev_obs,self.prev_targ = self.env.reset()
            self.prev_rew = 0
            self.prev_done = 0
            self.prev_start = 1
        else:
            self.model.h = self.prev_h
        obs = self.prev_obs.cuda()

        obsrs = [obs]
        hs = [self.model.h]
        targs = [self.prev_targ]
        rews  = [self.prev_rew]
        dones = [0]
        starts = [1]

        loc_preds = []
        color_preds = []
        shape_preds = []

        with torch.no_grad():
            while len(rews) < n_tsteps:
                tup = self.model(obs[None])
                pred,color_pred,shape_pred = tup

                obs,targ,rew,done,_ = self.env.step(pred)
                start = 0
                done = int(done)
                obs = obs.cuda()

                loc_preds.append(pred)
                if obj_recog:
                    color_preds.append(color_pred)
                    shape_preds.append(shape_pred)

                obsrs.append(obs)
                hs.append(self.model.h)
                targs.append(targ)
                rews.append(rew)
                starts.append(start)
                dones.append(done)

                if done>0 and len(rews) < n_tsteps:
                    # Finish out last step
                    tup = self.model(obs[None])
                    pred,color_pred,shape_pred = tup
                    loc_preds.append(pred)
                    if obj_recog:
                        color_preds.append(color_pred)
                        shape_preds.append(shape_pred)

                    obs,targ = self.env.reset()
                    rew = 0
                    done = 0
                    start = 1
                    self.model.reset_h()
                    obs = obs.cuda()

                    obsrs.append(obs)
                    hs.append(self.model.h)
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
        loc_targs,obj_targs = targs[:,:2],targs[:,2:].long()

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

        if validation:
            tup = self.model(obs[None])
            pred,color_pred,shape_pred = tup
            loc_preds.append(pred)
            if obj_recog:
                color_preds.append(color_pred)
                shape_preds.append(shape_pred)

            loc_preds = torch.vstack(loc_preds)
            if len(color_preds) > 0:
                color_preds = torch.vstack(color_preds)
                shape_preds = torch.vstack(shape_preds)
            color_targs,shape_targs = obj_targs[:,0],obj_targs[:,1]

            post_obj_preds = try_key(hyps,'post_obj_preds',False)
            loss_tup = calc_losses(loc_preds,color_preds,shape_preds,
                                   loc_targs,color_targs,shape_targs,
                                   starts,dones,
                                   post_obj_preds=post_obj_preds)
            loc_loss,color_loss,shape_loss,color_acc,shape_acc=loss_tup

            return loc_loss,color_loss,shape_loss,\
                    color_acc,shape_acc,rews.mean()

def calc_losses(loc_preds,color_preds,shape_preds,
                loc_targs,color_targs,shape_targs,
                starts,dones,post_obj_preds=False):
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
    """
    d_idxs = (1-dones).bool()
    s_idxs = (1-starts).bool()

    # Loc Loss
    loc_preds = loc_preds[d_idxs]
    loc_targs = loc_targs[s_idxs]
    loc_loss = F.mse_loss(loc_preds.cuda(), loc_targs.cuda())

    if len(color_preds) > 0:
        if post_obj_preds:
            d_idxs = s_idxs
        color_preds = color_preds[d_idxs]
        shape_preds = shape_preds[d_idxs]
        color_targs = color_targs[s_idxs]
        shape_targs = shape_targs[s_idxs]

        color_loss = F.cross_entropy(color_preds, color_targs)
        shape_loss = F.cross_entropy(shape_preds, shape_targs)
        with torch.no_grad():
            maxes = torch.argmax(color_preds,dim=-1)
            color_acc = (maxes==color_targs).float().mean()
            maxes = torch.argmax(shape_preds,dim=-1)
            shape_acc = (maxes==shape_targs).float().mean()
    else:
        color_loss = torch.zeros(1).to(DEVICE)
        shape_loss = torch.zeros(1).to(DEVICE)
        color_acc = torch.zeros(1)
        shape_acc = torch.zeros(1)
    return loc_loss, color_loss, shape_loss, color_acc, shape_acc

def bptt(hyps, model, obsrs, hs, starts):
    """
    Used to include dependencies over time. It is assumed each rollout
    is of fixed length.

    R = Number of runs
    N = Number of steps per run

    obsrs: torch FloatTensor (R*N,C,H,W)
        MDP states at each timestep t
    hs: FloatTensor (R*N,H)
        Recurrent states at timestep t
    starts: torch LongTensor (R*N,)
        Binary array denoting what indices are new episodes
    """
    n_runs = hyps['n_runs']
    n_tsteps = hyps['n_tsteps']
    b_size = n_runs*n_tsteps
    assert len(obsrs) == b_size

    obsrs = obsrs.reshape(n_runs,n_tsteps,*obsrs.shape[1:])
    starts = starts.reshape(n_runs,n_tsteps,1)
    resets = 1-starts
    h_inits = model.reset_h(batch_size=n_runs)
    model.h = hs.reshape(n_runs,n_tsteps,-1)[:,0]
    loc_preds = []
    color_preds = []
    shape_preds = []
    for i in range(n_tsteps):
        obs = obsrs[:,i]
        h = model.h
        h = h*resets[:,i].data+h_inits.data*starts[:,i].data
        #if starts[:,i].sum() > 0:
        #    new_hs = []
        #    for j in range(len(starts)):
        #        new_h = h[j] if starts[j,i] < 1 else h_inits[j].data
        #        new_hs.append(new_h)
        #    h = torch.stack(new_hs)
        loc_pred,color_pred,shape_pred = model(obs,h)
        loc_preds.append(loc_pred)
        color_preds.append(color_pred)
        shape_preds.append(shape_pred)
    shape = (b_size, loc_pred.shape[-1])
    loc_preds = torch.stack(loc_preds,dim=1).reshape(shape)
    if len(color_pred) > 0:
        shape = (b_size,color_pred.shape[-1])
        color_preds = torch.stack(color_preds,dim=1).reshape(shape)
        shape = (b_size,shape_pred.shape[-1])
        shape_preds = torch.stack(shape_preds,dim=1).reshape(shape)
    else:
        color_preds,shape_preds = [],[]
    return loc_preds, color_preds, shape_preds

#def calc_losses(loc_preds,color_preds,shape_preds,
#                loc_targs,color_targs,shape_targs,
#                starts,dones,post_obj_preds=False):
#    """
#    loc_preds: FloatTensor (N,2)
#        the location predictions
#    color_preds: FloatTensor (N, N_COLORS)
#        the color predictions
#    shape_preds: FloatTensor (N, N_SHAPES)
#        the shape predictions
#    loc_targs: FloatTensor (N,2)
#        the location targets
#    color_targs: LongTensor (N,)
#        the color targets
#    shape_targs: LongTensor (N,)
#        the shape targets
#    dones: FloatTensor (N,)
#        the done signals
#    starts: FloatTensor (N,)
#        the start signals
#    post_obj_preds: bool
#        if the object predictions come after the movement
#    """
#    # Loc Loss
#    loc_loss = F.mse_loss(loc_preds.cuda(), loc_targs.cuda())
#
#    # Color Loss
#    if len(color_preds) > 0:
#        color_loss = F.cross_entropy(color_preds, color_targs)
#        shape_loss = F.cross_entropy(shape_preds, shape_targs)
#        with torch.no_grad():
#            maxes = torch.argmax(color_preds,dim=-1)
#            color_acc = (maxes==color_targs).float().mean()
#            maxes = torch.argmax(shape_preds,dim=-1)
#            shape_acc = (maxes==shape_targs).float().mean()
#    else:
#        color_loss = torch.zeros(1).cuda()
#        shape_loss = torch.zeros(1).cuda()
#        color_acc = torch.zeros(1)
#        shape_acc = torch.zeros(1)
#    return loc_loss, color_loss, shape_loss, color_acc, shape_acc

#def bptt(hyps, model, obsrs, hs, dones):
#    """
#    Used to include dependencies over time. It is assumed each rollout
#    is of fixed length.
#
#    R = Number of runs
#    N = Number of steps per run
#
#    obsrs: torch FloatTensor (R*N,C,H,W)
#        MDP states at each timestep t
#    hs: FloatTensor (R*N,H)
#        Recurrent states at timestep t
#    dones: torch LongTensor (R*N,)
#        Binary array denoting what indices are new episodes
#    """
#    n_runs = hyps['n_runs']
#    n_tsteps = hyps['n_tsteps']
#    b_size = n_runs*n_tsteps
#    assert len(obsrs) == b_size
#
#    # Reshape by run (Run, Tsteps, ..)
#    obsrs = obsrs.reshape(n_runs,n_tsteps,*obsrs.shape[1:])
#    dones = dones.reshape(n_runs,n_tsteps,1).float()
#    resets = (1-dones).float()
#    h_inits = model.reset_h(batch_size=n_runs)
#    h = hs.reshape(n_runs,n_tsteps,-1)[:,0]
#    loc_preds = []
#    color_preds = []
#    shape_preds = []
#    for i in range(n_tsteps):
#        obs = obsrs[:,i]
#        loc_pred,color_pred,shape_pred = model(obs,h)
#        loc_preds.append(loc_pred)
#        color_preds.append(color_pred)
#        shape_preds.append(shape_pred)
#        #h = h*resets[:,i].data+h_inits.data*dones[:,i].data
#        h = model.h
#        if dones[:,i].sum() > 0:
#            new_hs = []
#            for j in range(len(dones)):
#                new_h = h[j] if dones[j,i] < 1 else h_inits[j].data
#                new_hs.append(new_h)
#            h = torch.stack(new_hs)
#    shape = (b_size, loc_pred.shape[-1])
#    loc_preds = torch.stack(loc_preds,dim=1).reshape(shape)
#    if len(color_pred) > 0:
#        shape = (b_size,color_pred.shape[-1])
#        color_preds = torch.stack(color_preds,dim=1).reshape(shape)
#        shape = (b_size,shape_pred.shape[-1])
#        shape_preds = torch.stack(shape_preds,dim=1).reshape(shape)
#    else:
#        color_preds,shape_preds = [],[]
#    return loc_preds, color_preds, shape_preds

