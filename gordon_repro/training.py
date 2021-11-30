import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import PoissonNLLLoss, MSELoss
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,MultiStepLR
import time
from tqdm import tqdm
import torch.multiprocessing as mp
import ml_utils.save_io as io
from ml_utils.training import get_exp_num, record_session, get_save_folder, get_resume_checkpt
from ml_utils.utils import try_key, load_json
import gordon_repro.models as models
import gordon_repro.environments as environments
from gordon_repro.experience import ExperienceReplay
import matplotlib.pyplot as plt
from datetime import datetime
from torch.distributions import kl_divergence, Normal

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
    print("Saving to", hyps['save_folder'])
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
    model.cuda()
    model.share_memory()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'],
                                           weight_decay=hyps['l2'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                                                    patience=6,
                                                    verbose=True)
    fwd_dynamics = try_key(hyps,'use_fwd_dynamics',True) and countOut
    fwd_model = None
    if fwd_dynamics:
        fwd_model = getattr(models, hyps['fwd_class'])(**hyps)
        fwd_model.cuda()
        fwd_model.share_memory()
        fwd_optim = torch.optim.RMSprop(fwd_model.parameters(),
                                     lr=hyps['fwd_lr'],
                                     weight_decay=hyps['fwd_l2'])
        fwd_scheduler = ReduceLROnPlateau(fwd_optim, 'min', factor=0.5,
                                                     patience=6,
                                                     verbose=True)
        exp_replay = ExperienceReplay(max_size=hyps['exp_size'])

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
    if hyps['n_runs'] >= hyps['batch_size']:
        hyps['batch_size'] = 2*hyps['n_runs']
    hyps['n_tsteps'] = hyps['batch_size']//hyps['n_runs']
    # The total number of steps included in the update
    hyps['batch_size'] = hyps['n_tsteps']*hyps['n_runs']
    shared_data = {
            'obsrs':     torch.zeros(hyps['batch_size'],*env.shape),
            'rews':      torch.zeros(hyps['batch_size']),
            "hs":   torch.zeros(hyps['batch_size'],model.h_shape[-1]),
            "fwd_hs":   torch.zeros(hyps['batch_size']),
            "loc_targs": torch.zeros(hyps['batch_size'],2),
            "count_idxs":torch.zeros(hyps['batch_size']).long(),
            "longs":     torch.zeros(hyps['batch_size'],5).long(),
            #"color_idxs": idx 0
            #"shape_idxs": idx 1
            #"starts":     idx 2
            #"dones":      idx 3
            #"resets":     idx 4
            }
    if fwd_dynamics:
        shared_data['fwd_hs'] = torch.zeros(hyps['batch_size'],
                                              fwd_model.h_shape[-1])
    shared_data = {k:v.share_memory_() for k,v in shared_data.items()}
    shared_data = {k:v.cuda() for k,v in shared_data.items()}
    shared_data['obsrs'] = shared_data['obsrs'].cpu()

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
            args = (model,True,fwd_model)
            proc = mp.Process(target=runners[i].run, args=args)
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
    if not os.path.exists("./imgs"):
        os.mkdir("./imgs")

    if hyps['exp_name'] == "test":
        hyps['n_epochs'] = 2
        hyps['fwd_epochs'] = 2
    epoch = -1 if checkpt is None else checkpt['epoch']

    alpha = try_key(hyps,'alpha',.5)
    rew_alpha = try_key(hyps,'rew_alpha',.9)
    obj_recog = try_key(hyps,'obj_recog',False)
    best_val_rew = -np.inf
    fwd_hs = None
    print()
    # Start the runners
    for i in range(hyps['n_runs']):
        gate_q.put(i)
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

        first_avg_loc_loss = 0
        first_avg_obj_loss = 0
        first_avg_color_loss = 0
        first_avg_color_acc = 0
        first_avg_shape_loss = 0
        first_avg_shape_acc = 0
        first_avg_rew_loss = 0
        first_avg_obj_acc = 0

        last_avg_loc_loss = 0
        last_avg_obj_loss = 0
        last_avg_color_loss = 0
        last_avg_color_acc = 0
        last_avg_shape_loss = 0
        last_avg_shape_acc = 0
        last_avg_rew_loss = 0
        last_avg_obj_acc = 0

        model.train()
        print("Training...")
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        # Collect new rollouts
        done = False
        for rollout in range(hyps['n_rollouts']):
            iter_start = time.time()
            if len(runners) > 1:
                # Wait for all runners to stop
                for i in range(hyps['n_runs']):
                    stop_q.get()

            # Collect data from runners
            rews = shared_data['rews']
            hs = shared_data['hs']
            obsrs = shared_data['obsrs']
            loc_targs = shared_data['loc_targs']
            count_idxs = shared_data['count_idxs']
            #"color_idxs": idx 0
            #"shape_idxs": idx 1
            #"starts":     idx 2
            #"dones":      idx 3
            #"resets":     idx 4
            color_idxs = shared_data['longs'][:,0]
            shape_idxs = shared_data['longs'][:,1]
            starts =     shared_data['longs'][:,2]
            dones =      shared_data['longs'][:,3]
            resets =     shared_data['longs'][:,4]

            # Make predictions
            if try_key(hyps,"use_bptt",False):
                pred_tup = bptt(hyps=hyps,model=model,obsrs=obsrs,
                                          hs=hs,
                                          dones=dones,
                                          color_idxs=color_idxs,
                                          shape_idxs=shape_idxs,
                                          count_idxs=count_idxs)
            else:
                pred_tup = model(obsrs.cuda(), h=hs.cuda(),
                                               color_idx=color_idxs,
                                               shape_idx=shape_idxs,
                                               count_idx=count_idxs)
            loc_preds,color_preds,shape_preds,rew_preds = pred_tup

            # Calc Losses
            post_obj_preds = try_key(hyps,'post_obj_preds',False)
            post_rew_preds = try_key(hyps,'post_rew_preds',False)
            loss_tup = calc_losses(loc_preds=loc_preds,
                                   color_preds=color_preds,
                                   shape_preds=shape_preds,
                                   rew_preds=rew_preds,
                                   loc_targs=loc_targs,
                                   color_targs=color_idxs,
                                   shape_targs=shape_idxs,
                                   rew_targs=rews,
                                   starts=starts,dones=dones,
                                   post_obj_preds=post_obj_preds,
                                   post_rew_preds=post_rew_preds,
                                   hyps=hyps, firsts=resets)
            loc_loss,color_loss,shape_loss,rew_loss = loss_tup[:4]
            color_acc,shape_acc = loss_tup[4:6]
            first_loc_loss,first_color_loss=loss_tup[6:8]
            first_shape_loss, first_rew_loss = loss_tup[8:10]
            first_color_acc,first_shape_acc = loss_tup[10:12]
            last_loc_loss,last_color_loss=loss_tup[12:14]
            last_shape_loss, last_rew_loss = loss_tup[14:16]
            last_color_acc,last_shape_acc = loss_tup[16:18]

            loss = rew_alpha*loc_loss + (1-rew_alpha)*rew_loss
            obj_loss = (color_loss + shape_loss)/2
            obj_acc = ((color_acc + shape_acc)/2)
            first_obj_loss = (first_color_loss + first_shape_loss)/2
            first_obj_acc = ((first_color_acc +  first_shape_acc)/2)
            last_obj_loss = (last_color_loss + last_shape_loss)/2
            last_obj_acc = ((last_color_acc +  last_shape_acc)/2)
            loss = alpha*loss + (1-alpha)*obj_loss

            back_loss = loss / hyps['n_loss_loops']
            back_loss.backward()

            if fwd_dynamics:
                exp_replay.add_data(shared_data)

            # Start the runners again so they collect in the background
            if len(runners) > 1:
                for i in range(hyps['n_runs']):
                    gate_q.put(i)
            else:
                runner.run(model, multi_proc=False)

            avg_loss            += loss.item()
            avg_rew             += rews.mean()
            avg_loc_loss        += loc_loss.item()
            avg_obj_loss        += obj_loss.item()
            avg_color_loss      += color_loss.item()
            avg_shape_loss      += shape_loss.item()
            avg_rew_loss        += rew_loss.item()
            avg_color_acc       += color_acc.item()
            avg_shape_acc       += shape_acc.item()
            avg_obj_acc         += obj_acc.item()
            first_avg_loc_loss  += first_loc_loss.item()
            first_avg_obj_loss  += first_obj_loss.item()
            first_avg_color_loss+= first_color_loss.item()
            first_avg_shape_loss+= first_shape_loss.item()
            first_avg_rew_loss  += first_rew_loss.item()
            first_avg_color_acc += first_color_acc.item()
            first_avg_shape_acc += first_shape_acc.item()
            first_avg_obj_acc   += first_obj_acc.item()
            last_avg_loc_loss  += last_loc_loss.item()
            last_avg_obj_loss  += last_obj_loss.item()
            last_avg_color_loss+= last_color_loss.item()
            last_avg_shape_loss+= last_shape_loss.item()
            last_avg_rew_loss  += last_rew_loss.item()
            last_avg_color_acc += last_color_acc.item()
            last_avg_shape_acc += last_shape_acc.item()
            last_avg_obj_acc   += last_obj_acc.item()

            if rollout % hyps['n_loss_loops'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            s = "LocL:{:.5f} | Obj:{:.5f} | {:.0f}% | t:{:.2f}"
            s = s.format(loc_loss.item(), obj_loss.item(),
                                   rollout/hyps['n_rollouts']*100,
                                   time.time()-iter_start)
            print(s, end=len(s)//4*" " + "\r")
            if hyps['exp_name'] == "test" and rollout>=2: break
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

        first_train_loc_loss = first_avg_loc_loss / hyps['n_rollouts']
        first_train_color_loss=first_avg_color_loss / hyps['n_rollouts']
        first_train_shape_loss=first_avg_shape_loss / hyps['n_rollouts']
        first_train_rew_loss = first_avg_rew_loss / hyps['n_rollouts']
        first_train_obj_loss = first_avg_obj_loss / hyps['n_rollouts']
        first_train_color_acc =first_avg_color_acc / hyps['n_rollouts']
        first_train_shape_acc =first_avg_shape_acc / hyps['n_rollouts']
        first_train_obj_acc =  first_avg_obj_acc / hyps['n_rollouts']

        last_train_loc_loss = last_avg_loc_loss / hyps['n_rollouts']
        last_train_color_loss=last_avg_color_loss / hyps['n_rollouts']
        last_train_shape_loss=last_avg_shape_loss / hyps['n_rollouts']
        last_train_rew_loss = last_avg_rew_loss / hyps['n_rollouts']
        last_train_obj_loss = last_avg_obj_loss / hyps['n_rollouts']
        last_train_color_acc =last_avg_color_acc / hyps['n_rollouts']
        last_train_shape_acc =last_avg_shape_acc / hyps['n_rollouts']
        last_train_obj_acc =  last_avg_obj_acc / hyps['n_rollouts']

        s = "Train- Loss:{:.5f} | Loc:{:.5f} | Rew:{:.5f}\n"
        s +="Train- Obj Loss:{:.5f} | Obj Acc:{:.5f}\n"
        stats_string = s.format(train_avg_loss, train_loc_loss,
                                                train_avg_rew,
                                                train_obj_loss,
                                                train_obj_acc)
        # Sample images
        rand = int(np.random.randint(0,len(obsrs)))
        obs = obsrs[rand].permute(1,2,0).cpu().data.numpy()/6+0.5
        plt.imsave("imgs/sample"+str(epoch)+".png", obs)

        # Fwd dynamics loss
        train_fwd_loss,train_obs_loss,train_state_loss = 0,0,0
        train_state_pred_loss,train_over_loss = 0,0
        if fwd_dynamics:
            print("Calculating Fwd Loss")
            model.cpu()
            fwd_model.cuda()
            fwd_model.train()
            tup = fwd_train_loop(hyps, fwd_model, fwd_optim,
                                                  exp_replay,
                                                  verbose=True)
            model.cuda()
            train_obs_loss,train_state_loss = tup[:2]
            train_state_pred_loss,train_over_loss,obs_preds = tup[2:5]
            train_fwd_loss = train_obs_loss + train_state_loss
            train_fwd_loss += train_state_pred_loss
            s = "Train- FwdObs: {:.5f} | FwdState: {:.5f}\n"
            stats_string += s.format(train_obs_loss,
                                     train_state_loss)
            s = "Train- FwdPred:{:.5f} | FwdOver:{:.5f}\n"
            stats_string += s.format(train_state_pred_loss,
                                     train_over_loss)

            # Sample images
            obs_preds = obs_preds.reshape(-1,*obs_preds.shape[-3:])
            rand = int(np.random.randint(0,len(obs_preds)))
            #obs = preds[rand].permute(1,2,0).cpu().data.numpy()/6+0.5
            obs = obs_preds[rand].permute(1,2,0).cpu().data.numpy()
            if not try_key(hyps,'end_sigmoid', False): obs = obs/6+0.5
            obs = np.clip(obs, 0, 1)
            path = os.path.join(hyps['save_folder'],
                               "pred_sample"+str(epoch)+".png")
            plt.imsave(path, obs)
        print("Evaluating")
        done = True
        model.eval()
        val_runner.model = model
        val_runner.fwd_model = DummyFwdModel() if fwd_model is None\
                                               else fwd_model
        with torch.no_grad():
            loss_tup = val_runner.rollout(0,validation=True,n_tsteps=200)
            loss_tup = [x.item() for x in loss_tup]
            val_loc_loss,val_color_loss,val_shape_loss = loss_tup[:3]
            val_rew_loss,val_color_acc,val_shape_acc = loss_tup[3:6]
            val_rew,val_fwd_loss = loss_tup[6:8]

            first_val_loc_loss,first_val_color_loss = loss_tup[8:10]
            first_val_shape_loss,first_val_rew_loss = loss_tup[10:12]
            first_val_color_acc,first_val_shape_acc = loss_tup[12:14]
            val_obs_loss,val_state_loss = loss_tup[14:16]
            val_state_pred_loss,val_over_loss = loss_tup[16:18]
            last_val_loc_loss,last_val_color_loss = loss_tup[18:20]
            last_val_shape_loss,last_val_rew_loss = loss_tup[20:22]
            last_val_color_acc,last_val_shape_acc = loss_tup[22:24]

            val_obj_loss = ((val_color_loss + val_shape_loss)/2)
            val_obj_acc = ((val_color_acc + val_shape_acc)/2)
            first_val_obj_loss = ((first_val_color_loss+\
                                   first_val_shape_loss)/2)
            first_val_obj_acc = ( (first_val_color_acc +\
                                   first_val_shape_acc)/2)
            last_val_obj_loss = ((last_val_color_loss+\
                                   last_val_shape_loss)/2)
            last_val_obj_acc = ( (last_val_color_acc +\
                                   last_val_shape_acc)/2)
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
            "train_obs_loss":train_obs_loss,
            "train_state_loss":train_state_loss,
            "train_state_pred_loss":train_state_pred_loss,
            "train_over_loss":train_over_loss,

            "first_train_loc_loss":   first_train_loc_loss,
            "first_train_color_loss": first_train_color_loss,
            "first_train_shape_loss": first_train_shape_loss,
            "first_train_rew_loss":   first_train_rew_loss,
            "first_train_obj_loss":   first_train_obj_loss,
            "first_train_color_acc":  first_train_color_acc,
            "first_train_shape_acc":  first_train_shape_acc,
            "first_train_obj_acc":    first_train_obj_acc,

            "last_train_loc_loss":   last_train_loc_loss,
            "last_train_color_loss": last_train_color_loss,
            "last_train_shape_loss": last_train_shape_loss,
            "last_train_rew_loss":   last_train_rew_loss,
            "last_train_obj_loss":   last_train_obj_loss,
            "last_train_color_acc":  last_train_color_acc,
            "last_train_shape_acc":  last_train_shape_acc,
            "last_train_obj_acc":    last_train_obj_acc,

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
            "val_obs_loss":val_obs_loss,
            "val_state_loss":val_state_loss,
            "val_state_pred_loss":val_state_pred_loss,
            "val_over_loss":val_over_loss,

            "first_val_loc_loss":   first_val_loc_loss,
            "first_val_color_loss": first_val_color_loss,
            "first_val_shape_loss": first_val_shape_loss,
            "first_val_rew_loss":   first_val_rew_loss,
            "first_val_obj_loss":   first_val_obj_loss,
            "first_val_color_acc":  first_val_color_acc,
            "first_val_shape_acc":  first_val_shape_acc,
            "first_val_obj_acc":    first_val_obj_acc,

            "last_val_loc_loss":   last_val_loc_loss,
            "last_val_color_loss": last_val_color_loss,
            "last_val_shape_loss": last_val_shape_loss,
            "last_val_rew_loss":   last_val_rew_loss,
            "last_val_obj_loss":   last_val_obj_loss,
            "last_val_color_acc":  last_val_color_acc,
            "last_val_shape_acc":  last_val_shape_acc,
            "last_val_obj_acc":    last_val_obj_acc,

            "train_rew":train_avg_rew,
            "val_rew":val_rew,
            "state_dict":model.state_dict(),
            "optim_dict":optimizer.state_dict(),
        }
        if fwd_dynamics:
            save_dict['fwd_state_dict'] = fwd_model.state_dict()
            save_dict['fwd_optim_dict'] = fwd_optim.state_dict()
        if epoch == int(hyps['n_epochs']//2):
            save_name = "halfway"
            save_name = os.path.join(hyps['save_folder'],save_name)
            io.save_checkpt(save_dict, save_name, epoch, ext=".pt",
                                       del_prev_sd=False, best=False)
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
    if len(runners) > 1:
        # Wait for all runners to stop
        for i in range(hyps['n_runs']):
            stop_q.get()
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

class DummyFwdModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.h = torch.zeros(1)

    def reset_h(self,batch_size=1):
        self.h = torch.zeros(batch_size)
        return self.h

    def decode(self,*args,**kwargs):
        return None

    def forward(self,*args,**kwargs):
        return self.h.data,self.h.data,self.h.data,self.h.data,\
                                                   self.h.data

class Runner:
    def __init__(self, rank, hyps, shared_data, gate_q, stop_q, end_q):
        """
        rank: int
            the id of the runner
        hyps: dict
            dict of hyperparams
        shared_data: dict of shared tensors
            keys: str
                'rews':      shared tensor
                "hs":        shared tensor
                "loc_targs": shared tensor
                "count_idxs": shared tensor
                "longs": shared tensor
                    #"color_idxs": idx 0
                    #"shape_idxs": idx 1
                    #"starts":     idx 2
                    #"dones":      idx 3
                    #"resets":     idx 4
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
        self.fwd_h = None

    def run(self, model, multi_proc=True, fwd_model=None):
        """
        Call this function for starting the process. If the class is
        being used in the same process as it is being created, set
        multi_proc to false.
        
        model: torch Module
        multi_proc: bool
            If the class is being used in the same process as it is
            being created, set multi_proc to false. Otherwise true.
        fwd_model: torch Module (optional)
            model that makes forward state predictions
        """
        self.model = model
        self.fwd_model = DummyFwdModel() if fwd_model is None\
                                         else fwd_model
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
        rew_recog = try_key(hyps,'rew_recog',False)
        post_obj_preds = try_key(hyps,'post_obj_preds',False)
        post_rew_preds = try_key(hyps,'post_rew_preds',False)
        n_tsteps = hyps['n_tsteps'] if n_tsteps is None else n_tsteps

        self.model.eval()
        self.fwd_model.eval()

        self.model.reset_h(batch_size=1)
        self.fwd_model.reset_h(batch_size=1)
        # Prev h will only be None if this is the first rollout of the
        # training. If we ended on a done in the last session, the env
        # hasn't been restarted yet. So, we can reset here.
        if self.prev_h is None or self.prev_done:
            self.prev_obs,self.prev_targ = self.env.reset()
            self.prev_rew = 0
            self.prev_done = 0
            self.prev_start = 1
            resets = [1]
        else:
            self.model.h = self.prev_h
            self.fwd_model.h = self.fwd_h
            resets = [0]
        obs = self.prev_obs.cuda()

        obsrs = [obs]
        hs = [self.model.h]
        fwd_hs = [self.fwd_model.h]
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
                    count_idx = torch.LongTensor(temp[:,4:5]).cuda()
                else:
                    count_idx = None
                tup = self.model(obs[None], None, color_idx.cuda(),
                                                  shape_idx.cuda(),
                                                  count_idx)
                pred,color_pred,shape_pred,rew_pred = tup
                _ = self.fwd_model(obs[None],h=None,
                                             color_idx=color_idx.cuda(),
                                             shape_idx=shape_idx.cuda(),
                                             count_idx=count_idx)

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
                fwd_hs.append(self.fwd_model.h)
                resets.append(0)
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
                        count_idx = torch.LongTensor(temp[:,4:5]).cuda()
                    else:
                        count_idx = None
                    tup = self.model(obs[None], None, color_idx.cuda(),
                                                      shape_idx.cuda(),
                                                      count_idx)
                    pred,color_pred,shape_pred,rew_pred = tup
                    loc_preds.append(pred)
                    if len(color_pred)>0:
                        color_preds.append(color_pred)
                        shape_preds.append(shape_pred)
                    if rew_recog:
                        rew_preds.append(rew_pred)

                    _ = self.fwd_model(obs[None], h=None,
                                       color_idx=color_idx.cuda(),
                                       shape_idx=shape_idx.cuda(),
                                       count_idx=count_idx)

                    obs,targ = self.env.reset()
                    rew = 0
                    done = 0
                    start = 1
                    self.model.reset_h()
                    self.fwd_model.reset_h()
                    obs = obs.cuda()

                    obsrs.append(obs)
                    hs.append(self.model.h)
                    fwd_hs.append(self.fwd_model.h)
                    resets.append(1)
                    targs.append(targ)
                    rews.append(rew)
                    starts.append(start)
                    dones.append(done)
        dones[-1] = 1

        self.prev_h = self.model.h
        self.fwd_h = self.fwd_model.h
        self.prev_obs = obs
        self.prev_targ = targ
        self.prev_rew = rew
        self.prev_done = int(done)
        self.prev_start = start

        rews = torch.FloatTensor(rews).cuda()
        hs = torch.vstack(hs).cuda()
        fwd_hs = torch.vstack(fwd_hs).cuda().squeeze()
        dones = torch.LongTensor(dones).cuda()
        starts = torch.LongTensor(starts).cuda()
        obsrs = torch.stack(obsrs)
        targs = torch.stack(targs).cuda()
        resets = torch.LongTensor(resets).cuda()
        loc_targs = targs[:,:2]
        color_idxs,shape_idxs = targs[:,2].long(), targs[:,3].long()
        count_idxs = None
        if targs.shape[1] > 4:
            count_idxs = targs[:,4].long()

        if not validation:
            #"color_idxs": idx 0
            #"shape_idxs": idx 1
            #"starts":     idx 2
            #"dones":      idx 3
            #"resets":     idx 4
            cat_arr = [color_idxs,
                       shape_idxs,
                       starts,
                       dones,
                       resets]
            longs = torch.stack(cat_arr,dim=1)
            # Send data to main proc
            startx = idx*n_tsteps
            endx = (idx+1)*n_tsteps
            self.shared_data['rews'][startx:endx] = rews
            self.shared_data['hs'][startx:endx] = hs
            self.shared_data['fwd_hs'][startx:endx] = fwd_hs
            self.shared_data['obsrs'][startx:endx] = obsrs
            self.shared_data['loc_targs'][startx:endx] = loc_targs
            self.shared_data['longs'][startx:endx] = longs
            if count_idxs is not None:
                self.shared_data['count_idxs'][startx:endx] = count_idxs

        if validation:
            color_idx = color_idxs[-1:]
            shape_idx = shape_idxs[-1:]
            count_idx = None
            if count_idxs is not None:
                count_idx = count_idxs[-1:].cuda()
            tup = self.model(obs[None], None, color_idx.cuda(),
                                              shape_idx.cuda(),
                                              count_idx)
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

            loss_tup = calc_losses(loc_preds=loc_preds,
                                   color_preds=color_preds,
                                   shape_preds=shape_preds,
                                   rew_preds=rew_preds,
                                   loc_targs=loc_targs,
                                   color_targs=color_idxs,
                                   shape_targs=shape_idxs,
                                   rew_targs=rews,
                                   starts=starts, dones=dones,
                                   post_obj_preds=post_obj_preds,
                                   post_rew_preds=post_rew_preds,
                                   hyps=self.hyps,
                                   firsts=resets)
            loc_loss,color_loss,shape_loss,rew_loss = loss_tup[:4]
            color_acc,shape_acc = loss_tup[4:6]
            first_loc_loss,first_color_loss = loss_tup[6:8]
            first_shape_loss, first_rew_loss = loss_tup[8:10]
            first_color_acc,first_shape_acc = loss_tup[10:12]
            last_loc_loss,last_color_loss = loss_tup[12:14]
            last_shape_loss, last_rew_loss = loss_tup[14:16]
            last_color_acc,last_shape_acc = loss_tup[16:18]

            fwd_loss = torch.zeros(1)
            obs_loss = torch.zeros(1)
            state_loss = torch.zeros(1)
            state_pred_loss = torch.zeros(1)
            over_loss = torch.zeros(1)
            if not isinstance(self.fwd_model,DummyFwdModel):
                # Nones are required to make batch size of 1
                data = {"obs_seq":   obsrs[None].clone(),
                        "h_seq":     fwd_hs[None].clone(),
                        "start_seq": starts[None].clone(),
                        "reset_seq": resets[None].clone(),
                        "color_seq": color_idxs[None].clone(),
                        "shape_seq": shape_idxs[None].clone(),
                        "count_seq": count_idxs[None].clone()}
                self.fwd_model.cuda()
                tup = fwd_preds(hyps, self.fwd_model, data=data)
                self.fwd_model.cpu()
                obs_preds,hs,mus,sigmas,mu_preds,sigma_preds = tup
                if try_key(hyps,'end_sigmoid',False):
                    data['obs_seq'] = data['obs_seq']/6+0.5
                obs_loss,state_loss,state_pred_loss = calc_fwd_loss(
                                               obs_preds=obs_preds,
                                               mu_truths=mus,
                                               sigma_truths=sigmas,
                                               mu_preds=mu_preds,
                                               sigma_preds=sigma_preds,
                                               data=data)
                fwd_loss = obs_loss + state_loss + state_pred_loss
                over_loss = torch.zeros(1)
                if try_key(hyps,"overshoot",False):
                    self.fwd_model.cuda()
                    tup = fwd_preds(hyps,self.fwd_model,
                                    data=data,overshoot=True)
                    _,_,_,_,mu_preds,sigma_preds = tup
                    self.fwd_model.cpu()
                    over_loss = calc_overshoot_loss(mu_truths=mus,
                                             sigma_truths=sigmas,
                                             mu_preds=mu_preds,
                                             sigma_preds=sigma_preds)

            return loc_loss,color_loss,shape_loss,rew_loss,\
                    color_acc,shape_acc,rews.mean(),fwd_loss,\
                    first_loc_loss,first_color_loss,first_shape_loss,\
                    first_rew_loss,first_color_acc,first_shape_acc,\
                    obs_loss,state_loss,state_pred_loss,over_loss,\
                    last_loc_loss,last_color_loss,last_shape_loss,\
                    last_rew_loss,last_color_acc,last_shape_acc

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
    torch.cuda.empty_cache()
    smooth_movement = False
    if hyps is not None:
        smooth_movement = hyps["float_params"]["smoothMovement"]
    d_idxs = (1-dones).bool()
    s_idxs = (1-starts).bool()

    # Loc Loss
    l_preds = loc_preds[d_idxs]
    l_targs = loc_targs[s_idxs]
    # TODO HEADS UP: Added a multiplcation factor of 10
    loc_loss = 10*F.mse_loss(l_preds.cuda(), l_targs.cuda())
    if firsts is not None and not smooth_movement:
        assert hyps is not None, "if using firsts, must argue hyps"
        n_runs = hyps['n_runs']
        n_tsteps = hyps['n_tsteps']
        b_size = n_runs*n_tsteps
        # Case of validation
        if len(d_idxs) != b_size:
            n_runs = 1
            n_tsteps = len(d_idxs)
        firsts = firsts.reshape(n_runs,n_tsteps).clone().bool()
        lasts = firsts.clone()

        # First Move Calculations
        firsts[:,-1] = 0 # don't care about ending firsts
        roll = torch.roll(firsts,shifts=1,dims=1).clone().bool()
        firsts = firsts.reshape(-1)
        roll = roll.reshape(-1)
        with torch.no_grad():
            l_preds = loc_preds[firsts]
            l_targs = loc_targs[roll]
            first_loc_loss = 10*F.mse_loss(l_preds.cuda(),
                                           l_targs.cuda())
        # Last Move Calculations
        lasts[:,0] = 0 # can't look behind starting firsts
        lastroll = torch.roll(lasts,shifts=-1,dims=1).clone().bool()
        # don't care about starting lasts (this shouldn't happen anyway)
        lastroll[:,0] = 0
        lasts = torch.roll(lastroll,shifts=-1,dims=1).clone().bool()
        lasts = lasts.reshape(-1)
        lastroll = lastroll.reshape(-1)
        with torch.no_grad():
            l_preds = loc_preds[lasts]
            l_targs = loc_targs[lastroll]
            last_loc_loss = 10*F.mse_loss(l_preds.cuda(),
                                          l_targs.cuda())
    else:
        first_loc_loss = torch.zeros(1).cuda()
        last_loc_loss = torch.zeros(1).cuda()

    if len(color_preds) > 0:
        idxs = d_idxs
        if post_obj_preds:
            idxs = s_idxs
        c_preds = color_preds[idxs].squeeze().cuda()
        s_preds = shape_preds[idxs].squeeze().cuda()
        c_targs = color_targs[s_idxs].squeeze().cuda()
        s_targs = shape_targs[s_idxs].squeeze().cuda()

        color_loss = F.cross_entropy(c_preds, c_targs)
        shape_loss = F.cross_entropy(s_preds, s_targs)
        with torch.no_grad():
            maxes = torch.argmax(c_preds,dim=-1)
            color_acc = (maxes==c_targs).float().mean()
            maxes = torch.argmax(s_preds,dim=-1)
            shape_acc = (maxes==s_targs).float().mean()
        if firsts is not None and not smooth_movement:
            with torch.no_grad():
                # Firsts
                if post_obj_preds:
                    c_preds = color_preds[roll].squeeze().cuda()
                    s_preds = shape_preds[roll].squeeze().cuda()
                else:
                    c_preds = color_preds[firsts].squeeze().cuda()
                    s_preds = shape_preds[firsts].squeeze().cuda()
                c_targs = color_targs[roll].squeeze().cuda()
                s_targs = shape_targs[roll].squeeze().cuda()
                if c_targs.nelement() > 0:
                    first_color_loss = F.cross_entropy(c_preds, c_targs)
                    first_shape_loss = F.cross_entropy(s_preds, s_targs)
                    maxes = torch.argmax(c_preds,dim=-1).long()
                    first_color_acc = (maxes==c_targs).float().mean()
                    maxes = torch.argmax(s_preds,dim=-1).long()
                    first_shape_acc = (maxes==s_targs).float().mean()
                else:
                    first_color_loss = torch.zeros(1)
                    first_shape_loss = torch.zeros(1)
                    first_color_acc = torch.zeros(1)
                    first_shape_acc = torch.zeros(1)

                # Lasts
                # lastroll is one step ahead of lasts
                if post_obj_preds:
                    c_preds = color_preds[lastroll].squeeze().cuda()
                    s_preds = shape_preds[lastroll].squeeze().cuda()
                else:
                    c_preds = color_preds[lasts].squeeze().cuda()
                    s_preds = shape_preds[lasts].squeeze().cuda()
                c_targs = color_targs[lastroll].squeeze().cuda()
                s_targs = shape_targs[lastroll].squeeze().cuda()
                if c_targs.nelement() > 0:
                    last_color_loss = F.cross_entropy(c_preds, c_targs)
                    last_shape_loss = F.cross_entropy(s_preds, s_targs)
                    maxes = torch.argmax(c_preds,dim=-1).long()
                    last_color_acc = (maxes==c_targs).float().mean()
                    maxes = torch.argmax(s_preds,dim=-1).long()
                    last_shape_acc = (maxes==s_targs).float().mean()
                else:
                    last_color_loss = torch.zeros(1)
                    last_shape_loss = torch.zeros(1)
                    last_color_acc = torch.zeros(1)
                    last_shape_acc = torch.zeros(1)
        else:
            first_color_loss = torch.zeros(1)
            first_shape_loss = torch.zeros(1)
            first_color_acc = torch.zeros(1)
            first_shape_acc = torch.zeros(1)
            last_color_loss = torch.zeros(1)
            last_shape_loss = torch.zeros(1)
            last_color_acc = torch.zeros(1)
            last_shape_acc = torch.zeros(1)
    else:
        color_loss = torch.zeros(1).cuda()
        shape_loss = torch.zeros(1).cuda()
        color_acc = torch.zeros(1)
        shape_acc = torch.zeros(1)
        first_color_loss = torch.zeros(1)
        first_shape_loss = torch.zeros(1)
        first_color_acc = torch.zeros(1)
        first_shape_acc = torch.zeros(1)
        last_color_loss = torch.zeros(1)
        last_shape_loss = torch.zeros(1)
        last_color_acc = torch.zeros(1)
        last_shape_acc = torch.zeros(1)

    if len(rew_preds) > 0:
        idxs = d_idxs
        if post_rew_preds:
            idxs = s_idxs
        r_preds = rew_preds[d_idxs]
        r_targs = rew_targs[s_idxs]
        rew_loss = F.mse_loss(r_preds.squeeze().cuda(),
                              r_targs.squeeze().cuda())
        if firsts is not None and not smooth_movement:
            with torch.no_grad():
                # Firsts
                if post_obj_preds:
                    r_preds = rew_preds[roll]
                else:
                    r_preds = rew_preds[firsts]
                r_targs = rew_targs[roll]
                if r_targs.nelement() > 0:
                    first_rew_loss = F.mse_loss(r_preds.squeeze().cuda(),
                                      r_targs.squeeze().cuda())
                else:
                    first_rew_loss = torch.zeros(1)
                # Lasts
                if post_obj_preds:
                    r_preds = rew_preds[lastroll]
                else:
                    r_preds = rew_preds[lasts]
                r_targs = rew_targs[lastroll]
                if r_targs.nelement() > 0:
                    last_rew_loss = F.mse_loss(r_preds.squeeze().cuda(),
                                      r_targs.squeeze().cuda())
                else:
                    last_rew_loss = torch.zeros(1)
        else:
            first_rew_loss = torch.zeros(1)
            last_rew_loss = torch.zeros(1)
    else:
        rew_loss = torch.zeros(1).cuda()
        first_rew_loss = torch.zeros(1).cuda()
        last_rew_loss = torch.zeros(1)

    return loc_loss,color_loss,shape_loss,rew_loss,color_acc,shape_acc,\
            first_loc_loss,first_color_loss,first_shape_loss,\
            first_rew_loss,first_color_acc,first_shape_acc,\
            last_loc_loss,last_color_loss,last_shape_loss,\
            last_rew_loss,last_color_acc,last_shape_acc,


def bptt(hyps, model, obsrs, hs, dones, color_idxs, shape_idxs,
                                                    count_idxs):
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
    color_idxs: long tensor (R*N,1)
        the color indexes
    shape_idxs: long tensor (R*N,1)
        the shape indexes
    count_idxs: long tensor (R*N,1)
        the indices of the number of objects to touch
    """
    torch.cuda.empty_cache()
    n_runs = hyps['n_runs']
    n_tsteps = hyps['n_tsteps']
    b_size = n_runs*n_tsteps
    assert len(obsrs) == b_size

    obsrs = obsrs.reshape(n_runs,n_tsteps,*obsrs.shape[1:])
    dones = dones.reshape(n_runs,n_tsteps,1)
    resets = 1-dones
    h_inits = model.reset_h(batch_size=n_runs)
    model.h = hs.reshape(n_runs,n_tsteps,-1)[:,0]
    color_idxs = color_idxs.reshape(n_runs,n_tsteps,1)
    shape_idxs = shape_idxs.reshape(n_runs,n_tsteps,1)
    count_idxs = count_idxs.reshape(n_runs,n_tsteps,1)
    loc_preds = []
    color_preds = []
    shape_preds = []
    rew_preds = []
    h = model.h
    for i in range(n_tsteps):
        obs = obsrs[:,i]
        color_idx = color_idxs[:,i]
        shape_idx = shape_idxs[:,i]
        count_idx = count_idxs[:,i]
        loc_pred,color_pred,shape_pred,rew_pred = model(obs.cuda(), 
                                            h.cuda(),
                                            color_idx=color_idx.cuda(),
                                            shape_idx=shape_idx.cuda(),
                                            count_idx=count_idx.cuda())
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
                new_h = h[j] if dones[j,i] < 1 else h_inits[j]
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

def fwd_train_loop(hyps,fwd_model,fwd_optim,exp_replay,verbose=False):
    """
    This function performs a training loop to train the fwd_dynamics
    model.

    hyps: dict of hyperparameters
    fwd_model: Module
        must have a decode function
    fwd_optim: Optimizer
        for the forward model parameters
    exp_replay: ExperienceReplay object
        this holds all the data to be trained on
    """
    torch.cuda.empty_cache()
    fwd_model.train()
    grad_norm = try_key(hyps,'fwd_grad_norm',None)
    horizon = hyps['fwd_horizon']
    bsize = hyps['fwd_bsize']
    n_loops = len(exp_replay)//bsize
    total_obs_loss = 0
    total_state_loss = 0
    total_state_pred_loss = 0
    total_over_loss = 0
    for epoch in range(hyps['fwd_epochs']):
        perm = torch.randperm(len(exp_replay)-horizon-1)
        avg_obs_loss = 0
        avg_state_loss = 0
        avg_state_pred_loss = 0
        avg_over_loss = 0
        iter_start = time.time()
        for b in range(n_loops):
            idxs = perm[b*bsize:(b+1)*bsize]
            data = exp_replay.get_data(idxs, horizon=horizon)
            tup = fwd_preds(hyps, fwd_model, data=data)
            obs_preds,hs,mus,sigmas,mu_preds,sigma_preds = tup
            exp_replay.update_hs(idxs, hs.data)
            if try_key(hyps,'end_sigmoid',False):
                data['obs_seq'] = data['obs_seq']/6+0.5
            obs_loss,state_loss,state_pred_loss = calc_fwd_loss(
                                                obs_preds=obs_preds,
                                                mu_truths=mus,
                                                sigma_truths=sigmas,
                                                mu_preds=mu_preds,
                                                sigma_preds=sigma_preds,
                                                data=data)
            fwd_loss = obs_loss + state_loss + state_pred_loss
            fwd_loss.backward()
            over_loss = torch.zeros(1)
            if try_key(hyps,'overshoot',False):
                tup = fwd_preds(hyps,fwd_model,data,overshoot=True)
                _,_,_,_,mu_preds,sigma_preds = tup
                over_loss = calc_overshoot_loss(mu_truths=mus,
                                         sigma_truths=sigmas,
                                         mu_preds=mu_preds,
                                         sigma_preds=sigma_preds)
                over_loss.backward()

            if grad_norm is not None and grad_norm > 0:
                params = fwd_optim.param_groups[0]['params']
                nn.utils.clip_grad_norm_(params, grad_norm, norm_type=2)
            fwd_optim.step()
            fwd_optim.zero_grad()
            avg_obs_loss += obs_loss.item()
            avg_state_loss += state_loss.item()
            avg_state_pred_loss += state_pred_loss.item()
            avg_over_loss += over_loss.item()
            if verbose:
                s ="Obs:{:.5f} | State:{:.5f} | Pred:{:.5f} "
                s += "| Over:{:.5f} | {:.0f}% | t:{:.2f}"
                percent = (epoch*n_loops+b)/(hyps['fwd_epochs']*n_loops)
                s = s.format(obs_loss.item(), state_loss.item(),
                                       state_pred_loss.item(),
                                       over_loss.item(),
                                       percent*100,
                                       time.time()-iter_start)
                print(s, end=len(s)//4*" " + "\r")
            if hyps['exp_name']=="test" and b > 1: 
                break
        arr = [data['obs_seq'][0,1].cpu(),obs_preds[0,1].cpu()]
        temp = torch.cat(arr,dim=-1).permute(1,2,0).data.numpy()
        if try_key(hyps,'end_sigmoid',False):
            plt.imsave("imgs/debug.png", np.clip(temp,0,1))
        else:
            plt.imsave("imgs/debug.png", np.clip(temp/6+0.5,0,1))
        total_obs_loss += avg_obs_loss/n_loops
        total_state_loss += avg_state_loss/n_loops
        total_state_pred_loss += avg_state_pred_loss/n_loops
        total_over_loss += avg_over_loss/n_loops
    if verbose:
        s ="Fwd Obs: {:.5f} | State: {:.5f} | {:.0f}% | t:{:.2f}"
        s = s.format(avg_obs_loss, avg_state_loss,
                               epoch/hyps['fwd_epochs']*100,
                               time.time()-iter_start)
        print(s, end=len(s)//4*" " + "\r")
        print()
    total_obs_loss = total_obs_loss/hyps['fwd_epochs']
    total_state_loss = total_state_loss/hyps['fwd_epochs']
    total_state_pred_loss = total_state_pred_loss/hyps['fwd_epochs']
    total_over_loss = total_over_loss/hyps['fwd_epochs']
    return total_obs_loss, total_state_loss, total_state_pred_loss,\
                           total_over_loss,  obs_preds.data.cpu()

def fwd_preds(hyps, fwd_model, data, overshoot=False):
    """
    Used to include dependencies over time. It is assumed each rollout
    is of fixed length.

    B = batch size
    S = horizon length

    data: dict
        obs_seq: torch FloatTensor (B,S,C,H,W)
            MDP states at each timestep t
        h_seq: FloatTensor (B,S,H)
            Recurrent states at timestep t
        start_seq: torch LongTensor (B,S)
            Binary array denoting the indices in which the stored h
            vector should be used
        reset_seq: torch LongTensor (B,S)
            Binary array denoting the indices in which a fresh h should be
            used
        color_seq: long tensor (B,S)
            the color indexes
        shape_seq: long tensor (B,S)
            the shape indexes
        count_seq: long tensor (B,S)
            the indices of the number of objects to touch
    overshoot: bool
        if true, overshot state predictions are returned as mu_truth
        and sigma_truth. All other returns are none. Remember that
        when overshooting, all mu and sigma predictions are shifted
        over one in the prediction direction!!
    """
    torch.cuda.empty_cache()
    obs_seq =   data['obs_seq'].data
    h_seq =     data['h_seq'].data
    resets =    data['reset_seq'].data
    starts =    data['start_seq'].data
    color_seq = data['color_seq'].data
    shape_seq = data['shape_seq'].data
    count_seq = data['count_seq'].data

    obs_preds = []
    fwd_model.reset_h(batch_size=len(starts))
    hs = []
    mus = []
    sigmas = []
    mu_preds = []
    sigma_preds = []
    mu_pred = None
    sigma_pred = None
    resets = (resets.bool()|starts.bool()).float()
    h = h_seq[:,0].data # (B,H)
    for i in range(obs_seq.shape[1]):
        # Here we set the h vector to the stored h vector if the start
        # value is 1. This is because starts denote a new segment of
        # rollout but do not necessarily denote the start of a rollout.
        # resets do, however, mark the start of a new rollout, so if
        # reset is true, we want to get the gradient update for the
        # h_init vector. Sadly, we will have stale h vectors for many
        # rollouts, but this is the best we can do. Setting the max
        # experience replay size can help cycle data through to avoid
        # stale h vectors
        if resets[:,i].sum() > 0:
            new_hs = []
            for j in range(len(resets)):
                if resets[j,i] > 1:
                    new_h = h_seq[j,i].data
                else:
                    new_h = h[j]
                new_hs.append(new_h)
            h = torch.stack(new_hs)
        hs.append(h.cpu())
        if overshoot:
            h,mu,sigma,mu_pred,sigma_pred=fwd_model(obs_seq[:,i].cuda(),
                                      h=h.cuda(),
                                      color_idx=color_seq[:,i].cuda(),
                                      shape_idx=shape_seq[:,i].cuda(),
                                      count_idx=count_seq[:,i].cuda(),
                                      prev_mu=mu_pred,
                                      prev_sigma=sigma_pred,
                                      resets=resets[:,i].cuda())
        else:
            h,mu,sigma,mu_pred,sigma_pred=fwd_model(obs_seq[:,i].cuda(),
                                      h=h.cuda(),
                                      color_idx=color_seq[:,i].cuda(),
                                      shape_idx=shape_seq[:,i].cuda(),
                                      count_idx=count_seq[:,i].cuda())
        mu_preds.append(mu_pred)
        sigma_preds.append(sigma_pred)
        if not overshoot:
            mus.append(mu)
            sigmas.append(sigma)
    mu_preds = torch.stack(mu_preds,dim=1)
    sigma_preds = torch.stack(sigma_preds,dim=1)
    if not overshoot:
        hs.append(h.cpu())
        hs = torch.stack(hs,dim=1)
        mus = torch.stack(mus,dim=1)
        sigmas = torch.stack(sigmas,dim=1)
        B,S = mus.shape[:2]

        # Obs preds are made for same time step as the true states
        mu_flat = mus.reshape(-1,mus.shape[-1])
        sigma_flat = sigmas.reshape(-1,mus.shape[-1])
        s = models.sample_s(mu_flat,sigma_flat)
        h_flat = hs[:,1:].reshape(-1,hs.shape[-1])
        obs_preds = fwd_model.decode(s,h_flat.cuda())
        obs_preds = obs_preds.reshape(B,S,*obs_preds.shape[1:])
        hs = hs[:,:-1]
    return obs_preds, hs, mus, sigmas, mu_preds, sigma_preds

def calc_fwd_loss(obs_preds, mu_truths, sigma_truths, mu_preds,
                                                      sigma_preds,
                                                      data):
    """
    A function to calculate the fwd dynamics loss

    obs_preds: torch Float Tensor (B,S,C,H,W)
        decoded observation predictions
    mu_truths: torch Float Tensor (B,S,E)
    sigma_truths: torch Float Tensor (B,S,E)
    mu_preds: torch Float Tensor (B,S,E)
    sigma_preds: torch Float Tensor (B,S,E)
    data: dict
        "obs_seq":      torch float tensor (B,S,C,H,W)
        "rew_seq:       torch float tensor (B,S)
        "count_seq":    torch long tensor  (B,S)
        "color_seq":    torch long tensor  (B,S)
        "shape_seq":    torch long tensor  (B,S)
        "done_seq":     torch long tensor  (B,S)
        "reset_seq":    torch long tensor  (B,S)
    """
    torch.cuda.empty_cache()
    obs_targs = data['obs_seq'].cuda()
    obs_loss = F.mse_loss(obs_preds.cuda(), obs_targs)

    normal = Normal(torch.zeros_like(mu_truths),
                    torch.ones_like(sigma_truths))
    true_normal = Normal(mu_truths,sigma_truths)
    state_loss  = kl_divergence(true_normal, normal).mean()

    # Must shift preds and truths by 1 space so that they align
    N = mu_truths.shape[0]*(mu_truths.shape[1]-1)
    mu_truths =    mu_truths[:,1:].reshape(N,-1)
    sigma_truths = sigma_truths[:,1:].reshape(N,-1)
    mu_preds =     mu_preds[:,:-1].reshape(N,-1)
    sigma_preds =  sigma_preds[:,:-1].reshape(N,-1)
    true_normal_nograd = Normal(mu_truths.data,sigma_truths.data)
    pred_normal = Normal(mu_preds,sigma_preds)
    state_pred_loss=kl_divergence(pred_normal,true_normal_nograd).mean()

    return obs_loss, state_loss, state_pred_loss

def calc_overshoot_loss(mu_truths, sigma_truths, mu_preds, sigma_preds):
    """
    A function to calculate the fwd dynamics loss

    mu_truths: torch Float Tensor (B,S,E)
    sigma_truths: torch Float Tensor (B,S,E)
    mu_preds: torch Float Tensor (B,S,E)
    sigma_preds: torch Float Tensor (B,S,E)
    """
    torch.cuda.empty_cache()
    # Must shift preds and truths by 1 space so that they align
    N = mu_truths.shape[0]*(mu_truths.shape[1]-1)
    mu_truths =    mu_truths[:,1:].reshape(N,-1)
    sigma_truths = sigma_truths[:,1:].reshape(N,-1)
    mu_preds =     mu_preds[:,:-1].reshape(N,-1)
    sigma_preds =  sigma_preds[:,:-1].reshape(N,-1)
    true_normal_nograd = Normal(mu_truths.data,sigma_truths.data)
    pred_normal = Normal(mu_preds,sigma_preds)
    overshoot_loss=kl_divergence(pred_normal, true_normal_nograd).mean()
    return overshoot_loss
