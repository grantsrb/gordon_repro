import os
import numpy as np
from skimage.color import rgb2grey
import time
from ml_utils.utils import try_key
from a2c.utils import next_state, sample_action, cuda_if
import torch
import gym
import torch.nn.functional as F
from collections import deque
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

class GymEnv:
    def __init__(self, env_name, prep_fxn, seed=int(time.time()),
                                           worker_id=None,
                                           float_params=dict(),
                                           **kwargs):
        """
        env_name: str
            the name of the environment
        prep_fxn: str
            the name of the preprocessing function to be used on each
            of the observations
        seed: int
            the random seed for the environment
        worker_id: int
            must specify a unique worker id for each unity process
            on this machine
        float_params: dict or None
            this should be a dict of argument settings for the unity
            environment
            keys: varies by environment
        """
        self.env_name = env_name
        self.prep_fxn = globals()[prep_fxn]
        self.seed = seed
        self.worker_id = worker_id
        self.float_params = float_params

        self.env = gym.make(self.env_name)
        obs,action_targ = self.reset()
        self.shape = obs.shape
        self.targ_shape = action_targ.shape
        # Discrete action spaces are not yet implemented
        self.is_discrete = True

    def prep_obs(self, obs):
        """
        obs: list or ndarray
            the observation returned by the environment
        """
        obs = self.prep_fxn(obs.transpose(2,0,1))
        return [torch.from_numpy(obs),torch.zeros(4)]

    def reset(self):
        obs = self.env.reset()
        return self.prep_obs(obs)

    def step(self,pred):
        """
        action: list, vector, or int
            the action to take in this step. type can vary depending
            on the environment type
        """
        action = self.get_action(pred)
        obs,rew,done,info = self.env.step(action)
        obs,targ = self.prep_obs(obs)
        targ[:2] = np.clip(targ[:2],-1,1)
        return obs, targ, rew, done, info

    def get_action(self, preds):
        """
        Action data types can vary from evnironment to environment.
        This function handles converting outputs from the model
        to actions of the appropriate form for the environment.

        preds: torch tensor (..., N)
            the outputs from the model
        """
        if self.is_discrete:
            probs = F.softmax(preds, dim=-1)
            action = sample_action(probs.data)
            return int(action.item())
        else:
            preds = preds.squeeze().cpu().data.numpy()
            return preds

    def close(self):
        self.env.close()

class UnityGymEnv:
    def __init__(self, env_name, prep_fxn, seed=int(time.time()),
                                           worker_id=None,
                                           float_params=dict(),
                                           **kwargs):
        """
        env_name: str
            the name of the environment
        prep_fxn: str
            the name of the preprocessing function to be used on each
            of the observations
        seed: int
            the random seed for the environment
        worker_id: int
            must specify a unique worker id for each unity process
            on this machine
        float_params: dict or None
            this should be a dict of argument settings for the unity
            environment
            keys: varies by environment
        """
        self.env_name = env_name
        self.prep_fxn = globals()[prep_fxn]
        self.seed = seed
        self.worker_id = worker_id
        self.float_params = float_params

        self.env = self.make_unity_env(env_name,
                                       seed=self.seed,
                                       worker_id=self.worker_id,
                                       float_params=float_params,
                                       **kwargs)
        obs,action_targ = self.reset()
        self.shape = obs.shape
        self.targ_shape = action_targ.shape
        # Discrete action spaces are not yet implemented
        self.is_discrete = False

    def make_unity_env(self, env_name, float_params=dict(), time_scale=1,
                                                      seed=time.time(),
                                                      worker_id=None,
                                                      **kwargs):
        """
        creates a gym environment from a unity game

        env_name: str
            the path to the game
        float_params: dict or None
            this should be a dict of argument settings for the unity
            environment
            keys: varies by environment
        time_scale: float
            argument to set Unity's time scale. This applies less to
            gym wrapped versions of Unity Environments, I believe..
            but I'm not sure
        seed: int
            the seed for randomness
        worker_id: int
            must specify a unique worker id for each unity process
            on this machine
        """
        seed = int(seed)
        if float_params is None: float_params = dict()
        path = os.path.expanduser(env_name)
        channel = EngineConfigurationChannel()
        env_channel = EnvironmentParametersChannel()
        channel.set_configuration_parameters(time_scale = 1)
        for k,v in float_params.items():
            if k=="validation" and v>=1:
                print("Game in validation mode")
            env_channel.set_float_parameter(k, float(v))
        if worker_id is None: worker_id = seed%500+1
        env_made = False
        n_loops = 0
        while not env_made and n_loops < 50:
            try:
                env = UnityEnvironment(file_name=path,
                                   side_channels=[channel,env_channel],
                                   worker_id=int(worker_id),
                                   seed=int(seed))
                env_made = True
                env = UnityToGymWrapper(env, allow_multiple_obs=True)
            except:
                s = "Error encountered making environment, "
                s += "trying new worker_id"
                print(s)
                worker_id =int(worker_id+1+np.random.random()*100)%500
                try: env.close()
                except: pass
                n_loops += 1
        return env

    def prep_obs(self, obs):
        """
        obs: list or ndarray
            the observation returned by the environment
        """
        if not isinstance(obs, list):
            obs = obs.transpose(2,0,1)
            return torch.from_numpy(self.prep_fxn(obs))
        prepped_obs = self.prep_fxn(obs[0].transpose(2,0,1))
        # Handles the additional observations passed by the env
        prepped_obs = [prepped_obs, *obs[1:]]
        prepped_obs = [torch.from_numpy(x) for x in prepped_obs]
        return prepped_obs

    def reset(self):
        obs = self.env.reset()
        return self.prep_obs(obs)

    def step(self,pred):
        """
        action: list, vector, or int
            the action to take in this step. type can vary depending
            on the environment type
        """
        action = self.get_action(pred)
        obs,rew,done,info = self.env.step(action)
        obs,targ = self.prep_obs(obs)
        targ[:2] = np.clip(targ[:2],-1,1)
        return obs, targ, rew, done, info

    def get_action(self, preds):
        """
        Action data types can vary from evnironment to environment.
        This function handles converting outputs from the model
        to actions of the appropriate form for the environment.

        preds: torch tensor (..., N)
            the outputs from the model
        """
        if self.is_discrete:
            probs = F.softmax(preds, dim=-1)
            action = sample_action(probs.data)
            return int(action.item())
        else:
            preds = preds.squeeze().cpu().data.numpy()
            return preds
    
    def close(self):
        self.env.close()

def pong_prep(pic):
    pic = pic[35:195] # crop
    pic = pic[::2,::2,0] # downsample by factor of 2
    pic[pic == 144] = 0 # erase background (background type 1)
    pic[pic == 109] = 0 # erase background (background type 2)
    pic[pic != 0] = 1 # everything else (paddles, ball) just set to 1
    return pic[None]

def breakout_prep(pic):
    pic = pic[35:195,8:-8] # crop
    pic = pic[::2,::2,0] # downsample by factor of 2
    pic = rgb2grey(pic)
    return pic[None]

def snake_prep(pic):
    new_pic = np.zeros(pic.shape[:2],dtype=np.float32)
    new_pic[:,:][pic[:,:,0]==1] = 1
    new_pic[:,:][pic[:,:,0]==255] = 1.5
    new_pic[:,:][pic[:,:,1]==255] = 0
    new_pic[:,:][pic[:,:,2]==255] = .33
    pic = new_pic
    return new_pic[None]

def center_zero2one(obs):
    """
    obs: ndarray (C, H, W)
        values must range from 0-1
    """
    obs = obs.astype(np.float32)
    obs = 3*(obs-.5)/.5
    if len(obs.shape)==2:
        return obs[None]
    return obs

def null_prep(obs):
    obs = obs.astype(np.float32)
    if len(obs.shape)==2:
        return obs[None]
    return obs

def get_env(hyps):
    if hyps['env_name'][:4] == "gym:":
        og_name = hyps['env_name']
        del hyps['env_name']
        env_name = og_name.split(":")[-1].strip()
        env = GymEnv(env_name=env_name,**hyps)
        hyps['env_name'] = og_name
        return env
    return UnityGymEnv(**hyps)
