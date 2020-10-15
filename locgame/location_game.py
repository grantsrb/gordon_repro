import time
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os

game_path = os.path.expanduser("~/countsort_data/LocationGameLinux.x86_64")
seed = 0
channel = EngineConfigurationChannel()
env_channel = EnvironmentParametersChannel()
env = UnityEnvironment(file_name=game_path, side_channels=[channel,env_channel], seed=seed)
channel.set_configuration_parameters(time_scale = 1)
env_channel.set_float_parameter("validation", 0)
env_channel.set_float_parameter("egoCentered", 0)

env = UnityToGymWrapper(env, allow_multiple_obs=True)
print("wrapped env")
obs = env.reset()
print("entering loop")
done = False
while True:
    print("stepping")
    x,z = np.random.random()*2-1, np.random.random()*2-1
    # The obs is a list of length 2 in which the first element is the image and the second is the goal coordinate
    # Reward in this case is the difference between the action location and the nearest object to the action location
    obs, rew, done, _ = env.step([x,z])
    print(obs[0])
    print(obs[1])
    print("rew:", rew)
    print("done:", done)
    time.sleep(1)
    if done:
        obs = env.reset()
        print("resetting")


