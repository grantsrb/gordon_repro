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
env_channel.set_float_parameter("egoCentered", 1)

env = UnityToGymWrapper(env, allow_multiple_obs=True)
obs = env.reset()
print("initl targ:", obs[1])
plt.imshow(obs[0])
plt.show()
done = False
while True:
    print("stepping")
    x,z = [float(y.strip()) for y in str(input("action: ")).split(",")]
    # The obs is a list of length 2 in which the first element is the image and the second is the goal coordinate
    # Reward in this case is the difference between the action location and the nearest object to the action location
    obs, rew, done, _ = env.step([x,z])
    print("targ:", obs[1])
    print("rew:", rew)
    print("done:", done)
    plt.imshow(obs[0])
    plt.show()
    if done:
        obs = env.reset()
        print("resetting")
        plt.imshow(obs[0])
        plt.show()


