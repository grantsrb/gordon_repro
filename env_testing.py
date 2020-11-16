import time
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import locgame.environments as environments
import locgame.save_io as io
import h5py as h5
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import os
import sys

if len(sys.argv) > 1:
    checkpt = io.load_checkpoint(sys.argv[1])
    hyps = checkpt['hyps']
    params = hyps['float_params']
    if "absoluteCoords" not in hyps['float_params']:
        params['absoluteCoords'] = float(not params["egoCentered"])
    seed = hyps['seed']
else:
    params = {"validation": 0,
              "egoCentered": 1,
              "absoluteCoords": 0,
              "smoothMovement": 0,
              "restrictCamera": 0}
    seed = 0

print("Seed:", seed)
print("Params:", params)
torch.manual_seed(seed)
np.random.seed(seed)

game_path = os.path.expanduser("~/loc_games/LocationGameLinux_1/LocationGameLinux.x86_64")
channel = EngineConfigurationChannel()
env_channel = EnvironmentParametersChannel()
env = UnityEnvironment(file_name=game_path,
                       side_channels=[channel,env_channel],
                       seed=seed)
channel.set_configuration_parameters(time_scale = 1)
for k,v in params.items():
    env_channel.set_float_parameter(k, v)
env = UnityToGymWrapper(env, allow_multiple_obs=True)

matplotlib.use("tkagg")
obs = env.reset()
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


