import Grid2OpOld
from Grid2OpOld.grid2op.Backend.pandaPowerBackend import PandaPowerBackend
from Grid2OpOld.grid2op.Reward.baseReward import BaseReward
from Grid2OpOld.grid2op.Reward.flatReward import FlatReward
import gym
from Grid2OpOld.grid2op.Environment.outage_env import OutageEnv

env = OutageEnv()


# import gymnasium as gym
n_episodes=10000
# env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

from stable_baselines3 import PPO
# model = PPO("MlpPolicy", env, n_steps=25, verbose=0, tensorboard_log="D:/jupyter/electric log")

model = PPO(env=env,
               learning_rate=0.0001,
               policy="MlpPolicy",
               policy_kwargs={"net_arch":[100,100,100]},# [256, 128,256]},
               n_steps=256, #256,
               batch_size=64,
               verbose=False,
               tensorboard_log='D:/jupyter/electric log'
               )

# Train the agent
model.learn(total_timesteps=10000,log_interval=1)

model.save("PPOOLD"+str(n_episodes))